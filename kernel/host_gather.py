# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).

"""GPU-driven gather from PINNED HOST memory over UVA (zero-copy PCIe reads).

The B2-B4 flagship runs fully characterized expert prefetch: the predictor is
right (93% pre/post-attention router agreement), the bytes are nearly free
(1.07x), and every CPU-mediated way of ISSUING the copies loses — B3 paid ~94
added GPU->CPU syncs/token, B4 paid GIL contention (0.57x). The remaining
shape is this: the copy itself becomes a KERNEL, indexed by GPU-resident
expert ids, reading the pinned host store directly (cudaHostAlloc'd memory is
UVA-addressable from device code). No CPU knows the ids; nothing synchronizes.

``gather_expert_rows(dst, host_ptr, ids, have_ids=None)`` copies ``ids[j]``'s
row-block from the host stack into ``dst[j]`` for j in [0, k). When
``have_ids`` is given, slots where ``have_ids[j] == ids[j]`` are skipped —
that is the GPU-side miss-correction: no PCIe traffic for predicted hits.

The host pointer is passed as a plain int64 and bitcast to a pointer inside
the kernel (``tl.cast(..., tl.pointer_type(...))``); the 4-layer smoke run
on the pod validates this primitive before any expensive build.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gather_rows(
    dst_ptr,          # cuda, [k, row_words] int64-viewed destination
    src_addr,         # int64 scalar: host UVA base address of [E, row_words]
    ids_ptr,          # cuda int32 [k] — which expert each slot wants
    have_ptr,         # cuda int32 [k] — expert currently in the slot (-1 = none)
    row_words,        # words (int64 units) per expert row-block
    CHECK_HAVE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    slot = tl.program_id(0)
    chunk = tl.program_id(1)
    want = tl.load(ids_ptr + slot)
    if CHECK_HAVE:
        have = tl.load(have_ptr + slot)
        if have == want:
            return
    offs = chunk * BLOCK + tl.arange(0, BLOCK)
    mask = offs < row_words
    src = tl.cast(src_addr, tl.pointer_type(tl.int64))
    vals = tl.load(src + want.to(tl.int64) * row_words + offs, mask=mask)
    tl.store(dst_ptr + slot.to(tl.int64) * row_words + offs, vals, mask=mask)


def _as_words(t: torch.Tensor) -> int:
    nbytes = t[0].numel() * t[0].element_size()
    assert nbytes % 8 == 0, "row-block must be 8-byte aligned"
    return nbytes // 8


_NEG1 = {}


def _neg1(dev):
    key = str(dev)
    if key not in _NEG1:
        _NEG1[key] = torch.full((64,), -1, dtype=torch.int32, device=dev)
    return _NEG1[key]


def gather_expert_rows(dst: torch.Tensor, host: torch.Tensor, ids: torch.Tensor,
                       have: torch.Tensor | None = None, block: int = 2048):
    """dst [k, ...] cuda <- host [E, ...] pinned, rows selected by ids (cuda int32).

    have (cuda int32 [k], optional): current slot contents; matching slots are
    skipped (miss-correction mode). All launch parameters are id-INDEPENDENT,
    so the CPU never needs the ids: this call enqueues and returns."""
    k = dst.shape[0]
    row_words = _as_words(dst)
    assert _as_words(host) == row_words
    grid = (k, triton.cdiv(row_words, block))
    _gather_rows[grid](
        dst.view(torch.int64).view(k, -1),
        host.data_ptr(),
        ids,
        have if have is not None else _neg1(dst.device)[:k],
        row_words,
        CHECK_HAVE=have is not None,
        BLOCK=block,
        num_warps=4,
    )
