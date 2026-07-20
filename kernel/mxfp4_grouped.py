# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Grouped W4A16 GEMM over NATIVE MXFP4 expert stacks — the table-swap of
nf4_grouped (Phase-0 seam map; Phase-1 oracle-adjudicated). Computes, in one
launch, ``out[t] = a[t] @ dequant_mxfp4(B[e(t)]).T`` for tokens grouped by
expert.

Only the decode primitives differ from nf4_grouped, and only in the four ways
the seam map named (verify: `git diff` against the NF4 kernels shows exactly
these):
  1. codebook: FP4_VALUES (e2m1) instead of NF4_LUT — a different `lut` pointer.
  2. nibble interleave: element 2j = LOW nibble (`kk%2==0 -> blk & 0xF`),
     2j+1 = HIGH — OPPOSITE bnb/NF4 (Phase-1 oracle lock).
  3. scale: per-32 e8m0 byte -> `exp2(e - 127)` multiply, instead of per-64
     fp32 absmax. (real checkpoint scales are finite; the 0xFF ldexp edge from
     pack_ref cannot arise on GPU — guarded by the exact-decode gate.)
  4. block geometry: BLOCK_K = 32, group index g0 = k0 // 32.

The grouped-ragged mainloop, tiling, device-id calling convention, fp32
accumulation, and single bf16 epilogue downcast are byte-identical in shape to
nf4_grouped (R1: anchor, don't restructure). No split-K in v1 (correctness
first; starved-grid split is an occupancy optimization, added post-gate).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

# e2m1 codebook (verbatim transformers FP4_VALUES; Phase-1 verified).
FP4_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
              -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
MX_BLOCK = 32
E8M0_BIAS = 127

from nf4_grouped import build_group_tiles, _prefill_block_m  # reuse verbatim  # noqa: E402

_LUT_CACHE: dict = {}


def _lut(device):
    key = str(device)
    if key not in _LUT_CACHE:
        _LUT_CACHE[key] = torch.tensor(FP4_VALUES, dtype=torch.float32, device=device)
    return _LUT_CACHE[key]


@triton.jit
def _gemm_mxfp4_grouped(
    a_ptr, b_ptr, scale_ptr, out_ptr, lut_ptr,
    t_row0_ptr, t_rows_ptr, t_group_ptr, expert_ids_ptr,
    K, N,
    stride_be, stride_bn, stride_se, stride_sn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    VARIANT: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    row0 = tl.load(t_row0_ptr + pid_m)
    rows = tl.load(t_rows_ptr + pid_m)
    grp = tl.load(t_group_ptr + pid_m)
    eid = tl.load(expert_ids_ptr + grp)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < rows
    n_mask = offs_n < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    offs_k = tl.arange(0, BLOCK_K)
    b_base = b_ptr + eid * stride_be + offs_n[:, None] * stride_bn
    a_base = a_ptr + (row0 + offs_m)[:, None] * K
    if VARIANT == 1:
        lut_reg = tl.load(lut_ptr + tl.arange(0, 16))

    for k0 in range(0, K, BLOCK_K):
        kk = k0 + offs_k
        bytes_ = tl.load(b_base + (kk[None, :] // 2), mask=n_mask[:, None], other=0).to(tl.int32)
        # MXFP4: element 2j = LOW nibble, 2j+1 = HIGH (opposite bnb/NF4)
        nib = tl.where((kk[None, :] % 2) == 0, bytes_ & 0xF, (bytes_ >> 4) & 0xF)
        if VARIANT == 1:
            w = tl.reshape(tl.gather(lut_reg, tl.reshape(nib, [BLOCK_N * BLOCK_K]), 0),
                           [BLOCK_N, BLOCK_K])
        else:
            w = tl.load(lut_ptr + nib)
        g0 = k0 // BLOCK_K
        e8 = tl.load(scale_ptr + eid * stride_se + offs_n * stride_sn + g0,
                     mask=n_mask, other=0).to(tl.int32)
        scale = tl.exp2((e8 - 127).to(tl.float32))          # e8m0 -> 2^(e-127)
        w = w * scale[:, None]
        a = tl.load(a_base + kk[None, :], mask=m_mask[:, None], other=0.0).to(tl.float32)
        acc += tl.dot(a, tl.trans(w))

    out_ptrs = out_ptr + (row0 + offs_m)[:, None] * N + offs_n[None, :]
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def _gemv_mxfp4_grouped(
    a_ptr, b_ptr, scale_ptr, out_ptr, lut_ptr, expert_ids_ptr,
    K, N,
    stride_be, stride_bn, stride_se, stride_sn,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Decode reduction: one token per group (M==1). out[g,n] = sum_k a[g,k]*w[n,k]."""
    g = tl.program_id(0)
    pid_n = tl.program_id(1)
    eid = tl.load(expert_ids_ptr + g)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    offs_k = tl.arange(0, BLOCK_K)
    b_base = b_ptr + eid * stride_be + offs_n[:, None] * stride_bn
    a_base = a_ptr + g * K

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + offs_k
        bytes_ = tl.load(b_base + (kk[None, :] // 2), mask=n_mask[:, None], other=0).to(tl.int32)
        nib = tl.where((kk[None, :] % 2) == 0, bytes_ & 0xF, (bytes_ >> 4) & 0xF)
        w = tl.load(lut_ptr + nib)
        g0 = k0 // BLOCK_K
        e8 = tl.load(scale_ptr + eid * stride_se + offs_n * stride_sn + g0,
                     mask=n_mask, other=0).to(tl.int32)
        scale = tl.exp2((e8 - 127).to(tl.float32))
        a = tl.load(a_base + kk).to(tl.float32)
        acc += tl.sum(w * a[None, :], axis=1) * scale

    tl.store(out_ptr + g * N + offs_n, acc.to(tl.bfloat16), mask=n_mask)


def gemm_mxfp4_grouped(a_cat, blocks, scales, sizes, expert_ids,
                       block_m: int | None = None, prefill_variant: int | None = None):
    """Single-launch grouped MXFP4 GEMM. ``a_cat [T,K]`` bf16/fp16 group-sorted;
    ``blocks [E, N, K//2]`` uint8 (native gpt-oss blocks flattened);
    ``scales [E, N, K//32]`` uint8 (e8m0); ``sizes`` per-group token counts
    (all > 0); ``expert_ids [G]`` int32/list/device-tensor. Returns ``[T, N]``
    bf16, same group order. Decode (all sizes==1) uses the GEMV reduction."""
    E, N, _ = blocks.shape
    T, K = a_cat.shape
    assert sum(sizes) == T, (sum(sizes), T)
    assert scales.shape == (E, N, K // MX_BLOCK), (scales.shape, (E, N, K // MX_BLOCK))
    assert blocks.dtype == torch.uint8 and scales.dtype == torch.uint8
    dev = a_cat.device
    eids = (expert_ids if torch.is_tensor(expert_ids)
            else torch.tensor(expert_ids, dtype=torch.int32, device=dev)).to(torch.int32)
    out = torch.empty(T, N, dtype=torch.bfloat16, device=dev)
    lut = _lut(dev)
    if max(sizes) == 1:
        bn, warps = 64, 2
        grid = (T, triton.cdiv(N, bn))
        _gemv_mxfp4_grouped[grid](
            a_cat, blocks, scales, out, lut, eids, K, N,
            blocks.stride(0), blocks.stride(1), scales.stride(0), scales.stride(1),
            BLOCK_N=bn, BLOCK_K=MX_BLOCK, num_warps=warps, num_stages=3)
        return out
    if block_m is None:
        block_m = _prefill_block_m(max(sizes))
    if prefill_variant is None:
        prefill_variant = 1 if hasattr(tl, "gather") else 0
    block_n = 128
    t_row0, t_rows, t_group = build_group_tiles(sizes, block_m, dev)
    grid = (t_row0.numel(), triton.cdiv(N, block_n))
    _gemm_mxfp4_grouped[grid](
        a_cat, blocks, scales, out, lut, t_row0, t_rows, t_group, eids, K, N,
        blocks.stride(0), blocks.stride(1), scales.stride(0), scales.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=MX_BLOCK,
        VARIANT=prefill_variant, num_warps=(8 if block_m >= 128 else 4),
        num_stages=(3 if block_m >= 128 else 2))
    return out
