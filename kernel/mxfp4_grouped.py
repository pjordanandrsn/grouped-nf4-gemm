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
# ``triton`` is a Linux-only dependency (pyproject marks it
# ``platform_system == 'Linux'``), so a supported macOS install has none and a
# bare ``import triton`` here made the whole module unimportable there. The shim
# binds the real thing when it exists — this file is unchanged below in that
# case — and otherwise lets the kernels still DEFINE while a launch raises.
from _triton_shim import tl, triton  # noqa: F401  (re-exported names)

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
    # int64 BEFORE any stride product: eid * stride_be overflows signed int32
    # the moment the packed stack passes 2^31 bytes -- same boundary nf4_grouped
    # measured (256 x 8 MiB passes, 257 faults); the MXFP4 port had dropped the
    # cast and G1's 300-row transient pool (stride 8.8 MB, slots to 255) hit it.
    eid = eid.to(tl.int64)

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
    # int64 BEFORE any stride product: eid * stride_be overflows signed int32
    # the moment the packed stack passes 2^31 bytes -- same boundary nf4_grouped
    # measured (256 x 8 MiB passes, 257 faults); the MXFP4 port had dropped the
    # cast and G1's 300-row transient pool (stride 8.8 MB, slots to 255) hit it.
    eid = eid.to(tl.int64)

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
    bf16, same group order. Decode (all sizes==1) uses the GEMV reduction.

    Use it when the experts are released as MXFP4 (gpt-oss, DeepSeek-V4, Kimi lineage) and
    must be computed on the checkpoint's own bytes: ``blocks [E, N, K//2]`` uint8 (e2m1,
    low nibble first) and ``scales [E, N, K//32]`` uint8 (e8m0), as ``mxfp4_loader.to_kernel_shapes``
    views them. Returns ``[T, N]`` bf16 in group order. Needs a CUDA GPU (sm_80+) and Triton.
    See ``docs/solutions/native-mxfp4-moe-inference.md``.
    """
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


# ------------------------------------------- decode-grade GEMV (B=1) --
@triton.jit
def _gemv_mxfp4_b32(xq_ptr, xs_ptr, w_ptr, ws_ptr, eid_ptr, out_ptr,
                    N, K: tl.constexpr, R: tl.constexpr,
                    BLOCK_N: tl.constexpr, SK: tl.constexpr,
                    KU: tl.constexpr):
    """The int4-b32 decode GEMV structure (split-K over 32-wide groups,
    KU groups per iteration, fp32 partials the wrapper reduces) on the
    NATIVE MXFP4 store: element 2j = LOW nibble, 2j+1 = HIGH, e8m0
    scale per (row, 32-group). Activations are the int4-b32 int8 rows
    with a per-32 fp32 scale (``quant_x_rows``), so the block dot is
    an EXACT int32 sum: an e2m1 nibble ``s|e1e0|m`` decodes to twice
    its value as the integer ``m`` (e == 0) or ``(2 + m) << (e - 1)``
    -- {0,1,2,3,4,6,8,12} -- and the 0.5 goes into the scale with
    ``exp2(e8 - 127)``. Same grid/plan as ``_gemv_int4_b32`` (receipts
    int4port): grid (cdiv(N, BN), R, SK); ``out[(sk*R + e)*N + n]``."""
    pid = tl.program_id(0)
    e = tl.program_id(1)
    sk = tl.program_id(2)
    eid = tl.load(eid_ptr + e).to(tl.int64)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    KB: tl.constexpr = K // 32
    span: tl.constexpr = (KB + SK * KU - 1) // (SK * KU)
    ku = tl.arange(0, KU)
    pair = tl.arange(0, 16)
    wbase = w_ptr + eid * N * (K // 2)
    sbase = ws_ptr + eid * N * KB
    for kbi in range(0, span):
        kb0 = (sk * span + kbi) * KU
        if kb0 < KB:
            xoff = e * K + kb0 * 32 + ku[:, None] * 32 + 2 * pair[None, :]
            xe = tl.load(xq_ptr + xoff).to(tl.int32)
            xo = tl.load(xq_ptr + xoff + 1).to(tl.int32)
            xsv = tl.load(xs_ptr + e * KB + kb0 + ku)
            wb = tl.load(wbase + offs_n[:, None] * (K // 2) + kb0 * 16
                         + tl.arange(0, KU * 16)[None, :],
                         mask=n_mask[:, None], other=0).to(tl.int32)
            wb = tl.reshape(wb, (BLOCK_N, KU, 16))
            lo = wb & 0xF
            hi = (wb >> 4) & 0xF
            # e2m1 -> 2*value as an exact integer, sign applied after
            lo_e = (lo >> 1) & 0x3
            lo_m = lo & 0x1
            lo_v = tl.where(lo_e == 0, lo_m,
                            (2 + lo_m) << tl.maximum(lo_e - 1, 0))
            lo_v = tl.where((lo & 0x8) != 0, -lo_v, lo_v)
            hi_e = (hi >> 1) & 0x3
            hi_m = hi & 0x1
            hi_v = tl.where(hi_e == 0, hi_m,
                            (2 + hi_m) << tl.maximum(hi_e - 1, 0))
            hi_v = tl.where((hi & 0x8) != 0, -hi_v, hi_v)
            d = tl.sum(lo_v * xe[None, :, :], axis=2) \
              + tl.sum(hi_v * xo[None, :, :], axis=2)
            e8 = tl.load(sbase + offs_n[:, None] * KB + kb0 + ku[None, :],
                         mask=n_mask[:, None], other=127).to(tl.int32)
            ws = tl.exp2((e8 - 128).to(tl.float32))     # 0.5 * 2^(e8-127)
            acc += tl.sum(d.to(tl.float32) * (ws * xsv[None, :]), axis=1)
    tl.store(out_ptr + (sk * R + e) * N + offs_n, acc, mask=n_mask)


def gemv_mxfp4_b32(xq, xs, blocks, scales, eids, N: int, K: int,
                   part: torch.Tensor | None = None):
    """Grouped decode GEMV on the native MXFP4 store: ``R = eids.numel()``
    activation rows (``quant_x_rows`` int8 + fp32 per-32 scales), row
    ``e`` against expert ``eids[e]``. ``blocks [E, N, K//2]`` uint8,
    ``scales [E, N, K//32]`` uint8 (e8m0). Returns ``[R, N]`` bf16;
    ``part`` may be a preallocated ``[SK*R, N]`` fp32 buffer (pass it
    under capture). Plan shared with ``gemv_int4_b32``.

    Use it for decode rows (a handful per call) on the same MXFP4 store: the int4-b32 split-K
    GEMV structure with an exact int32 e2m1 dot over ``quant_x_rows`` activations. Above a
    handful of rows it re-streams the weights per row and loses to the grouped GEMM or the
    consumer's NF4 path. See ``docs/solutions/native-mxfp4-moe-inference.md``.
    """
    from int4_b32 import _plan
    R = eids.numel()
    assert blocks.dtype == torch.uint8 and scales.dtype == torch.uint8
    assert scales.shape[-1] == K // MX_BLOCK, (scales.shape, K)
    bn, wp, sk, ku = _plan(N, K)
    if part is None:
        part = torch.empty(sk * R, N, dtype=torch.float32, device=xq.device)
    _gemv_mxfp4_b32[(triton.cdiv(N, bn), R, sk)](
        xq, xs, blocks, scales, eids, part,
        N, K=K, R=R, BLOCK_N=bn, SK=sk, KU=ku, num_warps=wp)
    from int4_b32 import reduce_partials
    return reduce_partials(part, sk, R, N)
