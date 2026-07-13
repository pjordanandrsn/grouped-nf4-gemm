# Copyright (c) 2026 Cerin Amroth LLC. All rights reserved.
# Private until Gate 2 (KERNEL_CONTRACT.md).

"""Grouped W4A16 GEMM over fused NF4 expert stacks — dequant inside the mainloop.

Computes, in ONE launch, ``out[t, :] = a[t, :] @ dequant_nf4(B[e(t)]).T`` for
tokens grouped by expert (KERNEL_CONTRACT.md: `bitsandbytes::gemm_4bit_grouped`
= the #1949 conventions + (group_offsets, expert_ids) + expert-major B/absmax).
The bf16 weight is never materialized in global memory: packed nibbles are
LUT-decoded to fp32 in registers, scaled by the fp32 blockwise absmax, and fed
to tensor-core ``tl.dot`` (TF32 on sm_86: 10-bit-mantissa inputs — *less*
rounding than the dequant path's bf16 materialization, which is the P-fid
mechanism) with fp32 accumulation and a single bf16 downcast at the epilogue.

Jagged grouping: the host expands groups into fixed-size M-tiles and passes
three small int32 descriptor arrays (tile→row0, tile→valid-rows, tile→expert);
the grid is (m_tiles, N/BLOCK_N). Empty groups never reach the kernel (the
caller drops them — a grouped GEMM never launches a 0-row tile). BLOCK_K == the
quant blocksize (64), so each (n, k-step) needs exactly one absmax scalar.

v1 scope per the contract: plain fp32 absmax (nested/`compress_statistics`
states are de-nested on the host at repack), no bias, nf4 only.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# The NF4 codebook (QLoRA appendix / bitsandbytes source). Code 7 decodes to
# exactly 0.0 — the zero-decode byte 0x77 the e4b mask fix relies on. The
# property suite asserts EXACT agreement (values and nibble order) against the
# installed bitsandbytes' dequantize_4bit, so a drift there fails loudly.
NF4_LUT = [
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
]
BLOCKSIZE = 64  # quant blocksize; K % 64 == 0 enforced (locked e4b design)


def repack_from_bnb(packed_list, states, N: int, K: int):
    """bnb per-expert quantize_4bit output -> the contract's expert-major tensors.

    Returns ``B [E, N, K//2] uint8`` and ``absmax [E, N, K//64] fp32``. bnb's
    packed tensor is the row-major flat [N*K/2, 1]; its absmax is flat
    [N*K/64] over the same row-major order, so both reshape cleanly when
    K % 64 == 0. Nested (compress_statistics) states are de-nested here —
    v1 of the kernel takes plain fp32 absmax per the contract."""
    assert K % BLOCKSIZE == 0, f"K={K} not a multiple of blocksize {BLOCKSIZE}"
    E = len(packed_list)
    dev = packed_list[0].device
    B = torch.empty(E, N, K // 2, dtype=torch.uint8, device=dev)
    A = torch.empty(E, N, K // BLOCKSIZE, dtype=torch.float32, device=dev)
    for e in range(E):
        B[e] = packed_list[e].reshape(N, K // 2)
        st = states[e]
        am = st.absmax
        if getattr(st, "nested", False):
            from bitsandbytes import functional as F

            am = F.dequantize_blockwise(st.absmax, st.state2) + st.offset
        A[e] = am.to(torch.float32).reshape(N, K // BLOCKSIZE)
    return B, A


def build_group_tiles(sizes, block_m: int, device):
    """Expand jagged group sizes into fixed M-tiles: (row0, valid_rows, group_idx)."""
    t_row0, t_rows, t_group = [], [], []
    row = 0
    for g, m in enumerate(sizes):
        left = m
        while left > 0:
            take = min(block_m, left)
            t_row0.append(row + (m - left))
            t_rows.append(take)
            t_group.append(g)
            left -= take
        row += m
    mk = lambda x: torch.tensor(x, dtype=torch.int32, device=device)  # noqa: E731
    return mk(t_row0), mk(t_rows), mk(t_group)


@triton.jit
def _gemm_nf4_grouped(
    a_ptr,
    b_ptr,
    amax_ptr,
    out_ptr,
    lut_ptr,
    t_row0_ptr,
    t_rows_ptr,
    t_group_ptr,
    expert_ids_ptr,
    K,
    N,
    stride_be,
    stride_bn,  # B strides (bytes dim contiguous)
    stride_ae,
    stride_an,  # absmax strides (block dim contiguous)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
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

    for k0 in range(0, K, BLOCK_K):
        kk = k0 + offs_k
        bytes_ = tl.load(b_base + (kk[None, :] // 2), mask=n_mask[:, None], other=0).to(
            tl.int32
        )
        # bnb packs element 2j into the HIGH nibble, 2j+1 into the LOW nibble
        nib = tl.where((kk[None, :] % 2) == 0, (bytes_ >> 4) & 0xF, bytes_ & 0xF)
        w = tl.load(lut_ptr + nib)  # [BN, BK] fp32 codebook gather
        am = tl.load(
            amax_ptr + eid * stride_ae + offs_n * stride_an + (k0 // BLOCK_K),
            mask=n_mask,
            other=0.0,
        )
        w = w * am[:, None]
        a = tl.load(a_base + kk[None, :], mask=m_mask[:, None], other=0.0).to(
            tl.float32
        )
        acc += tl.dot(a, tl.trans(w))  # TF32 tensor cores on sm_86, fp32 acc

    out_ptrs = out_ptr + (row0 + offs_m)[:, None] * N + offs_n[None, :]
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def _gemv_nf4_grouped(
    a_ptr,
    b_ptr,
    amax_ptr,
    out_ptr,
    lut_ptr,
    expert_ids_ptr,
    K,
    N,
    stride_be,
    stride_bn,
    stride_ae,
    stride_an,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Decode-specialized path: one token per group (M==1 everywhere), so a
    tensor-core M-tile would waste 15/16 of its lanes. Straight reduction:
    program (g, n-tile) accumulates out[g, n] = sum_k a[g,k] * w[n,k]."""
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
        bytes_ = tl.load(b_base + (kk[None, :] // 2), mask=n_mask[:, None], other=0).to(
            tl.int32
        )
        nib = tl.where((kk[None, :] % 2) == 0, (bytes_ >> 4) & 0xF, bytes_ & 0xF)
        w = tl.load(lut_ptr + nib)
        am = tl.load(
            amax_ptr + eid * stride_ae + offs_n * stride_an + (k0 // BLOCK_K),
            mask=n_mask,
            other=0.0,
        )
        a = tl.load(a_base + kk).to(tl.float32)
        acc += tl.sum(w * a[None, :], axis=1) * am

    tl.store(out_ptr + g * N + offs_n, acc.to(tl.bfloat16), mask=n_mask)


_LUT_CACHE: dict = {}


def _lut(device):
    key = str(device)
    if key not in _LUT_CACHE:
        _LUT_CACHE[key] = torch.tensor(NF4_LUT, dtype=torch.float32, device=device)
    return _LUT_CACHE[key]


def gemm_4bit_grouped(
    a_cat,
    B,
    absmax,
    sizes,
    expert_ids,
    block_m: int | None = None,
    decode_config: tuple | None = None,
):
    """Single-launch grouped NF4 GEMM. ``a_cat [T,K]`` bf16/fp16 in group-sorted
    order, ``B [E,N,K//2]`` uint8, ``absmax [E,N,K//64]`` fp32, ``sizes`` the
    per-group token counts (all > 0), ``expert_ids [G]`` int32/list. Returns
    ``[T, N]`` bf16 in the same group order. ``decode_config`` overrides the
    decode path's (BLOCK_N, num_warps) — benchmark/ablation support only."""
    E, N, _ = B.shape
    T, K = a_cat.shape
    assert sum(sizes) == T, (sum(sizes), T)
    dev = a_cat.device
    eids = (
        expert_ids
        if torch.is_tensor(expert_ids)
        else torch.tensor(expert_ids, dtype=torch.int32, device=dev)
    ).to(torch.int32)
    out = torch.empty(T, N, dtype=torch.bfloat16, device=dev)
    if max(sizes) == 1:
        # decode: every group is one token; the reduction path skips the M-tile.
        # Config is a single constant, (BLOCK_N=64, num_warps=2). Basis: a dense
        # 360-cell (N, K, T) sweep x 14 configs on TWO sm_86 devices (A5000
        # 64 SM + A2000 26 SM; bench/phase2/decode_config_sweep.py) — 64/2 is
        # oracle on ~2/3 of cells and within a few % on ~90% (median regret
        # 1.000 on both devices, p95 1.05/1.11, max 1.32/1.67), and within 10%
        # of oracle on all 16 real model shapes measured. The previous
        # exact-(N, K) dict (census-tuned, 8 shapes) had default-config regret
        # up to 2.6x off-census and did not transfer across instances — the
        # Gate-2 blind confirmatory caught it; this replaces it.
        bn, warps = decode_config or (64, 2)
        grid = (T, triton.cdiv(N, bn))
        _gemv_nf4_grouped[grid](
            a_cat,
            B,
            absmax,
            out,
            _lut(dev),
            eids,
            K,
            N,
            B.stride(0),
            B.stride(1),
            absmax.stride(0),
            absmax.stride(1),
            BLOCK_N=bn,
            BLOCK_K=BLOCKSIZE,
            num_warps=warps,
            num_stages=3,
        )
        return out
    if block_m is None:
        block_m = 16 if max(sizes) <= 16 else 64
    t_row0, t_rows, t_group = build_group_tiles(sizes, block_m, dev)
    block_n = 64
    grid = (t_row0.numel(), triton.cdiv(N, block_n))
    _gemm_nf4_grouped[grid](
        a_cat,
        B,
        absmax,
        out,
        _lut(dev),
        t_row0,
        t_rows,
        t_group,
        eids,
        K,
        N,
        B.stride(0),
        B.stride(1),
        absmax.stride(0),
        absmax.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=BLOCKSIZE,
        num_warps=4,
        num_stages=3,
    )
    return out


def dequant_ref(packed_row_major: torch.Tensor, absmax: torch.Tensor, N: int, K: int):
    """Pure-torch reference decode (same LUT + nibble order as the kernel) —
    the property suite asserts this matches bnb's dequantize_4bit EXACTLY,
    which pins both the codebook values and the high-nibble-first order."""
    lut = _lut(packed_row_major.device)
    flat = packed_row_major.reshape(-1).to(torch.int32)
    hi = (flat >> 4) & 0xF
    lo = flat & 0xF
    codes = torch.stack([hi, lo], dim=1).reshape(-1)  # element 2j = high nibble
    vals = lut[codes]
    am = absmax.to(torch.float32).reshape(-1).repeat_interleave(BLOCKSIZE)
    return (vals * am).reshape(N, K)
