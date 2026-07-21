# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).

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

import os

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
    GROUPS: tl.constexpr,   # quant groups per K-step (BLOCK_K // 64)
    VARIANT: tl.constexpr,  # 0 = v5 mainloop; 1 = register-LUT tl.gather;
                            # 3 = OPT-IN bf16 MMA (documented looser P-fid)
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
        lut_reg = tl.load(lut_ptr + tl.arange(0, 16))  # codebook in registers

    for k0 in range(0, K, BLOCK_K):
        kk = k0 + offs_k
        bytes_ = tl.load(b_base + (kk[None, :] // 2), mask=n_mask[:, None], other=0).to(
            tl.int32
        )
        # bnb packs element 2j into the HIGH nibble, 2j+1 into the LOW nibble
        nib = tl.where((kk[None, :] % 2) == 0, (bytes_ >> 4) & 0xF, bytes_ & 0xF)
        if VARIANT == 1:
            # register-resident codebook: shuffle-gather, no per-element L1 LDG
            w = tl.reshape(
                tl.gather(lut_reg, tl.reshape(nib, [BLOCK_N * BLOCK_K]), 0),
                [BLOCK_N, BLOCK_K],
            )
        else:
            w = tl.load(lut_ptr + nib)  # [BN, BK] fp32 codebook gather
        g0 = k0 // 64
        if GROUPS == 1:
            am = tl.load(
                amax_ptr + eid * stride_ae + offs_n * stride_an + g0,
                mask=n_mask,
                other=0.0,
            )
            scale = am[:, None]
        else:  # two quant groups per K-step (wrapper guarantees K % BLOCK_K == 0)
            am0 = tl.load(
                amax_ptr + eid * stride_ae + offs_n * stride_an + g0,
                mask=n_mask,
                other=0.0,
            )
            am1 = tl.load(
                amax_ptr + eid * stride_ae + offs_n * stride_an + (g0 + 1),
                mask=n_mask,
                other=0.0,
            )
            scale = tl.where(offs_k[None, :] < 64, am0[:, None], am1[:, None])
        if VARIANT == 3:
            # OPT-IN bf16 MMA: weight rounding matches the dequant baseline
            # (P-fid parity, not the fp32/TF32 edge); full-rate HMMA on sm_86.
            wb = (w * scale).to(tl.bfloat16)
            a = tl.load(a_base + kk[None, :], mask=m_mask[:, None], other=0.0)
            acc += tl.dot(a, tl.trans(wb))
        else:
            w = w * scale
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


@triton.jit
def _gemv_nf4_grouped_splitk(
    a_ptr,
    b_ptr,
    amax_ptr,
    ws_ptr,
    lut_ptr,
    expert_ids_ptr,
    K,
    N,
    T,
    KBLOCKS_PER_SPLIT,
    stride_be,
    stride_bn,
    stride_ae,
    stride_an,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Split-K variant of the decode reduction for occupancy-starved grids
    (few groups x few n-tiles, e.g. top_k=1): program (g, n-tile, k-split)
    accumulates a PARTIAL fp32 sum over its span of whole absmax blocks into
    ``ws[k_split, g, n]``; the host reduces ``ws.sum(0)`` (deterministic
    two-pass, no atomics) and downcasts once. Decode math is identical to
    ``_gemv_nf4_grouped``. A split whose span starts past K stores zeros."""
    g = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    eid = tl.load(expert_ids_ptr + g)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    offs_k = tl.arange(0, BLOCK_K)
    b_base = b_ptr + eid * stride_be + offs_n[:, None] * stride_bn
    a_base = a_ptr + g * K

    k_lo = pid_k * KBLOCKS_PER_SPLIT * BLOCK_K
    k_hi = tl.minimum(k_lo + KBLOCKS_PER_SPLIT * BLOCK_K, K)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(k_lo, k_hi, BLOCK_K):
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

    tl.store(ws_ptr + pid_k * (T * N) + g * N + offs_n, acc, mask=n_mask)


_LUT_CACHE: dict = {}
_SM_CACHE: dict = {}


def _sm_count(device) -> int:
    key = str(device)
    if key not in _SM_CACHE:
        # CPU / interpreter mode (correctness testing, no GPU) can't query cuda;
        # the count only feeds split-K planning, so a nominal value is safe there.
        # On a real cuda device the true count is still used (no perf-path change).
        if torch.cuda.is_available() and "cuda" in key:
            _SM_CACHE[key] = torch.cuda.get_device_properties(device).multi_processor_count
        else:
            _SM_CACHE[key] = 64
    return _SM_CACHE[key]


# v4 dispatch constants (each carries its measured basis; see the prereg/results
# docs for the runs behind the numbers):
# - DECODE_MIN_FUSED_BYTES: below this per-call weight+absmax traffic the fused
#   launch loses OUTRIGHT to the dequant path (v3 blind: Switch-Base cells at
#   1.3-2.7 MB ran 0.24-0.35x speed and 4-7x energy on both devices, while
#   granite down at 3.5 MB kept a 1.13-1.97x win). Product integrations should
#   route below-floor calls to the dequant path via decode_dispatch().
# - SPLITK_MIN_BLOCKS: a split must own at least this many absmax blocks —
#   v3 blind showed the starvation-only trigger splitting 12-block cells hurt
#   (Switch gu paired 0.655 on the A5000) while >=32-block splits helped
#   (Scout down 1.46x, Hunyuan down 1.18x paired).
DECODE_MIN_FUSED_BYTES = 3_000_000
SPLITK_MIN_BLOCKS = 32


def _decode_plan(N: int, K: int, T: int, sm_count: int):
    """Decode launch plan: (BLOCK_N, num_warps, split_k).

    Config is the universal constant (64, 2) — dense 2-device (N, K, T)
    sweeps put it at median regret 1.000 on both grids
    (bench/phase2/sweeps/); the v3 confirmatory showed the v2-era A2000
    preference for 128/4 did not reproduce (config deltas on the 26-SM card
    are run-context noise), so the SM-conditional branch is reverted.

    Split-K engages only for truly starved grids (programs < 2*SM — census
    cells never split) AND only when each split owns >= SPLITK_MIN_BLOCKS
    absmax blocks (v3: splitting tiny-K cells hurt). fp32 partials are
    host-reduced; power-of-2, capped at 8."""
    bn, warps = 64, 2
    programs = T * -(-N // bn)
    split_k = 1
    if programs < 2 * sm_count:
        want = -(-(4 * sm_count) // programs)
        while split_k < want and split_k < 8:
            split_k *= 2
        kblocks = max(K // BLOCKSIZE, 1)
        while split_k > 1 and kblocks // split_k < SPLITK_MIN_BLOCKS:
            split_k //= 2
    return bn, warps, split_k


def decode_dispatch(N: int, K: int, T: int, sm_count: int):
    """Product-layer path choice for one decode call: ``("dequant",)`` when
    the call is below the fused floor (tiny cells belong to the dequant
    path — v3 measured them losing outright), else
    ``("fused", BLOCK_N, num_warps, split_k)``.

    The op itself (gemm_4bit_grouped) always runs fused — an op that
    silently ran a different algorithm would be a contract violation — so
    integrations consult this helper and call the dequant path themselves
    for below-floor cells. The benchmark's ``fused_routed`` backend does
    exactly that."""
    traffic = T * N * (K // 2 + K // 16)  # packed nibbles + fp32 absmax bytes
    if traffic < DECODE_MIN_FUSED_BYTES:
        return ("dequant",)
    return ("fused", *_decode_plan(N, K, T, sm_count))


def _lut(device):
    key = str(device)
    if key not in _LUT_CACHE:
        _LUT_CACHE[key] = torch.tensor(NF4_LUT, dtype=torch.float32, device=device)
    return _LUT_CACHE[key]


def _prefill_block_m(max_rows: int) -> int:
    """Group-size-keyed M-tile height (sweep basis in the wrapper comment)."""
    if max_rows <= 16:
        return 16
    if max_rows <= 32:
        return 32
    if max_rows <= 64:
        return 64
    return 128


def gemm_4bit_grouped(
    a_cat,
    B,
    absmax,
    sizes,
    expert_ids,
    block_m: int | None = None,
    decode_config: tuple | None = None,
    split_k: int | None = None,
    prefill_config: tuple | None = None,
    prefill_variant: int | None = None,
    prefill_groups: int = 1,
):
    """Single-launch grouped NF4 GEMM. ``a_cat [T,K]`` bf16/fp16 in group-sorted
    order, ``B [E,N,K//2]`` uint8, ``absmax [E,N,K//64]`` fp32, ``sizes`` the
    per-group token counts (all > 0), ``expert_ids [G]`` int32/list. Returns
    ``[T, N]`` bf16 in the same group order. ``decode_config`` overrides the
    decode path's (BLOCK_N, num_warps); ``split_k`` overrides the decode
    split-K factor (None = plan, 1 = off); ``prefill_config`` overrides the
    M-tile path's (BLOCK_N, num_warps, num_stages) — benchmark/ablation
    support only. ``prefill_variant``: None = auto (register-LUT mainloop
    when triton has ``tl.gather``, else the v5 loop), 0 = force v5 loop,
    1 = register-LUT tl.gather (the v6 default: fidelity-identical, kills
    the per-element L1 codebook gather), 3 = OPT-IN bf16 MMA (P-fid parity
    with the dequant baseline, not the fp32/TF32 edge — measured slower
    than variant 1 everywhere in the v6 matrix; see RESULTS-phase2-v1.1 and
    bench/phase2/v6_prefill_matrix.py). ``prefill_groups``: quant groups
    per K-step (2 = BLOCK_K 128: dead on sm_86 — SMEM blowout; kept for
    ablation only)."""
    E, N, _ = B.shape
    T, K = a_cat.shape
    assert sum(sizes) == T, (sum(sizes), T)
    dev = a_cat.device
    # CUDA-only in real use; TRITON_INTERPRET=1 runs the kernel on CPU tensors
    # (the interpreter-contract suite), so exempt that path from the guard.
    if dev.type != "cuda" and os.environ.get("TRITON_INTERPRET") != "1":
        raise RuntimeError(
            f"gemm_4bit_grouped runs the fused Triton kernel and requires CUDA tensors "
            f"(got device '{dev.type}'). For a CPU-checkable decode of the same NF4 bytes, "
            f"use dequant_ref(packed, absmax, N, K) — the pure-torch reference the property "
            f"suite pins the kernel against."
        )
    eids = (
        expert_ids
        if torch.is_tensor(expert_ids)
        else torch.tensor(expert_ids, dtype=torch.int32, device=dev)
    ).to(torch.int32)
    out = torch.empty(T, N, dtype=torch.bfloat16, device=dev)
    if max(sizes) == 1:
        # decode: every group is one token; the reduction path skips the M-tile.
        # Launch plan (_decode_plan): SM-conditional constant + split-K for
        # starved grids. Each part carries its measured basis in the plan's
        # docstring; the ablation kwargs let a harness force either off.
        bn, warps, sk = _decode_plan(N, K, T, _sm_count(dev))
        if decode_config is not None:
            bn, warps = decode_config
        if split_k is not None:
            sk = split_k
        if sk <= 1:
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
        kblocks = -(-K // BLOCKSIZE)
        span = -(-kblocks // sk)
        ws = torch.empty(sk, T, N, dtype=torch.float32, device=dev)
        grid = (T, triton.cdiv(N, bn), sk)
        _gemv_nf4_grouped_splitk[grid](
            a_cat,
            B,
            absmax,
            ws,
            _lut(dev),
            eids,
            K,
            N,
            T,
            span,
            B.stride(0),
            B.stride(1),
            absmax.stride(0),
            absmax.stride(1),
            BLOCK_N=bn,
            BLOCK_K=BLOCKSIZE,
            num_warps=warps,
            num_stages=3,
        )
        out.copy_(ws.sum(dim=0))  # fp32 partial reduce, single bf16 downcast
        return out
    if block_m is None:
        block_m = _prefill_block_m(max(sizes))
    if prefill_variant is None:
        prefill_variant = 1 if hasattr(tl, "gather") else 0
    if prefill_config is not None:
        block_n, warps, stages = prefill_config
    elif prefill_variant == 1:
        # v6 register-LUT mainloop rule (bench/phase2/v6_prefill_matrix.py,
        # A5000): bn=128/w4/s3 with the group-size-keyed BLOCK_M is the
        # per-cell oracle on 6/8 census prefill cells, worst regret 1.034.
        # Under it the M-tile path runs 1.20-2.88x the dequant baseline on
        # every census cell except OLMoE gate_up (0.62x, the remaining
        # known loser; was 0.38x on the v5 loop).
        block_n = 128
        warps = 4
        stages = 3
    else:
        # v4 group-size-keyed rule for the v5 (VARIANT=0) loop
        # (bench/phase2/sweeps/v4_prefill_*.json): 128/128/w8/s3 for m >= 128
        # groups, 64-and-below groups want the narrower 64-row tile at w4/s2.
        # Rule regret vs per-cell oracle: worst 1.058, 13/16 cells at 1.00-1.02.
        block_n = 128
        warps = 8 if block_m >= 128 else 4
        stages = 3 if block_m >= 128 else 2
    if prefill_variant == 1 and not hasattr(tl, "gather"):
        raise RuntimeError("prefill_variant=1 needs triton with tl.gather")
    block_k = BLOCKSIZE * prefill_groups
    if prefill_groups != 1:
        assert prefill_groups == 2 and K % block_k == 0, (prefill_groups, K)
    t_row0, t_rows, t_group = build_group_tiles(sizes, block_m, dev)
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
        BLOCK_K=block_k,
        GROUPS=prefill_groups,
        VARIANT=prefill_variant,
        num_warps=warps,
        num_stages=stages,
    )
    return out


def dequant_ref(packed_row_major: torch.Tensor, absmax: torch.Tensor, N: int, K: int):
    """Pure-torch reference decode (same LUT + nibble order as the kernel) —
    the property suite asserts this matches bnb's dequantize_4bit EXACTLY,
    which pins both the codebook values and the high-nibble-first order.
    Runs on CPU (no CUDA/Triton), so it is the checkable oracle for the kernel.

    Example:
        >>> from nf4_pack_ref import quantize_pack_nf4
        >>> from nf4_grouped import dequant_ref
        >>> packed, absmax = quantize_pack_nf4(torch.randn(128, 256))
        >>> w = dequant_ref(packed, absmax, 128, 256)      # [128, 256] fp32
    """
    lut = _lut(packed_row_major.device)
    flat = packed_row_major.reshape(-1).to(torch.int32)
    hi = (flat >> 4) & 0xF
    lo = flat & 0xF
    codes = torch.stack([hi, lo], dim=1).reshape(-1)  # element 2j = high nibble
    vals = lut[codes]
    am = absmax.to(torch.float32).reshape(-1).repeat_interleave(BLOCKSIZE)
    return (vals * am).reshape(N, K)
