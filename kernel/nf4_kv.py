# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).

"""NF4 KV cache — attention that reads 4-bit keys/values without materializing them.

The weights got the tiered-4-bit treatment; the KV cache is the *other* memory
consumer and it grows linearly in context (measured: 188.0 KB/token for
Qwen3-235B, docs/context-budgets.md). This module applies the same primitive to
it: quantize K and V on write, then dequant-in-the-mainloop inside the two
attention GEMMs so the bf16 cache is never materialized.

Why this is the same problem as the expert GEMM. A per-head KV row is
``[head_dim]`` contiguous, and ``head_dim`` is a multiple of the 64-element
quant blocksize on every architecture measured (64 gpt-oss, 128 Qwen3/OLMoE,
256 Gemma-4) — so the cache is exactly the ``[N, K]`` -> ``[N, K/2] u8`` +
``[N, K/64] f32`` layout family that ``gemm_4bit_grouped`` already consumes,
with (N -> tokens x kv_heads, K -> head_dim). Same codebook, same nibble order,
same blockwise absmax; the property suite asserts that agreement.

Two kernels, one per attention matmul (decode, one query token):

  scores   : out[h, t] = q[h, :] . dequant(K[t, h_kv, :])          (reduce over D)
  weighted : out[h, d] = sum_t p[h, t] * dequant(V[t, h_kv, d])    (reduce over T)

GQA is an index map, not a broadcast: query head ``h`` reads kv head
``h // (H_q // H_kv)``, so a 4-kv-head model stores 4 rows per token, not 64.

Scope (v1): decode (one query token per call), fp32 accumulation, nf4 only,
plain fp32 absmax. Prefill and paged/blocked layouts are deliberately out —
this is the memory-footprint primitive, not a full attention replacement.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from nf4_grouped import BLOCKSIZE, NF4_LUT, _device_shared_limit, _lut

__all__ = [
    "quantize_kv",
    "quantize_kv_perchannel",
    "PERCHANNEL_GROUP",
    "dequant_kv_ref",
    "dequant_kv_fused",
    "kv_scores_nf4",
    "kv_weighted_sum_nf4",
    "attend_nf4_kv",
    "attend_nf4_kv_fused",
    "attend_nf4_kv_split",
    "attend_nf4_kv_gqa",
    "kv_cache_bytes",
    "kv_cache_bytes_perchannel",
]


#: Token-group size for per-channel scaling. Setting it equal to ``BLOCKSIZE``
#: is what makes this free: both schemes then store exactly one fp32 scale per
#: 64 quantized values — per-token groups 64 channels within a token,
#: per-channel groups 64 tokens within a channel — so the side channel is
#: byte-for-byte identical for ANY head_dim (T*H*D/16 either way). Per-channel
#: keys are therefore a pure fidelity change, not a memory trade. Any other
#: group size breaks the equality: G < 64 costs more, G > 64 costs less and
#: scales over more tokens.
PERCHANNEL_GROUP = BLOCKSIZE


def _check(head_dim: int) -> None:
    if head_dim % BLOCKSIZE != 0:
        raise ValueError(
            f"head_dim={head_dim} must be a multiple of the quant blocksize "
            f"{BLOCKSIZE}; every architecture measured satisfies this (64/128/256)."
        )


def quantize_kv(x: torch.Tensor, chunk_rows: int = 1 << 14):
    """``[T, H_kv, D]`` float -> (``[T, H_kv, D//2]`` u8, ``[T, H_kv, D//64]`` f32).

    Blockwise along ``D`` (the contiguous per-head row), matching
    ``nf4_pack_ref.quantize_pack_nf4``: even index -> high nibble, odd -> low.
    Chunked over rows because the code search materializes ``[rows, D, 16]``.
    """
    if x.dim() != 3:
        raise ValueError(f"expected [T, H_kv, D]; got {tuple(x.shape)}")
    T, H, D = x.shape
    _check(D)
    lut = _lut(x.device)
    rows = T * H
    flat = x.reshape(rows, D).float()
    packed = torch.empty(rows, D // 2, dtype=torch.uint8, device=x.device)
    absmax = torch.empty(rows, D // BLOCKSIZE, dtype=torch.float32, device=x.device)
    for lo_r in range(0, rows, chunk_rows):
        hi_r = min(lo_r + chunk_rows, rows)
        blocks = flat[lo_r:hi_r].reshape(-1, D // BLOCKSIZE, BLOCKSIZE)
        am = blocks.abs().amax(dim=2).clamp_min(1e-12)
        scaled = (blocks / am[:, :, None]).reshape(-1, D, 1)
        codes = (scaled - lut).abs().argmin(dim=2)          # nearest codebook entry
        packed[lo_r:hi_r] = (codes[:, 0::2].to(torch.uint8) << 4) | codes[
            :, 1::2
        ].to(torch.uint8)
        absmax[lo_r:hi_r] = am
    return (packed.reshape(T, H, D // 2).contiguous(),
            absmax.reshape(T, H, D // BLOCKSIZE).contiguous())


def quantize_kv_perchannel(x: torch.Tensor, group: int = PERCHANNEL_GROUP):
    """``[T, H, D]`` -> (``[T, H, D//2]`` u8, ``[ceil(T/group), H, D]`` f32).

    Groups the absmax along **tokens** instead of along the head row, giving
    every channel its own scale. Measured motivation (docs/context-budgets.md
    finding #7): keys carry per-channel outliers, so under per-token blockwise
    scaling one loud channel sets the absmax for the 63 quiet channels sharing
    its block and costs them precision. Values do not have this problem, which
    is why this is offered for K and not applied blanket.

    The packed layout is byte-identical to :func:`quantize_kv` — only the
    grouping of the scales differs — so the same kernels read both.
    """
    if x.dim() != 3:
        raise ValueError(f"expected [T, H, D]; got {tuple(x.shape)}")
    T, H, D = x.shape
    _check(D)
    lut = _lut(x.device)
    n_grp = (T + group - 1) // group
    pad = n_grp * group - T
    xp = x.float()
    if pad:
        # zero-pad the last group; zeros cannot raise an amax, and an all-zero
        # group is caught by the clamp below
        xp = torch.cat([xp, xp.new_zeros(pad, H, D)], dim=0)
    absmax = xp.abs().reshape(n_grp, group, H, D).amax(dim=1).clamp_min(1e-12)
    scaled = (xp / absmax.repeat_interleave(group, dim=0))[:T]
    codes = (scaled.reshape(-1, D, 1) - lut).abs().argmin(dim=2).reshape(T, H, D)
    packed = ((codes[..., 0::2] << 4) | codes[..., 1::2]).to(torch.uint8)
    return packed.contiguous(), absmax.contiguous()


def _require_inner_contig(**tensors) -> None:
    """The kernels index the head-dim as ``base + t*stride_t + h*stride_h + j``,
    i.e. they assume the innermost dim is packed. Token/head strides ARE passed,
    so an outer-sliced view is fine; a strided inner dim is not.

    This raises rather than calling ``.contiguous()`` on purpose. ``q`` and
    ``probs`` are one row per head and free to copy, but the cache is the large
    object — silently materializing a contiguous copy of a 32K cache would
    double peak memory, which is precisely the cost this module exists to avoid.
    A caller that hits this wants to fix its layout, not pay for a hidden copy.
    """
    for name, t in tensors.items():
        if t.stride(-1) != 1:
            raise ValueError(
                f"{name} must have a contiguous innermost dim (stride(-1)==1); "
                f"got shape {tuple(t.shape)} strides {t.stride()}. Slice along "
                "tokens/heads (those strides are honoured) or re-pack; this is "
                "not copied for you because the cache is the big allocation."
            )


def dequant_kv_ref(packed: torch.Tensor, absmax: torch.Tensor, D: int,
                   dtype=torch.float32, token_group: int | None = None) -> torch.Tensor:
    """Reference dequant of a packed cache -> ``[T, H_kv, D]``. Test oracle."""
    T, H, _ = packed.shape
    # Same layout contract as the kernels. Without this the oracle would
    # silently accept (via reshape's copy) inputs the kernels reject, so the
    # two would disagree about the valid DOMAIN rather than just the values --
    # and a test comparing them on such an input would be comparing a number
    # against an exception.
    _require_inner_contig(packed=packed, absmax=absmax)
    _check_cache(packed, absmax, D, token_group, "cache")
    lut = _lut(packed.device)
    b = packed.reshape(-1, D // 2).to(torch.int32)
    hi = (b >> 4) & 0xF
    lo = b & 0xF
    codes = torch.stack([hi, lo], dim=2).reshape(-1, D)      # even=hi, odd=lo
    vals = lut[codes].reshape(T, H, D)
    if token_group is None:
        am = absmax.reshape(-1, D // BLOCKSIZE).repeat_interleave(BLOCKSIZE, dim=1)
        return (vals.reshape(-1, D) * am).reshape(T, H, D).to(dtype)
    am = absmax.repeat_interleave(token_group, dim=0)[:T]     # [T, H, D]
    return (vals * am).to(dtype)


@triton.jit
def _dequant_kv_kernel(packed_ptr, absmax_ptr, out_ptr, lut_ptr, n_rows,
                       D: tl.constexpr, BLOCKSIZE_C: tl.constexpr,
                       ROWS: tl.constexpr, HALF: tl.constexpr,
                       NBLK: tl.constexpr):
    """One pass: unpack nibbles, gather the codebook, scale, store.

    The oracle this replaces materializes seven full-size intermediates -- an
    int32 widening, two masks, a stack, a float32 LUT gather, a
    repeat_interleave that expands each scale into BLOCKSIZE copies, and the
    product -- for a two-byte result. Nothing here leaves registers.
    """
    pid = tl.program_id(0)
    rows = pid * ROWS + tl.arange(0, ROWS)
    live = rows < n_rows
    i = tl.arange(0, HALF)                                  # byte index in a row

    b = tl.load(packed_ptr + rows[:, None] * HALF + i[None, :],
                mask=live[:, None], other=0).to(tl.int32)
    hi = (b >> 4) & 0xF                                     # even element
    lo = b & 0xF                                            # odd element
    vhi = tl.load(lut_ptr + hi)
    vlo = tl.load(lut_ptr + lo)

    # element 2i lives in block (2i)//BLOCKSIZE, element 2i+1 in (2i+1)//BLOCKSIZE
    am_hi = tl.load(absmax_ptr + rows[:, None] * NBLK + ((2 * i) // BLOCKSIZE_C)[None, :],
                    mask=live[:, None], other=0.0)
    am_lo = tl.load(absmax_ptr + rows[:, None] * NBLK + ((2 * i + 1) // BLOCKSIZE_C)[None, :],
                    mask=live[:, None], other=0.0)

    base = rows[:, None] * D
    tl.store(out_ptr + base + (2 * i)[None, :], vhi * am_hi, mask=live[:, None])
    tl.store(out_ptr + base + (2 * i + 1)[None, :], vlo * am_lo, mask=live[:, None])


def dequant_kv_fused(packed: torch.Tensor, absmax: torch.Tensor, D: int,
                     dtype=torch.float32) -> torch.Tensor:
    """Fused dequant of a packed cache -> ``[T, H_kv, D]``.

    Bit-identical to :func:`dequant_kv_ref` by construction: both multiply an
    fp32 codebook value by an fp32 scale and round once at the store. The
    property suite asserts equality rather than a tolerance, because anything
    else would mean the two disagree about arithmetic and not just speed.

    Per-channel (``token_group``) scaling is NOT handled -- its absmax is
    grouped over runs of tokens rather than within a row, which is a different
    indexing problem. Callers fall back to the reference, as they do elsewhere
    for that dial.
    """
    _require_inner_contig(packed=packed, absmax=absmax)
    _check_cache(packed, absmax, D, None, "cache")
    T, H, _ = packed.shape
    n_rows = T * H
    out = torch.empty(n_rows, D, dtype=dtype, device=packed.device)
    rows_per_prog = 4
    grid = (triton.cdiv(n_rows, rows_per_prog),)
    _dequant_kv_kernel[grid](
        packed.reshape(n_rows, D // 2), absmax.reshape(n_rows, D // BLOCKSIZE),
        out, _lut(packed.device), n_rows,
        D=D, BLOCKSIZE_C=BLOCKSIZE, ROWS=rows_per_prog,
        HALF=D // 2, NBLK=D // BLOCKSIZE, num_warps=4)
    return out.reshape(T, H, D)


def kv_cache_bytes(n_tokens: int, kv_heads: int, head_dim: int,
                   nf4: bool = True) -> int:
    """Bytes for a cache of ``n_tokens`` (K and V together)."""
    if not nf4:
        return 2 * n_tokens * kv_heads * head_dim * 2          # bf16 K + V
    per = n_tokens * kv_heads * (head_dim // 2 + (head_dim // BLOCKSIZE) * 4)
    return 2 * per                                             # K and V


def kv_cache_bytes_perchannel(n_tokens: int, kv_heads: int, head_dim: int,
                              group: int = PERCHANNEL_GROUP) -> int:
    """Bytes for ONE per-channel-scaled tensor (K or V), nibbles + grouped absmax.

    Equals the per-token blockwise cost exactly when ``group == BLOCKSIZE``
    (see :data:`PERCHANNEL_GROUP`), for any head_dim. Asserted in the suite
    rather than left as prose, because "free fidelity" is only true where the
    arithmetic actually lands.
    """
    n_grp = (n_tokens + group - 1) // group
    return n_tokens * kv_heads * (head_dim // 2) + n_grp * kv_heads * head_dim * 4


@triton.jit
def _kv_scores_nf4(
    q_ptr, k_ptr, ka_ptr, out_ptr, lut_ptr,
    T, D, GQA,
    stride_kt, stride_kh, stride_at, stride_ah,
    BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr, QBLK: tl.constexpr,
    TGRP: tl.constexpr,
):
    """scores[h, t] = q[h, :] . dequant(K[t, h//GQA, :]) — reduce over head_dim.

    D <= BLOCK_D so the whole head row is one register tile; the reduction is
    over the (short) head dim and the grid covers tokens.
    """
    h = tl.program_id(0)
    pid_t = tl.program_id(1)
    hkv = h // GQA
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = offs_t < T
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < D

    q = tl.load(q_ptr + h * D + offs_d, mask=d_mask, other=0.0).to(tl.float32)
    kb = k_ptr + offs_t[:, None] * stride_kt + hkv * stride_kh
    byts = tl.load(kb + (offs_d[None, :] // 2),
                   mask=t_mask[:, None] & d_mask[None, :], other=0).to(tl.int32)
    nib = tl.where((offs_d[None, :] % 2) == 0, (byts >> 4) & 0xF, byts & 0xF)
    w = tl.load(lut_ptr + nib)
    # One index expression covers both groupings: per-token blockwise is
    # (TGRP=1, QBLK=64) -> a scale per 64 channels of each token; per-channel is
    # (TGRP=G, QBLK=1) -> a scale per channel, shared across G tokens.
    am = tl.load(ka_ptr + (offs_t[:, None] // TGRP) * stride_at + hkv * stride_ah
                 + (offs_d[None, :] // QBLK),
                 mask=t_mask[:, None] & d_mask[None, :], other=0.0)
    acc = tl.sum(w * am * q[None, :], axis=1)
    tl.store(out_ptr + h * T + offs_t, acc, mask=t_mask)


@triton.jit
def _kv_wsum_nf4(
    p_ptr, v_ptr, va_ptr, out_ptr, lut_ptr,
    T, D, GQA,
    stride_vt, stride_vh, stride_at, stride_ah,
    BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr, QBLK: tl.constexpr,
    TGRP: tl.constexpr,
):
    """out[h, d] = sum_t p[h, t] * dequant(V[t, h//GQA, d]) — reduce over tokens."""
    h = tl.program_id(0)
    hkv = h // GQA
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < D
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for t0 in range(0, T, BLOCK_T):
        offs_t = t0 + tl.arange(0, BLOCK_T)
        t_mask = offs_t < T
        p = tl.load(p_ptr + h * T + offs_t, mask=t_mask, other=0.0).to(tl.float32)
        vb = v_ptr + offs_t[:, None] * stride_vt + hkv * stride_vh
        byts = tl.load(vb + (offs_d[None, :] // 2),
                       mask=t_mask[:, None] & d_mask[None, :], other=0).to(tl.int32)
        nib = tl.where((offs_d[None, :] % 2) == 0, (byts >> 4) & 0xF, byts & 0xF)
        w = tl.load(lut_ptr + nib)
        am = tl.load(va_ptr + (offs_t[:, None] // TGRP) * stride_at + hkv * stride_ah
                     + (offs_d[None, :] // QBLK),
                     mask=t_mask[:, None] & d_mask[None, :], other=0.0)
        acc += tl.sum(w * am * p[:, None], axis=0)
    tl.store(out_ptr + h * D + offs_d, acc, mask=d_mask)


def _check_cache(packed: torch.Tensor, absmax: torch.Tensor, D: int,
                 token_group: int | None, name: str) -> None:
    """Validate packed/absmax shapes against the declared head_dim and grouping.

    Load-bearing since per-channel scaling landed: there are now TWO legal
    absmax layouts for identical packed bytes — ``[T, H, D/64]`` per-token and
    ``[ceil(T/G), H, D]`` per-channel — so passing one while the kernel is
    configured for the other silently reads the wrong scales, and on the
    per-token path a per-channel absmax is also SHORTER than the kernel's index
    range, i.e. an out-of-bounds read rather than merely a wrong answer.
    """
    if packed.dim() != 3 or packed.shape[-1] != D // 2:
        raise ValueError(
            f"{name}_packed last dim must be head_dim//2 = {D // 2}; "
            f"got shape {tuple(packed.shape)} (head_dim inferred as {D})")
    T, H = packed.shape[0], packed.shape[1]
    if token_group is None:
        want = (T, H, D // BLOCKSIZE)
        hint = "per-token blockwise"
    else:
        want = ((T + token_group - 1) // token_group, H, D)
        hint = f"per-channel, group {token_group}"
    if tuple(absmax.shape) != want:
        raise ValueError(
            f"{name}_absmax shape {tuple(absmax.shape)} does not match {hint} "
            f"scaling of a [{T}, {H}, {D}] cache (expected {want}). Pass "
            f"token_group= to match how the tensor was quantized.")


def _gqa(n_q_heads: int, n_kv_heads: int) -> int:
    if n_q_heads % n_kv_heads != 0:
        raise ValueError(f"q heads {n_q_heads} not divisible by kv heads {n_kv_heads}")
    return n_q_heads // n_kv_heads


def kv_scores_nf4(q: torch.Tensor, k_packed: torch.Tensor, k_absmax: torch.Tensor,
                  block_t: int = 128,
                  token_group: int | None = None) -> torch.Tensor:
    """``q [H_q, D]`` against a packed key cache -> ``scores [H_q, T]`` fp32."""
    H_q, D = q.shape
    T, H_kv, _ = k_packed.shape
    _check(D)
    _require_inner_contig(k_packed=k_packed, k_absmax=k_absmax)
    _check_cache(k_packed, k_absmax, D, token_group, "k")
    gqa = _gqa(H_q, H_kv)
    out = torch.empty(H_q, T, dtype=torch.float32, device=q.device)
    block_d = triton.next_power_of_2(D)
    grid = (H_q, triton.cdiv(T, block_t))
    _kv_scores_nf4[grid](
        q.contiguous(), k_packed, k_absmax, out, _lut(q.device),
        T, D, gqa,
        k_packed.stride(0), k_packed.stride(1),
        k_absmax.stride(0), k_absmax.stride(1),
        BLOCK_T=block_t, BLOCK_D=block_d,
        QBLK=1 if token_group else BLOCKSIZE, TGRP=token_group or 1,
    )
    return out


def kv_weighted_sum_nf4(probs: torch.Tensor, v_packed: torch.Tensor,
                        v_absmax: torch.Tensor, head_dim: int,
                        block_t: int = 128,
                        token_group: int | None = None) -> torch.Tensor:
    """``probs [H_q, T]`` against a packed value cache -> ``out [H_q, D]`` fp32."""
    H_q, T = probs.shape
    _, H_kv, _ = v_packed.shape
    _check(head_dim)
    _require_inner_contig(v_packed=v_packed, v_absmax=v_absmax)
    _check_cache(v_packed, v_absmax, head_dim, token_group, "v")
    gqa = _gqa(H_q, H_kv)
    out = torch.empty(H_q, head_dim, dtype=torch.float32, device=probs.device)
    _kv_wsum_nf4[(H_q,)](
        probs.contiguous(), v_packed, v_absmax, out, _lut(probs.device),
        T, head_dim, gqa,
        v_packed.stride(0), v_packed.stride(1),
        v_absmax.stride(0), v_absmax.stride(1),
        BLOCK_T=block_t, BLOCK_D=triton.next_power_of_2(head_dim),
        QBLK=1 if token_group else BLOCKSIZE, TGRP=token_group or 1,
    )
    return out


def attend_nf4_kv(q: torch.Tensor, k_packed: torch.Tensor, k_absmax: torch.Tensor,
                  v_packed: torch.Tensor, v_absmax: torch.Tensor,
                  scale: float | None = None, block_t: int = 128,
                  k_token_group: int | None = None,
                  v_token_group: int | None = None) -> torch.Tensor:
    """One decode attention step over a 4-bit cache. ``q [H_q, D]`` -> ``[H_q, D]``.

    Softmax runs in fp32 on the fp32 scores; only the two matmuls touch the
    packed cache. No causal mask argument: a decode step attends to every cached
    token by construction.
    """
    H_q, D = q.shape
    scale = D ** -0.5 if scale is None else scale
    if k_packed.shape[0] != v_packed.shape[0]:
        raise ValueError(
            f"K and V hold different token counts ({k_packed.shape[0]} vs "
            f"{v_packed.shape[0]}); the softmax over K would not align with V.")
    if k_packed.shape[1] != v_packed.shape[1]:
        # Each kernel derives its own GQA factor from its own tensor, so a
        # mismatch silently maps one query head onto DIFFERENT kv heads for
        # scores and for the weighted sum -- wrong attention, no error.
        raise ValueError(
            f"K and V have different kv-head counts ({k_packed.shape[1]} vs "
            f"{v_packed.shape[1]}); each kernel would derive a different GQA "
            "map and query heads would read mismatched K/V rows.")
    scores = kv_scores_nf4(q, k_packed, k_absmax, block_t=block_t,
                           token_group=k_token_group) * scale
    probs = torch.softmax(scores, dim=-1)
    return kv_weighted_sum_nf4(probs, v_packed, v_absmax, D, block_t=block_t,
                               token_group=v_token_group)


@triton.jit
def _kv_attend_fused(
    q_ptr, k_ptr, ka_ptr, v_ptr, va_ptr, out_ptr, lut_ptr,
    T, D, GQA, scale,
    stride_kt, stride_kh, stride_kat, stride_kah,
    stride_vt, stride_vh, stride_vat, stride_vah,
    BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr, QBLK: tl.constexpr,
    TGRP: tl.constexpr,
):
    """One decode step, ONE pass over the cache: online-softmax flash decode.

    The v1 path reads the cache twice — once for scores, once for the weighted
    sum — which is why it cost 2.5-3x fp16 SDPA despite moving 4x fewer bytes.
    Here each token block's K and V are loaded once and consumed immediately,
    carrying the running max/denominator so the softmax never needs a second
    look at the scores.
    """
    h = tl.program_id(0)
    hkv = h // GQA
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < D
    q = tl.load(q_ptr + h * D + offs_d, mask=d_mask, other=0.0).to(tl.float32)

    m_i = float("-inf")                      # running max
    l_i = 0.0                                # running denominator
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    for lo in range(0, T, BLOCK_T):
        offs_t = lo + tl.arange(0, BLOCK_T)
        t_mask = offs_t < T
        m2 = t_mask[:, None] & d_mask[None, :]

        kb = k_ptr + offs_t[:, None] * stride_kt + hkv * stride_kh
        kby = tl.load(kb + (offs_d[None, :] // 2), mask=m2, other=0).to(tl.int32)
        knib = tl.where((offs_d[None, :] % 2) == 0, (kby >> 4) & 0xF, kby & 0xF)
        kam = tl.load(ka_ptr + (offs_t[:, None] // TGRP) * stride_kat
                      + hkv * stride_kah + (offs_d[None, :] // QBLK),
                      mask=m2, other=0.0)
        s = tl.sum(tl.load(lut_ptr + knib) * kam * q[None, :], axis=1) * scale
        s = tl.where(t_mask, s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, axis=0))
        corr = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new)
        p = tl.where(t_mask, p, 0.0)

        vb = v_ptr + offs_t[:, None] * stride_vt + hkv * stride_vh
        vby = tl.load(vb + (offs_d[None, :] // 2), mask=m2, other=0).to(tl.int32)
        vnib = tl.where((offs_d[None, :] % 2) == 0, (vby >> 4) & 0xF, vby & 0xF)
        vam = tl.load(va_ptr + (offs_t[:, None] // TGRP) * stride_vat
                      + hkv * stride_vah + (offs_d[None, :] // QBLK),
                      mask=m2, other=0.0)
        v = tl.load(lut_ptr + vnib) * vam

        acc = acc * corr + tl.sum(p[:, None] * v, axis=0)
        l_i = l_i * corr + tl.sum(p, axis=0)
        m_i = m_new

    tl.store(out_ptr + h * D + offs_d, acc / l_i, mask=d_mask)


def attend_nf4_kv_fused(q: torch.Tensor, k_packed: torch.Tensor, k_absmax: torch.Tensor,
                        v_packed: torch.Tensor, v_absmax: torch.Tensor,
                        scale: float | None = None, block_t: int = 128,
                        k_token_group: int | None = None,
                        v_token_group: int | None = None) -> torch.Tensor:
    """Single-pass equivalent of :func:`attend_nf4_kv`. ``q [H_q, D]`` -> ``[H_q, D]``.

    Requires K and V to share a scaling mode, since one kernel reads both with
    the same constexpr divisors. Mixed modes must use the two-pass path.
    """
    H_q, D = q.shape
    T, H_kv, _ = k_packed.shape
    _check(D)
    _require_inner_contig(k_packed=k_packed, k_absmax=k_absmax,
                          v_packed=v_packed, v_absmax=v_absmax)
    _check_cache(k_packed, k_absmax, D, k_token_group, "k")
    _check_cache(v_packed, v_absmax, D, v_token_group, "v")
    if k_token_group != v_token_group:
        raise ValueError(
            f"fused path needs one scaling mode for K and V (got {k_token_group} "
            "and {v_token_group}); use attend_nf4_kv for mixed modes.")
    if k_packed.shape[0] != v_packed.shape[0] or k_packed.shape[1] != v_packed.shape[1]:
        raise ValueError("K and V must agree on token count and kv-head count")
    gqa = _gqa(H_q, H_kv)
    out = torch.empty(H_q, D, dtype=torch.float32, device=q.device)
    _kv_attend_fused[(H_q,)](
        q.contiguous(), k_packed, k_absmax, v_packed, v_absmax, out, _lut(q.device),
        T, D, gqa, D ** -0.5 if scale is None else scale,
        k_packed.stride(0), k_packed.stride(1), k_absmax.stride(0), k_absmax.stride(1),
        v_packed.stride(0), v_packed.stride(1), v_absmax.stride(0), v_absmax.stride(1),
        BLOCK_T=block_t, BLOCK_D=triton.next_power_of_2(D),
        QBLK=1 if k_token_group else BLOCKSIZE, TGRP=k_token_group or 1,
    )
    return out


@triton.jit
def _kv_attend_split(
    q_ptr, k_ptr, ka_ptr, v_ptr, va_ptr,
    om_ptr, ol_ptr, oacc_ptr, lut_ptr,
    T, D, GQA, scale, SPLIT_T,
    stride_kt, stride_kh, stride_kat, stride_kah,
    stride_vt, stride_vh, stride_vat, stride_vah,
    BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr, QBLK: tl.constexpr,
    TGRP: tl.constexpr, S: tl.constexpr,
):
    """Flash-decoding pass 1: partial (m, l, acc) over one slice of the tokens.

    The single-program-per-head version of this kernel was measured SLOWER than
    the two-pass path at 32K (0.79x) because 64 programs cannot fill 26 SMs
    while each walks 256 blocks serially. Splitting the token axis restores
    parallelism to H_q x S without putting the scores intermediate back in
    memory: the partials are [H_q, S, D], ~2 MB at S=8 against 33.6 MB.
    """
    h = tl.program_id(0)
    sp = tl.program_id(1)
    hkv = h // GQA
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < D
    q = tl.load(q_ptr + h * D + offs_d, mask=d_mask, other=0.0).to(tl.float32)

    t_start = sp * SPLIT_T
    t_end = tl.minimum(t_start + SPLIT_T, T)
    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    for lo in range(t_start, t_end, BLOCK_T):
        offs_t = lo + tl.arange(0, BLOCK_T)
        t_mask = offs_t < t_end
        m2 = t_mask[:, None] & d_mask[None, :]

        kb = k_ptr + offs_t[:, None] * stride_kt + hkv * stride_kh
        kby = tl.load(kb + (offs_d[None, :] // 2), mask=m2, other=0).to(tl.int32)
        knib = tl.where((offs_d[None, :] % 2) == 0, (kby >> 4) & 0xF, kby & 0xF)
        kam = tl.load(ka_ptr + (offs_t[:, None] // TGRP) * stride_kat
                      + hkv * stride_kah + (offs_d[None, :] // QBLK),
                      mask=m2, other=0.0)
        s = tl.sum(tl.load(lut_ptr + knib) * kam * q[None, :], axis=1) * scale
        s = tl.where(t_mask, s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, axis=0))
        corr = tl.exp(m_i - m_new)
        p = tl.where(t_mask, tl.exp(s - m_new), 0.0)

        vb = v_ptr + offs_t[:, None] * stride_vt + hkv * stride_vh
        vby = tl.load(vb + (offs_d[None, :] // 2), mask=m2, other=0).to(tl.int32)
        vnib = tl.where((offs_d[None, :] % 2) == 0, (vby >> 4) & 0xF, vby & 0xF)
        vam = tl.load(va_ptr + (offs_t[:, None] // TGRP) * stride_vat
                      + hkv * stride_vah + (offs_d[None, :] // QBLK),
                      mask=m2, other=0.0)
        acc = acc * corr + tl.sum(p[:, None] * (tl.load(lut_ptr + vnib) * vam), axis=0)
        l_i = l_i * corr + tl.sum(p, axis=0)
        m_i = m_new

    tl.store(om_ptr + h * S + sp, m_i)
    tl.store(ol_ptr + h * S + sp, l_i)
    tl.store(oacc_ptr + (h * S + sp) * D + offs_d, acc, mask=d_mask)


@triton.jit
def _kv_combine(m_ptr, l_ptr, acc_ptr, out_ptr, D,
                BLOCK_D: tl.constexpr, S_P2: tl.constexpr, S: tl.constexpr):
    """Flash-decoding pass 2: log-sum-exp merge of the per-split partials.

    An empty split leaves m = -inf and l = 0, so its exp(m - M) weight is 0 and
    it drops out without a special case — provided at least one split saw a
    token, which the launcher guarantees by construction.
    """
    h = tl.program_id(0)
    sp = tl.arange(0, S_P2)
    s_mask = sp < S
    m = tl.load(m_ptr + h * S + sp, mask=s_mask, other=float("-inf"))
    l = tl.load(l_ptr + h * S + sp, mask=s_mask, other=0.0)
    M = tl.max(m, axis=0)
    w = tl.where(s_mask, tl.exp(m - M), 0.0)
    denom = tl.sum(l * w, axis=0)

    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < D
    a = tl.load(acc_ptr + (h * S + sp[:, None]) * D + offs_d[None, :],
                mask=s_mask[:, None] & d_mask[None, :], other=0.0)
    num = tl.sum(a * w[:, None], axis=0)
    tl.store(out_ptr + h * D + offs_d, num / denom, mask=d_mask)


def attend_nf4_kv_split(q: torch.Tensor, k_packed: torch.Tensor, k_absmax: torch.Tensor,
                        v_packed: torch.Tensor, v_absmax: torch.Tensor,
                        scale: float | None = None, block_t: int = 128,
                        token_group: int | None = None,
                        splits: int | None = None) -> torch.Tensor:
    """Flash-decoding: split the token axis, then combine. ``q [H_q, D]``.

    ``splits`` defaults to whatever brings the program count to roughly 512,
    bounded by the number of token blocks available — one program per block is
    the finest split that does any good.
    """
    H_q, D = q.shape
    T, H_kv, _ = k_packed.shape
    _check(D)
    _require_inner_contig(k_packed=k_packed, k_absmax=k_absmax,
                          v_packed=v_packed, v_absmax=v_absmax)
    _check_cache(k_packed, k_absmax, D, token_group, "k")
    _check_cache(v_packed, v_absmax, D, token_group, "v")
    if k_packed.shape[:2] != v_packed.shape[:2]:
        raise ValueError("K and V must agree on token count and kv-head count")
    gqa = _gqa(H_q, H_kv)
    n_blocks = max(1, (T + block_t - 1) // block_t)
    if splits is None:
        splits = max(1, min(n_blocks, -(-512 // H_q)))
    split_t = -(-T // splits) if splits > 1 else T
    split_t = max(split_t, block_t)                  # never smaller than a block
    splits = max(1, -(-T // split_t))

    m = torch.empty(H_q, splits, dtype=torch.float32, device=q.device)
    l = torch.empty(H_q, splits, dtype=torch.float32, device=q.device)
    acc = torch.empty(H_q, splits, D, dtype=torch.float32, device=q.device)
    out = torch.empty(H_q, D, dtype=torch.float32, device=q.device)
    bd = triton.next_power_of_2(D)
    _kv_attend_split[(H_q, splits)](
        q.contiguous(), k_packed, k_absmax, v_packed, v_absmax, m, l, acc,
        _lut(q.device), T, D, gqa, D ** -0.5 if scale is None else scale, split_t,
        k_packed.stride(0), k_packed.stride(1), k_absmax.stride(0), k_absmax.stride(1),
        v_packed.stride(0), v_packed.stride(1), v_absmax.stride(0), v_absmax.stride(1),
        BLOCK_T=block_t, BLOCK_D=bd,
        QBLK=1 if token_group else BLOCKSIZE, TGRP=token_group or 1, S=splits,
    )
    _kv_combine[(H_q,)](m, l, acc, out, D, BLOCK_D=bd,
                        S_P2=triton.next_power_of_2(splits), S=splits)
    return out


@triton.jit
def _kv_scores_gqa(q_ptr, k_ptr, ka_ptr, out_ptr, lut_ptr,
                   T, D, GQA, stride_kt, stride_kh, stride_at, stride_ah,
                   BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr,
                   BLOCK_M: tl.constexpr, QBLK: tl.constexpr, TGRP: tl.constexpr,
                   PREC: tl.constexpr):
    """scores for ALL query heads sharing one kv head, dequantizing K once.

    The measured problem with ``grid=(H_q, ...)``: dequant work scaled with
    QUERY heads while the bytes are indexed by KV heads, so GQA 16:1 paid 16x
    redundant ALU on byte-identical input (5.59 ms at 1:1 vs 13.83 ms at 16:1,
    same 4 kv heads). Here the grid is over kv heads, the block is dequantized
    once, and the per-head dot products become one [GQA, D] x [D, BLOCK_T] dot.
    """
    hkv = tl.program_id(0)
    pid_t = tl.program_id(1)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = offs_t < T
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < D
    offs_m = tl.arange(0, BLOCK_M)
    m_mask = offs_m < GQA

    kb = k_ptr + offs_t[:, None] * stride_kt + hkv * stride_kh
    m2 = t_mask[:, None] & d_mask[None, :]
    byts = tl.load(kb + (offs_d[None, :] // 2), mask=m2, other=0).to(tl.int32)
    nib = tl.where((offs_d[None, :] % 2) == 0, (byts >> 4) & 0xF, byts & 0xF)
    am = tl.load(ka_ptr + (offs_t[:, None] // TGRP) * stride_at + hkv * stride_ah
                 + (offs_d[None, :] // QBLK), mask=m2, other=0.0)
    kblk = tl.load(lut_ptr + nib) * am                       # [BLOCK_T, BLOCK_D]

    # q rows for the GQA heads that map to this kv head
    qrow = hkv * GQA + offs_m
    q = tl.load(q_ptr + qrow[:, None] * D + offs_d[None, :],
                mask=m_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    # input_precision matters here: tl.dot defaults to TF32 on Ampere (~10-bit
    # mantissa), which measured 2.7e-3 relative error at logits ~40x -- small,
    # but it is a SECOND error source stacked on the quantization error this
    # module exists to characterize, and the two would be inseparable. "ieee"
    # keeps them separable; the speed cost is measured, not assumed.
    acc = tl.dot(q, tl.trans(kblk), input_precision=PREC)     # [BLOCK_M, BLOCK_T]
    tl.store(out_ptr + qrow[:, None] * T + offs_t[None, :], acc,
             mask=m_mask[:, None] & t_mask[None, :])


@triton.jit
def _kv_wsum_gqa(p_ptr, v_ptr, va_ptr, out_ptr, lut_ptr,
                 T, D, GQA, SPLIT_T, stride_vt, stride_vh, stride_at, stride_ah,
                 BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr,
                 BLOCK_M: tl.constexpr, QBLK: tl.constexpr, TGRP: tl.constexpr,
                 S: tl.constexpr, PREC: tl.constexpr):
    """Weighted sum for all query heads of one kv head, V dequantized once.

    Split over tokens for occupancy (only H_kv programs otherwise — 4 on the
    shapes measured); partials are summed outside, which keeps the reduction
    deterministic rather than relying on atomics.
    """
    hkv = tl.program_id(0)
    sp = tl.program_id(1)
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < D
    offs_m = tl.arange(0, BLOCK_M)
    m_mask = offs_m < GQA
    qrow = hkv * GQA + offs_m
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    t_start = sp * SPLIT_T
    t_end = tl.minimum(t_start + SPLIT_T, T)
    for lo in range(t_start, t_end, BLOCK_T):
        offs_t = lo + tl.arange(0, BLOCK_T)
        t_mask = offs_t < t_end
        m2 = t_mask[:, None] & d_mask[None, :]
        vb = v_ptr + offs_t[:, None] * stride_vt + hkv * stride_vh
        byts = tl.load(vb + (offs_d[None, :] // 2), mask=m2, other=0).to(tl.int32)
        nib = tl.where((offs_d[None, :] % 2) == 0, (byts >> 4) & 0xF, byts & 0xF)
        am = tl.load(va_ptr + (offs_t[:, None] // TGRP) * stride_at + hkv * stride_ah
                     + (offs_d[None, :] // QBLK), mask=m2, other=0.0)
        vblk = tl.load(lut_ptr + nib) * am                    # [BLOCK_T, BLOCK_D]
        p = tl.load(p_ptr + qrow[:, None] * T + offs_t[None, :],
                    mask=m_mask[:, None] & t_mask[None, :], other=0.0)
        acc += tl.dot(p, vblk, input_precision=PREC)
    tl.store(out_ptr + ((hkv * S + sp) * BLOCK_M + offs_m[:, None]) * D + offs_d[None, :],
             acc, mask=m_mask[:, None] & d_mask[None, :])


def attend_nf4_kv_gqa(q: torch.Tensor, k_packed: torch.Tensor, k_absmax: torch.Tensor,
                      v_packed: torch.Tensor, v_absmax: torch.Tensor,
                      scale: float | None = None, block_t: int = 128,
                      token_group: int | None = None,
                      splits: int | None = None,
                      precision: str = "ieee") -> torch.Tensor:
    """Decode attention over a 4-bit cache, batched over the GQA group.

    Query heads are assumed contiguous per kv head (head h reads kv head
    ``h // GQA``), which is the mapping the rest of this module already uses.
    """
    H_q, D = q.shape
    T, H_kv, _ = k_packed.shape
    _check(D)
    _require_inner_contig(k_packed=k_packed, k_absmax=k_absmax,
                          v_packed=v_packed, v_absmax=v_absmax)
    _check_cache(k_packed, k_absmax, D, token_group, "k")
    _check_cache(v_packed, v_absmax, D, token_group, "v")
    if k_packed.shape[:2] != v_packed.shape[:2]:
        raise ValueError("K and V must agree on token count and kv-head count")
    gqa = _gqa(H_q, H_kv)
    bd, bm = triton.next_power_of_2(D), max(16, triton.next_power_of_2(gqa))
    qb, tg = (1 if token_group else BLOCKSIZE), (token_group or 1)
    scale = D ** -0.5 if scale is None else scale

    # This kernel stages a dequantized [BLOCK_T, BLOCK_D] fp32 tile: 64 KB at
    # 128x128 before the dot operands, against a ~99 KB cap on this device. A
    # static estimate is not enough -- triton's num_stages pipelining multiplies
    # the staging by a factor the caller does not control, so the analytic
    # figure (82 KB here) can pass while the launch still fails. The device cap
    # prunes the obviously-too-large configs, and an OutOfResources retry
    # handles the rest; num_stages is pinned low to keep the multiplier small.
    cap = _device_shared_limit(q.device)
    if cap:
        while block_t > 16 and (block_t * bd + bm * bd + bm * block_t) * 4 > cap:
            block_t //= 2

    from triton.runtime.errors import OutOfResources
    qc, lut = q.contiguous(), _lut(q.device)
    while True:
        scores = torch.zeros(H_q, T, dtype=torch.float32, device=q.device)
        try:
            _kv_scores_gqa[(H_kv, triton.cdiv(T, block_t))](
                qc, k_packed, k_absmax, scores, lut,
                T, D, gqa, k_packed.stride(0), k_packed.stride(1),
                k_absmax.stride(0), k_absmax.stride(1),
                BLOCK_T=block_t, BLOCK_D=bd, BLOCK_M=bm, QBLK=qb, TGRP=tg,
                PREC=precision, num_stages=2)
            break
        except OutOfResources:
            if block_t <= 16:
                raise
            block_t //= 2
    probs = torch.softmax(scores * scale, dim=-1)

    n_blocks = max(1, triton.cdiv(T, block_t))
    if splits is None:
        splits = max(1, min(n_blocks, -(-512 // max(H_kv, 1))))
    split_t = max(block_t, -(-T // splits))
    splits = max(1, -(-T // split_t))
    while True:
        part = torch.empty(H_kv, splits, bm, D, dtype=torch.float32, device=q.device)
        try:
            _kv_wsum_gqa[(H_kv, splits)](
                probs, v_packed, v_absmax, part, lut,
                T, D, gqa, split_t, v_packed.stride(0), v_packed.stride(1),
                v_absmax.stride(0), v_absmax.stride(1),
                BLOCK_T=block_t, BLOCK_D=bd, BLOCK_M=bm, QBLK=qb, TGRP=tg,
                S=splits, PREC=precision, num_stages=2)
            break
        except OutOfResources:
            if block_t <= 16:
                raise
            block_t //= 2
    return part.sum(1)[:, :gqa, :].reshape(H_q, D)
