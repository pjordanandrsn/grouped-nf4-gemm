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

from nf4_grouped import BLOCKSIZE, NF4_LUT, _lut

__all__ = [
    "quantize_kv",
    "quantize_kv_perchannel",
    "PERCHANNEL_GROUP",
    "dequant_kv_ref",
    "kv_scores_nf4",
    "kv_weighted_sum_nf4",
    "attend_nf4_kv",
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
