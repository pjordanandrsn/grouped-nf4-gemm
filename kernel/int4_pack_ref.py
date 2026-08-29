# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Pure-torch pack/decode for the uniform int4-b32 format (no triton).

The CPU-checkable half of :mod:`int4_b32`, split out the way
``nf4_pack_ref`` is from ``nf4_grouped`` so packing and its reference
decode import anywhere. See int4_b32's docstring for the format's
measured basis and the lm_head exclusion.
"""
from __future__ import annotations

import torch

BLOCK = 32


def pack_int4_b32(w: torch.Tensor):
    """``w [N, K] float`` -> ``(packed [N, K//2] uint8, scales [N, K//32] fp16)``.

    Symmetric per-32-block absmax grid, levels -8..7 stored offset-binary
    (0..15); even k in the LOW nibble. Pack from SOURCE weights only --
    quantising onto an already-quantised grid measured ~7x the pure grid's
    own ppl cost (the composition lesson, receipts INT4GATE/PUREINT4).
    """
    N, K = w.shape
    if K % BLOCK:
        raise ValueError(f"K={K} must be a multiple of {BLOCK}")
    b = w.detach().float().reshape(N, K // BLOCK, BLOCK)
    s = b.abs().amax(-1).clamp_min(1e-12) / 7.0
    q = (b / s[..., None]).round().clamp(-8, 7).to(torch.int8)
    qq = (q + 8).to(torch.uint8).reshape(N, K)
    packed = (qq[:, 0::2] | (qq[:, 1::2] << 4)).contiguous()
    return packed, s.to(torch.float16).contiguous()


def dequant_int4_ref(packed: torch.Tensor, scales: torch.Tensor,
                     N: int, K: int) -> torch.Tensor:
    """Pure-torch reference decode of :func:`pack_int4_b32` bytes (fp32).
    Runs on CPU; the property tests pin the kernels against it."""
    lo = (packed.to(torch.int16) & 0xF) - 8
    hi = ((packed.to(torch.int16) >> 4) & 0xF) - 8
    q = torch.stack([lo, hi], dim=-1).reshape(N, K).float()
    return (q.reshape(N, K // BLOCK, BLOCK)
            * scales.float()[..., None]).reshape(N, K)


