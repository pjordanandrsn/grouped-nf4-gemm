# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Pure-torch NF4 packer — bnb-free fixture generation for CPU-only CI.

Matches the contract's storage semantics (the same layout `repack_from_bnb`
produces): per 64-element K-block, absmax = max|w|; each element quantizes to
the nearest NF4 codebook entry of w/absmax; element 2j packs into the HIGH
nibble of byte j, 2j+1 into the LOW nibble.

This does NOT claim bit-exactness with bitsandbytes' quantizer (ties can
round differently); bnb equivalence is pinned by the GPU property suite
(test_nf4_grouped.py) on real hardware. What this enables is DEVICE-FREE
contract testing: pack -> kernel -> compare against dequant_ref of the same
bytes, which is self-consistent regardless of how the bytes were chosen.
"""
import torch

from nf4_grouped import BLOCKSIZE, NF4_LUT


def quantize_pack_nf4(w: torch.Tensor):
    """w [N, K] float -> (packed [N, K//2] uint8, absmax [N, K//64] fp32).

    Example:
        >>> import torch
        >>> from nf4_pack_ref import quantize_pack_nf4
        >>> from nf4_grouped import dequant_ref
        >>> w = torch.randn(256, 512)
        >>> packed, absmax = quantize_pack_nf4(w)          # [256, 256] u8, [256, 8] f32
        >>> wq = dequant_ref(packed, absmax, 256, 512)     # round-trips to ~0.09 rel-err
    """
    if w.dim() != 2:
        raise ValueError(
            f"expected 2-D [N, K] per-expert weight; got shape {tuple(w.shape)} — "
            "pack per-expert and stack, or use make_stack() for synthetic fixtures."
        )
    N, K = w.shape
    assert K % BLOCKSIZE == 0, f"K={K} must be a multiple of {BLOCKSIZE}"
    lut = torch.tensor(NF4_LUT, dtype=torch.float32, device=w.device)
    blocks = w.float().reshape(N, K // BLOCKSIZE, BLOCKSIZE)
    absmax = blocks.abs().amax(dim=2).clamp_min(1e-12)          # [N, K/64]
    scaled = blocks / absmax[:, :, None]                          # in [-1, 1]
    codes = (scaled.reshape(N, K, 1) - lut).abs().argmin(dim=2)   # [N, K] int64
    hi = codes[:, 0::2].to(torch.uint8)
    lo = codes[:, 1::2].to(torch.uint8)
    packed = (hi << 4) | lo                                       # [N, K/2]
    return packed.contiguous(), absmax.to(torch.float32).contiguous()


def make_stack(E: int, N: int, K: int, seed: int = 0, device: str = "cpu"):
    """E experts of random weights -> (B [E,N,K/2] u8, absmax [E,N,K/64] f32)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    B = torch.empty(E, N, K // 2, dtype=torch.uint8)
    A = torch.empty(E, N, K // BLOCKSIZE, dtype=torch.float32)
    for e in range(E):
        w = torch.randn(N, K, generator=g, dtype=torch.float32)
        B[e], A[e] = quantize_pack_nf4(w)
    return B.to(device), A.to(device)
