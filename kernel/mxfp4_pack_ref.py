# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Pure-torch MXFP4 (e2m1 element / e8m0 scale) reference — the executable spec,
mirroring nf4_pack_ref's role for the native gpt-oss checkpoint format.

Sources (R6, Phase-0 seam map): the e2m1 codebook is `FP4_VALUES` as used by
`transformers.integrations.mxfp4` (the A4 reference/oracle path) — 8 magnitudes
per sign, ±{0, .5, 1, 1.5, 2, 3, 4, 6}; the block size is k=32 and the scale is
e8m0 (OCP MX v1.0: unsigned biased-float32 exponent, bias 127, 0xFF reserved =
NaN). The checkpoint stores each expert projection as `blocks [.., N, n_blk, 16]`
uint8 (16 bytes = 32 packed fp4) and `scales [.., N, n_blk]` uint8 (one e8m0 per
block). Flattened, `n_blk*16 == K//2` — the same packed width as NF4's B tensor.

This does NOT assert bit-exactness by fiat; the nibble interleave order is the
Phase-0 STOP item and is ADJUDICATED against the oracle in test_mxfp4_oracle.py.
`NIBBLE_LOW_FIRST` below is the locked outcome of that adjudication (the test
fails loudly if it is ever wrong).
"""
import torch

# e2m1 codebook, index = (sign<<3)|magnitude; verbatim from transformers FP4_VALUES.
FP4_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
              -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
MX_BLOCK = 32          # elements per e8m0 scale (OCP MX v1.0)
E8M0_BIAS = 127        # OCP MX v1.0
E8M0_NAN = 0xFF        # reserved by OCP MX v1.0 — but see below

# Adjudicated against the oracle (transformers _convert_moe_packed_tensors),
# Phase-1: element 2j is the LOW nibble (`sub[:,0::2]=lut[blk & 0x0F]`), 2j+1
# the HIGH. (bnb/NF4 is the opposite — 2j=HIGH.) Locked True; the oracle test
# fails loudly if this is ever wrong.
NIBBLE_LOW_FIRST = True

# The oracle applies the scale with `torch.ldexp(x, s-127)` (= x * 2^(s-127))
# and does NOT implement the OCP `0xFF`->NaN reservation: 0xFF is just exponent
# 128, so a 0xFF block -> ±inf (nonzero) / 0 (zero element). We match the oracle
# EXACTLY (ldexp, not exp2-multiply) so the reference == ground truth bit-for-
# bit, including the 0xFF edge where `ldexp(0,128)=0` but `0*2**128=NaN`. The
# GPU kernel (Phase 2) may use exp2-multiply: real checkpoint scales are finite
# so the edge never arises there (noted; guarded by the GPU exact-decode gate).


def _lut(device, dtype=torch.float32):
    return torch.tensor(FP4_VALUES, dtype=dtype, device=device)


def dequant_mxfp4(blocks: torch.Tensor, scales: torch.Tensor,
                  dtype=torch.float32) -> torch.Tensor:
    """blocks [..., n_blk, 16] uint8, scales [..., n_blk] uint8 (e8m0)
    -> [..., n_blk*32] dequantized. `dtype` is the accumulation/output dtype."""
    assert blocks.dtype == torch.uint8 and scales.dtype == torch.uint8
    assert blocks.shape[-1] == 16 and blocks.shape[:-1] == scales.shape, \
        (blocks.shape, scales.shape)
    dev = blocks.device
    lut = _lut(dev, dtype)
    lo = (blocks & 0x0F).long()          # [..., n_blk, 16]
    hi = (blocks >> 4).long()
    # interleave two nibbles per byte -> 32 elements per block, in K order
    a, b = (lo, hi) if NIBBLE_LOW_FIRST else (hi, lo)
    nibs = torch.stack([a, b], dim=-1).reshape(*blocks.shape[:-1], 32)  # [..., n_blk, 32]
    vals = lut[nibs].to(dtype)           # codebook lookup -> [..., n_blk, 32]
    # e8m0 -> ldexp(x, s-127), matching the oracle exactly (0*2^n == 0, not NaN)
    exp = (scales.to(torch.int32) - E8M0_BIAS)[..., None]               # [..., n_blk, 1]
    out = torch.ldexp(vals, exp)         # [..., n_blk, 32]
    return out.reshape(*blocks.shape[:-2], blocks.shape[-2] * 32)


def quantize_pack_mxfp4(w: torch.Tensor):
    """w [..., K] float -> (blocks [..., K//32, 16] u8, scales [..., K//32] u8).
    Fixture generation for CI: per 32-block, choose the e8m0 scale as the
    power-of-two that puts max|w| at the top of e2m1's range (max 6.0), then
    round each scaled element to the nearest codebook entry. Not claimed
    bit-identical to any producer; self-consistent with dequant_mxfp4."""
    *lead, K = w.shape
    assert K % MX_BLOCK == 0, f"K={K} must be a multiple of {MX_BLOCK}"
    nb = K // MX_BLOCK
    wb = w.float().reshape(*lead, nb, MX_BLOCK)
    amax = wb.abs().amax(dim=-1).clamp_min(1e-20)                 # [..., nb]
    # target: amax * 2^-e <= 6.0  => e = ceil(log2(amax/6)); clamp to e8m0 range
    e = torch.ceil(torch.log2(amax / 6.0)).clamp(-E8M0_BIAS, E8M0_BIAS)
    scale = torch.exp2(e)                                          # [..., nb]
    scaled = wb / scale[..., None]                                # ~[-6,6]
    lut = _lut(w.device)
    codes = (scaled[..., None] - lut).abs().argmin(dim=-1)        # [..., nb, 32]
    lo_el = codes[..., 0::2].to(torch.uint8)                      # elements 2j
    hi_el = codes[..., 1::2].to(torch.uint8)                      # elements 2j+1
    if NIBBLE_LOW_FIRST:
        packed = (hi_el << 4) | lo_el   # 2j in low nibble, 2j+1 in high
    else:
        packed = (lo_el << 4) | hi_el   # 2j in high nibble (bnb convention)
    scales = (e + E8M0_BIAS).to(torch.uint8)
    return packed.contiguous(), scales.contiguous()
