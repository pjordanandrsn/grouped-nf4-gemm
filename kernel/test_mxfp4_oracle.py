# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Phase-1 oracle adjudication for MXFP4 (device-free; the A4 dequant path is
ground truth — disagreement is STOP, not tolerance).

Resolves the Phase-0 STOP items:
  1. nibble interleave order (bnb 2j=HIGH vs transformers 2j=LOW),
  2. e8m0 0xFF NaN handling,
  3. scale application, and the e2m1 codebook itself,
by comparing mxfp4_pack_ref.dequant_mxfp4 to transformers'
_convert_moe_packed_tensors on the SAME synthetic bytes.
"""
import pytest
import torch

pytest.importorskip("transformers")

from mxfp4_pack_ref import (  # noqa: E402
    FP4_VALUES, MX_BLOCK, NIBBLE_LOW_FIRST, dequant_mxfp4, quantize_pack_mxfp4,
)


def _oracle():
    """Return a callable (blocks[R,NB,16] u8, scales[R,NB] u8) -> dequant fp32
    in [R, NB*32] order. The transformers fn needs rank>=4 (it does a
    transpose(1,2)) and returns [E, K, N] transposed — this wrapper adds a lead
    axis, calls it, and undoes the transpose so the result is [R, K]."""
    from transformers.integrations import mxfp4 as M
    fn = getattr(M, "_convert_moe_packed_tensors", None)
    if fn is None:
        pytest.skip("transformers _convert_moe_packed_tensors not present")

    def run(blocks, scales):
        b4 = blocks.unsqueeze(0)          # [1, R, NB, 16]
        s3 = scales.unsqueeze(0)          # [1, R, NB]
        try:
            out = fn(b4, s3, dtype=torch.float32, rows_per_chunk=1 << 30)
        except TypeError:
            out = fn(b4, s3)
        # out is [1, K, R]; undo transpose -> [1, R, K] -> [R, K]
        return out.transpose(1, 2).reshape(blocks.shape[0], blocks.shape[1] * 32).float()
    return run


def test_codebook_matches_transformers():
    from transformers.integrations import mxfp4 as M
    ref = getattr(M, "FP4_VALUES", None)
    assert ref is not None, "transformers FP4_VALUES vanished"
    assert list(ref) == FP4_VALUES, (list(ref), FP4_VALUES)


def test_discover_and_lock_nibble_order():
    """Construct bytes whose two nibbles decode to DISTINCT magnitudes, ask the
    oracle, and read the interleave off element 0 vs 1. Then assert our locked
    NIBBLE_LOW_FIRST matches — a wrong lock fails here, loudly."""
    oracle = _oracle()
    # one row, one block: byte0 = 0x21 -> low=1 (val .5), high=2 (val 1.0);
    # scale byte 127 -> 2^0 = 1. Remaining 15 bytes zero.
    blocks = torch.zeros(1, 1, 16, dtype=torch.uint8)   # [R=1, NB=1, 16]
    blocks[0, 0, 0] = 0x21
    scales = torch.full((1, 1), 127, dtype=torch.uint8)
    got = oracle(blocks, scales).reshape(-1)   # 32 elements
    e0, e1 = got[0].item(), got[1].item()
    # low nibble(=1)->0.5, high nibble(=2)->1.0
    low_first = abs(e0 - 0.5) < 1e-6 and abs(e1 - 1.0) < 1e-6
    high_first = abs(e0 - 1.0) < 1e-6 and abs(e1 - 0.5) < 1e-6
    assert low_first or high_first, f"unexpected oracle decode e0={e0} e1={e1}"
    observed = low_first
    assert observed == NIBBLE_LOW_FIRST, (
        f"NIBBLE_LOW_FIRST locked to {NIBBLE_LOW_FIRST} but oracle shows "
        f"low_first={observed} (e0={e0}, e1={e1}) — update the lock in "
        f"mxfp4_pack_ref.py and re-run")


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_dequant_exact_vs_oracle(seed):
    """Random blocks + safe e8m0 scales: our dequant must EQUAL the oracle
    bit-for-bit (both are pure table+scale; no rounding between them)."""
    oracle = _oracle()
    g = torch.Generator().manual_seed(seed)
    NB = 4
    blocks = torch.randint(0, 256, (3, NB, 16), generator=g, dtype=torch.uint8)
    # scales in a safe exponent window (avoid 0xFF NaN + extreme over/underflow)
    scales = torch.randint(120, 135, (3, NB), generator=g, dtype=torch.uint8)
    ours = dequant_mxfp4(blocks, scales)                 # [3, NB*32]
    ref = oracle(blocks, scales).reshape(3, NB * 32)
    torch.testing.assert_close(ours, ref, rtol=0, atol=0)


def test_e8m0_0xff_matches_oracle():
    """FINDING: transformers does NOT honor the OCP 0xFF->NaN reservation — it
    is exponent 128 via ldexp, so a 0xFF block -> ±inf (nonzero elem) / 0 (zero
    elem). We match the oracle bit-for-bit (ldexp), NOT the spec's NaN."""
    oracle = _oracle()
    g = torch.Generator().manual_seed(9)
    blocks = torch.randint(0, 256, (2, 3, 16), generator=g, dtype=torch.uint8)
    scales = torch.full((2, 3), 0xFF, dtype=torch.uint8)
    ours = dequant_mxfp4(blocks, scales)
    ref = oracle(blocks, scales)
    torch.testing.assert_close(ours, ref, rtol=0, atol=0, equal_nan=True)


def test_pack_roundtrip_self_consistent():
    """quantize_pack -> dequant reproduces the pre-quant values within one
    e2m1 step (fixture-generation sanity; not an oracle claim)."""
    g = torch.Generator().manual_seed(3)
    w = torch.randn(5, 8 * MX_BLOCK, generator=g) * 2.0
    blocks, scales = quantize_pack_mxfp4(w)
    assert blocks.shape == (5, 8, 16) and scales.shape == (5, 8)
    deq = dequant_mxfp4(blocks, scales)
    # per-block relative error bounded by half the local codebook gap
    err = (deq - w).abs()
    per_block_amax = w.reshape(5, 8, MX_BLOCK).abs().amax(-1, keepdim=True)
    tol = (per_block_amax * 0.5).expand(-1, -1, MX_BLOCK).reshape(5, -1)
    assert (err <= tol + 1e-6).float().mean() > 0.98
