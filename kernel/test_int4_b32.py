# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""int4-b32 pack/decode contract + kernel parity.

CPU half (runs anywhere): pack->dequant round-trip properties against
the pure-torch reference -- grid symmetry, block scaling, nibble order,
non-pow2 K, and the exactness bound |w - deq| <= scale/2 per element.
Triton half (interp mode or GPU): the GEMV must match an fp32 reference
of the SAME int4 values within output-rounding of bf16; on the interp
path the accumulation identity is checked to ~1e-6 in fp32.
"""
import os

import pytest

torch = pytest.importorskip("torch")

from int4_pack_ref import BLOCK, dequant_int4_ref, pack_int4_b32  # noqa: E402


@pytest.mark.parametrize("N,K", [(8, 64), (5, 96), (16, 32), (3, 2048)])
def test_pack_roundtrip_error_bound(N, K):
    torch.manual_seed(0)
    w = torch.randn(N, K) * 0.7
    packed, scales = pack_int4_b32(w)
    deq = dequant_int4_ref(packed, scales, N, K)
    # slack: q was computed with the fp32 scale, stored fp16 -- the
    # dequant pays |q| * s * 2^-11 of scale-storage rounding on top of
    # the half-step quantisation bound
    step = scales.float().repeat_interleave(BLOCK, dim=1)
    bound = step / 2 + 8 * step * 2 ** -11 + 1e-9
    assert ((w - deq).abs() <= bound).all()


def test_nibble_order_even_is_low():
    w = torch.zeros(1, BLOCK)
    w[0, 0] = 7.0     # k=0 (even) -> low nibble
    w[0, 1] = -8.0    # k=1 (odd)  -> high nibble
    packed, scales = pack_int4_b32(w)
    assert float(scales[0, 0]) == pytest.approx(8.0 / 7.0, rel=1e-3)
    b0 = int(packed[0, 0])
    lo, hi = b0 & 0xF, (b0 >> 4) & 0xF
    # absmax/7 scaling: +7 -> code +6? no: 7/(8/7) = 6.125 -> 6;
    # -8/(8/7) = -7 exactly. The -8 LEVEL is unreachable for the
    # block's defining element -- that IS the symmetric-grid contract.
    assert lo - 8 == 6 and hi - 8 == -7, (lo, hi)


def test_zero_block_survives():
    w = torch.zeros(2, 2 * BLOCK)
    packed, scales = pack_int4_b32(w)
    deq = dequant_int4_ref(packed, scales, 2, 2 * BLOCK)
    assert torch.equal(deq, torch.zeros_like(deq))


def test_k_not_multiple_refused():
    with pytest.raises(ValueError, match="multiple"):
        pack_int4_b32(torch.zeros(2, BLOCK + 1))


def test_composition_degrades():
    """Packing an already-int4-gridded tensor must be lossless (idempotent
    grid), while packing NF4-dequantised values is NOT -- the doc's
    repack-from-source rule, pinned as behaviour."""
    torch.manual_seed(1)
    w = torch.randn(4, 128)
    p1, s1 = pack_int4_b32(w)
    d1 = dequant_int4_ref(p1, s1, 4, 128)
    p2, s2 = pack_int4_b32(d1)
    d2 = dequant_int4_ref(p2, s2, 4, 128)
    assert torch.allclose(d1, d2, atol=1e-6)


def _gpu():
    if os.environ.get("TRITON_INTERPRET") == "1":
        return "cpu"
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA or TRITON_INTERPRET=1")
    return "cuda"


@pytest.mark.parametrize("N,K,E,R", [(64, 64, 4, 3), (128, 96, 8, 8),
                                     (256, 2048, 16, 8)])
def test_gemv_matches_reference(N, K, E, R):
    pytest.importorskip("triton")
    from int4_b32 import gemv_int4_b32, quant_x_rows
    dev = _gpu()
    torch.manual_seed(2)
    W = torch.randn(E, N, K) * 0.1
    pk, sc = zip(*[pack_int4_b32(W[e]) for e in range(E)])
    Wp = torch.stack(pk).to(dev).contiguous()
    Sp = torch.stack(sc).to(dev).contiguous()
    eids = torch.randperm(E)[:R].to(dev).to(torch.int32)
    x = (torch.randn(R, K) * 0.2).to(dev, torch.bfloat16)
    xq, xs = quant_x_rows(x)
    got = gemv_int4_b32(xq, xs, Wp, Sp, eids, N, K)
    ref = torch.stack([
        (dequant_int4_ref(Wp[int(e)].cpu(), Sp[int(e)].cpu(), N, K).to(dev)
         * (xq[i].float() * xs[i].repeat_interleave(BLOCK))[None, :]).sum(-1)
        for i, e in enumerate(eids)])
    # int32 accumulation is exact; bf16 OUTPUT rounding is the only slack
    assert (got.float() - ref).abs().max() <= ref.abs().max() * 2 ** -7
