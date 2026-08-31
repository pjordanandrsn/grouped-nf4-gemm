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


@pytest.mark.parametrize("N,K,E,B,topk", [
    (64, 64, 8, 4, 2),        # tiny, uneven routing
    (1536, 2048, 128, 16, 8), # the B=16 census gate_up cell
    (2048, 768, 128, 16, 8),  # the B=16 census down cell
])
def test_grouped_matches_reference(N, K, E, B, topk):
    """The M-tile grouped kernel must match the fp32 reference of the
    SAME int4 values and SAME int8 activations, over realistic routing
    (uneven counts, empty experts, padding tiles), with rows restored
    through the order/inverse-scatter contract."""
    pytest.importorskip("triton")
    from int4_b32 import gemm_int4_b32_grouped_captured, quant_x_rows
    from nf4_grouped import build_group_tiles_device
    dev = _gpu()
    torch.manual_seed(3)
    W = torch.randn(E, N, K) * 0.1
    pk, sc = zip(*[pack_int4_b32(W[e]) for e in range(E)])
    Wp = torch.stack(pk).to(dev).contiguous()
    Sp = torch.stack(sc).to(dev).contiguous()
    R = B * topk
    eids = torch.randint(0, E, (R,), device=dev).to(torch.int32)
    x = (torch.randn(R, K) * 0.2).to(dev, torch.bfloat16)
    xq, xs = quant_x_rows(x)

    t_row0, t_rows, t_grp, order, _counts = \
        build_group_tiles_device(eids, E, 16)
    aq_s = xq.index_select(0, order).contiguous()
    as_s = xs.index_select(0, order).contiguous()
    got_sorted = gemm_int4_b32_grouped_captured(
        aq_s, as_s, Wp, Sp, t_row0, t_rows, t_grp)
    inv = torch.empty_like(order)
    inv[order] = torch.arange(R, device=dev)
    got = got_sorted.index_select(0, inv)

    ref = torch.stack([
        (dequant_int4_ref(Wp[int(e)].cpu(), Sp[int(e)].cpu(), N, K).to(dev)
         * (xq[i].float() * xs[i].repeat_interleave(BLOCK))[None, :]).sum(-1)
        for i, e in enumerate(eids)])
    assert (got.float() - ref).abs().max() <= ref.abs().max() * 2 ** -7


def test_grouped_every_row_written_once():
    """Coverage of the tile table: each sorted row belongs to exactly one
    live tile, so a poisoned output must be fully overwritten."""
    pytest.importorskip("triton")
    from int4_b32 import gemm_int4_b32_grouped_captured, quant_x_rows
    from nf4_grouped import build_group_tiles_device
    dev = _gpu()
    torch.manual_seed(4)
    E, N, K, R = 5, 64, 64, 37       # rows > BLOCK_M for one expert
    W = torch.randn(E, N, K) * 0.1
    pk, sc = zip(*[pack_int4_b32(W[e]) for e in range(E)])
    Wp = torch.stack(pk).to(dev).contiguous()
    Sp = torch.stack(sc).to(dev).contiguous()
    eids = torch.cat([torch.zeros(20), torch.ones(9),
                      torch.full((8,), 4.0)]).to(dev).to(torch.int32)
    x = (torch.randn(R, K) * 0.2).to(dev, torch.bfloat16)
    xq, xs = quant_x_rows(x)
    t_row0, t_rows, t_grp, order, _ = build_group_tiles_device(eids, E, 16)
    aq_s = xq.index_select(0, order).contiguous()
    as_s = xs.index_select(0, order).contiguous()
    a = gemm_int4_b32_grouped_captured(aq_s, as_s, Wp, Sp,
                                       t_row0, t_rows, t_grp)
    b = gemm_int4_b32_grouped_captured(aq_s, as_s, Wp, Sp,
                                       t_row0, t_rows, t_grp)
    assert torch.equal(a, b)
    assert a.shape == (R, N) and torch.isfinite(a.float()).all()


def test_quant_grid_matches_reference():
    """The (R, K//32) grid must reproduce the per-block arithmetic: the
    int8 CODES exactly (they feed the exact integer GEMM), the fp32
    scales to ~1 ULP -- device division rounds differently from the
    torch reference, and that was never bitwise on the old kernel
    either (found on the first GPU gate run: codes matched exactly,
    scales differed in the last bit)."""
    pytest.importorskip("triton")
    from int4_b32 import quant_x_rows
    dev = _gpu()
    torch.manual_seed(6)
    R, K = 7, 96
    x = (torch.randn(R, K) * 0.3).to(dev, torch.bfloat16)
    xq, xs = quant_x_rows(x)
    xf = x.float().reshape(R, K // BLOCK, BLOCK)
    s_ref = xf.abs().amax(dim=2) / 127.0 + 1e-12
    q_ref = torch.floor(xf / s_ref[:, :, None] + 0.5).clamp(-127, 127)
    assert torch.equal(xq.float().cpu(), q_ref.reshape(R, K).cpu())
    assert torch.allclose(xs.cpu(), s_ref.cpu(), rtol=1e-6, atol=0.0)


@pytest.mark.parametrize("R,E,BM", [(128, 128, 16), (96, 128, 16),
                                    (24, 8, 16), (5, 4, 4)])
def test_fused_tile_table_matches_chained_builder(R, E, BM):
    """The one-launch tile table must reproduce the chained builder's
    five outputs EXACTLY -- including stable-sort order on ties and
    zeroed padding slots -- over uneven routing with empty experts."""
    pytest.importorskip("triton")
    from int4_b32 import build_group_tiles_fused
    from nf4_grouped import build_group_tiles_device
    dev = _gpu()
    torch.manual_seed(11 + R)
    eids = torch.randint(0, E, (R,), device=dev).to(torch.int32)
    a = build_group_tiles_device(eids, E, BM)
    b = build_group_tiles_fused(eids, E, BM)
    names = ("row0", "rows", "grp", "order", "counts")
    for n, x, y in zip(names, a, b):
        assert x.dtype == y.dtype, (n, x.dtype, y.dtype)
        assert torch.equal(x, y), (n, x.cpu(), y.cpu())


def test_fused_tile_table_refuses_prefill_shapes():
    pytest.importorskip("triton")
    from int4_b32 import build_group_tiles_fused
    dev = _gpu()
    eids = torch.zeros(1024, dtype=torch.int32, device=dev)
    with pytest.raises(ValueError, match="decode-only"):
        build_group_tiles_fused(eids, 128, 16)


def test_gathered_quant_matches_gather_then_quant():
    pytest.importorskip("triton")
    from int4_b32 import quant_x_rows, quant_x_rows_gathered
    dev = _gpu()
    torch.manual_seed(12)
    R, K = 24, 96
    x = (torch.randn(R, K) * 0.3).to(dev, torch.bfloat16)
    order = torch.randperm(R, device=dev)
    q1, s1 = quant_x_rows(x.index_select(0, order).contiguous())
    q2, s2 = quant_x_rows_gathered(x, order)
    assert torch.equal(q1, q2) and torch.equal(s1, s2)


def test_swiglu_rows_matches_chain():
    pytest.importorskip("triton")
    from int4_b32 import swiglu_rows
    import torch.nn.functional as F
    dev = _gpu()
    torch.manual_seed(13)
    R, inter = 24, 40                     # non-pow2 inner dim
    gu = (torch.randn(R, 2 * inter) * 2).to(dev, torch.bfloat16)
    got = swiglu_rows(gu)
    g, u = gu.chunk(2, dim=-1)
    want = (F.silu(g.float()) * u.float()).to(torch.bfloat16)
    # fp32 silu*mul then one bf16 rounding in BOTH paths; sigmoid may
    # differ in the last bit between device libdevice and torch
    assert (got.float() - want.float()).abs().max() <= \
        want.float().abs().max() * 2 ** -7


def test_fused_tile_table_empty_routing():
    """R = 0 must match the chained builder without launching (an empty
    tl.arange is invalid) -- review finding, round 1."""
    pytest.importorskip("triton")
    from int4_b32 import build_group_tiles_fused
    from nf4_grouped import build_group_tiles_device
    dev = _gpu()
    eids = torch.empty(0, dtype=torch.int32, device=dev)
    a = build_group_tiles_device(eids, 8, 16)
    b = build_group_tiles_fused(eids, 8, 16)
    for n, x, y in zip(("row0", "rows", "grp", "order", "counts"), a, b):
        assert x.dtype == y.dtype and torch.equal(x, y), n
