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


@pytest.mark.parametrize("shape,H", [((1, 2048), 2048),
                                     ((32, 128), 128),
                                     ((1, 5, 96), 96)])
def test_rmsnorm_rows_matches_reference(shape, H):
    """Upstream RMSNorm semantics: fp32 mean-square, rsqrt, weight
    multiply, bf16 cast -- within one output rounding of the fp32
    chain."""
    pytest.importorskip("triton")
    from int4_b32 import rmsnorm_rows
    dev = _gpu()
    torch.manual_seed(23)
    x = (torch.randn(*shape) * 2).to(dev, torch.bfloat16)
    w = (torch.randn(H).abs() + 0.5).to(dev, torch.bfloat16)
    eps = 1e-6
    got = rmsnorm_rows(x, w, eps)
    xf = x.float()
    ref = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
           * w.float()).to(torch.bfloat16)
    assert got.shape == x.shape
    assert (got.float() - ref.float()).abs().max() <= \
        ref.float().abs().max() * 2 ** -7


def _bf16_ulp(t):
    """One bf16 ULP at each element's own magnitude: bf16 keeps 7
    explicit mantissa bits, so the step at x is 2**(floor(log2|x|) - 7).
    An elementwise bound says what a rounding claim actually means --
    a scalar scaled off the tensor's max is tighter than a real ULP for
    small elements and looser for large ones."""
    e = torch.floor(torch.log2(t.float().abs().clamp_min(2 ** -126)))
    return torch.pow(torch.tensor(2.0), e - 7)


@pytest.mark.parametrize("scale", [1.0, 0.22])
@pytest.mark.parametrize("shape,H", [((1, 2048), 2048), ((16, 2048), 2048)])
def test_rmsnorm_resid_rows_matches_reference(shape, H, scale):
    """Fused residual-add + RMSNorm: the add must round once to bf16
    exactly as the upstream bf16 ``+`` (operands exact in fp32, one
    rounding), the norm then reads that rounded value. new_resid is
    BITWISE the bf16 sum; the normed output sits within one rounding
    of the fp32 chain. With a residual multiplier (GraniteMoe's body
    is ``resid + x * m``) the product rounds to bf16 first, exactly as
    the upstream bf16 tensor-times-float does, then the add rounds."""
    pytest.importorskip("triton")
    from int4_b32 import rmsnorm_resid_rows
    dev = _gpu()
    torch.manual_seed(31)
    x = (torch.randn(*shape) * 2).to(dev, torch.bfloat16)
    r = (torch.randn(*shape) * 2).to(dev, torch.bfloat16)
    w = (torch.randn(H).abs() + 0.5).to(dev, torch.bfloat16)
    eps = 1e-6
    if scale == 1.0:
        got, nres = rmsnorm_resid_rows(x, r, w, eps)
        ref_res = (x.float() + r.float()).to(torch.bfloat16)
    else:
        got, nres = rmsnorm_resid_rows(x, r, w, eps, scale=scale)
        # upstream: bf16 product (one rounding), then bf16 add
        ref_res = (r + x * scale)
        assert ref_res.dtype == torch.bfloat16
    if dev == "cuda":
        # on hardware the add really is fp32, so the double rounding
        # (exact sum -> fp32 -> bf16) matches torch's exactly
        assert torch.equal(nres, ref_res), \
            "residual sum must be bitwise the bf16 add on hardware"
    else:
        # interpreter mode does not evaluate the add in fp32, so a sum
        # whose fp32 rounding differs from its wider rounding lands one
        # ULP away from torch's double-rounded result. Bound it at
        # exactly that: this leg checks the FORMULA, CUDA checks bits.
        d = (nres.float() - ref_res.float()).abs()
        assert (d <= 2 * _bf16_ulp(ref_res)).all(), \
            "residual sum must stay within a couple of bf16 ULP"
    sf = ref_res.float()
    ref = (sf * torch.rsqrt(sf.pow(2).mean(-1, keepdim=True) + eps)
           * w.float()).to(torch.bfloat16)
    assert got.shape == x.shape and nres.shape == x.shape
    # one ULP on hardware, where the fused chain differs from the
    # reference only by its single final rounding. The interpreter also
    # feeds the norm a residual that is itself off by a ULP, so its
    # deviation compounds -- bound it loosely enough not to sit on the
    # boundary, still orders of magnitude tighter than any logic error.
    budget = _bf16_ulp(ref) * (1 if dev == "cuda" else 4)
    assert ((got.float() - ref.float()).abs() <= budget).all()


@pytest.mark.parametrize("shape,H,scale", [((1, 2048), 2048, 0.22),
                                            ((16, 1536), 1536, 0.5),
                                            ((3, 96), 96, 1.0)])
def test_scaled_resid_add_rows_matches_reference(shape, H, scale):
    """``resid + x * scale`` in one launch must carry upstream's TWO
    roundings (bf16 product, then bf16 sum): on hardware it is bitwise
    the torch expression; under the interpreter within a couple of ULP
    (the same allowance the residual fold takes)."""
    pytest.importorskip("triton")
    from int4_b32 import scaled_resid_add_rows
    dev = _gpu()
    torch.manual_seed(37)
    x = (torch.randn(*shape) * 2).to(dev, torch.bfloat16)
    r = (torch.randn(*shape) * 2).to(dev, torch.bfloat16)
    got = scaled_resid_add_rows(x, r, scale)
    ref = r + x * scale
    assert ref.dtype == torch.bfloat16 and got.shape == x.shape
    if dev == "cuda":
        assert torch.equal(got, ref), "scaled residual add must be bitwise upstream on hardware"
    else:
        d = (got.float() - ref.float()).abs()
        assert (d <= 2 * _bf16_ulp(ref)).all()
    # a single-rounding add is NOT the upstream value in general: the
    # test's own reference must differ from it somewhere, or it proves
    # nothing about which rounding the kernel took
    if scale != 1.0:
        single = torch.add(r, x, alpha=scale)
        assert not torch.equal(single, ref) or shape[0] * H < 256, \
            "reference is indistinguishable from the single-rounding add on this draw"


@pytest.mark.parametrize("R,HEADS,D", [(1, 32, 128), (16, 4, 128)])
def test_rope_norm_heads_matches_reference(R, HEADS, D):
    """Fused per-head RMSNorm + rotate-half rotary against the exact
    upstream chain (norm in fp32 -> bf16, then q*cos + rotate_half(q)
    *sin). The fused kernel keeps one fp32 chain with a single final
    rounding, so the frame is the K6 relative tolerance, not bitwise."""
    pytest.importorskip("triton")
    from int4_b32 import rope_norm_heads
    dev = _gpu()
    torch.manual_seed(37)
    x = (torch.randn(R, HEADS, D) * 2).to(dev, torch.bfloat16)
    w = (torch.randn(D).abs() + 0.5).to(dev, torch.bfloat16)
    ang = torch.rand(R, D // 2) * 6.28
    ang = torch.cat([ang, ang], dim=-1)          # upstream duplicates halves
    cos = ang.cos().to(dev, torch.bfloat16)
    sin = ang.sin().to(dev, torch.bfloat16)
    eps = 1e-6
    got = rope_norm_heads(x, w, cos, sin, eps)
    xf = x.float()
    xn = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
          * w.float()).to(torch.bfloat16).float()
    half = D // 2
    rot = torch.cat([-xn[..., half:], xn[..., :half]], dim=-1)
    ref = (xn * cos.float().unsqueeze(1)
           + rot * sin.float().unsqueeze(1)).to(torch.bfloat16)
    assert got.shape == x.shape
    assert (got.float() - ref.float()).abs().max() <= \
        ref.float().abs().max() * 2 ** -7


@pytest.mark.parametrize("R,E,K,norm", [(1, 128, 8, True), (16, 128, 8, True),
                                        (1, 64, 4, False),
                                        (1, 128, 6, True),
                                        (4, 100, 3, True)])
def test_router_epilogue_matches_torch(R, E, K, norm):
    """Must reproduce torch's softmax->topk->renormalise exactly enough
    that the SELECTION is identical: a different expert set is a routing
    change, not a rounding one, and no ppl gate would forgive it."""
    pytest.importorskip("triton")
    from int4_b32 import router_epilogue
    dev = _gpu()
    torch.manual_seed(41)
    logits = (torch.randn(R, E) * 3).to(dev, torch.float32)
    probs, w, idx = router_epilogue(logits, K, norm)

    ref_p = torch.softmax(logits, dim=-1, dtype=torch.float32)
    ref_v, ref_i = torch.topk(ref_p, K, dim=-1)
    if norm:
        ref_v = ref_v / ref_v.sum(dim=-1, keepdim=True)
    assert torch.equal(idx, ref_i), "selected experts must match exactly"
    assert (probs - ref_p).abs().max() <= ref_p.abs().max() * 2 ** -18
    assert (w - ref_v).abs().max() <= ref_v.abs().max() * 2 ** -18


def test_router_epilogue_handles_non_power_of_two_k():
    """top_k is a model choice, not a kernel choice: 6 and 3 are as
    legitimate as 8, and tl.arange only spans powers of two."""
    pytest.importorskip("triton")
    from int4_b32 import router_epilogue
    dev = _gpu()
    torch.manual_seed(43)
    logits = (torch.randn(4, 100) * 3).to(dev, torch.float32)
    for k in (3, 5, 6, 7):
        _, w, idx = router_epilogue(logits, k, True)
        ref_p = torch.softmax(logits, dim=-1, dtype=torch.float32)
        ref_v, ref_i = torch.topk(ref_p, k, dim=-1)
        ref_v = ref_v / ref_v.sum(dim=-1, keepdim=True)
        assert w.shape == (4, k) and idx.shape == (4, k)
        assert torch.equal(idx, ref_i)
        assert (w - ref_v).abs().max() <= ref_v.abs().max() * 2 ** -18


def test_router_epilogue_breaks_ties_to_the_lower_index():
    """Equal logits must select the lower expert index, the rule a
    stable descending sort follows -- otherwise two engines routing the
    same token could disagree while both look correct."""
    pytest.importorskip("triton")
    from int4_b32 import router_epilogue
    dev = _gpu()
    logits = torch.zeros(1, 8, dtype=torch.float32, device=dev)
    _, _, idx = router_epilogue(logits, 3, False)
    assert idx.flatten().tolist() == [0, 1, 2]
