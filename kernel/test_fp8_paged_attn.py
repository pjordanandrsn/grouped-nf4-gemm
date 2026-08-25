# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""FP8 paged decode attention vs the reference oracle: permuted block
tables (actually paged, not accidentally contiguous), ragged sequence
lengths with partial tail blocks, grouped key scales, GQA group sizes
below the tensor-core minimum, and the in-kernel E4M3 decode against the
reference dequant bit for bit."""

import os
import sys

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, os.path.dirname(__file__))

from fp8_kv import (  # noqa: E402
    dequant_kv_fp8_ref,
    pack_kv_block,
    quantize_kv_fp8,
    unpack_kv_block_grouped,
)
from fp8_paged_attn import (  # noqa: E402
    fp8_paged_decode_attention,
    paged_attn_available,
    paged_attn_ref,
)

needs_gpu = pytest.mark.skipif(not paged_attn_available(),
                               reason="needs CUDA + triton")

BT = 16


def _build(B, hq, hkv, d, seq_lens, k_groups=1, v_groups=1, seed=0,
           permute=True, layout="tokens"):
    """Random sequences packed into pools through the REAL pack path, with
    a permuted block table so contiguity is never accidental."""
    g = torch.Generator().manual_seed(seed)
    max_blocks = max((t + BT - 1) // BT for t in seq_lens)
    total_rows = sum((t + BT - 1) // BT for t in seq_lens)
    k_row = BT * hkv * d + BT * hkv * k_groups * 4
    v_row = BT * hkv * d + BT * hkv * v_groups * 4
    k_pool = torch.zeros(total_rows * k_row, dtype=torch.uint8)
    v_pool = torch.zeros(total_rows * v_row, dtype=torch.uint8)
    perm = (torch.randperm(total_rows, generator=g) if permute
            else torch.arange(total_rows))
    table = torch.zeros(B, max_blocks, dtype=torch.int32)
    raw_k, raw_v = [], []
    row_i = 0
    for b, t in enumerate(seq_lens):
        n_blk = (t + BT - 1) // BT
        kt = torch.randn(n_blk * BT, hkv, d, generator=g) * 1.5
        vt = torch.randn(n_blk * BT, hkv, d, generator=g)
        raw_k.append(kt)
        raw_v.append(vt)
        for i in range(n_blk):
            row = int(perm[row_i])
            table[b, i] = row
            qk, sk = quantize_kv_fp8(kt[i * BT:(i + 1) * BT],
                                     group=d // k_groups)
            pack_kv_block(qk, sk, k_pool[row * k_row:(row + 1) * k_row],
                          layout=layout)
            qv, sv = quantize_kv_fp8(vt[i * BT:(i + 1) * BT],
                                     group=d // v_groups)
            pack_kv_block(qv, sv, v_pool[row * v_row:(row + 1) * v_row],
                          layout=layout)
            row_i += 1
    q = torch.randn(B, hq, d, generator=g, dtype=torch.float32) \
        .to(torch.bfloat16)
    lens = torch.tensor(seq_lens, dtype=torch.int32)
    return q, k_pool, v_pool, table, lens


FP8_DOT_OK = (torch.cuda.is_available()
              and torch.cuda.get_device_capability() >= (8, 9))


def _modes():
    """Kernel paths to run every shape test through. compute="fp8" needs
    fp8 MMA hardware; on older cards the shape tests still cover both
    non-fp8 paths."""
    modes = [("split", {}), ("packed", {"pack_heads": True})]
    if FP8_DOT_OK:
        modes.append(("f8dot", {"compute": "fp8"}))
        modes.append(("pf8", {"pack_heads": True, "compute": "fp8"}))
    return modes


def _run_both(B, hq, hkv, d, seq_lens, mode_kw=None, **kw):
    q, kp, vp, tab, lens = _build(B, hq, hkv, d, seq_lens, **kw)
    dev = lambda t: t.cuda()  # noqa: E731
    got = fp8_paged_decode_attention(
        dev(q), dev(kp), dev(vp), dev(tab), dev(lens),
        n_kv_heads=hkv, head_dim=d, **(mode_kw or {}),
        layout=kw.get("layout", "tokens"),
        k_groups=kw.get("k_groups", 1), v_groups=kw.get("v_groups", 1))
    want = paged_attn_ref(q, kp, vp, tab, lens, n_kv_heads=hkv, head_dim=d,
                          k_groups=kw.get("k_groups", 1),
                          v_groups=kw.get("v_groups", 1),
                          layout=kw.get("layout", "tokens"))
    return got.cpu().float(), want.float()


def _close(got, want, mode="split"):
    # fp32-accumulated online softmax vs an fp32 oracle over IDENTICAL
    # dequantized values: agreement to bf16-output rounding plus tf32 dot.
    # The f8dot path adds one e4m3 rounding on q and one on p (the K/V
    # bytes are identical either way). P rounding is the dominant term
    # and is WORST when one or two tokens carry the softmax mass: probed
    # exact at T=1 (p == 1 -> 448, representable), 0.087 worst-element at
    # T=2, 0.034 at T=33, shrinking as contributions average over tokens.
    # These shape tests run adversarial tiny-T layouts, so the f8dot
    # bound is the measured worst-element envelope — structural bugs
    # (mis-folded scales) are O(1) and still fail it — while the
    # serving-shape distributional bound lives in
    # test_f8dot_error_is_bounded_and_reported (invariant 4-prime;
    # storage quality was certified by the G7 oracle, fp8 COMPUTE
    # quality is owed separately if it becomes the default).
    tol = 1.5e-1 if mode in ("f8dot", "pf8") else 2e-2
    torch.testing.assert_close(got, want, rtol=tol, atol=tol)


@needs_gpu
@pytest.mark.parametrize("mode,mkw", _modes())
def test_permuted_paged_tables_match_reference(mode, mkw):
    got, want = _run_both(3, 16, 4, 64, [64, 128, 96], mode_kw=mkw)
    _close(got, want, mode)


@needs_gpu
@pytest.mark.parametrize("mode,mkw", _modes())
def test_partial_tail_blocks_are_masked_not_scored(mode, mkw):
    """Ragged lengths ending mid-block: the tail block's unwritten rows
    hold zeros in the pool; a kernel that scores them shifts the softmax
    and fails this."""
    got, want = _run_both(4, 16, 4, 64, [17, 33, 1, 47], mode_kw=mkw)
    _close(got, want, mode)


@needs_gpu
@pytest.mark.parametrize("mode,mkw", _modes())
def test_grouped_key_scales(mode, mkw):
    """The quality-passing configuration: K scales per 32 channels, V per
    row — two different scale layouts read in one kernel launch."""
    got, want = _run_both(2, 16, 4, 128, [96, 160], k_groups=4, v_groups=1, mode_kw=mkw)
    _close(got, want, mode)


@needs_gpu
@pytest.mark.parametrize("mode,mkw", _modes())
def test_small_gqa_group_is_padded_not_wrong(mode, mkw):
    """G below the 16-row dot minimum (Llama-class 32q/8kv -> G=4): the
    group tile is padded and masked; the pad rows must not leak."""
    got, want = _run_both(2, 32, 8, 64, [64, 80], mode_kw=mkw)
    _close(got, want, mode)


@needs_gpu
@pytest.mark.parametrize("mode,mkw", _modes())
def test_single_token_context(mode, mkw):
    got, want = _run_both(2, 16, 4, 64, [1, 2], mode_kw=mkw)
    _close(got, want, mode)


@needs_gpu
def test_kernel_dequant_matches_reference_bitwise():
    """The in-kernel E4M3 decode against dequant_kv_fp8_ref, isolated from
    attention: run a 1-token, 1-head attention whose softmax weight is
    exactly 1, so the output IS the dequantized V row."""
    g = torch.Generator().manual_seed(7)
    d = 64
    vt = torch.randn(BT, 1, d, generator=g) * 3
    qv, sv = quantize_kv_fp8(vt)
    v_row = BT * 1 * d + BT * 1 * 4
    v_pool = torch.zeros(v_row, dtype=torch.uint8)
    pack_kv_block(qv, sv, v_pool)
    k_pool = v_pool.clone()
    table = torch.zeros(1, 1, dtype=torch.int32)
    lens = torch.tensor([1], dtype=torch.int32)
    q = torch.ones(1, 16, d, dtype=torch.bfloat16)
    got = fp8_paged_decode_attention(
        q.cuda(), k_pool.cuda(), v_pool.cuda(), table.cuda(), lens.cuda(),
        n_kv_heads=1, head_dim=d)
    want = dequant_kv_fp8_ref(qv, sv, dtype=torch.float32)[0, 0]
    torch.testing.assert_close(got.cpu().float()[0, 0], want,
                               rtol=1e-2, atol=1e-2)


@needs_gpu
def test_pool_roundtrip_through_grouped_unpack():
    """The test builder writes through the real pack path; prove the pools
    it builds read back exactly."""
    g = torch.Generator().manual_seed(3)
    x = torch.randn(BT, 4, 128, generator=g)
    qk, sk = quantize_kv_fp8(x, group=32)
    row = torch.zeros(BT * 4 * 128 + BT * 4 * 4 * 4, dtype=torch.uint8)
    pack_kv_block(qk, sk, row)
    q2, s2 = unpack_kv_block_grouped(row, BT, 4, 128, 4)
    assert torch.equal(q2.view(torch.uint8), qk.view(torch.uint8))
    assert torch.equal(s2, sk)


@needs_gpu
@pytest.mark.skipif(not FP8_DOT_OK, reason="needs fp8 MMA (sm_89+)")
def test_f8dot_error_is_bounded_and_reported():
    """The fp8-compute path's ACTUAL error distribution vs the f32
    oracle at serving-shape lengths (hundreds of tokens, grouped keys,
    ragged tails). Printed so receipts can quote measured numbers;
    asserted distributionally so honest per-element fp8 rounding passes
    while structural bugs (mis-folded scales are O(1) everywhere)
    cannot."""
    got, want = _run_both(4, 64, 4, 128, [512, 731, 288, 512], k_groups=4,
                          v_groups=1, mode_kw={"compute": "fp8"})
    err = (got - want).abs()
    q99 = err.flatten().kthvalue(int(err.numel() * 0.99)).values.item()
    print(f"f8dot mean {err.mean().item():.5f} p99 {q99:.5f} "
          f"max {err.max().item():.5f}")
    assert err.mean().item() < 5e-3
    assert q99 < 5e-2
    assert err.max().item() < 2e-1


@needs_gpu
@pytest.mark.skipif(not FP8_DOT_OK, reason="needs fp8 MMA (sm_89+)")
def test_f8dot_refuses_unrolled_group_counts():
    """k_groups=8 with head_dim=256 passes the >=32-wide check but is
    outside the kernels' constexpr unroll — it must refuse, not silently
    score only the first four groups (Bugbot, HIGH)."""
    q, kp, vp, tab, lens = _build(1, 16, 4, 256, [32], k_groups=8)
    with pytest.raises(AssertionError, match="unrolls k_groups"):
        fp8_paged_decode_attention(
            q.cuda(), kp.cuda(), vp.cuda(), tab.cuda(), lens.cuda(),
            n_kv_heads=4, head_dim=256, k_groups=8, compute="fp8")
    with pytest.raises(AssertionError, match="unrolls k_groups"):
        fp8_paged_decode_attention(
            q.cuda(), kp.cuda(), vp.cuda(), tab.cuda(), lens.cuda(),
            n_kv_heads=4, head_dim=256, k_groups=8, compute="fp8",
            pack_heads=True)


@needs_gpu
@pytest.mark.skipif(not FP8_DOT_OK, reason="needs fp8 MMA (sm_89+)")
def test_fuse_counters_are_per_stream():
    """Concurrent streams of the same shape must not share arrival slots
    (a shared slot could combine early or never — review); one stream's
    repeated launches must reuse one buffer (that is the whole point)."""
    from fp8_paged_attn import _fuse_counters, reset_fuse_counters
    reset_fuse_counters()
    dev = torch.device("cuda", torch.cuda.current_device())
    a = _fuse_counters(8, dev)
    assert _fuse_counters(8, dev) is a
    s2 = torch.cuda.Stream(dev)
    with torch.cuda.stream(s2):
        b = _fuse_counters(8, dev)
    assert b is not a
    reset_fuse_counters()


@needs_gpu
@pytest.mark.skipif(not FP8_DOT_OK, reason="needs fp8 MMA (sm_89+)")
def test_f8dot_two_streams_concurrent_correct():
    """Two f8dot launches racing on different streams, each checked
    against the oracle — the failure mode of shared counters is a wrong
    or unwritten output tensor."""
    builds = [_build(2, 16, 4, 64, [64, 48], k_groups=2)
              for _ in range(2)]
    outs = []
    streams = [torch.cuda.Stream() for _ in range(2)]
    for (q, kp, vp, tab, lens), s in zip(builds, streams):
        with torch.cuda.stream(s):
            outs.append(fp8_paged_decode_attention(
                q.cuda(), kp.cuda(), vp.cuda(), tab.cuda(), lens.cuda(),
                n_kv_heads=4, head_dim=64, k_groups=2, compute="fp8"))
    torch.cuda.synchronize()
    for (q, kp, vp, tab, lens), got in zip(builds, outs):
        want = paged_attn_ref(q, kp, vp, tab, lens, n_kv_heads=4,
                              head_dim=64, k_groups=2)
        _close(got.cpu().float(), want.float(), "f8dot")


def test_heads_layout_roundtrips_bitwise():
    """Heads-major rows must dequantize to exactly what tokens-major rows
    hold — same bytes, different order; any drift is the permute math."""
    from fp8_kv import kv_block_bytes
    g = torch.Generator().manual_seed(11)
    x = torch.randn(BT, 4, 128, generator=g) * 1.5
    qk, sk = quantize_kv_fp8(x, group=32)
    row_t = torch.zeros(kv_block_bytes(BT, 4, 128) + BT * 4 * 3 * 4,
                        dtype=torch.uint8)
    row_h = row_t.clone()
    pack_kv_block(qk, sk, row_t, layout="tokens")
    pack_kv_block(qk, sk, row_h, layout="heads")
    assert not torch.equal(row_t, row_h), "layouts should differ on disk"
    from fp8_kv import unpack_kv_block_grouped
    qt, st = unpack_kv_block_grouped(row_t, BT, 4, 128, 4)
    qh, sh = unpack_kv_block_grouped(row_h, BT, 4, 128, 4, layout="heads")
    assert torch.equal(qt.view(torch.uint8), qh.view(torch.uint8))
    assert torch.equal(st, sh)


@needs_gpu
@pytest.mark.skipif(not FP8_DOT_OK, reason="needs fp8 MMA (sm_89+)")
@pytest.mark.parametrize("shape", [
    (3, 16, 4, 64, [64, 128, 96], 2),
    (4, 16, 4, 64, [17, 33, 1, 47], 1),
    (2, 16, 4, 128, [96, 160], 4),
    (2, 64, 4, 128, [200, 71], 4),
])
def test_f8dot_heads_layout_matches_reference(shape):
    """The fp8 kernel over heads-major pools vs the oracle reading the
    same heads-major bytes — permuted tables, ragged tails, grouped
    scales."""
    B, hq, hkv, d, lens, kg = shape
    got, want = _run_both(B, hq, hkv, d, lens, k_groups=kg,
                          layout="heads", mode_kw={"compute": "fp8"})
    _close(got, want, "f8dot")


@needs_gpu
def test_f32_split_fused_combine_bitwise():
    """PREREG-f2-tail T1: the in-kernel combine on the DEFAULT f32
    split path must be bitwise-equal to the two-kernel path -- same
    fixed 0..n_split reduction order, only the launch disappears.
    Real pack path, permuted tables, several splits, odd lengths."""
    for trial, (b, hq, hkv, lens) in enumerate([
            (1, 16, 4, [513]),
            (2, 16, 4, [96, 511]),
            (3, 8, 2, [17, 250, 384])]):
        q, kp, vp, tab, sl = _build(b, hq, hkv, 64, lens,
                                    seed=100 + trial)
        dev = lambda t: t.cuda()                     # noqa: E731
        for n_split in (1, 2, 5):
            a = fp8_paged_decode_attention(
                dev(q), dev(kp), dev(vp), dev(tab), dev(sl),
                n_kv_heads=hkv, head_dim=64, n_split=n_split,
                fuse_combine=False)
            f = fp8_paged_decode_attention(
                dev(q), dev(kp), dev(vp), dev(tab), dev(sl),
                n_kv_heads=hkv, head_dim=64, n_split=n_split,
                fuse_combine=True)
            assert torch.equal(a, f), (
                f"fused combine diverges (trial={trial} "
                f"n_split={n_split})")


def test_f32_fuse_default_is_env_gated(monkeypatch):
    """Default ON since RESULTS-f2-tail (PARTIAL ships both
    treatments); GNF4_F32_FUSE_COMBINE=0 is the rollback. Explicit
    True/False from a caller always wins (the bitwise gate test above
    passes both explicitly)."""
    from fp8_paged_attn import _f32_fuse_default
    monkeypatch.delenv("GNF4_F32_FUSE_COMBINE", raising=False)
    assert _f32_fuse_default() is True
    monkeypatch.setenv("GNF4_F32_FUSE_COMBINE", "1")
    assert _f32_fuse_default() is True
    monkeypatch.setenv("GNF4_F32_FUSE_COMBINE", "0")
    assert _f32_fuse_default() is False


def test_wrapper_resolves_none_per_path():
    """Source guard: the signature default must stay None and the body
    must resolve it through _f32_fuse_default() for f32 -- a future
    edit restoring `= True` would silently ship T1 ahead of its
    verdict through e4b's kwargs passthrough."""
    import inspect

    import fp8_paged_attn as m
    sig = inspect.signature(m.fp8_paged_decode_attention)
    assert sig.parameters["fuse_combine"].default is None
    src = inspect.getsource(m.fp8_paged_decode_attention)
    assert "_f32_fuse_default()" in src
    assert 'compute == "fp8" else _f32_fuse_default()' in src
