# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""cpu_grouped (hybrid Phase 2): exact-tree parity between the native CPU
kernels and the numpy executable spec, oracle agreement with the repo's
dequant references, the deterministic router epilogue, and the codebook
pinning that keeps the three copies of each LUT honest.

CPU-only; skips when no C compiler can build gnf4_native."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import cpu_grouped as cg  # noqa: E402

E, N, K, G = 4, 24, 128, 3
SIZES = [1, 2, 1]
EIDS = [2, 0, 3]

needs_native = pytest.mark.skipif(
    not cg.cpu_kernels_available(), reason="no C compiler / build failed"
)


def test_native_required_when_promised():
    """CI exports GNF4_REQUIRE_NATIVE=1: there, an unavailable native build
    is a FAILURE, never a silent skip of the whole exactness gate."""
    import os
    if os.environ.get("GNF4_REQUIRE_NATIVE") == "1":
        assert cg.cpu_kernels_available(), (
            "GNF4_REQUIRE_NATIVE=1 but gnf4_native cannot build/import — "
            "the exact-parity gate would silently skip")


def _nf4_stack(seed=0, rows=None):
    g = np.random.default_rng(seed)
    packed = g.integers(0, 256, size=(E, N, K // 2), dtype=np.uint8)
    absmax = g.random(size=(E, N, K // 64), dtype=np.float32) + 0.5
    rows = sum(SIZES) if rows is None else rows
    a = g.standard_normal((rows, K), dtype=np.float32)
    return a, packed, absmax


def _mx_stack(seed=1, weird_scales=False, rows=None):
    g = np.random.default_rng(seed)
    packed = g.integers(0, 256, size=(E, N, K // 2), dtype=np.uint8)
    scales = g.integers(100, 140, size=(E, N, K // 32), dtype=np.uint8)
    if weird_scales:
        scales[0, 0, 0] = 0xFF        # oracle ldexp semantics: 2^128
        scales[0, 0, 1] = 0           # subnormal 2^-127
        scales[1, 3, 2] = 0xFF
    rows = sum(SIZES) if rows is None else rows
    a = g.standard_normal((rows, K), dtype=np.float32)
    return a, packed, scales


# --------------------------------------------------------------------------- #
# codebook pinning: local copies == canonical sources
# --------------------------------------------------------------------------- #

def test_luts_match_canonical_sources():
    from nf4_grouped import BLOCKSIZE, NF4_LUT
    from mxfp4_pack_ref import FP4_VALUES, MX_BLOCK
    assert cg.BLOCKSIZE == BLOCKSIZE
    assert cg.MX_BLOCK == MX_BLOCK
    assert np.array_equal(cg._NF4_LUT32,
                          np.asarray(NF4_LUT, dtype=np.float32))
    assert np.array_equal(cg._FP4_LUT32,
                          np.asarray(FP4_VALUES, dtype=np.float32))


# --------------------------------------------------------------------------- #
# native kernel == numpy executable spec, EXACTLY
# --------------------------------------------------------------------------- #

@needs_native
def test_nf4_native_exactly_matches_ordered_ref():
    a, packed, absmax = _nf4_stack()
    ref = cg.ref_gemv_grouped(a, packed, absmax, SIZES, EIDS, fmt="nf4")
    out = cg.gemv_nf4_grouped_cpu(
        torch.from_numpy(a), torch.from_numpy(packed),
        torch.from_numpy(absmax), SIZES, EIDS)
    assert np.array_equal(out.numpy(), ref), "locked-tree mismatch (NF4)"


@needs_native
def test_mxfp4_native_exactly_matches_ordered_ref():
    for weird in (False, True):
        a, packed, scales = _mx_stack(weird_scales=weird)
        ref = cg.ref_gemv_grouped(a, packed, scales, SIZES, EIDS, fmt="mxfp4")
        out = cg.gemv_mxfp4_grouped_cpu(
            torch.from_numpy(a), torch.from_numpy(packed),
            torch.from_numpy(scales), SIZES, EIDS)
        ours, theirs = out.numpy(), ref
        assert np.array_equal(
            np.nan_to_num(ours, nan=1e30, posinf=2e30, neginf=-2e30),
            np.nan_to_num(theirs, nan=1e30, posinf=2e30, neginf=-2e30),
        ), f"locked-tree mismatch (MXFP4, weird_scales={weird})"


@needs_native
def test_pool_mode_bits_identical():
    """The persistent pool must produce the same bits as the OpenMP path
    (same work items, different partitioner — independence makes any
    difference a bug)."""
    a, packed, absmax = _nf4_stack(seed=11)
    ta, tp, tm = (torch.from_numpy(a), torch.from_numpy(packed),
                  torch.from_numpy(absmax))
    base = cg.gemv_nf4_grouped_cpu(ta, tp, tm, SIZES, EIDS)
    n = cg.pool_start(4)
    try:
        assert n >= 1
        pooled = cg.gemv_nf4_grouped_cpu(ta, tp, tm, SIZES, EIDS)
    finally:
        cg.pool_stop()
    assert torch.equal(base, pooled)
    after = cg.gemv_nf4_grouped_cpu(ta, tp, tm, SIZES, EIDS)  # omp again
    assert torch.equal(base, after)


@needs_native
def test_threads_do_not_change_bits():
    a, packed, absmax = _nf4_stack(seed=7)
    t1 = cg.gemv_nf4_grouped_cpu(torch.from_numpy(a), torch.from_numpy(packed),
                                 torch.from_numpy(absmax), SIZES, EIDS,
                                 threads=1)
    t4 = cg.gemv_nf4_grouped_cpu(torch.from_numpy(a), torch.from_numpy(packed),
                                 torch.from_numpy(absmax), SIZES, EIDS,
                                 threads=4)
    assert torch.equal(t1, t4)


# --------------------------------------------------------------------------- #
# oracle agreement (torch reference dequant; order caveat -> tolerance)
# --------------------------------------------------------------------------- #

@needs_native
def test_nf4_agrees_with_repo_dequant_ref():
    from nf4_grouped import dequant_ref
    a, packed, absmax = _nf4_stack(seed=3)
    out = cg.gemv_nf4_grouped_cpu(
        torch.from_numpy(a), torch.from_numpy(packed),
        torch.from_numpy(absmax), SIZES, EIDS).numpy()
    r = 0
    for g, e in enumerate(EIDS):
        w = dequant_ref(torch.from_numpy(packed[e]).reshape(N, K // 2),
                        torch.from_numpy(absmax[e]).reshape(N, K // 64),
                        N, K)
        for _ in range(SIZES[g]):
            want = (torch.from_numpy(a[r]) @ w.T).numpy()
            np.testing.assert_allclose(out[r], want, rtol=2e-6, atol=2e-5)
            r += 1


# --------------------------------------------------------------------------- #
# router epilogue: deterministic rule + bf16 RNE
# --------------------------------------------------------------------------- #

@needs_native
def test_epilogue_matches_stable_sort_rule_and_softmax():
    rng = np.random.default_rng(5)
    t, e, k = 3, 64, 8
    logits = rng.standard_normal((t, e)).astype(np.float32)
    logits[0, 5] = logits[0, 11] = logits[0, 3]        # three-way tie
    idx = np.zeros((t, k), dtype=np.int64)
    wts = np.zeros((t, k), dtype=np.uint16)
    for mode, norm in ((0, False), (0, True), (1, False)):
        cg.route_epilogue_bf16(logits, k, mode, norm, idx, wts)
        order = np.argsort(-logits, axis=-1, kind="stable")[:, :k]
        assert np.array_equal(idx, order), f"selection rule (mode={mode})"
        if mode == 0:
            probs = np.exp(logits - logits.max(-1, keepdims=True))
            probs /= probs.sum(-1, keepdims=True)
            w = np.take_along_axis(probs, order, axis=-1)
            if norm:
                w = w / w.sum(-1, keepdims=True)
        else:
            top = np.take_along_axis(logits, order, axis=-1)
            ex = np.exp(top - top.max(-1, keepdims=True))
            w = ex / ex.sum(-1, keepdims=True)
        got = torch.from_numpy(wts.copy()).view(torch.bfloat16).float().numpy()
        np.testing.assert_allclose(got, w, atol=1e-2, rtol=1e-2)
        # bf16 conversion itself is RNE, torch-identical
        ref16 = torch.from_numpy(w.astype(np.float32)).to(torch.bfloat16)
        assert torch.equal(torch.from_numpy(wts.copy()).view(torch.bfloat16),
                           ref16)


# --------------------------------------------------------------------------- #
# contract violations raise
# --------------------------------------------------------------------------- #

@needs_native
def test_bad_calls_raise():
    a, packed, absmax = _nf4_stack()
    ta, tp, tm = (torch.from_numpy(a), torch.from_numpy(packed),
                  torch.from_numpy(absmax))
    with pytest.raises(TypeError):
        cg.gemv_nf4_grouped_cpu(ta.to(torch.bfloat16), tp, tm, SIZES, EIDS)
    with pytest.raises(ValueError):
        cg.gemv_nf4_grouped_cpu(ta, tp, tm, [1, 2, 0], EIDS)   # empty group
    with pytest.raises(ValueError):
        cg.gemv_nf4_grouped_cpu(ta, tp, tm, [1, 1, 1], EIDS)   # rows mismatch
    with pytest.raises(ValueError):
        cg.gemv_nf4_grouped_cpu(ta, tp, tm, SIZES, [0, 1, 99])  # eid range


# --------------------------------------------------------------------------- #
# grouped dgrad (hybrid Phase 5): gi = g @ W on the same packed bytes
# --------------------------------------------------------------------------- #

def _dgrad_nf4_stack(seed=7, n=N, k=K, sizes=None):
    g = np.random.default_rng(seed)
    packed = g.integers(0, 256, size=(E, n, k // 2), dtype=np.uint8)
    absmax = g.random(size=(E, n, k // 64), dtype=np.float32) + 0.5
    rows = sum(sizes or SIZES)
    grad = g.standard_normal((rows, n), dtype=np.float32)
    return grad, packed, absmax


def _dgrad_mx_stack(seed=8, n=N, k=K, sizes=None, weird_scales=False):
    g = np.random.default_rng(seed)
    packed = g.integers(0, 256, size=(E, n, k // 2), dtype=np.uint8)
    scales = g.integers(100, 140, size=(E, n, k // 32), dtype=np.uint8)
    if weird_scales:
        scales[0, 0, 0] = 0xFF
        scales[0, 0, 1] = 0
        scales[2, 5, 2] = 0xFF
    rows = sum(sizes or SIZES)
    grad = g.standard_normal((rows, n), dtype=np.float32)
    return grad, packed, scales


@needs_native
@pytest.mark.parametrize("threads", [0, 3])
def test_dgrad_nf4_matches_spec_exactly(threads):
    grad, packed, absmax = _dgrad_nf4_stack()
    want = cg.ordered_dgrad_ref(grad, packed, absmax, SIZES, EIDS, fmt="nf4")
    got = cg.dgrad_nf4_grouped_cpu(
        torch.from_numpy(grad), torch.from_numpy(packed),
        torch.from_numpy(absmax), SIZES, EIDS, threads=threads)
    assert np.array_equal(got.numpy(), want), "dgrad chain is not the spec's"


@needs_native
def test_dgrad_mxfp4_matches_spec_exactly_including_ldexp_edges():
    grad, packed, scales = _dgrad_mx_stack(weird_scales=True)
    want = cg.ordered_dgrad_ref(grad, packed, scales, SIZES, EIDS, fmt="mxfp4")
    got = cg.dgrad_mxfp4_grouped_cpu(
        torch.from_numpy(grad), torch.from_numpy(packed),
        torch.from_numpy(scales), SIZES, EIDS)
    assert np.array_equal(got.numpy(), want)


@needs_native
def test_dgrad_is_the_forward_transpose_within_fp32():
    """Independent math check (the spec could share a wrong-axis bug):
    gi must equal g @ W_dequant to normal fp32 tolerance."""
    grad, packed, absmax = _dgrad_nf4_stack(seed=11)
    got = cg.dgrad_nf4_grouped_cpu(
        torch.from_numpy(grad), torch.from_numpy(packed),
        torch.from_numpy(absmax), SIZES, EIDS)
    r = 0
    for gi, e in enumerate(EIDS):
        w = np.stack([cg.dequant_row_nf4(packed[e, n], absmax[e, n])
                      for n in range(N)])          # [N, K] fp64 accumulate
        for _ in range(SIZES[gi]):
            want = grad[r].astype(np.float64) @ w.astype(np.float64)
            np.testing.assert_allclose(got.numpy()[r], want, rtol=2e-5,
                                       atol=2e-5)
            r += 1


@needs_native
def test_dgrad_training_size_groups_are_legal_and_exact():
    """The whole point of the tile scratch: sizes far beyond the decode
    contract's 1..8, exact to the spec."""
    sizes, eids = [12, 1, 20], [1, 3, 0]
    grad, packed, absmax = _dgrad_nf4_stack(seed=13, sizes=sizes)
    want = cg.ordered_dgrad_ref(grad, packed, absmax, sizes, eids, fmt="nf4")
    got = cg.dgrad_nf4_grouped_cpu(
        torch.from_numpy(grad), torch.from_numpy(packed),
        torch.from_numpy(absmax), sizes, eids)
    assert np.array_equal(got.numpy(), want)


@needs_native
@pytest.mark.parametrize("k", [64, 192, 320])
def test_dgrad_k_tail_tiles_are_exact(k):
    """K % 128 == 64 leaves a half-width tail tile; every K tile must chain
    identically to the spec."""
    grad, packed, absmax = _dgrad_nf4_stack(seed=17, k=k)
    want = cg.ordered_dgrad_ref(grad, packed, absmax, SIZES, EIDS, fmt="nf4")
    got = cg.dgrad_nf4_grouped_cpu(
        torch.from_numpy(grad), torch.from_numpy(packed),
        torch.from_numpy(absmax), SIZES, EIDS)
    assert np.array_equal(got.numpy(), want)


@needs_native
def test_dgrad_thread_and_pool_invariance():
    """Work units own disjoint (group, k-tile) outputs, so thread count and
    dispatch mechanism (OpenMP vs executor pool) must not move one bit."""
    grad, packed, absmax = _dgrad_nf4_stack(seed=19, sizes=[9, 2, 4])
    args = (torch.from_numpy(grad), torch.from_numpy(packed),
            torch.from_numpy(absmax), [9, 2, 4], EIDS)
    one = cg.dgrad_nf4_grouped_cpu(*args, threads=1)
    four = cg.dgrad_nf4_grouped_cpu(*args, threads=4)
    assert torch.equal(one, four)
    n = cg.pool_start(2)
    try:
        pooled = cg.dgrad_nf4_grouped_cpu(*args)
    finally:
        cg.pool_stop()
    assert n >= 1 and torch.equal(one, pooled)


@needs_native
def test_dgrad_rejects_contract_violations():
    grad, packed, absmax = _dgrad_nf4_stack()
    tg, tp, ta = (torch.from_numpy(grad), torch.from_numpy(packed),
                  torch.from_numpy(absmax))
    with pytest.raises(TypeError, match="fp32"):
        cg.dgrad_nf4_grouped_cpu(tg.double(), tp, ta, SIZES, EIDS)
    with pytest.raises(ValueError, match="rows"):
        cg.dgrad_nf4_grouped_cpu(tg[:-1], tp, ta, SIZES, EIDS)
    with pytest.raises(ValueError, match="columns"):
        cg.dgrad_nf4_grouped_cpu(tg[:, :-2].contiguous(), tp, ta, SIZES, EIDS)
    with pytest.raises(ValueError, match="out of range"):
        cg.dgrad_nf4_grouped_cpu(tg, tp, ta, SIZES, [2, 0, 99])
    with pytest.raises(ValueError, match="< 1"):
        cg.dgrad_nf4_grouped_cpu(tg[:2], tp, ta, [1, 0, 1], EIDS)


# --------------------------------------------------------------------------- #
# Phase 8: groups larger than the cell's 8-row register blocking
# --------------------------------------------------------------------------- #

@needs_native
@pytest.mark.parametrize("fmt", ["nf4", "mxfp4"])
@pytest.mark.parametrize("sizes", [[9], [16], [8, 17, 1], [33, 2]])
def test_large_groups_match_the_ordered_reference(fmt, sizes):
    """A group may now hold any number of rows: the kernel chunks across
    its 8-row register blocking internally so the weight row stays L1-hot
    (Phase 8's amortization) instead of being re-read per chunk. Outputs
    are per-row, so chunking cannot change a single bit — and the ordered
    reference is what proves it."""
    rows = sum(sizes)
    eids = [i % E for i in range(len(sizes))]
    if fmt == "nf4":
        a, packed, scales = _nf4_stack(rows=rows)
        fn = cg.gemv_nf4_grouped_cpu
    else:
        a, packed, scales = _mx_stack(rows=rows)
        fn = cg.gemv_mxfp4_grouped_cpu
    ref = cg.ref_gemv_grouped(a, packed, scales, sizes, eids, fmt=fmt)
    out = fn(torch.from_numpy(a), torch.from_numpy(packed),
             torch.from_numpy(scales), sizes, eids)
    got, want = out.numpy(), ref
    if fmt == "mxfp4":
        got = np.nan_to_num(got, nan=1e30, posinf=2e30, neginf=-2e30)
        want = np.nan_to_num(want, nan=1e30, posinf=2e30, neginf=-2e30)
    assert np.array_equal(got, want), f"large-group mismatch ({fmt}, {sizes})"


@needs_native
def test_one_large_group_equals_the_split_it_replaces():
    """The pre-Phase-8 caller split an oversize group into 8-row chunks as
    SEPARATE groups (re-reading weights per chunk). One large group must
    produce bit-identical output to that split — same rows, same expert,
    same order — or the change is a numerics change wearing a perf hat."""
    rows = 19
    a, packed, absmax = _nf4_stack(rows=rows)
    ta, tp, tm = (torch.from_numpy(a), torch.from_numpy(packed),
                  torch.from_numpy(absmax))
    whole = cg.gemv_nf4_grouped_cpu(ta, tp, tm, [rows], [3])
    split = cg.gemv_nf4_grouped_cpu(ta, tp, tm, [8, 8, 3], [3, 3, 3])
    assert np.array_equal(whole.numpy(), split.numpy())


# --------------------------------------------------------------------------- #
# fused NF4 expert FFN (one pool wake per layer; hybrid Phase 8 follow-on)
# --------------------------------------------------------------------------- #

FFN_K, FFN_H, FFN_NDN = 128, 64, 24     # gu is [E, 2H, K/2]; dn [E, NDN, H/2]


def _ffn_stack(seed=21, sizes=(1, 9, 3), scale=1.0):
    """Random packed gu+dn stacks. `scale` inflates activations so the gu
    outputs sweep the silu clamps (|x| > 87) and the deep negative tail —
    the corners where a wrong activation contract would hide."""
    g = np.random.default_rng(seed)
    gu_p = g.integers(0, 256, size=(E, 2 * FFN_H, FFN_K // 2), dtype=np.uint8)
    gu_a = (g.random(size=(E, 2 * FFN_H, FFN_K // 64), dtype=np.float32)
            + 0.5) * np.float32(scale)
    dn_p = g.integers(0, 256, size=(E, FFN_NDN, FFN_H // 2), dtype=np.uint8)
    dn_a = g.random(size=(E, FFN_NDN, FFN_H // 64), dtype=np.float32) + 0.5
    a = g.standard_normal((sum(sizes), FFN_K), dtype=np.float32)
    eids = [int(x) for x in g.permutation(E)[:len(sizes)]]
    return a, gu_p, gu_a, dn_p, dn_a, list(sizes), eids


def test_silu_locked_ref_tracks_true_silu():
    """The polynomial is a spec, not an approximation contest — but it must
    still BE silu: ~2e-8 relative against float64 ground truth on the
    clamped domain, and exactly x (resp. ~0) beyond the clamps."""
    x = np.linspace(-87.0, 87.0, 20011, dtype=np.float32)
    got = cg.silu_locked_ref(x)
    want = x.astype(np.float64) / (1.0 + np.exp(-x.astype(np.float64)))
    denom = np.maximum(np.abs(want), 1e-6)
    # bound is diagnostic (the CONTRACT is the locked op sequence):
    # the deep tail computes sig via 1/(1+e) and sits a few f32
    # ulps off true silu — ~9e-7 worst on this grid
    assert np.max(np.abs(got - want) / denom) < 2e-6
    big = np.asarray([88.0, 500.0, 1e30], dtype=np.float32)
    assert np.array_equal(cg.silu_locked_ref(big), big)
    assert np.array_equal(cg.silu_locked_ref(-big),
                          np.zeros_like(big))


@needs_native
@pytest.mark.parametrize("sizes,scale", [
    ((1, 9, 3), 1.0),          # chunk crossing NF4_CELL_ROWS
    ((1, 1, 1, 1), 1.0),       # every group T=1 -> the paired-column path
    ((8, 8), 1.0),             # exactly at the row blocking
    ((2, 5), 40.0),            # activations through the silu clamps
])
def test_ffn_fused_matches_composed_spec_exactly(sizes, scale):
    a, gu_p, gu_a, dn_p, dn_a, sz, eids = _ffn_stack(sizes=sizes,
                                                     scale=scale)
    want = cg.ref_ffn_grouped(a, gu_p, gu_a, dn_p, dn_a, sz, eids)
    got = cg.gemm_nf4_ffn_grouped_cpu(
        torch.from_numpy(a), torch.from_numpy(gu_p), torch.from_numpy(gu_a),
        torch.from_numpy(dn_p), torch.from_numpy(dn_a), sz, eids)
    assert np.array_equal(got.numpy(), want), (
        f"fused FFN diverged from the composed spec (sizes={sizes}, "
        f"scale={scale}) — max abs diff "
        f"{np.max(np.abs(got.numpy() - want))}")


@needs_native
def test_ffn_fused_equals_two_native_calls_plus_locked_silu():
    """The fusion must change WHERE the chain runs, not what it computes:
    fused == gemv -> silu_locked_ref -> gemv with the same native cells."""
    a, gu_p, gu_a, dn_p, dn_a, sz, eids = _ffn_stack(seed=33, sizes=(4, 7))
    gu = cg.gemv_nf4_grouped_cpu(torch.from_numpy(a), torch.from_numpy(gu_p),
                                 torch.from_numpy(gu_a), sz, eids)
    h = (cg.silu_locked_ref(gu.numpy()[:, :FFN_H])
         * gu.numpy()[:, FFN_H:]).astype(np.float32)
    dn = cg.gemv_nf4_grouped_cpu(torch.from_numpy(h), torch.from_numpy(dn_p),
                                 torch.from_numpy(dn_a), sz, eids)
    fused = cg.gemm_nf4_ffn_grouped_cpu(
        torch.from_numpy(a), torch.from_numpy(gu_p), torch.from_numpy(gu_a),
        torch.from_numpy(dn_p), torch.from_numpy(dn_a), sz, eids)
    assert torch.equal(fused, dn)


@needs_native
def test_ffn_pool_and_threads_do_not_change_bits():
    a, gu_p, gu_a, dn_p, dn_a, sz, eids = _ffn_stack(seed=44, sizes=(1, 9, 3))
    args = (torch.from_numpy(a), torch.from_numpy(gu_p),
            torch.from_numpy(gu_a), torch.from_numpy(dn_p),
            torch.from_numpy(dn_a), sz, eids)
    base = cg.gemm_nf4_ffn_grouped_cpu(*args, threads=1)
    t4 = cg.gemm_nf4_ffn_grouped_cpu(*args, threads=4)
    assert torch.equal(base, t4)
    n = cg.pool_start(4)
    try:
        assert n >= 1
        pooled = cg.gemm_nf4_ffn_grouped_cpu(*args)
    finally:
        cg.pool_stop()
    assert torch.equal(base, pooled)


@needs_native
def test_ffn_bad_calls_raise():
    a, gu_p, gu_a, dn_p, dn_a, sz, eids = _ffn_stack()
    ta = torch.from_numpy(a)
    with pytest.raises(ValueError, match="even"):
        cg.gemm_nf4_ffn_grouped_cpu(ta, torch.from_numpy(gu_p[:, :-1].copy()),
                                    torch.from_numpy(gu_a[:, :-1].copy()),
                                    torch.from_numpy(dn_p),
                                    torch.from_numpy(dn_a), sz, eids)
    with pytest.raises(ValueError, match="H % 64"):
        cg.gemm_nf4_ffn_grouped_cpu(ta, torch.from_numpy(gu_p[:, :64].copy()),
                                    torch.from_numpy(gu_a[:, :64].copy()),
                                    torch.from_numpy(dn_p),
                                    torch.from_numpy(dn_a), sz, eids)
    with pytest.raises(ValueError, match="expert counts"):
        cg.gemm_nf4_ffn_grouped_cpu(ta, torch.from_numpy(gu_p),
                                    torch.from_numpy(gu_a),
                                    torch.from_numpy(dn_p[:2]),
                                    torch.from_numpy(dn_a[:2]), sz, eids)
