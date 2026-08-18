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
