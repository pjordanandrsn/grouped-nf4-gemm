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


def _nf4_stack(seed=0):
    g = np.random.default_rng(seed)
    packed = g.integers(0, 256, size=(E, N, K // 2), dtype=np.uint8)
    absmax = g.random(size=(E, N, K // 64), dtype=np.float32) + 0.5
    rows = sum(SIZES)
    a = g.standard_normal((rows, K), dtype=np.float32)
    return a, packed, absmax


def _mx_stack(seed=1, weird_scales=False):
    g = np.random.default_rng(seed)
    packed = g.integers(0, 256, size=(E, N, K // 2), dtype=np.uint8)
    scales = g.integers(100, 140, size=(E, N, K // 32), dtype=np.uint8)
    if weird_scales:
        scales[0, 0, 0] = 0xFF        # oracle ldexp semantics: 2^128
        scales[0, 0, 1] = 0           # subnormal 2^-127
        scales[1, 3, 2] = 0xFF
    rows = sum(SIZES)
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
        cg.gemv_nf4_grouped_cpu(ta, tp, tm, [1, 2, 9], EIDS)   # size > 8
    with pytest.raises(ValueError):
        cg.gemv_nf4_grouped_cpu(ta, tp, tm, [1, 1, 1], EIDS)   # rows mismatch
    with pytest.raises(ValueError):
        cg.gemv_nf4_grouped_cpu(ta, tp, tm, SIZES, [0, 1, 99])  # eid range
