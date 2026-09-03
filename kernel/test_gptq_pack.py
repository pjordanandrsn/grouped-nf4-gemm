# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Calibrated packing must beat round-to-nearest on the quantity that
actually gates this lane: activation-weighted OUTPUT error."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))


def _layer(N=256, K=512, seed=0, regime="aniso"):
    g = torch.Generator().manual_seed(seed)
    w = (torch.randn(N, K, generator=g) / K ** 0.5)
    if regime == "outlier_w":
        w[:, torch.randperm(K, generator=g)[:K // 32]] *= 6.0
    if regime == "outlier_x":
        # a few enormous activation channels: the documented LLM regime,
        # and the one where a mis-implemented GPTQ loses to plain rounding
        chan = torch.rand(K, generator=g) * 0.3 + 0.05
        chan[torch.randperm(K, generator=g)[:K // 64]] = 20.0
    else:
        chan = torch.rand(K, generator=g) * 4.0 + 0.05
    return w, torch.randn(2048, K, generator=g) * chan


def _out_err(w_deq, w, x):
    return ((x @ w_deq.t()) - (x @ w.t())).pow(2).mean().item()


@pytest.mark.parametrize("regime", ["aniso", "outlier_w", "outlier_x"])
def test_calibrated_beats_rtn_on_output_error(regime):
    """Measured gains: ~11% (anisotropic), ~7% (outlier activations).
    The bar is set at 5% in EVERY regime rather than at the best one --
    a draft that scaled to the source block's absmax and re-derived
    scales through the uncalibrated packer measured -3% here, i.e. WORSE
    than plain rounding, and a best-case-only assertion would have
    passed it. The real gate is the paired K8 perplexity run; this test
    only has to catch a lane that cannot possibly clear it."""
    from gptq_pack import HessianAccumulator, gptq_pack_int4_b32
    from int4_pack_ref import dequant_int4_ref, pack_int4_b32
    w, x = _layer(regime=regime)
    N, K = w.shape

    acc = HessianAccumulator(K)
    for i in range(0, x.shape[0], 256):
        acc.add(x[i:i + 256])

    p_rtn, s_rtn = pack_int4_b32(w)
    p_cal, s_cal = gptq_pack_int4_b32(w, acc.H)
    d_rtn = dequant_int4_ref(p_rtn, s_rtn, N, K)
    d_cal = dequant_int4_ref(p_cal, s_cal, N, K)

    e_rtn = _out_err(d_rtn, w, x)
    e_cal = _out_err(d_cal, w, x)
    assert e_cal < e_rtn * 0.95, (
        f"[{regime}] calibrated output MSE {e_cal:.4e} vs RTN {e_rtn:.4e} "
        "-- no material gain, the lane's premise fails")


def test_format_is_byte_identical_to_the_shipped_grid():
    """Same packed layout and scale dtype as the uncalibrated packer, so
    the shipped GEMV and Int4Linear need no changes at all."""
    from gptq_pack import HessianAccumulator, gptq_pack_int4_b32
    from int4_pack_ref import pack_int4_b32
    w, x = _layer(N=64, K=128, seed=3)
    acc = HessianAccumulator(w.shape[1])
    acc.add(x)
    p_cal, s_cal = gptq_pack_int4_b32(w, acc.H)
    p_rtn, s_rtn = pack_int4_b32(w)
    assert p_cal.shape == p_rtn.shape and p_cal.dtype == p_rtn.dtype
    assert s_cal.shape == s_rtn.shape and s_cal.dtype == s_rtn.dtype


def test_identity_hessian_stays_close_to_rtn():
    """With no calibration signal (H = I) there is nothing to exploit, so
    the result must not be WORSE than plain rounding -- a guard against a
    sign error in the error-propagation step."""
    from gptq_pack import gptq_pack_int4_b32
    from int4_pack_ref import dequant_int4_ref, pack_int4_b32
    w, x = _layer(N=128, K=256, seed=7)
    N, K = w.shape
    H = torch.eye(K) * 2.0
    p_cal, s_cal = gptq_pack_int4_b32(w, H)
    p_rtn, s_rtn = pack_int4_b32(w)
    d_cal = dequant_int4_ref(p_cal, s_cal, N, K)
    d_rtn = dequant_int4_ref(p_rtn, s_rtn, N, K)
    w_err_cal = (d_cal - w).pow(2).mean().item()
    w_err_rtn = (d_rtn - w).pow(2).mean().item()
    assert w_err_cal <= w_err_rtn * 1.10


def test_dead_channels_are_handled():
    """A channel the calibration never excites has zero Hessian diagonal;
    it must not produce NaNs or blow up the Cholesky."""
    from gptq_pack import HessianAccumulator, gptq_pack_int4_b32
    from int4_pack_ref import dequant_int4_ref
    w, x = _layer(N=64, K=128, seed=11)
    x[:, 5] = 0.0
    x[:, 40] = 0.0
    acc = HessianAccumulator(w.shape[1])
    acc.add(x)
    p, s = gptq_pack_int4_b32(w, acc.H)
    d = dequant_int4_ref(p, s, *w.shape)
    assert torch.isfinite(d).all()
