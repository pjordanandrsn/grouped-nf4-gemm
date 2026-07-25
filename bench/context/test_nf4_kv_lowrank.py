# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).

"""Tests for the low-rank KV codes path.

The load-bearing claim here is the *absorption identity* — that attention over
rank-r codes equals attention over the reconstructed cache, exactly, so the
up-projection never has to touch stored tokens. These tests pin that identity
against a reference that does reconstruct, which is the only way to catch an
algebra error: a wrong basis application still produces plausible-looking
numbers, it just answers a different question.

Real-data structure (does rank-64 actually retain the energy?) is deliberately
NOT tested here — it needs real weights and belongs in
``bench/context/lowrank_probe.py``, which writes a receipt. Synthetic tests
cannot answer it: iid data has no low-rank structure by construction, so a test
built on it would either be vacuous or would bake in a fixture that flatters
the method.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "kernel"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nf4_grouped import BLOCKSIZE
from nf4_kv import dequant_kv_ref, quantize_kv
from nf4_kv_lowrank import (attend_lowrank_nf4, calibrate_basis,
                            energy_retained, lowrank_cache_bytes,
                            pack_lowrank_kv, project_to_codes, reconstruct_ref)

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _lowrank_sample(T, H, D, rank, seed=0, device="cuda"):
    """Data that genuinely lives in a rank-`rank` subspace, so truncation error
    is zero by construction and any deviation is the code's fault, not the
    data's."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    basis = torch.linalg.qr(torch.randn(H, D, D, generator=g))[0][:, :rank, :]
    coeff = torch.randn(T, H, rank, generator=g)
    x = torch.einsum("thr,hrd->thd", coeff, basis)
    return (x / x.abs().amax() * 0.6).to(device, torch.bfloat16)


@cuda
def test_basis_is_orthonormal():
    x = _lowrank_sample(256, 4, 128, 64, seed=1)
    B = calibrate_basis(x, 64)
    assert B.shape == (4, 64, 128)
    for h in range(4):
        gram = B[h] @ B[h].T
        assert torch.allclose(gram, torch.eye(64, device=gram.device), atol=1e-4)


@cuda
def test_full_rank_projection_is_lossless():
    """rank == head_dim must round-trip exactly (up to fp): the projection is
    then a change of basis, nothing more. If this drifts, the einsum indices are
    transposed somewhere."""
    x = _lowrank_sample(128, 4, 128, 128, seed=2)
    B = calibrate_basis(x, 128)
    back = reconstruct_ref(project_to_codes(x, B), B)
    rel = ((back - x.float()).norm() / x.float().norm()).item()
    assert rel < 1e-3, f"full-rank round-trip lost {rel:.2e}"


@cuda
def test_energy_retained_is_one_at_full_rank_and_monotone():
    x = _lowrank_sample(256, 4, 128, 96, seed=3)
    assert energy_retained(x, 128) == pytest.approx(1.0, abs=1e-4)
    e64, e128 = energy_retained(x, 64), energy_retained(x, 128)
    assert e64 <= e128 + 1e-6
    # data built in a rank-96 subspace: rank-128 keeps everything, rank-64 does not
    assert e64 < 0.999


@cuda
def test_energy_retained_predicts_truncation_error():
    """The gate reads `energy_retained`, so it has to actually predict the
    error it is used to bound: for an orthonormal basis, relative error is
    sqrt(1 - energy)."""
    x = _lowrank_sample(256, 4, 128, 128, seed=4)
    B = calibrate_basis(x, 64)
    back = reconstruct_ref(project_to_codes(x, B), B)
    measured = ((back - x.float()).norm() / x.float().norm()).item()
    predicted = (1.0 - energy_retained(x, 64)) ** 0.5
    assert measured == pytest.approx(predicted, rel=0.05), (measured, predicted)


@cuda
@pytest.mark.parametrize("H_q,H_kv,D,rank", [(4, 4, 128, 64), (32, 4, 128, 64),
                                             (16, 2, 256, 128), (8, 8, 128, 128)])
def test_absorption_identity_matches_reconstructing_reference(H_q, H_kv, D, rank):
    """THE test. `attend_lowrank_nf4` never reconstructs the cache; the
    reference does. They must agree to fp tolerance, because the rewrite

        q @ (C @ B).T == (q @ B.T) @ C.T      and      p @ (C @ B) == (p @ C) @ B

    is exact algebra. Any disagreement is a real bug in the absorption, not a
    fidelity trade-off — the quantization is common to both sides."""
    T = 192
    kx = _lowrank_sample(T, H_kv, D, rank, seed=5)
    vx = _lowrank_sample(T, H_kv, D, rank, seed=6)
    Bk, Bv = calibrate_basis(kx, rank), calibrate_basis(vx, rank)
    kp, ka = pack_lowrank_kv(kx, Bk)
    vp, va = pack_lowrank_kv(vx, Bv)
    q = torch.randn(H_q, D, device="cuda", dtype=torch.float32) * 0.5

    got = attend_lowrank_nf4(q, Bk, kp, ka, Bv, vp, va)

    # reference: reconstruct the SAME quantized codes, then dense attention
    k_hat = reconstruct_ref(dequant_kv_ref(kp, ka, rank), Bk)
    v_hat = reconstruct_ref(dequant_kv_ref(vp, va, rank), Bv)
    rep = H_q // H_kv
    k_hat = k_hat.repeat_interleave(rep, dim=1)
    v_hat = v_hat.repeat_interleave(rep, dim=1)
    scores = torch.einsum("hd,thd->ht", q, k_hat) * D ** -0.5
    ref = torch.einsum("ht,thd->hd", torch.softmax(scores, dim=-1), v_hat)

    rel = ((got - ref).norm() / ref.norm()).item()
    assert rel < 2e-3, f"absorption diverged from reconstruction: {rel:.2e}"


@cuda
def test_scale_uses_original_head_dim_not_rank():
    """A projection is a change of basis, not of scale — softmax temperature
    must not silently change when rank != head_dim. Catching this matters
    because a wrong scale still yields a valid-looking distribution."""
    T, H, D, rank = 128, 4, 128, 64
    kx, vx = _lowrank_sample(T, H, D, rank, 7), _lowrank_sample(T, H, D, rank, 8)
    Bk, Bv = calibrate_basis(kx, rank), calibrate_basis(vx, rank)
    kp, ka = pack_lowrank_kv(kx, Bk)
    vp, va = pack_lowrank_kv(vx, Bv)
    q = torch.randn(H, D, device="cuda", dtype=torch.float32)
    default = attend_lowrank_nf4(q, Bk, kp, ka, Bv, vp, va)
    explicit = attend_lowrank_nf4(q, Bk, kp, ka, Bv, vp, va, scale=D ** -0.5)
    wrong = attend_lowrank_nf4(q, Bk, kp, ka, Bv, vp, va, scale=rank ** -0.5)
    assert torch.allclose(default, explicit, atol=1e-5)
    assert not torch.allclose(default, wrong, atol=1e-3), "scale arg is inert"


@cuda
def test_rank_must_be_blocksize_multiple():
    x = _lowrank_sample(64, 2, 128, 64, seed=9)
    with pytest.raises(ValueError, match=f"multiple of the quant blocksize {BLOCKSIZE}"):
        calibrate_basis(x, 40)
    with pytest.raises(ValueError, match="exceeds head_dim"):
        calibrate_basis(x, 192)


def test_footprint_arithmetic_and_basis_amortization():
    """The basis is a per-layer constant, so the ratio must IMPROVE with context
    and be honest (possibly < 1) at short context. A footprint function that
    hid the basis would overstate the saving exactly where it is weakest."""
    kw = dict(kv_heads=4, rank=64, head_dim=128, n_layers=1)
    short = lowrank_cache_bytes(n_tokens=64, **kw)
    long = lowrank_cache_bytes(n_tokens=32768, **kw)
    # per token: 2 * 4 heads * (32 nibble-bytes + 1 absmax * 4) = 288 B
    assert long["codes_bytes"] == 288 * 32768
    assert long["basis_bytes"] == 2 * 4 * 64 * 128 * 4            # fp32, K and V
    assert long["ratio_vs_fp16"] > short["ratio_vs_fp16"]
    # The ASYMPTOTE is 1024 B/token fp16 / 288 B/token codes = 7.11x. Real
    # contexts do not reach it, because the basis never amortizes away:
    asymptote = 2 * 4 * 128 * 2 / 288
    assert asymptote == pytest.approx(7.111, abs=0.01)
    mid = lowrank_cache_bytes(n_tokens=4096, **kw)
    assert long["ratio_vs_fp16"] == pytest.approx(6.92, abs=0.02)    # 32K: basis is 2.7%
    assert mid["ratio_vs_fp16"] == pytest.approx(5.82, abs=0.02)     # 4K:  basis is 18%
    assert short["ratio_vs_fp16"] < 1.0, "at 64 tokens the basis COSTS more than it saves"
