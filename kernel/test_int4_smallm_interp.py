# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""K16 small-M int4-b32 GEMM: the correctness contract, device-free (Triton interpreter mode, CPU).

Guards: (1) within one bf16 output ulp of ``x @ dequant_int4_ref(packed, scales).T`` on the registered
projection shapes (scaled down in K only where the interpreter would crawl) and on the checkout shapes
(K=64, 96; N not a multiple of BLOCK_N; M < 16); (2) deterministic for a fixed config; (3) within one
bf16 ulp across SK; (4) the plan legaliser refuses what the format cannot express and never changes
arithmetic silently; (5) the counter is left zeroed (the launch is re-armed). Set by conftest/CI:
TRITON_INTERPRET=1.
"""
import os
os.environ.setdefault("TRITON_INTERPRET", "1")

import pytest
import torch

pytest.importorskip("triton", reason="interpreter mode needs triton (Linux-only dependency)")

from int4_pack_ref import dequant_int4_ref, pack_int4_b32          # noqa: E402
from int4_smallm import gemm_int4_b32_smallm, plan_smallm, smallm_workspace  # noqa: E402


def _case(N, K, M, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(N, K, generator=g) * 0.7
    packed, scales = pack_int4_b32(w)
    w_deq = dequant_int4_ref(packed, scales, N, K).float()
    x = (torch.randn(M, K, generator=g) * 0.5).to(torch.bfloat16)
    ref = (x.float() @ w_deq.t())
    return x, packed.contiguous(), scales.contiguous(), ref


def _ulp_ok(y, ref):
    # within one bf16 ulp of the reference value, plus fp32 accumulation slack scaled to the row magnitude
    yb = y.float(); rb = ref.to(torch.bfloat16).float()
    ulp = torch.maximum(rb.abs(), torch.full_like(rb, 1e-6)) * 2 ** -7
    slack = ref.abs().max() * 2e-6
    return bool(((yb - rb).abs() <= ulp + slack).all()), float((yb - ref).abs().max())


@pytest.mark.parametrize("N,K,M,bn,kc,sk", [
    (64, 256, 16, 64, 128, 2),       # a 2-split, two chunks each
    (100, 128, 16, 64, 128, 1),      # N not a multiple of BLOCK_N, single chunk
    (48, 64, 5, 32, 32, 2),          # checkout K=64 -> KC 32, M < 16 (padded rows)
    (40, 96, 3, 32, 32, 3),          # checkout K=96 -> KC 32, SK 3 (odd split)
    (256, 512, 16, 64, 256, 2),      # fat chunk
    (128, 1024, 16, 128, 128, 8),    # 8-way split
])
def test_matches_dequant_reference(N, K, M, bn, kc, sk):
    x, packed, scales, ref = _case(N, K, M)
    y = gemm_int4_b32_smallm(x, packed, scales, block_n=bn, kc=kc, sk=sk)
    assert y.shape == (M, N) and y.dtype == torch.bfloat16
    ok, err = _ulp_ok(y, ref)
    assert ok, f"max |y - ref| {err} exceeds one bf16 ulp on ({N},{K},{M}) bn{bn} kc{kc} sk{sk}"


def test_deterministic_and_within_ulp_across_sk():
    x, packed, scales, ref = _case(128, 1024, 16)
    ys = {}
    for sk in (1, 2, 4, 8):
        a = gemm_int4_b32_smallm(x, packed, scales, block_n=64, kc=128, sk=sk)
        b = gemm_int4_b32_smallm(x, packed, scales, block_n=64, kc=128, sk=sk)
        assert torch.equal(a, b), f"sk={sk}: two launches of one config differ"
        ys[sk] = a.float()
    base = ys[1].to(torch.bfloat16).float()
    ulp = torch.maximum(base.abs(), torch.full_like(base, 1e-6)) * 2 ** -7
    for sk in (2, 4, 8):
        assert ((ys[sk] - base).abs() <= ulp).all(), f"sk={sk} differs from sk=1 by more than one bf16 ulp"


def test_counter_rearmed_and_workspace_reuse():
    x, packed, scales, ref = _case(96, 512, 16)
    ws = smallm_workspace(96, block_n=32, sk=4, device=x.device)
    y1 = gemm_int4_b32_smallm(x, packed, scales, block_n=32, kc=128, sk=4, workspace=ws)
    assert int(ws[1].abs().sum()) == 0, "the counter must be left zeroed for the next launch"
    y2 = gemm_int4_b32_smallm(x, packed, scales, block_n=32, kc=128, sk=4, workspace=ws)
    assert torch.equal(y1, y2)
    ok, _ = _ulp_ok(y1, ref)
    assert ok


def test_plan_refuses_or_legalises_never_silently_rearithmetics():
    assert plan_smallm(4096, 2048) == (64, 128, 4)
    assert plan_smallm(512, 96, kc=128, sk=4) == (64, 32, 1)          # K=96: only KC=32 divides; 3 chunks -> sk falls to 1
    assert plan_smallm(512, 96, kc=32, sk=3) == (64, 32, 3)           # an odd split that divides is kept
    with pytest.raises(ValueError):
        plan_smallm(64, 48)                                           # not a multiple of the scale block
    with pytest.raises(ValueError):
        gemm_int4_b32_smallm(torch.zeros(17, 64, dtype=torch.bfloat16), *pack_int4_b32(torch.randn(8, 64)))
