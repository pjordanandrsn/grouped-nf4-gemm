# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""DeepSeek-V4's epilogue on the NVMe engine — pure python, no GPU.

`Mxfp4NvmeResidencyV4` exists because V4 is neither of its parents: gpt-oss's *clamps* with
SwiGLU's *combination*, over a clean-concat `gate_up`. Every one of those three choices is
silently wrong if taken from the wrong parent — the shapes agree either way — so each gets
its own assertion against a transcription of the checkpoint's `inference/model.py`.

`_glu` reads only `limit` and `cd`, so a stub stands in for the engine and these run without
an arena, CUDA, or triton.
"""
import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))

from mxfp4_residency import (  # noqa: E402
    V4_RESIDENCY_KINDS,
    Mxfp4NvmeResidencyV4,
)

LIMIT = 2.0          # far below the real 10.0 so the clamp actually binds on test data


class _Stub:
    """`_glu` touches only these two attributes."""

    def __init__(self, limit=LIMIT, cd=torch.bfloat16):
        self.limit = limit
        self.cd = cd


def _glu(gu, limit=LIMIT, cd=torch.bfloat16):
    return Mxfp4NvmeResidencyV4._glu(_Stub(limit, cd), gu)


def _gu(rows=8, inter=16, seed=0, scale=3.0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(rows, 2 * inter, generator=g) * scale).bfloat16()


def _ref(gu, limit=LIMIT, *, interleaved=False, two_sided_gate=False, gpt_oss=False):
    """inference/model.py's Expert.forward, transcribed."""
    if interleaved:
        gate, up = gu[..., ::2].float(), gu[..., 1::2].float()
    else:
        gate, up = (t.float() for t in gu.chunk(2, dim=-1))
    if two_sided_gate:
        gate = gate.clamp(min=-limit, max=limit)
    else:
        gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    if gpt_oss:
        return (up + 1) * (gate * torch.sigmoid(gate * 1.702))
    return F.silu(gate) * up


def _rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-6)).item()


def test_matches_the_reference_expert_forward():
    gu = _gu()
    assert _rel(_glu(gu), _ref(gu).bfloat16()) < 1e-6


def test_gate_clamp_is_one_sided():
    """gate clamps only from above; up clamps both ways. Taking gpt-oss's two-sided clamp
    for the gate is the kind of mistake that changes nothing structural."""
    gu = _gu(seed=1)
    one, two = _ref(gu), _ref(gu, two_sided_gate=True)
    assert _rel(one, two) > 1e-2, "test data never drives gate below -limit; raise scale"
    assert _rel(_glu(gu), one.bfloat16()) < 1e-6
    assert _rel(_glu(gu), two.bfloat16()) > 1e-2


def test_is_not_the_gptoss_glu():
    """V4 borrows gpt-oss's clamps but NOT its combination. Guards a refactor that
    collapses this class into its parent."""
    gu = _gu(seed=2)
    assert _rel(_glu(gu), _ref(gu).bfloat16()) < 1e-6
    assert _rel(_glu(gu), _ref(gu, gpt_oss=True).bfloat16()) > 1e-2


def test_split_is_clean_concat_not_interleaved():
    """gpt-oss ships gate/up interleaved by column; V4 (like K3) ships them concatenated.
    Reading the wrong one produces a same-shaped tensor of nonsense."""
    gu = _gu(seed=3)
    assert _rel(_glu(gu), _ref(gu).bfloat16()) < 1e-6
    assert _rel(_glu(gu), _ref(gu, interleaved=True).bfloat16()) > 1e-2


def test_glu_is_evaluated_in_fp32():
    """V4's reference computes the whole GLU in fp32 and casts only before the down
    projection; the sibling epilogues stay in compute dtype because their references do.

    Asserted structurally, by the dtype `silu` is handed. A numeric version cannot see it:
    the result is cast back to `cd` on the way out, and that rounding is larger than the
    difference being measured — so a tolerance test here would measure noise.
    """
    seen = []
    real_silu = F.silu

    def spy(t, *a, **k):
        seen.append(t.dtype)
        return real_silu(t, *a, **k)

    F.silu = spy
    try:
        out = _glu(_gu(seed=4), cd=torch.bfloat16)
    finally:
        F.silu = real_silu
    assert seen == [torch.float32], f"GLU evaluated in {seen}, not fp32"
    assert out.dtype is torch.bfloat16, "must cast back to compute dtype for the down GEMM"


@pytest.mark.parametrize("cd", [torch.bfloat16, torch.float16, torch.float32])
def test_returns_the_compute_dtype(cd):
    assert _glu(_gu(seed=5), cd=cd).dtype is cd


def test_limit_is_load_bearing():
    gu = _gu(seed=6)
    assert _rel(_glu(gu, limit=0.5), _glu(gu, limit=50.0)) > 1e-2


def test_residency_kinds_cover_both_projections_and_their_scales():
    """The bake and the engine must agree on the segment list; a missing scale reads as a
    silent exponent of zero rather than an error."""
    assert set(V4_RESIDENCY_KINDS) == {
        "w1.weight", "w3.weight", "w1.scale", "w3.scale", "w2.weight", "w2.scale"}
    # weights and scales pair up, and w1/w3 precede w2 (gate_up before down)
    assert V4_RESIDENCY_KINDS.index("w1.weight") < V4_RESIDENCY_KINDS.index("w2.weight")
    for proj in ("w1", "w2", "w3"):
        assert f"{proj}.weight" in V4_RESIDENCY_KINDS
        assert f"{proj}.scale" in V4_RESIDENCY_KINDS
