# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The knobs must be provable from the run, not from the environment.

PREREG-m3 asks whether two certified knobs may ship ON by default. Its
four arms are OFF / DOTPAD / FP8 / BOTH, and every one of them is
selected by an env var. But an env var is a REQUEST. `GNF4_GEMV_DOTPAD=1`
engages the dot-pad kernel only if the shape is in `_DOTPAD_CONFIGS`
AND the part carries >= 160 SMs; miss either and the call quietly
takes the certified scalar path.

That failure is invisible from the outside. A "dotpad" arm that
silently ran scalar produces a step time equal to OFF and -- because
K6-B measured dot-pad as token-IDENTICAL at 127 tokens -- a
perplexity equal to OFF too. M3 would read "no quality cost" and flip
a default on the strength of an arm that never exercised the
mechanism: a pass no input could have refuted
([[check-the-result-could-have-failed]]).

Recording `os.environ` does not help, because that is the request
again. These tests pin the counters that record what actually
dispatched, and -- the part that matters -- prove the counters
DISCRIMINATE: on a part below the SM guard, the env var set, the
tally must show the scalar path.
"""
import os

import pytest
import torch

import nf4_grouped
from nf4_grouped import gemm_4bit_grouped
from nf4_pack_ref import make_stack

CUDA = torch.cuda.is_available()
needs_cuda = pytest.mark.skipif(not CUDA, reason="dispatch needs a device")

N, K, E, T = 1536, 2048, 4, 4


def test_tally_starts_zero_and_resets():
    nf4_grouped._DISPATCH_COUNTS["dotpad"] = 7
    nf4_grouped.reset_dispatch_counts()
    assert nf4_grouped.dispatch_counts() == {
        "dotpad": 0, "dotpad_splitk": 0, "scalar": 0, "scalar_splitk": 0}


def test_tally_accessor_returns_a_copy():
    """A caller must not be able to mutate the live tally by accident."""
    snap = nf4_grouped.dispatch_counts()
    snap["dotpad"] = 999
    assert nf4_grouped.dispatch_counts()["dotpad"] != 999


def _call(dev):
    B, A = make_stack(E, N, K, seed=3, device=dev)
    a = torch.randn(T, K, dtype=torch.float32).to(torch.bfloat16).to(dev)
    eids = torch.arange(T, dtype=torch.int32, device=dev) % E
    return gemm_4bit_grouped(a, B, A, [1] * T, eids)


@needs_cuda
def test_knob_off_dispatches_the_certified_scalar_path(monkeypatch):
    monkeypatch.delenv("GNF4_GEMV_DOTPAD", raising=False)
    monkeypatch.setattr(nf4_grouped, "_sm_count", lambda d: 200)
    nf4_grouped.reset_dispatch_counts()
    _call(torch.device("cuda"))
    c = nf4_grouped.dispatch_counts()
    assert c["dotpad"] == 0 and c["dotpad_splitk"] == 0, c
    assert c["scalar"] + c["scalar_splitk"] > 0, c


@needs_cuda
def test_knob_on_above_the_sm_guard_dispatches_dot_pad(monkeypatch):
    monkeypatch.setenv("GNF4_GEMV_DOTPAD", "1")
    monkeypatch.setattr(nf4_grouped, "_sm_count", lambda d: 200)
    nf4_grouped.reset_dispatch_counts()
    _call(torch.device("cuda"))
    c = nf4_grouped.dispatch_counts()
    assert c["dotpad"] + c["dotpad_splitk"] > 0, c


@needs_cuda
def test_the_tally_CATCHES_a_knob_that_was_silently_ignored(monkeypatch):
    """The whole reason this instrument exists.

    Env var set, part below the SM guard: the request is honoured by
    nothing. Without the tally this arm is indistinguishable from OFF
    and M3 reads it as free quality.
    """
    monkeypatch.setenv("GNF4_GEMV_DOTPAD", "1")
    monkeypatch.setattr(nf4_grouped, "_sm_count", lambda d: 26)
    nf4_grouped.reset_dispatch_counts()
    _call(torch.device("cuda"))
    c = nf4_grouped.dispatch_counts()
    assert c["dotpad"] == 0 and c["dotpad_splitk"] == 0, (
        "the SM guard should have refused dot-pad on a 26-SM part", c)
    assert c["scalar"] + c["scalar_splitk"] > 0, (
        "the call must still have gone somewhere -- a tally that is "
        "all-zero proves nothing about which path ran", c)


@needs_cuda
def test_an_unregistered_shape_also_falls_back(monkeypatch):
    """The second silent-fallback door: SM guard passes, shape misses."""
    monkeypatch.setenv("GNF4_GEMV_DOTPAD", "1")
    monkeypatch.setattr(nf4_grouped, "_sm_count", lambda d: 200)
    n, k = 128, 256
    assert (n, k) not in nf4_grouped._DOTPAD_CONFIGS
    B, A = make_stack(E, n, k, seed=4, device="cuda")
    a = torch.randn(T, k, dtype=torch.float32).to(torch.bfloat16).cuda()
    eids = torch.arange(T, dtype=torch.int32, device="cuda") % E
    nf4_grouped.reset_dispatch_counts()
    gemm_4bit_grouped(a, B, A, [1] * T, eids)
    c = nf4_grouped.dispatch_counts()
    assert c["dotpad"] + c["dotpad_splitk"] == 0, c
    assert c["scalar"] + c["scalar_splitk"] > 0, c


def test_attention_tally_starts_zero_and_resets():
    import fp8_paged_attn
    fp8_paged_attn._COMPUTE_COUNTS["fp8"] = 5
    fp8_paged_attn.reset_compute_counts()
    assert fp8_paged_attn.compute_counts() == {"f32": 0, "fp8": 0}


def test_attention_tally_accessor_returns_a_copy():
    import fp8_paged_attn
    snap = fp8_paged_attn.compute_counts()
    snap["fp8"] = 999
    assert fp8_paged_attn.compute_counts()["fp8"] != 999


def test_env_alone_is_not_treated_as_evidence():
    """Guard the reasoning, not just the counters.

    If someone later 'simplifies' the receipt back to reading the
    environment, this fails: the env says fp8, the tally says nothing
    ran, and those must not be the same statement.
    """
    import fp8_paged_attn
    os.environ["GNF4_ATTN_COMPUTE"] = "fp8"
    try:
        fp8_paged_attn.reset_compute_counts()
        assert fp8_paged_attn._compute_default() == "fp8"
        assert fp8_paged_attn.compute_counts() == {"f32": 0, "fp8": 0}, (
            "resolving the default must not be mistaken for a decode "
            "call; only an actual dispatch may increment the tally")
    finally:
        del os.environ["GNF4_ATTN_COMPUTE"]
