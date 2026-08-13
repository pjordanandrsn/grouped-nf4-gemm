#!/usr/bin/env python3
"""CPU wiring tests for the dequant-on-forward training arm.

These run without CUDA, bitsandbytes or triton, because the failures they catch
are shape/permutation/graph failures, and finding those on a rented card costs
money and a re-registration. The stack is stubbed: what is under test is the
arm's plumbing, not NF4 arithmetic (the property suite owns that).

The point of each test:
  * the routed probe and the sliced arm must agree EXACTLY — if they disagree,
    the probe's gather/scatter dropped or double-counted an expert, and a
    timing-only probe would have reported that as a speed difference;
  * gradients must reach the activations and both adapters in every arm — a
    dead arm is a fast arm;
  * the dequant counter must FAIL on an arm that hoists the dequant out of the
    forward. A gate that cannot fail is not a gate.
"""
from __future__ import annotations

import importlib.util as _iu
import os
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("DQF_REPO", str(_ROOT))
_spec = _iu.spec_from_file_location(
    "tdf", _ROOT / "bench" / "phase1" / "train_dequant_forward.py")
tdf = _iu.module_from_spec(_spec)
_spec.loader.exec_module(tdf)

E, K, N, R = 6, 16, 12, 4
SIZES = [3, 0, 5, 2, 4, 1]          # a zero-size group on purpose


class FakeStack:
    """Deterministic per-expert weights; counts nothing itself so the counter
    under test is the only thing counting."""

    class _S:
        N = N
        K = K
        E = E

    spec = _S()

    def __init__(self):
        g = torch.Generator().manual_seed(3)
        self.W = [(torch.randn(N, K, generator=g) * 0.05).to(torch.float32)
                  for _ in range(E)]

    def dequant_bf16(self, e):
        return self.W[e]

    def ref64(self, e):
        return self.W[e].to(torch.float64)


def _fixture():
    stack = FakeStack()
    groups = []
    g = torch.Generator().manual_seed(5)
    for e, n in enumerate(SIZES):
        groups.append((e, torch.randn(n, K, generator=g)))
    sizes = list(SIZES)
    eids = torch.tensor(list(range(E)), dtype=torch.int32)
    a_cat = torch.cat([a for _, a in groups]).detach().requires_grad_(True)
    T = a_cat.shape[0]

    lora_A = (torch.randn(E, R, K, generator=g) * 0.01).requires_grad_(True)
    lora_B = (torch.randn(E, N, R, generator=g) * 0.01).requires_grad_(True)

    grp_of_row = torch.repeat_interleave(torch.arange(len(groups)),
                                         torch.tensor(sizes))
    perm = torch.randperm(T, generator=torch.Generator().manual_seed(11))
    order = eids.to(torch.int64)[grp_of_row][perm]
    inv = torch.argsort(perm)
    a_tok = a_cat.detach()[perm].requires_grad_(True)
    return dict(stack=stack, groups=groups, sizes=sizes, eids=eids, a_cat=a_cat,
                lora_A=lora_A, lora_B=lora_B, a_tok=a_tok, order=order, inv=inv)


def _rows_differing(a, b):
    """The cell's gate, verbatim: per-row, scale-relative. Not bitwise — the
    gather hands the GEMM the same rows in a different order, so blocking
    differs and fp32 lands ~1e-8 apart."""
    return int(((a - b).abs().max(dim=1).values > 1e-2 * b.abs().max()).sum())


def test_routed_probe_matches_sliced_arm():
    f = _fixture()
    d = tdf.dequant_forward_arm(f["stack"], f["a_cat"], f["sizes"], f["eids"],
                                f["lora_A"], f["lora_B"])
    dr = tdf.d_routed_arm(f["stack"], f["a_tok"], f["order"], f["inv"],
                          f["sizes"], f["eids"], f["lora_A"], f["lora_B"],
                          f["a_cat"], E)
    od, orr = d(), dr()
    assert _rows_differing(orr, od) == 0
    assert (orr - od).abs().max() < 1e-5      # rounding only, nothing structural


def test_gate_REJECTS_a_probe_that_drops_an_expert():
    """Positive control on the equality gate. The plumbing failure that matters
    is a hit expert never entering the loop; the gate must catch it, and a
    bitwise-equality gate would have caught it only by accident."""
    f = _fixture()
    od = tdf.dequant_forward_arm(f["stack"], f["a_cat"], f["sizes"], f["eids"],
                                 f["lora_A"], f["lora_B"])()
    orig = tdf.d_routed_arm(f["stack"], f["a_tok"], f["order"], f["inv"],
                            f["sizes"], f["eids"], f["lora_A"], f["lora_B"],
                            f["a_cat"], E)()
    assert _rows_differing(orig, od) == 0

    dropped = f["order"].clone()
    dropped[dropped == 4] = 0          # expert 4's tokens re-routed to 0
    bad = tdf.d_routed_arm(f["stack"], f["a_tok"], dropped, f["inv"],
                           f["sizes"], f["eids"], f["lora_A"], f["lora_B"],
                           f["a_cat"], E)()
    assert _rows_differing(bad, od) > 0


def test_gradients_reach_activations_and_both_adapters():
    f = _fixture()
    for name, mk in (
        ("sliced", lambda: tdf.dequant_forward_arm(
            f["stack"], f["a_cat"], f["sizes"], f["eids"], f["lora_A"],
            f["lora_B"])),
        ("routed", lambda: tdf.d_routed_arm(
            f["stack"], f["a_tok"], f["order"], f["inv"], f["sizes"],
            f["eids"], f["lora_A"], f["lora_B"], f["a_cat"], E)),
    ):
        for t in (f["a_cat"], f["a_tok"], f["lora_A"], f["lora_B"]):
            t.grad = None
        mk()().float().pow(2).mean().backward()
        act = f["a_tok"] if name == "routed" else f["a_cat"]
        for label, t in (("act", act), ("lora_A", f["lora_A"]),
                         ("lora_B", f["lora_B"])):
            assert t.grad is not None, f"{name}/{label} got no gradient"
            assert torch.isfinite(t.grad).all(), f"{name}/{label} non-finite"
            assert t.grad.abs().sum() > 0, f"{name}/{label} all-zero gradient"


def test_dequant_counter_counts_every_hit_expert():
    f = _fixture()
    d = tdf.dequant_forward_arm(f["stack"], f["a_cat"], f["sizes"], f["eids"],
                                f["lora_A"], f["lora_B"])
    with tdf._DeqCount(f["stack"]) as c:
        d()
    assert c.n == sum(1 for s in SIZES if s > 0)


def test_dequant_counter_REJECTS_a_hoisted_dequant():
    """Positive control on the gate: an arm that materializes the weights once
    outside the forward is exactly the cheat the counter exists to catch, and
    it must fail the same assertion the real gate applies."""
    f = _fixture()
    stack = f["stack"]
    with tdf._DeqCount(stack) as c:
        cached = [stack.dequant_bf16(e) for e, _ in f["groups"]]  # hoisted
    hoisted_calls = c.n

    with tdf._DeqCount(stack) as c2:
        row = 0
        for g, n in enumerate(SIZES):
            if n:
                torch.nn.functional.linear(f["a_cat"][row:row + n], cached[g])
            row += n
    nonempty = sum(1 for s in SIZES if s > 0)
    assert c2.n == 0 and c2.n < nonempty          # the gate would reject this
    assert hoisted_calls == len(f["groups"])      # counter itself does fire


def test_counter_restores_the_real_method():
    f = _fixture()
    stack = f["stack"]
    before = stack.dequant_bf16(0)
    with tdf._DeqCount(stack):
        pass
    assert "dequant_bf16" not in stack.__dict__
    assert torch.equal(stack.dequant_bf16(0), before)


def test_fidelity_is_zero_for_an_exact_arm_and_positive_for_a_perturbed_one():
    f = _fixture()
    stack, groups, sizes = f["stack"], f["groups"], f["sizes"]
    exact = torch.cat([
        (a.to(torch.float64) @ stack.W[e].to(torch.float64).t()).to(torch.float32)
        for e, a in groups if a.shape[0]])
    b_exact = tdf._fidelity_sub(stack, groups, exact, sizes, 16, 32)
    assert b_exact < 1e-6
    b_off = tdf._fidelity_sub(stack, groups, exact * 1.01, sizes, 16, 32)
    assert b_off > 5e-3


def test_fidelity_caps_are_honoured():
    """A cap that silently does nothing would make the recorded caps a lie."""
    f = _fixture()
    stack, groups, sizes = f["stack"], f["groups"], f["sizes"]
    seen = []
    orig = stack.ref64
    stack.ref64 = lambda e: (seen.append(e), orig(e))[1]
    out = torch.zeros(sum(sizes), N)
    tdf._fidelity_sub(stack, groups, out, sizes, 1, 2)
    assert len(seen) == 2                      # 2 groups, not all 5 non-empty


if __name__ == "__main__":                     # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
