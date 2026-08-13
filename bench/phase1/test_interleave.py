#!/usr/bin/env python3
"""Tests for iteration-level interleaved pairing.

The point of the redesign is a claim about drift, so the tests inject drift and
check that the interleaved statistic survives it while the block statistic does
not. If those two came out the same, the redesign would be pointless and these
tests would say so.

No GPU: `interleaved_pairs` takes an injectable torch, and the reduction under
test is pure arithmetic on per-call times.
"""
from __future__ import annotations

import importlib.util as _iu
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_spec = _iu.spec_from_file_location("il", _ROOT / "bench" / "phase1" / "interleave.py")
il = _iu.module_from_spec(_spec)
_spec.loader.exec_module(il)

TRUE_RATIO = 1.60


def synth(n, drift_per_pair=0.0, noise=0.0, seed=0):
    """A and B timings under a common multiplicative drift.

    The drift is COMMON to both arms -- it is the box getting slower or faster,
    not one arm changing -- so the true B/A ratio is constant at TRUE_RATIO
    throughout. Any statistic that reports something else is reporting drift."""
    import random
    rng = random.Random(seed)
    ta, tb = [], []
    def jit():
        return 1.0 + (rng.random() - 0.5) * 2 * noise
    for i in range(n):
        scale = (1.0 + drift_per_pair) ** i
        ta.append(1.00 * scale * jit())
        tb.append(TRUE_RATIO * scale * jit())
    return ta, tb


def test_no_drift_both_statistics_are_right():
    ta, tb = synth(200)
    assert abs(il.pair_stats(ta, tb)["ratio_median"] - TRUE_RATIO) < 1e-9
    assert abs(il.block_ratio(ta, tb) - TRUE_RATIO) < 1e-9


def test_under_drift_the_INTERLEAVED_statistic_survives_and_the_BLOCK_one_does_not():
    """The whole premise. A box drifting 0.3%/pair over 200 pairs ends ~1.8x
    slower than it started -- the scale of drift leg 2 actually saw."""
    ta, tb = synth(200, drift_per_pair=0.003)
    inter = il.pair_stats(ta, tb)["ratio_median"]
    assert abs(inter - TRUE_RATIO) < 1e-9, f"interleaved must be exact: {inter}"

    # Block pairing on the SAME data: all of A first, then all of B. B is timed
    # later, so it is measured on a slower box and looks worse than it is.
    n = len(ta)
    drift = (1.003) ** n
    block = il.block_ratio(ta, [t * drift for t in tb])
    assert block > TRUE_RATIO * 1.5, f"block pairing should be badly biased: {block}"


def test_the_halves_check_is_blind_to_box_drift_and_catches_RATIO_drift():
    """After interleaving, a drifting box no longer threatens the ratio, so the
    stability gate must not fire on it -- otherwise it voids cells that are
    fine. It must still fire when the RATIO itself moves."""
    ta, tb = synth(200, drift_per_pair=0.003)          # box drifts, ratio does not
    assert abs(il.pair_stats(ta, tb)["halves_ratio"] - 1.0) < 1e-9

    ta, tb = synth(200)
    n = len(tb)
    tb = [t * (1.0 + 0.30 * (i >= n // 2)) for i, t in enumerate(tb)]  # ratio jumps 30%
    assert il.pair_stats(ta, tb)["halves_ratio"] > 1.25


def test_order_bias_control_detects_a_first_call_penalty():
    """If the first call of every pair paid a fixed penalty, alternation would
    hide it in the median but the order control would show it."""
    n = 200
    ta, tb, orders = [], [], []
    for i in range(n):
        o = "ba" if i % 2 else "ab"
        a, b = 1.0, TRUE_RATIO
        if o == "ab":
            a *= 1.20          # A ran first and paid the penalty
        else:
            b *= 1.20          # B ran first
        ta.append(a)
        tb.append(b)
        orders.append(o)
    s = il.pair_stats(ta, tb, orders)
    assert s["order_bias"] < 0.8, s.get("order_bias")
    # and the alternated median still lands between the two biased halves
    assert TRUE_RATIO / 1.2 < s["ratio_median"] < TRUE_RATIO * 1.2


def test_interleaved_pairs_alternates_and_calls_each_arm_once_per_pair():
    """Drives the real collector against a stub torch, so the ordering and call
    accounting are checked rather than assumed."""
    calls = []

    class _Ev:
        def __init__(self, **kw): self.t = None
        def record(self): self.t = len(calls)
        def elapsed_time(self, other): return float(other.t - self.t)

    class _Cuda:
        Event = _Ev
        @staticmethod
        def synchronize(): pass

    class _T:
        cuda = _Cuda

    ta, tb, orders = il.interleaved_pairs(
        lambda: calls.append("a"), lambda: calls.append("b"),
        pairs=6, warm=2, torch_mod=_T)
    assert orders == ["ab", "ba", "ab", "ba", "ab", "ba"]
    assert len(ta) == len(tb) == 6
    body = calls[2 * 2:]                       # drop the warm-up calls
    assert body.count("a") == 6 and body.count("b") == 6
    assert body[:4] == ["a", "b", "b", "a"]    # pair 0 is a,b; pair 1 is b,a


if __name__ == "__main__":  # pragma: no cover
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
