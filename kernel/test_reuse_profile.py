# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Gate 3's instrument: does it distinguish the four behaviours, and does it
settle R4 rather than assume it?

The classifier decides which experts get argued into a higher tier, so its
failure mode is a feedback loop that promotes on thin evidence and thrashes.
These tests build traces with a KNOWN shape and check the label comes back.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from reuse_profile import (BURST, HOT, ONE_SHOT, WARM,  # noqa: E402
                           ReuseProfile, _spearman)


def _feed(p, key, ticks, resurrect_at=()):
    for t in ticks:
        p.observe(key, t, resurrected=t in resurrect_at)


# ------------------------------------------------------- the four classes --

def test_a_single_routing_is_one_shot():
    p = ReuseProfile(window=16)
    _feed(p, "a", [3])
    assert p.classify("a") == ONE_SHOT


def test_clustered_then_silent_is_a_burst():
    """Locally hot, globally cold — the case reclaimable DRAM exists for."""
    p = ReuseProfile(window=16)
    _feed(p, "a", [10, 11, 12, 13])          # one tight cluster
    for t in range(0, 400, 7):               # keep the clock running
        p.observe("filler", t)
    assert p.classify("a") == BURST


def test_present_in_most_windows_is_hot():
    p = ReuseProfile(window=16)
    _feed(p, "a", list(range(0, 320, 4)))    # every window, repeatedly
    assert p.classify("a") == HOT


def test_spread_but_thin_is_warm_not_hot():
    """Promoting this to VRAM on thin evidence is how a feedback loop starts
    thrashing, so the classifier must not call it hot."""
    p = ReuseProfile(window=16)
    _feed(p, "a", [5, 100, 250])             # 3 windows out of ~16
    for t in range(0, 260, 3):
        p.observe("filler", t)
    assert p.classify("a") == WARM


# ------------------------------------------------------------- R4 itself --

def test_recency_beats_frequency_when_reuse_is_bursty():
    """The INSTRUMENT responds to R4's effect. Not evidence for R4.

    This trace is CONSTRUCTED so recency wins: two populations with equal
    routing counts, one clustered and resurrected, one spread and never
    reused. Passing shows the comparison can detect the effect when it is
    present — nothing more. R4 is a claim about REAL routing, and settling
    it needs a captured trace with real resurrection events, which no test
    in this file provides.

    Written this way deliberately: a rigged trace that "confirms" a
    registered prediction is the easiest way to fake a result, and the
    docstring is where that distinction has to live."""
    p = ReuseProfile(window=32)
    for i in range(8):                       # bursty: clustered + resurrected
        base = i * 64
        ticks = [base, base + 1, base + 2, base + 3, base + 4]
        _feed(p, ("burst", i), ticks, resurrect_at=set(ticks[1:]))
    for i in range(8):                       # spread: same picks, no reuse
        _feed(p, ("spread", i), [j * 64 + i for j in range(5)])
    s = p.predictor_scores()
    assert s["recency"] is not None and s["frequency"] is not None
    assert s["recency"] > s["frequency"], s


def test_predictor_scores_refuse_to_invent_a_correlation():
    """No resurrections means nothing to rank against. Returning 0.0 would
    read as 'both predictors are useless' rather than 'not measured'."""
    p = ReuseProfile(window=16)
    _feed(p, "a", [1, 2])
    _feed(p, "b", [3, 4])
    s = p.predictor_scores()
    assert s["recency"] is None and s["frequency"] is None
    assert "no resurrections" in s.get("note", "")


def test_a_single_expert_cannot_be_ranked():
    p = ReuseProfile(window=16)
    _feed(p, "a", [1, 2], resurrect_at={2})
    assert p.predictor_scores()["recency"] is None


# ------------------------------------------------------------ promotion --

def test_dram_candidates_are_ranked_by_retained_reuse():
    """DRAM promotion is argued by resurrections — bytes that were kept and
    then actually used — not by raw routing frequency."""
    p = ReuseProfile(window=32)
    for i, res in ((0, 5), (1, 1)):
        base = i * 100
        ticks = [base + j for j in range(6)]
        _feed(p, ("e", i), ticks, resurrect_at=set(ticks[1:1 + res]))
    for t in range(0, 400, 5):
        p.observe("filler", t)
    got = p.candidates(tier="dram")
    assert got and got[0][0] == ("e", 0), got


def test_vram_candidates_must_be_hot_not_merely_frequent():
    p = ReuseProfile(window=16)
    _feed(p, "hot", list(range(0, 320, 4)))
    _feed(p, "burst", [10, 11, 12, 13])
    for t in range(0, 320, 9):
        p.observe("filler", t)
    keys = {k for k, _ in p.candidates(tier="vram")}
    assert "hot" in keys and "burst" not in keys


def test_unknown_tier_is_a_named_error():
    with pytest.raises(ValueError, match="dram.*vram|tier must"):
        ReuseProfile().candidates(tier="nvme")


def test_window_must_be_meaningful():
    with pytest.raises(ValueError, match="window"):
        ReuseProfile(window=1)


# --------------------------------------------------------------- ranking --

def test_spearman_handles_the_ties_routing_counts_produce():
    assert _spearman([1, 1, 1], [5, 5, 5]) == 0.0     # no variance: no signal
    assert _spearman([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert _spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)


def test_stats_report_what_the_classifier_used():
    p = ReuseProfile(window=16)
    _feed(p, "a", [1, 2, 3], resurrect_at={3})
    s = p.stats("a")
    assert s.picks == 3 and s.resurrections == 1
    assert s.first_tick == 1 and s.last_tick == 3 and s.max_run_picks == 3
