"""Tests for the degenerate-generation detector.

The detector exists because Qwen's math trace was a period-2 repetition loop
that shipped into a results document before anyone noticed
(RESULTS-third-model.md). A detector that is only ever eyeballed against the
one trace it was written for is not much better, so both directions are
pinned: it fires on a loop, and it stays quiet on healthy generation.
"""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from reuse_overlap import n_experts, overlap, token_period  # noqa: E402


def recs(tokens):
    return [{"token": t, "routed": {"0": [0]}} for t in tokens]


def test_period_two_loop_is_found():
    assert token_period(recs([7, 9] * 64)) == 2


def test_period_one_loop_is_found():
    assert token_period(recs([5] * 64)) == 1


def test_period_three_loop_is_found():
    assert token_period(recs([1, 2, 3] * 40)) == 3


def test_healthy_generation_reports_no_cycle():
    # Distinct tokens throughout: a period would have to be a coincidence.
    assert token_period(recs(list(range(200)))) == 0


def test_mostly_healthy_with_a_few_repeats_is_not_a_cycle():
    t = list(range(200))
    for i in range(0, 200, 10):        # 10% incidental repeats
        t[i] = t[i - 2] if i >= 2 else t[i]
    assert token_period(recs(t)) == 0


def test_trace_without_tokens_is_undetectable_not_clean():
    """None, not 0 -- an older trace is unknown, and must not read as clean."""
    assert token_period([{"routed": {"0": [0]}} for _ in range(64)]) is None


def test_threshold_is_a_fraction_not_all_or_nothing():
    # A loop with a few tokens perturbed is still a loop.
    t = [7, 9] * 64
    t[10] = 11
    t[40] = 13
    assert token_period(recs(t)) == 2


def test_period_longer_than_max_is_not_reported():
    assert token_period(recs([i % 12 for i in range(240)]), max_p=8) == 0


def test_n_experts_prefers_metadata():
    assert n_experts({"n_experts": 64}, recs([1])) == (64, True)


def test_n_experts_falls_back_to_max_id_as_a_lower_bound():
    r = [{"token": 1, "routed": {"0": [0, 39], "1": [3, 7]}}]
    assert n_experts({"n_experts": None}, r) == (40, False)


def test_n_experts_treats_zero_as_missing():
    """0 is not a valid expert count and must not pass the truthiness gate."""
    r = [{"token": 1, "routed": {"0": [0, 5]}}]
    assert n_experts({"n_experts": 0}, r) == (6, False)


def test_overlap_is_a_fraction_of_the_current_step():
    S = [{1, 2, 3, 4}, {3, 4, 5, 6}]
    assert overlap(S, 1) == pytest.approx(0.5)


def test_overlap_of_identical_steps_is_one():
    S = [{1, 2}, {1, 2}, {1, 2}]
    assert overlap(S, 1) == pytest.approx(1.0)
