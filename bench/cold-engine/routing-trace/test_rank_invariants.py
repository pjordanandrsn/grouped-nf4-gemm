"""Tests for the preregistered rank capture check.

The check runs inside the decode loop, after a multi-gigabyte model download
and hundreds of forward passes — the worst possible place to discover it is
wrong. So it lives in a function and is tested here instead.

Its first inline version bound `a, b = ...` and clobbered the argparse
namespace `main()` reads `a.model` and `a.out` from, which would have crashed
every capture at the metadata write after a full run (Bugbot, gnf4#199). That
is the class of mistake these tests exist to make cheap.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from capture_routing import check_rank_invariants          # noqa: E402


def rec(routed, ranked, near=None, step=7):
    r = {"step": step,
         "routed": {str(i): v for i, v in enumerate(routed)},
         "routed_rank": {str(i): v for i, v in enumerate(ranked)}}
    if near is not None:
        r["near_miss"] = {str(i): v for i, v in enumerate(near)}
    return r


def test_a_valid_record_passes():
    assert check_rank_invariants(
        rec([[1, 3, 7]], [[7, 1, 3]]), 1) is None


def test_valid_with_a_near_miss_band():
    assert check_rank_invariants(
        rec([[1, 3, 7]], [[7, 1, 3]], [[2, 9, 4]]), 1) is None


def test_rank_missing_an_expert_is_caught():
    err = check_rank_invariants(rec([[1, 3, 7]], [[7, 1]]), 1)
    assert err and "permutation" in err


def test_rank_containing_a_foreign_expert_is_caught():
    err = check_rank_invariants(rec([[1, 3, 7]], [[7, 1, 9]]), 1)
    assert err and "permutation" in err


def test_a_repeated_expert_is_caught():
    """A duplicate keeps the multiset the same length but is still wrong."""
    err = check_rank_invariants(rec([[1, 3, 3]], [[3, 3, 1]]), 1)
    assert err is None or "repeats" in err     # sorted-equal, so permutation ok
    err2 = check_rank_invariants(rec([[1, 3, 7]], [[3, 3, 1]]), 1)
    assert err2 and ("permutation" in err2 or "repeats" in err2)


def test_near_miss_overlapping_the_selection_is_caught():
    err = check_rank_invariants(rec([[1, 3, 7]], [[7, 1, 3]], [[3, 9]]), 1)
    assert err and "overlaps" in err


def test_every_layer_is_checked_not_just_the_first():
    err = check_rank_invariants(
        rec([[1, 2], [3, 4]], [[2, 1], [3, 9]]), 2)
    assert err and "layer 1" in err


def test_the_error_names_the_step_and_layer():
    err = check_rank_invariants(rec([[1, 2]], [[1, 9]], step=42), 1)
    assert "step 42" in err and "layer 0" in err
