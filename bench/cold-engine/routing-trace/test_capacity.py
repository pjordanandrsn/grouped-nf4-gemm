"""Tests for the shared capacity helper.

Two harnesses computed `frac of arena` differently -- `int(arena * f)` in one,
`int(round(arena * f))` in the other -- and published two receipts for the same
experiment with different capacities at frac 0.7 (Bugbot, gnf4#177). The rule
is a FLOOR; the bug was that binary floating point makes the floor wrong when
the product is exact in decimal but not in binary.
"""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "kernel"))
sys.path.insert(0, HERE)

from score_policies import capacity, steps_capacity    # noqa: E402

FRACS = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
ARENAS = (1024, 1280, 1440)                            # olmoe, granite, qwen


def test_the_cell_that_started_this():
    """0.7 of 1440 is exactly 1008; the float product is 1007.9999999999999."""
    assert 1440 * 0.7 != 1008.0                        # the trap is real
    assert capacity(1440, 0.7) == 1008


def test_floor_is_kept_where_the_product_is_genuinely_fractional():
    assert capacity(1024, 0.7) == 716                  # 716.8 floors to 716
    assert capacity(1024, 0.15) == 153                 # 153.6 floors to 153
    assert capacity(1440, 0.15) == 216                 # 216.0 exactly


def test_it_never_exceeds_the_budget():
    for a in ARENAS:
        for f in FRACS:
            assert capacity(a, f) <= a * f + 1e-9


def test_full_arena_is_the_whole_arena():
    for a in ARENAS:
        assert capacity(a, 1.0) == a


def test_floor_argument_is_a_minimum_not_an_override():
    assert capacity(96, 0.5, floor=2) == 48            # 48 > 2, unaffected
    assert capacity(4, 0.01, floor=2) == 2             # 0.04 would floor to 0


def test_default_floor_keeps_capacity_positive():
    assert capacity(10, 0.001) == 1


def test_it_reproduces_int_everywhere_the_float_was_not_lying():
    """Only genuinely-corrupted cells may move, or this silently rewrites
    the derivation receipts it is supposed to leave alone."""
    moved = [(a, f) for a in ARENAS for f in FRACS
             if capacity(a, f) != max(1, int(a * f))]
    assert moved == [(1440, 0.7)]


STEPS = (0.5, 0.75, 0.9, 1.0, 1.25, 1.5)

# layers x top-k for every geometry captured or registered. 64 is Mixtral,
# which is the whole reason these tests are not written against 96 alone.
PER_STEP = {"olmoe": 128, "granite": 256, "qwen": 96, "mixtral": 64}


@pytest.mark.parametrize("per", sorted(PER_STEP.values()))
def test_steps_sweep_matches_the_registered_int_round(per):
    """The grid the preregistrations were written against, on every geometry.

    Checking only per=96 hid a real regression: floor and round agree there
    (0.9 x 96 = 86.4) and part at per=64 (57.6), so routing P1 through the
    flooring helper silently moved Mixtral's 0.9 cell to 57 while
    PREREG-fourth-model.md registered 58 (Bugbot, gnf4#177).
    """
    assert [steps_capacity(per, s) for s in STEPS] == \
        [max(2, int(round(per * s))) for s in STEPS]


def test_steps_sweep_reproduces_the_registered_qwen_grid():
    assert [steps_capacity(96, s) for s in STEPS] == [48, 72, 86, 96, 120, 144]


def test_steps_sweep_reproduces_the_registered_mixtral_grid():
    """PREREG-fourth-model.md registered {32, 48, 58, 64, 80, 96}."""
    assert [steps_capacity(64, s) for s in STEPS] == [32, 48, 58, 64, 80, 96]


def test_steps_rounds_where_the_arena_helper_floors():
    """The two are different quantities and must not be collapsed again."""
    assert steps_capacity(64, 0.9) == 58
    assert capacity(64, 0.9, floor=2) == 57


def test_no_registered_sweep_lands_on_an_exact_half():
    """steps_capacity is half-up; that only differs from Python's banker's
    rounding on an exact .5, and nothing registered hits one."""
    from fractions import Fraction
    for per in PER_STEP.values():
        for s in STEPS:
            prod = Fraction(str(s)) * per
            assert prod.denominator != 2, (per, s, prod)


def test_integer_fraction_and_string_agree():
    assert capacity(1440, 0.7) == capacity(1440, "0.7")


@pytest.mark.parametrize("arena", ARENAS)
def test_monotone_in_the_fraction(arena):
    caps = [capacity(arena, f) for f in FRACS]
    assert caps == sorted(caps)
