# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""VRAM reclaimable residency (R2, R3).

The failure mode this class exists to prevent is serving one expert's bytes
as another's, so the tests are mostly about WHEN a slot may be handed over.
The RETIRING gate is the sharp edge: a slot released while a kernel is still
reading it produces wrong numbers that look like a numerics bug, and no
amount of downstream tolerance will find it.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from vram_slots import (ABSENT, ACTIVE, RECLAIMABLE,  # noqa: E402
                        RETIRING, VramSlots)


# ------------------------------------------------------------- the basics --

def test_a_fresh_arena_gathers_everything_once():
    v = VramSlots(4)
    a, need = v.want([1, 2, 3])
    assert sorted(need) == [1, 2, 3] and len(set(a.values())) == 3
    assert v.stats()["gathers"] == 3


def test_an_active_expert_is_a_hit_with_no_gather():
    v = VramSlots(4)
    v.want([1, 2])                               # gathers, not hits
    assert v.stats()["active_hits"] == 0
    _, need = v.want([1, 2])
    assert need == [] and v.stats()["active_hits"] == 2


def test_default_budget_never_makes_a_slot_reclaimable():
    """protected=None reproduces a plain slot map exactly, so this cannot
    change behaviour unless it is asked for."""
    v = VramSlots(4)
    for e in (1, 2, 3, 4, 1, 2):
        v.want([e])
    s = v.stats()
    assert s["logical_evictions"] == 0 and s["resurrections"] == 0
    assert s["reuse_before_overwrite"] is None


# ------------------------------------------------------ the resurrection --

def test_a_demoted_expert_is_resurrected_without_a_gather():
    """The whole point: no NVMe read, no staging, no H2D, no gather — the
    bytes never left."""
    v = VramSlots(4, protected=1)
    v.want([1])
    v.want([2])                                  # 1 loses ownership
    assert v.state(v.slot_of(1)) == RECLAIMABLE
    _, need = v.want([1])
    assert need == [], "a surviving slot must not be re-gathered"
    assert v.stats()["resurrections"] == 1


def test_reclaimable_loses_every_allocation_contest():
    v = VramSlots(2, protected=1)
    v.want([1])
    v.want([2])
    assert v.state(v.slot_of(1)) == RECLAIMABLE
    v.want([3])                                  # must take 1's slot, not 2's
    assert v.slot_of(1) is None and v.slot_of(2) is not None
    assert v.stats()["overwritten"] == 1


def test_overwrite_resolves_the_eviction_the_losing_way():
    v = VramSlots(2, protected=1)
    v.want([1])
    v.want([2])
    v.want([3])
    assert v.stats()["reuse_before_overwrite"] == 0.0
    _, need = v.want([1])
    assert need == [1], "an overwritten expert must be gathered again"


# --------------------------------------------------------- the RETIRING gate --

def test_a_slot_with_readers_in_flight_is_never_handed_over():
    """A slot released while a kernel still reads it is a correctness bug
    that presents as a numerics bug. RETIRING must be untouchable."""
    v = VramSlots(2, protected=1)
    v.want([1], event_tag="ev1")
    v.want([2], event_tag="ev2")                 # 1 -> RETIRING, not free
    assert v.state(v.slot_of(1)) == RETIRING
    with pytest.raises(RuntimeError, match="RETIRING"):
        v.want([3, 4], event_tag="ev3")          # needs both slots


def test_settle_flips_retiring_only_when_the_event_says_so():
    v = VramSlots(2, protected=1)
    v.want([1], event_tag="ev1")
    v.want([2], event_tag="ev2")
    s = v.slot_of(1)
    assert v.settle(lambda tag: False) == 0      # nothing landed yet
    assert v.state(s) == RETIRING
    assert v.settle(lambda tag: True) == 1
    assert v.state(s) == RECLAIMABLE


def test_a_retiring_slot_is_still_a_hit_for_its_own_expert():
    """Its bytes are valid — only handing them to ANOTHER expert is unsafe."""
    v = VramSlots(2, protected=1)
    v.want([1], event_tag="ev1")
    v.want([2], event_tag="ev2")
    _, need = v.want([1], event_tag="ev3")
    assert need == [] and v.stats()["blocked_by_retiring"] == 1


def test_without_an_event_tag_demotion_goes_straight_to_reclaimable():
    """The caller asserting there is nothing in flight to wait for."""
    v = VramSlots(2, protected=1)
    v.want([1])
    v.want([2])
    assert v.state(v.slot_of(1)) == RECLAIMABLE


# ------------------------------------------------------------- contract --

def test_a_request_larger_than_the_arena_is_a_named_error():
    with pytest.raises(ValueError, match="into 2 slots"):
        VramSlots(2).want([1, 2, 3])


def test_protected_outside_the_arena_is_a_named_error():
    for bad in (0, 5):
        with pytest.raises(ValueError, match="protected"):
            VramSlots(4, protected=bad)


def test_generations_move_only_when_contents_do():
    v = VramSlots(2, protected=1)
    v.want([1])
    s = v.slot_of(1)
    g = v.generation(s)
    v.want([2])                                  # 1 demoted, not overwritten
    assert v.generation(s) == g, "demotion must not bump a generation"
    v.want([3])                                  # now 1's slot is refilled
    assert v.generation(s) != g


def test_a_requests_own_rows_are_never_demoted():
    v = VramSlots(4, protected=1)
    a, _ = v.want([1, 2, 3])
    for e in (1, 2, 3):
        assert v.state(a[e]) == ACTIVE
    assert v.stats()["logical_evictions"] == 0


def test_absent_slots_are_used_before_active_ones_are_disturbed():
    v = VramSlots(4)
    v.want([1])
    v.want([2])
    assert v.slot_of(1) is not None and v.slot_of(2) is not None
    assert sum(1 for s in range(4) if v.state(s) == ABSENT) == 2


def test_settle_never_releases_a_slot_resurrected_since_its_tag():
    """gnf4#128. A RETIRING self-hit makes the slot ACTIVE again, but the
    pending entry recorded before it could outlive the resurrection. settle()
    then wrote RECLAIMABLE over a LIVE assignment -- making it _claim's first
    pick, so another expert would be handed bytes the holder still owns."""
    v = VramSlots(4, protected=2)
    v.want([0, 1])
    v.want([2, 3], event_tag="ev1")          # 0,1 -> RETIRING under ev1
    assert v.state(v.slot_of(0)) == "retiring"

    a, _ = v.want([0, 2], event_tag="ev2")   # self-hit resurrects 0
    s0 = a[0]
    assert v.state(s0) == "active"

    v.settle(lambda t: t == "ev1")           # ev1's readers finish
    assert v.state(s0) == "active", "a live assignment was released"

    a2, _ = v.want([0, 2, 7], event_tag="ev3")
    assert a2[0] == s0 and a2[7] != s0, "expert 7 was handed live bytes"


def test_claim_refuses_a_live_active_slot_while_pipelining():
    """gnf4#128. With event tags in play the natural victims are RETIRING and
    get skipped, so the ACTIVE fallback took the recent working set -- whose
    readers may still be running and which carries no tag to prove otherwise.
    Refusing the allocation is correct; corrupting the reader is not."""
    v = VramSlots(2, protected=2)
    v.want([0, 1])                           # both ACTIVE, neither tagged
    with pytest.raises(RuntimeError, match="cannot prove are done"):
        v.want([5], event_tag="ev1")


def test_claim_still_takes_active_when_the_caller_asserts_quiescence():
    """The untagged path is the caller stating there are no readers in
    flight -- the same assertion that sends _demote's victims straight to
    RECLAIMABLE. It keeps the fallback."""
    v = VramSlots(2, protected=2)
    v.want([0, 1])
    a, need = v.want([5])                    # no tag -> ACTIVE is fair game
    assert need == [5] and a[5] in (0, 1)
