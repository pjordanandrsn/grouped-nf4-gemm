# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The device row cache — expert-keyed residency in front of the tier.

The property under test is the one the positional cache does not have: an
expert that is routed again is a hit no matter WHERE the router put it. The
rest of the file is about not lying — a stall counted rather than hidden, a
never-recorded event that must not read as complete, and cross-layer keys
that must not alias.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dev_row_cache import DevRowCache, StepTag       # noqa: E402


def _cache(rows=8, stride=256, protected=None):
    return DevRowCache(rows, stride, device="cpu", protected=protected)


def test_a_reroute_to_a_new_position_is_a_hit():
    """The whole point. The positional cache re-fetches an expert that moved;
    this one does not, and the slot it reports is stable so the engine's
    address-equality skip keeps working for the ones that did not move."""
    c = _cache()
    t1 = StepTag("cpu")
    a1, need1 = c.want(0, [3, 4, 5], t1, None)
    t1.record()
    assert sorted(need1) == [3, 4, 5]

    t2 = StepTag("cpu")
    a2, need2 = c.want(0, [5, 3, 4], t2, t1)          # same set, all moved
    assert need2 == [], "a re-routed expert was re-fetched"
    assert a2 == a1, "its slot moved, so the engine will copy anyway"


def test_layers_do_not_alias_on_expert_id():
    """Slot addresses are reused across layers, so the key must carry the
    layer — otherwise layer 1's expert 3 reads layer 0's bytes and the error
    is silent."""
    c = _cache()
    t1 = StepTag("cpu")
    c.want(0, [3], t1, None)
    t1.record()
    t2 = StepTag("cpu")
    a2, need2 = c.want(1, [3], t2, t1)
    assert need2 == [3], "layer 1 was served layer 0's row"


def test_a_logically_evicted_row_is_resurrected_not_refetched():
    c = _cache(rows=4, protected=1)
    t1 = StepTag("cpu")
    c.want(0, [0], t1, None)
    t1.record()
    t2 = StepTag("cpu")
    c.want(0, [1], t2, t1)      # 0 is logically evicted here
    t2.record()
    t3 = StepTag("cpu")
    _, need = c.want(0, [0], t3, t2)                            # back again
    assert need == [], "a row still on the device was read from disk again"
    assert c.stats()["resurrections"] >= 1


def test_a_stall_waits_for_the_previous_step_and_is_counted():
    """When the arena cannot serve a miss without touching a row whose reader
    may still be running, the honest move is to WAIT — not to raise, and
    certainly not to overwrite it. The wait is counted so an arena that is
    too small shows up as a number instead of a mystery slowdown."""
    class _Stub:
        def __init__(self):
            self.synced = False

        def done(self):
            return self.synced

        def sync(self):
            self.synced = True

    c = _cache(rows=2, protected=1)
    prev = _Stub()
    t1 = StepTag("cpu")
    c.want(0, [0], t1, None)
    c.slots.want([(0, 9)], event_tag=prev)      # fill the second slot, tagged
    c.slots._demote({(0, 9)}, prev)             # and retire the first

    t2 = StepTag("cpu")
    a, need = c.want(0, [7], t2, prev)
    assert prev.synced, "the allocator gave up instead of waiting"
    assert c.stats()["stalls"] == 1
    assert 7 in a


def test_an_unrecorded_step_never_reads_as_complete():
    """torch returns True from query() on an event that was never recorded.
    Taking that at face value releases rows whose gather never ran."""
    t = StepTag("cpu")
    assert not t.done(), "a step that never started looked finished"
    t.record()
    assert t.done()


def test_protected_at_rows_is_refused_with_the_reason():
    with pytest.raises(ValueError, match="nowhere to land"):
        _cache(rows=4, protected=4)


def test_the_arena_strides_the_padded_row():
    """row_stride, never row_bytes. Slot 1 onward is where a wrong stride
    starts reading mid-row, and nothing downstream would raise."""
    c = _cache(rows=4, stride=320)
    assert c.addr(1) - c.addr(0) == 320
    assert c.rowview().shape == (4, 320)
    assert c.buf.numel() == 4 * 320


def test_bytes_written_host_to_cache_are_reported():
    c = _cache()
    t1 = StepTag("cpu")
    _, need = c.want(0, [1, 2], t1, None)
    t1.record()
    c.note_filled(len(need))
    t2 = StepTag("cpu")
    _, need2 = c.want(0, [1, 2], t2, t1)
    c.note_filled(len(need2))
    assert c.stats()["host_to_cache_rows"] == 2, "a hit was billed as a fill"


def test_row_contents_survive_a_reroute():
    """Residency is only useful if the bytes are still the right bytes."""
    c = _cache(rows=4, stride=8)
    t1 = StepTag("cpu")
    a1, _ = c.want(0, [2], t1, None)
    t1.record()
    c.rowview()[a1[2]] = torch.arange(8, dtype=torch.uint8)
    t2 = StepTag("cpu")
    a2, need = c.want(0, [5, 2], t2, t1)
    assert need == [5]
    assert torch.equal(c.rowview()[a2[2]], torch.arange(8, dtype=torch.uint8))
