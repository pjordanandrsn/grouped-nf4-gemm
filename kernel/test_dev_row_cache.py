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
    a1, need1 = c.want(0, [3, 4, 5], t1)
    t1.record()
    assert sorted(need1) == [3, 4, 5]

    t2 = StepTag("cpu")
    a2, need2 = c.want(0, [5, 3, 4], t2)          # same set, all moved
    assert need2 == [], "a re-routed expert was re-fetched"
    assert a2 == a1, "its slot moved, so the engine will copy anyway"


def test_layers_do_not_alias_on_expert_id():
    """Slot addresses are reused across layers, so the key must carry the
    layer — otherwise layer 1's expert 3 reads layer 0's bytes and the error
    is silent."""
    c = _cache()
    t1 = StepTag("cpu")
    c.want(0, [3], t1)
    t1.record()
    t2 = StepTag("cpu")
    a2, need2 = c.want(1, [3], t2)
    assert need2 == [3], "layer 1 was served layer 0's row"


def test_a_logically_evicted_row_is_resurrected_not_refetched():
    c = _cache(rows=4, protected=1)
    t1 = StepTag("cpu")
    c.want(0, [0], t1)
    t1.record()
    t2 = StepTag("cpu")
    c.want(0, [1], t2)      # 0 is logically evicted here
    t2.record()
    t3 = StepTag("cpu")
    _, need = c.want(0, [0], t3)                            # back again
    assert need == [], "a row still on the device was read from disk again"
    assert c.stats()["resurrections"] >= 1


def test_a_stall_waits_for_the_previous_step_and_is_counted():
    """When the arena cannot serve a miss without touching a row whose reader
    may still be running, the honest move is to WAIT -- not to raise, and
    certainly not to overwrite it. The wait is counted so an arena that is
    too small shows up as a number instead of a mystery slowdown."""
    class _Stub:
        recorded = True                 # it DID record; it just has not landed

        def __init__(self):
            self.synced = False

        def done(self):
            return self.synced

        def sync(self):
            self.synced = True

    c = _cache(rows=2, protected=1)
    prev = _Stub()
    t1 = StepTag("cpu")
    c.want(0, [0], t1)
    c.slots.want([(0, 9)], event_tag=prev)      # fill the second slot, tagged
    c.slots._demote({(0, 9)}, prev)             # and retire the first
    c._last = prev                              # the cache's previous step

    t2 = StepTag("cpu")
    a, need = c.want(0, [7], t2)
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
    _, need = c.want(0, [1, 2], t1)
    t1.record()
    c.note_filled(len(need))
    t2 = StepTag("cpu")
    _, need2 = c.want(0, [1, 2], t2)
    c.note_filled(len(need2))
    assert c.stats()["host_to_cache_rows"] == 2, "a hit was billed as a fill"


def test_row_contents_survive_a_reroute():
    """Residency is only useful if the bytes are still the right bytes."""
    c = _cache(rows=4, stride=8)
    t1 = StepTag("cpu")
    a1, _ = c.want(0, [2], t1)
    t1.record()
    c.rowview()[a1[2]] = torch.arange(8, dtype=torch.uint8)
    t2 = StepTag("cpu")
    a2, need = c.want(0, [5, 2], t2)
    assert need == [5]
    assert torch.equal(c.rowview()[a2[2]], torch.arange(8, dtype=torch.uint8))


def test_a_second_layer_can_evict_the_first_from_a_shared_arena():
    """gnf4#131. The previous step belongs to the CACHE, not to an engine.
    Tracked per-engine, every engine after the first started with no previous
    step, so it neither settled nor stall-waited -- and a shared arena could
    not evict another layer's working set. The second layer to touch a full
    cache simply raised."""
    c = _cache(rows=4, protected=2)
    for step in range(3):                      # layer 0 fills the arena
        t = StepTag("cpu")
        c.want(0, [step * 2, step * 2 + 1], t)
        t.record()
    t = StepTag("cpu")
    a, need = c.want(1, [100, 101], t)         # a DIFFERENT layer arrives
    assert sorted(need) == [100, 101]
    assert len(set(a.values())) == 2


def test_an_abandoned_step_does_not_retire_rows_forever():
    """A step that raises never reaches record(). Its rows would stay RETIRING
    against an event nobody will ever fire, so the arena leaks capacity until
    it cannot allocate. The next want() closes the abandoned step out."""
    c = _cache(rows=4, protected=1)
    t1 = StepTag("cpu")
    c.want(0, [0], t1)                          # deliberately NOT recorded
    t2 = StepTag("cpu")
    c.want(0, [1], t2)
    assert c.stats()["abandoned_steps"] == 1
    t2.record()
    t3 = StepTag("cpu")
    _, need = c.want(0, [2], t3)                # would raise if 0 stayed stuck
    assert need == [2]


def test_a_discarded_row_is_a_miss_not_a_hit():
    """gnf4#131. A fill that raised leaves the row mapped but unwritten.
    Unless it is unpublished, the next request finds it and calls it a hit --
    serving garbage as an expert, with nothing downstream to trip on."""
    c = _cache(rows=4)
    t1 = StepTag("cpu")
    _, need = c.want(0, [3, 4], t1)
    t1.record()
    assert sorted(need) == [3, 4]
    assert c.discard(0, [3]) == 1
    t2 = StepTag("cpu")
    _, need2 = c.want(0, [3, 4], t2)
    assert need2 == [3], "the discarded row was served as a hit"


def test_a_protected_budget_that_starves_the_next_step_is_refused():
    """gnf4#131. `rows >= 2*k` assumed the previous step leaves exactly k rows
    ACTIVE, but _demote reduces the ACTIVE set to `protected`. A budget above
    rows-k leaves fewer than k rows demotable, so an all-miss step cannot be
    served -- and settle() cannot free them, because they are ACTIVE rather
    than RETIRING. The engine refuses the pairing instead of crashing mid
    forward."""
    c = _cache(rows=8, protected=6)              # legal for the cache alone
    keep = []
    for step in range(4):                        # drive it to a steady state
        t = StepTag("cpu")
        c.want(0, list(range(step * 2, step * 2 + 2)), t)
        t.record()
        keep.append(t)
    active = sum(1 for s in range(8) if c.slots.state(s) == "active")
    assert active == 6, ("_demote holds `protected` ACTIVE, not k -- this is "
                         f"the premise the engine guard has to encode: {active}")


class _FakeEngine:
    """Only the fields _init_dev_cache reads. Lets the sizing rules be tested
    without a GPU, which is where they are cheapest to get wrong."""
    k = 4
    row_stride = 256


def _guard(cache):
    from mxfp4_residency import Mxfp4NvmeResidency
    Mxfp4NvmeResidency._init_dev_cache(_FakeEngine(), cache)


def test_the_engine_refuses_an_arena_smaller_than_two_routed_sets():
    with pytest.raises(ValueError, match=r"at least 2\*k"):
        _guard(_cache(rows=6, protected=2))


def test_the_engine_refuses_a_budget_that_starves_the_next_step():
    """gnf4#131: rows >= 2*k does not imply it. protected=6 of 8 leaves 2
    demotable rows for a step that may miss on 4."""
    with pytest.raises(ValueError, match="rows-k"):
        _guard(_cache(rows=8, protected=6))


def test_a_correctly_sized_arena_is_accepted():
    _guard(_cache(rows=8, protected=4))
    _guard(_cache(rows=16, protected=12))


def test_the_engine_refuses_a_stride_that_is_not_the_tier_s():
    with pytest.raises(ValueError, match="row_stride"):
        _guard(_cache(rows=8, protected=4, stride=512))


def test_record_is_idempotent():
    """gnf4#131. A tag names ONE point in time. An all-hot step left the
    previous cold tag in place and _commit recorded it again, moving it
    forward on the stream; rows retiring under it then waited on a position
    that receded with every repeat."""
    class _Counting(StepTag):
        def __init__(self):
            super().__init__("cpu")
            self.records = 0

        def record(self):
            before = self.recorded
            super().record()
            if not before:
                self.records += 1

    t = _Counting()
    t.record()
    t.record()
    t.record()
    assert t.records == 1 and t.done()


def test_the_stall_path_waits_on_every_pending_tag():
    """Rows blocking a request may be retiring under an OLDER step than the
    most recent one, and syncing only the last tag leaves them stuck."""
    class _Stub:
        recorded = True

        def __init__(self):
            self.synced = False

        def done(self):
            return self.synced

        def sync(self):
            self.synced = True

    c = _cache(rows=2, protected=1)
    old, last = _Stub(), _Stub()
    t1 = StepTag("cpu")
    c.want(0, [0], t1)
    c.slots.want([(0, 9)], event_tag=old)      # retired under the OLDER tag
    c.slots._demote({(0, 9)}, old)
    c._last = last                              # a different, newer tag

    t2 = StepTag("cpu")
    c.want(0, [7], t2)
    assert old.synced, "the older tag's rows would have stayed stuck"


def test_the_cache_reports_whether_it_can_retain_at_all():
    """A cache smaller than ONE decode step's routed set is evicted before its
    own next request, and then its extra host->cache write per miss makes it
    worse than the positional cache already in the engine. Measured across two
    models and four prompts, rows >= per_step separated every configuration
    where it helped from every one where it lost, 24 of 24. The cache learns
    per_step from what arrives rather than being told, because the engine that
    drives it does not know how many layers share it."""
    c = _cache(rows=16, protected=8)
    assert c.stats()["steps_held"] is None, "nothing seen yet; no ratio"
    assert c.stats()["too_small_to_retain"] is None

    for _step in range(2):                       # 4 layers x 8 = 32 > 16 rows
        for layer in range(4):
            t = StepTag("cpu")
            c.want(layer, list(range(layer * 8, layer * 8 + 8)), t)
            t.record()
    st = c.stats()
    assert st["per_step_rows"] == 32
    assert st["steps_held"] == 0.5
    assert st["too_small_to_retain"] is True

    big = _cache(rows=64, protected=56)
    for _step in range(2):
        for layer in range(4):
            t = StepTag("cpu")
            big.want(layer, list(range(layer * 8, layer * 8 + 8)), t)
            t.record()
    st = big.stats()
    assert st["steps_held"] == 2.0 and st["too_small_to_retain"] is False


def test_a_layer_that_stops_asking_is_not_counted_forever():
    """gnf4#165. The engine skips want() for a layer with no cold experts.
    Summing every layer's last-seen count would keep that layer in the total
    forever, reporting historical demand as though it were one step."""
    c = _cache(rows=64, protected=56)
    for _step in range(2):                       # 4 layers asking
        for layer in range(4):
            t = StepTag("cpu")
            c.want(layer, list(range(layer * 8, layer * 8 + 8)), t)
            t.record()
    assert c.stats()["per_step_rows"] == 32

    for _step in range(3):                       # layers 2 and 3 go all-hot
        for layer in range(2):
            t = StepTag("cpu")
            c.want(layer, list(range(layer * 8, layer * 8 + 8)), t)
            t.record()
    st = c.stats()
    assert st["per_step_rows"] == 16, "a silent layer stayed in the total"
    assert st["steps_held"] == 4.0
    # the worst step is still the heavy one, and capacity is judged on it
    assert st["per_step_rows_max"] == 32
    assert st["too_small_to_retain"] is False


def test_a_layer_becoming_active_does_not_fold_two_steps_together():
    """gnf4#165, the mirror of the silent-layer case. If a layer that was
    quiet becomes active with a LOWER index than the last one seen, a
    boundary rule watching only for repeats folds its demand into the
    previous step and finalises a count spanning two."""
    c = _cache(rows=64, protected=56)
    for _step in range(2):                       # only layers 1 and 2 ask
        for layer in (1, 2):
            t = StepTag("cpu")
            c.want(layer, list(range(layer * 8, layer * 8 + 8)), t)
            t.record()
    assert c.stats()["per_step_rows"] == 16

    for _step in range(2):                       # layer 0 wakes up, index 0
        for layer in (0, 1, 2):
            t = StepTag("cpu")
            c.want(layer, list(range(layer * 8, layer * 8 + 8)), t)
            t.record()
    st = c.stats()
    assert st["per_step_rows"] == 24, "layer 0's demand landed in the old step"
    assert st["per_step_rows_max"] == 24


def test_a_single_layer_cache_closes_a_step_every_request():
    c = _cache(rows=32, protected=24)
    for _ in range(3):
        t = StepTag("cpu")
        c.want(0, [1, 2, 3, 4], t)
        t.record()
    assert c.stats()["per_step_rows"] == 4
