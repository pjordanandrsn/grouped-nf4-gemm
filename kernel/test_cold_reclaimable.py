# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Stage-3 contract tests: reclaimable residency over the cold tier.

The hypothesis under test is that the interval between *logical eviction*
(ownership revoked) and *physical overwrite* (slot refilled) holds reusable
information. These tests pin the mechanism that makes the interval exist and
measurable; the PREREG's R1-R10 measure whether it pays.

Correctness first — a resurrected row must be byte-identical to the arena,
because the entire claim is that no bytes moved. Then the state machine, then
the accounting discipline (an unresolved eviction counts on neither side).
Pure stdlib + the toy arena, same as test_nvme_residency.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from nvme_arena import bake, load_index  # noqa: E402
from nvme_reader import alloc_landing  # noqa: E402
from nvme_residency import ColdTier  # noqa: E402
from test_nvme_arena import E, L, make_snapshot  # noqa: E402


@pytest.fixture()
def arena(tmp_path):
    snap = tmp_path / "snap"
    make_snapshot(str(snap))
    path = str(tmp_path / "toy.arena")
    bake(str(snap), path, align=4096, log=lambda *a: None)
    return path, load_index(path)


def _tier(path, index, hot_rows, protected=None):
    return ColdTier(path, hot_rows=hot_rows, pinned=False, index=index,
                    protected_rows=protected)


def _reads(t):
    return t.reader.traffic()["reads"]


# ------------------------------------------------------------ the default --

def test_no_budget_means_no_row_is_ever_reclaimable(arena):
    """protected_rows=None must leave the pre-Stage-3 tier untouched: the
    reclaimable set stays empty, so the victim ranking's leading term is
    constant and every path below it is the code that shipped."""
    path, index = arena
    with _tier(path, index, 2) as t:
        for lay in range(L):
            for e in range(E):
                t.ensure(lay, [e])
        s = t.stats()
        assert s["protected_rows"] == t.hot_rows
        assert s["reclaimable_rows"] == 0
        assert s["logical_evictions"] == 0
        assert s["resurrections"] == 0
        # no eviction has RESOLVED, so the probability is undefined, not zero
        assert s["reuse_before_overwrite"] is None
        assert s["evictions"] > 0          # physical overwrites still happen


def test_protected_rows_out_of_range_is_a_clean_error(arena):
    path, index = arena
    for bad in (0, -1, 5):
        with pytest.raises(ValueError, match="protected_rows"):
            _tier(path, index, 4, protected=bad).close()


# ------------------------------------------------------- the state machine --

def test_logical_eviction_revokes_ownership_without_touching_bytes(arena):
    """The load-bearing claim: a demoted row is still readable and still
    correct. If this fails, resurrection is reading garbage."""
    path, index = arena
    with _tier(path, index, 4, protected=1) as t:
        t.ensure(0, [0])
        t.ensure(0, [1])                   # rotates the window off expert 0
        assert t.resident(0, 0) and t.reclaimable(0, 0)
        ref, keep = alloc_landing(t.row_stride, pinned=False)
        t.reader.read_row_sync(0, 0, ref)
        assert bytes(t.row(0, 0)) == bytes(ref[:t.row_bytes])
        del keep


def test_resurrection_costs_no_disk_read(arena):
    path, index = arena
    with _tier(path, index, 4, protected=1) as t:
        t.ensure(0, [0])
        t.ensure(0, [1])
        assert t.reclaimable(0, 0)
        before = _reads(t)
        t.ensure(0, [0])                   # the resurrection
        assert _reads(t) == before, "a resurrection must not touch the disk"
        s = t.stats()
        assert s["resurrections"] == 1
        assert s["resurrection_bytes_saved"] == t.row_bytes
        assert not t.reclaimable(0, 0)     # back to ACTIVE
        assert s["reuse_before_overwrite"] == 1.0


def test_resurrected_row_is_byte_identical_to_the_arena(arena):
    """Resurrection is a metadata move. Bytes must be untouched, and the
    equivalence 'cold-cache reuse vs fresh read' is exactly this assert."""
    path, index = arena
    with _tier(path, index, 4, protected=1) as t:
        t.ensure(0, [2])
        t.ensure(0, [3])
        assert t.reclaimable(0, 2)
        t.ensure(0, [2])
        ref, keep = alloc_landing(t.row_stride, pinned=False)
        t.reader.read_row_sync(0, 2, ref)
        assert bytes(t.row(0, 2)) == bytes(ref[:t.row_bytes])
        del keep


def test_reclaimable_loses_every_allocation_contest(arena):
    """A reclaimable row is handed over before ANY active row, regardless of
    frequency — that is what makes it free capacity rather than a second
    pool. Expert 0 is routed hardest and would win on LFU alone."""
    path, index = arena
    with _tier(path, index, 2, protected=1) as t:
        for _ in range(5):
            t.ensure(0, [0])               # make expert 0 the LFU favourite
        t.ensure(0, [1])                   # 0 demotes; 1 is the active row
        assert t.reclaimable(0, 0) and not t.reclaimable(0, 1)
        t.ensure(0, [2])                   # needs a slot: must take 0's
        assert not t.resident(0, 0)
        assert t.resident(0, 1), "an ACTIVE row was taken before a reclaimable one"


def test_overwrite_resolves_the_eviction_the_losing_way(arena):
    path, index = arena
    with _tier(path, index, 2, protected=1) as t:
        t.ensure(0, [0])
        t.ensure(0, [1])
        t.ensure(0, [2])                   # overwrites reclaimable expert 0
        s = t.stats()
        assert s["reclaimable_overwritten"] == 1
        assert s["reuse_before_overwrite"] == 0.0
        assert s["mean_ticks_to_overwrite"] is not None
        before = _reads(t)
        t.ensure(0, [0])                   # now a genuine miss
        assert _reads(t) == before + 1


def test_logical_and_physical_eviction_counts_diverge(arena):
    """R8's mechanism: nominal placement misses stop predicting I/O. The two
    counters answer different questions and must not be conflated."""
    path, index = arena
    with _tier(path, index, 4, protected=1) as t:
        for e in (0, 1, 0, 1, 0, 1):
            t.ensure(0, [e])
        s = t.stats()
        assert s["logical_evictions"] > s["evictions"]
        assert s["resurrections"] > 0
        assert _reads(t) == 2, "two experts, two physical reads, the rest resurrected"


def test_demand_window_is_never_logically_evicted(arena):
    """A caller is between its ensure and its reads. Revoking there would
    count a resurrection for a row that was never at risk."""
    path, index = arena
    with _tier(path, index, 4, protected=1) as t:
        t.ensure(0, list(range(E)))        # window of E > protected_rows
        for e in range(E):
            assert not t.reclaimable(0, e)
        assert t.stats()["logical_evictions"] == 0


def test_ghost_working_set_exceeds_the_protected_budget(arena):
    """The systems claim: the pool serves more distinct rows without disk
    than it protects. Capacity ownership and information retention are not
    the same thing."""
    path, index = arena
    with _tier(path, index, L * E, protected=1) as t:
        for lay in range(L):
            for e in range(E):
                t.ensure(lay, [e])
        cold = _reads(t)
        for lay in range(L):               # every row again, none protected
            for e in range(E):
                t.ensure(lay, [e])
        assert _reads(t) == cold, "surviving reclaimable rows should serve these"
        assert t.stats()["resurrections"] == L * E
        assert t.protected_rows == 1


# ------------------------------------------------------------ generations --

def test_generation_bump_invalidates_a_stale_slot_reference(arena):
    path, index = arena
    with _tier(path, index, 2, protected=1) as t:
        slot = t.ensure(0, [0])[0]
        gen = t.generations([slot])[0]
        assert t.validate(0, 0, slot, gen)
        t.ensure(0, [1])
        t.ensure(0, [2])                   # claims expert 0's slot
        assert not t.validate(0, 0, slot, gen), \
            "a stale (slot, generation) must not validate — it is another " \
            "expert's bytes now"


def test_resurrection_does_not_bump_the_generation(arena):
    """No bytes moved, so no reference was invalidated. A generation bump
    here would force needless refetches on the very path that exists to
    avoid them."""
    path, index = arena
    with _tier(path, index, 4, protected=1) as t:
        slot = t.ensure(0, [0])[0]
        gen = t.generations([slot])[0]
        t.ensure(0, [1])
        assert t.reclaimable(0, 0)
        assert t.validate(0, 0, slot, gen), "a reclaimable row is still itself"
        t.ensure(0, [0])
        assert t.generations([slot])[0] == gen
        assert t.validate(0, 0, slot, gen)


def test_validate_rejects_an_out_of_range_slot(arena):
    path, index = arena
    with _tier(path, index, 2) as t:
        t.ensure(0, [0])
        assert not t.validate(0, 0, 99, 0)
        assert not t.validate(0, 0, -1, 0)


# ------------------------------------------------------------- invariants --

def test_every_logical_eviction_resolves_exactly_once(arena):
    """The accounting identity that makes `reuse_before_overwrite` honest:
    a demoted row is resurrected, overwritten, or still pending — never two
    of those, never none. If this drifts, the probability is fiction."""
    import random
    rng = random.Random(20260819)
    path, index = arena
    with _tier(path, index, 4, protected=2) as t:
        for _ in range(300):
            lay = rng.randrange(L)
            t.ensure(lay, [rng.randrange(E) for _ in range(rng.randint(1, 3))])
        s = t.stats()
        assert s["logical_evictions"] == (
            s["resurrections"] + s["spec_resurrections"]
            + s["reclaimable_overwritten"] + s["reclaimable_rows"])
        # and the reclaimable set never names a row the tier does not hold
        assert set(t._reclaimable) <= set(t._slot_of)
        assert s["logical_evictions"] > 0, "the workload must exercise it"


def test_speculative_churn_cannot_evict_the_demand_window(arena):
    """The Stage-1 concurrency contract, re-pinned with a protected budget:
    a demoted row losing every allocation contest must not let a prefetcher
    take a row the demand caller is between ensure and read on."""
    import threading
    path, index = arena
    with _tier(path, index, 4, protected=1) as t:
        stop = threading.Event()

        def spec():
            while not stop.is_set():
                for e in range(E):
                    t.ensure(0, [e], speculative=True)

        th = threading.Thread(target=spec, daemon=True)
        th.start()
        try:
            for _ in range(200):
                slots = t.ensure(1, [0, 1])
                assert len(slots) == 2
                for e in (0, 1):
                    t.row(1, e)            # KeyError here is the failure
        finally:
            stop.set()
            th.join(timeout=5)


# ------------------------------------------------- Bugbot follow-ups --

def test_a_nested_ensure_does_not_demote_the_outer_batchs_siblings(arena):
    """gnf4#112. A caller materializing a batch one expert at a time must not
    logically evict its own siblings: that inflates logical_evictions and
    resurrections, and drops the window protection that keeps a concurrent
    speculative ensure off rows still being copied out."""
    from nvme_residency import segment_into
    path, index = arena
    suf = next(g["suffix"] for g in index["segments"])
    with _tier(path, index, 4, protected=1) as t:
        t.ensure(0, range(E))                       # one batch, protected
        before = t.stats()
        out = __import__("torch").empty(
            (4, *next(g["shape_per_expert"] for g in index["segments"]
                      if g["suffix"] == suf)), dtype=__import__("torch").uint8)
        for e in range(E):
            segment_into(t, index, 0, [e], suf, out, rows=[e], ensure=False)
        after = t.stats()
        assert after["logical_evictions"] == before["logical_evictions"], (
            "materializing a batch must not evict its own members")
        assert after["resurrections"] == before["resurrections"]


def test_ensure_false_refuses_a_row_that_is_not_resident(arena):
    """The contract that makes ensure=False safe: it may not silently read a
    slot the caller never made resident."""
    import torch

    from nvme_residency import segment_into
    path, index = arena
    g0 = next(g for g in index["segments"])
    with _tier(path, index, 4) as t:
        out = torch.empty((1, *g0["shape_per_expert"]), dtype=torch.uint8)
        with pytest.raises(KeyError, match="not resident"):
            segment_into(t, index, 0, [3], g0["suffix"], out, rows=[0],
                         ensure=False)


def test_a_failing_landing_does_not_strand_reservations(arena):
    """gnf4#118. A synchronous submit failure must reclaim its slots and wake
    its waiters — otherwise the next ensure of that key waits on an event
    nothing will ever set."""
    path, index = arena
    t = ColdTier(path, hot_rows=4, pinned=False, index=index,
                 landing=lambda layer, e, slot: None)   # always fails
    try:
        with pytest.raises(RuntimeError, match="no views"):
            t.ensure(0, [0, 1])
        assert t._reserved == set(), "reservations leaked"
        assert t._pending == {}, "pending events leaked"
        # every slot is back on the free list, so the tier can still serve a
        # caller that fixes its landing -- the failure is not terminal
        assert len(t._free) == t.hot_rows
        assert t._slot_of == {}
    finally:
        t.close()


def test_a_late_attached_landing_frees_the_buffer_it_will_never_fill(arena):
    """gnf4#120. Construction with landing= allocates one row; a late attach
    must match, or it keeps a full landing nothing writes."""
    path, index = arena
    t = ColdTier(path, hot_rows=64, pinned=False, index=index)
    try:
        big = len(t.buffer)
        t.attach_landing(lambda layer, e, slot: None)
        assert len(t.buffer) == t.row_stride < big
    finally:
        t.close()
