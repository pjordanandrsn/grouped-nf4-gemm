# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""SegmentedRowPool: placement, displacement-as-refill, budget, shrink/grow.

CPU mode. Replacement policy lives INSIDE each segment (the shipped margin
machinery); the wrapper only places, so these tests assert the wrapper's
contracts: hits resolve to their owning segment, internal displacement
surfaces as a re-fill on next touch (never an error), budget-skipped keys
mutate nothing, shrink frees whole segments event-gated, grow restores.
"""
import pytest

from dev_row_cache import StepTag
from segmented_pool import SegmentedRowPool


def _pool(segments=2, seg_rows=12, routed=4):
    return SegmentedRowPool(segments, seg_rows, 64, device="cpu",
                            routed=routed)


def _want(p, layer, experts, budget=None):
    t = StepTag("cpu")
    out = p.want(layer, experts, t, budget=budget)
    t.record()
    return out


def test_insert_then_hit_no_refill():
    p = _pool()
    placed, need, skipped = _want(p, 0, [1, 2, 3])
    assert sorted(need) == [(0, 1), (0, 2), (0, 3)] and not skipped
    assert p.fills == 3
    placed2, need2, skipped2 = _want(p, 0, [1, 2, 3])
    assert not need2 and not skipped2 and p.fills == 3
    assert placed2 == placed


def test_inserts_rotate_across_segments():
    p = _pool(segments=3, seg_rows=12)
    _want(p, 0, list(range(6)))
    segs = {p._where[(0, e)][0] for e in range(6)}
    assert segs == {0, 1, 2}


def test_displacement_surfaces_as_refill():
    p = _pool(segments=1, seg_rows=12, routed=4)
    # protected = 8: pushing far past durable capacity forces the segment's
    # own LRU machinery to displace early keys
    for e in range(24):
        _want(p, 0, [e])
    total = p.fills + p.refills
    placed, need, _ = _want(p, 0, [0])
    assert placed and (0, 0) in placed
    if need:                       # displaced internally: re-fill, no error
        assert need == [(0, 0)]
        assert p.fills + p.refills == total + 1


def test_budget_skips_without_state_change():
    p = _pool()
    placed, need, skipped = _want(p, 0, [1, 2, 3, 4], budget=2)
    assert len(need) == 2 and len(skipped) == 2
    assert all(k not in p._where for k in skipped)
    _, need2, _ = _want(p, 0, [e for (_, e) in skipped], budget=None)
    assert len(need2) == 2


def test_shrink_frees_emptiest_first_and_survivors_refill():
    p = _pool(segments=3, seg_rows=12)
    _want(p, 0, list(range(12)))          # rotation: 4 keys per segment
    p.shrink(0)
    assert p.segments_alive() == 3
    freed = p.shrink(2)
    assert freed == 2 * p.seg_bytes() and p.segments_alive() == 1
    dropped = 12 - p.rows_resident()
    assert dropped == 8                    # two segments' keys dropped
    _, need, _ = _want(p, 0, list(range(12)))
    assert len(need) == dropped            # survivors hit, dropped refill


def test_shrunk_segment_views_raise():
    p = _pool(segments=2, seg_rows=12)
    p.shrink(1)
    dead = [si for si, s in enumerate(p._segs) if s is None][0]
    with pytest.raises(KeyError):
        p.views(dead, (8, 4), (8, 2), 32)


def test_grow_restores_capacity_and_serves():
    p = _pool(segments=2, seg_rows=12)
    p.shrink(1)
    assert p.rows_capacity() == 12
    assert p.grow(1) == 1
    assert p.rows_capacity() == 24 and p.segments_alive() == 2
    _, need, _ = _want(p, 0, list(range(8)))
    assert len(need) == 8


def test_fully_shrunk_pool_skips_everything():
    p = _pool(segments=1, seg_rows=12)
    p.shrink(1)
    placed, need, skipped = _want(p, 0, [1, 2])
    assert not placed and not need and len(skipped) == 2
