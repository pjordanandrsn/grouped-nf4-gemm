# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""N3 contract tests: residency over a baked arena.

Correctness first (a resident row must be byte-identical to the shipped bytes,
and eviction must never corrupt a row a request is about to use), then the
number that actually decides the design: hit rate vs hot-set fraction under
heavy-tailed routing. Pure stdlib + the toy arena from test_nvme_arena — no
torch, no GPU, no large files.
"""
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from nvme_arena import bake, load_index  # noqa: E402
from nvme_residency import ColdTier, capacity_for_bytes  # noqa: E402
from test_nvme_arena import E, L, make_snapshot  # noqa: E402


@pytest.fixture()
def arena(tmp_path):
    snap = tmp_path / "snap"
    ground = make_snapshot(str(snap))
    path = str(tmp_path / "toy.arena")
    bake(str(snap), path, align=4096, log=lambda *a: None)
    return path, load_index(path), ground


def _tier(path, index, hot_rows):
    # pinned=False: mmap landing, so these run on CPU CI with no CUDA.
    return ColdTier(path, hot_rows=hot_rows, pinned=False, index=index)


def test_resident_rows_are_byte_identical_to_the_arena(arena):
    path, index, _ = arena
    with _tier(path, index, L * E) as t:
        for lay in range(L):
            slots = t.ensure(lay, range(E))
            assert len(slots) == E and len(set(slots)) == E
            for e, slot in zip(range(E), slots):
                # read the same row straight off disk and compare
                ref, keep = __import__("nvme_reader").alloc_landing(
                    t.row_stride, pinned=False)
                t.reader.read_row_sync(lay, e, ref)
                assert bytes(t.row(lay, e)) == bytes(ref[:t.row_bytes])
                del keep


def test_hit_and_miss_accounting(arena):
    path, index, _ = arena
    with _tier(path, index, E) as t:
        t.ensure(0, range(E))                      # all cold
        s = t.stats()
        assert s["misses"] == E and s["hits"] == 0
        t.ensure(0, range(E))                      # all warm
        s = t.stats()
        assert s["hits"] == E and s["misses"] == E
        assert s["hit_rate"] == pytest.approx(0.5)
        assert s["disk_reads"] == E                # no re-read on hits


def test_lfu_keeps_the_hot_expert_resident(arena):
    """A repeatedly routed expert must survive churn from one-off experts."""
    path, index, _ = arena
    with _tier(path, index, 2) as t:
        for _ in range(5):
            t.ensure(0, [0])                       # expert 0 is hot
        for e in range(1, E):                      # churn through the rest
            t.ensure(0, [e])
        assert t.resident(0, 0), "LFU evicted the most-frequently-used row"


def test_request_never_evicts_its_own_rows(arena):
    """hot_rows exactly equal to the request size must still succeed: every
    slot is protected, so the victim search must not be asked for one."""
    path, index, _ = arena
    with _tier(path, index, E) as t:
        slots = t.ensure(0, range(E))
        assert len(set(slots)) == E
        for e in range(E):
            assert t.resident(0, e)


def test_request_larger_than_hot_set_is_a_clean_error(arena):
    path, index, _ = arena
    with _tier(path, index, 2) as t:
        with pytest.raises(ValueError, match="exceeds hot_rows"):
            t.ensure(0, range(E))


def test_same_expert_twice_in_one_request_shares_a_slot(arena):
    path, index, _ = arena
    with _tier(path, index, 3) as t:
        slots = t.ensure(0, [1, 1, 1])
        assert slots[0] == slots[1] == slots[2]
        assert t.stats()["disk_reads"] == 1


def test_capacity_for_bytes_floors_and_never_returns_zero():
    assert capacity_for_bytes(10 * 4096, 4096) == 10
    assert capacity_for_bytes(4095, 4096) == 1          # never 0 rows
    assert capacity_for_bytes(9000, 4096) == 2


def test_hit_rate_rises_with_hot_fraction_under_skewed_routing(arena):
    """The decision curve. Routing is heavy-tailed, so a hot set far smaller
    than the expert count should still absorb most picks — that is the whole
    premise of serving a 1.446 TB expert set from a 503 GB host."""
    path, index, _ = arena
    rng = random.Random(1689)
    # Zipf-ish draw over E experts, top-2 routed per step
    weights = [1.0 / (i + 1) ** 1.2 for i in range(E)]

    def draw(k=2):
        picks, seen = [], set()
        while len(picks) < k:
            e = rng.choices(range(E), weights=weights, k=1)[0]
            if e not in seen:
                seen.add(e); picks.append(e)
        return picks

    rates = {}
    for hot in (2, max(2, E // 2), E):
        rng.seed(1689)
        with _tier(path, index, hot) as t:
            for _ in range(200):
                t.ensure(0, draw())
            rates[hot] = t.stats()["hit_rate"]
    assert rates[E] >= rates[max(2, E // 2)] >= rates[2], rates
    assert rates[E] > 0.5, f"a fully-resident hot set should hit often: {rates}"
