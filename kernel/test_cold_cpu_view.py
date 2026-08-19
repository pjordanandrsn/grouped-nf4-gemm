# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Stage-3 contract tests: the CPU destination for cold experts.

The claim under test is that a cold expert can reach the native CPU kernels
from the same packed bytes the GPU path reads, with the tier owning residency
and the view owning only layout. Correctness first — the materialized stack
must equal `segment_tensor`, which is already pinned bit-identical to the
shipped checkpoint — then the generation discipline that decides when a
re-layout is needed, then the reclaimable-residency payoff arriving intact.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from nvme_arena import bake, load_index  # noqa: E402
from nvme_residency import ColdTier  # noqa: E402
from test_nvme_arena import E, L, make_snapshot  # noqa: E402

torch = pytest.importorskip("torch")

from cold_cpu_view import ColdCpuView  # noqa: E402
from nvme_residency import segment_tensor  # noqa: E402


@pytest.fixture()
def arena(tmp_path):
    snap = tmp_path / "snap"
    make_snapshot(str(snap))
    path = str(tmp_path / "toy.arena")
    bake(str(snap), path, align=4096, log=lambda *a: None)
    return path, load_index(path)


def _suffixes(index, n=2):
    return [g["suffix"] for g in index["segments"]][:n]


def _tier(path, index, hot_rows, protected=None):
    return ColdTier(path, hot_rows=hot_rows, pinned=False, index=index,
                    protected_rows=protected)


def _reads(t):
    return t.reader.traffic()["reads"]


# ----------------------------------------------------------- correctness --

def test_materialized_rows_equal_the_shipped_tensor(arena):
    """The equivalence 'NVMe->CPU vs resident CPU' reduces to this: the bytes
    the CPU kernel indexes are the bytes the arena holds, at the same dtype
    and shape. `segment_tensor` is already pinned bit-identical to the
    checkpoint, so equality against it closes the chain."""
    path, index = arena
    sufs = _suffixes(index)
    with _tier(path, index, L * E) as t:
        v = ColdCpuView(t, index, sufs)
        for lay in range(L):
            slots = v.ensure(lay, range(E))
            ref = {s: segment_tensor(t, index, lay, range(E), s) for s in sufs}
            for s in sufs:
                got = v.stack(s)[list(slots)]
                assert got.dtype == ref[s].dtype
                assert torch.equal(got, ref[s]), (
                    f"layer {lay} segment {s}: the CPU destination's bytes "
                    f"differ from the arena's")


def test_slots_are_the_kernel_expert_ids(arena):
    """`ensure` returns tier slots, and the stacks are slot-parallel, so the
    slot IS the kernel's expert id. Anything else would need a translation
    table on the hot path."""
    path, index = arena
    suf = _suffixes(index, 1)[0]
    with _tier(path, index, 4) as t:
        v = ColdCpuView(t, index, [suf])
        slots = v.ensure(0, [2, 3])
        for e, slot in zip((2, 3), slots):
            one = segment_tensor(t, index, 0, [e], suf)[0]
            assert torch.equal(v.stack(suf)[slot], one)
        assert v.stack(suf).shape[0] == t.hot_rows
        assert v.stack(suf).is_contiguous()


def test_repeats_in_one_request_are_laid_out_once(arena):
    path, index = arena
    with _tier(path, index, 4) as t:
        v = ColdCpuView(t, index, _suffixes(index, 1))
        slots = v.ensure(0, [1, 1, 1])
        assert len(set(slots)) == 1
        assert v.stats()["materializations"] == 1


# ------------------------------------------------------------ generations --

def test_a_reused_slot_is_relaid_out(arena):
    """The failure this prevents: slot reuse leaving a stale materialization,
    so the kernel multiplies expert 0's bytes while the router asked for
    expert 3. Plausible shapes, wrong answer."""
    path, index = arena
    suf = _suffixes(index, 1)[0]
    with _tier(path, index, 1) as t:            # one slot: guaranteed reuse
        v = ColdCpuView(t, index, [suf])
        v.ensure(0, [0])
        slot = v.ensure(0, [3])[0]
        assert torch.equal(v.stack(suf)[slot],
                           segment_tensor(t, index, 0, [3], suf)[0])
        assert v.stats()["materializations"] == 2


def test_unchanged_slot_is_not_relaid_out(arena):
    path, index = arena
    with _tier(path, index, 4) as t:
        v = ColdCpuView(t, index, _suffixes(index, 1))
        v.ensure(0, [0])
        v.ensure(0, [0])
        v.ensure(0, [0])
        s = v.stats()
        assert s["materializations"] == 1 and s["view_hits"] == 2


def test_holds_tracks_the_tier(arena):
    path, index = arena
    with _tier(path, index, 1) as t:
        v = ColdCpuView(t, index, _suffixes(index, 1))
        v.ensure(0, [0])
        assert v.holds(0, 0)
        v.ensure(0, [1])
        assert not v.holds(0, 0), "the slot was reused; the layout is stale"


# ------------------------------- the reclaimable payoff, arriving intact --

def test_a_resurrection_costs_neither_a_read_nor_a_relayout(arena):
    """The whole point of carrying generations across the boundary: a
    reclaimable row's bytes never moved, so the CPU destination's copy of
    them is still valid. If the view re-laid-out here, the resurrection
    would have saved a disk read and paid a memcpy for it."""
    path, index = arena
    with _tier(path, index, 4, protected=1) as t:
        v = ColdCpuView(t, index, _suffixes(index, 1))
        v.ensure(0, [0])
        v.ensure(0, [1])
        assert t.reclaimable(0, 0)
        reads, mats = _reads(t), v.stats()["materializations"]
        v.ensure(0, [0])
        assert _reads(t) == reads, "resurrection must not read the disk"
        assert v.stats()["materializations"] == mats, \
            "resurrection must not re-lay-out either"
        assert t.stats()["resurrections"] == 1


def test_an_overwritten_row_costs_both_again(arena):
    """The complement, so the test above is not passing by accident."""
    path, index = arena
    with _tier(path, index, 2, protected=1) as t:
        v = ColdCpuView(t, index, _suffixes(index, 1))
        v.ensure(0, [0])
        v.ensure(0, [1])
        v.ensure(0, [2])                        # overwrites reclaimable 0
        reads, mats = _reads(t), v.stats()["materializations"]
        v.ensure(0, [0])
        assert _reads(t) == reads + 1
        assert v.stats()["materializations"] == mats + 1


# -------------------------------------------------------------- contract --

def test_unmaterialized_segment_is_a_named_refusal(arena):
    path, index = arena
    with _tier(path, index, 2) as t:
        v = ColdCpuView(t, index, _suffixes(index, 1))
        with pytest.raises(KeyError, match="not materialized"):
            v.stack("no_such_segment")


def test_empty_segment_list_is_a_clean_error(arena):
    path, index = arena
    with _tier(path, index, 2) as t:
        with pytest.raises(ValueError, match="at least one segment"):
            ColdCpuView(t, index, [])


def test_cast_naming_an_unheld_segment_is_a_clean_error(arena):
    path, index = arena
    sufs = _suffixes(index, 1)
    with _tier(path, index, 2) as t:
        with pytest.raises(ValueError, match="does not hold"):
            ColdCpuView(t, index, sufs, casts={"elsewhere": torch.float32})
