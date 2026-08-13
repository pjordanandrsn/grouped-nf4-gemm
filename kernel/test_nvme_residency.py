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
from nvme_residency import (  # noqa: E402
    PINNED_ROW_FACTOR,
    ColdTier,
    capacity_for_bytes,
)
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
    # pinned=False is the plain arithmetic: a row costs exactly its stride.
    assert capacity_for_bytes(10 * 4096, 4096, pinned=False) == 10
    assert capacity_for_bytes(4095, 4096, pinned=False) == 1      # never 0 rows
    assert capacity_for_bytes(9000, 4096, pinned=False) == 2
    assert capacity_for_bytes(0, 4096, pinned=False) == 1         # still never 0


def test_capacity_for_bytes_defaults_to_the_pinned_cost():
    """The default must budget for what a PINNED row really costs.

    ColdTier pins by default, and a pinned row costs ~1.9x its stride of real
    host memory (measured by cap ladder, see PINNED_ROW_FACTOR). Dividing by the
    stride alone returns a hot_rows that OOMs partway through the first step, so
    the default has to be the conservative one -- a caller who wants the raw
    arithmetic asks for it.
    """
    budget, stride = 10 * 4096, 4096
    assert capacity_for_bytes(budget, stride) < capacity_for_bytes(budget, stride, pinned=False)
    assert capacity_for_bytes(budget, stride) == int(budget // (stride * PINNED_ROW_FACTOR))
    # An explicit factor overrides both, and 1.0 reproduces the unpinned answer.
    assert capacity_for_bytes(budget, stride, factor=1.0) == 10
    assert capacity_for_bytes(budget, stride, factor=2.0) == 5
    with pytest.raises(ValueError):
        capacity_for_bytes(budget, stride, factor=0)


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


# ------------------------------------------------ Bugbot #17 regressions --
def test_duplicates_do_not_count_against_capacity(arena):
    """`ensure(l, [7,7,7])` needs ONE row, not three: repeats share a slot, so
    the capacity check must count UNIQUE (layer, expert) pairs."""
    path, index, _ = arena
    with _tier(path, index, 1) as t:
        slots = t.ensure(0, [1, 1, 1])          # 3 ids, 1 unique, hot_rows=1
        assert slots == [slots[0]] * 3
        assert t.stats()["disk_reads"] == 1
    with _tier(path, index, 2) as t:            # still rejects genuinely-too-many
        with pytest.raises(ValueError, match="unique rows exceeds"):
            t.ensure(0, [0, 1, 2])


def test_failed_fill_is_never_published_as_resident(arena, monkeypatch):
    """A slot must not appear resident until its read LANDS. Publishing in pass 1
    let a failed (or in-flight) read expose partial bytes to the gather path."""
    path, index, _ = arena
    with _tier(path, index, E) as t:
        boom = RuntimeError("simulated NVMe failure")

        def fail(layer, expert, dst):
            fut = __import__("concurrent.futures", fromlist=["Future"]).Future()
            fut.set_exception(boom)
            return fut

        monkeypatch.setattr(t.reader, "read_row", fail)
        with pytest.raises(RuntimeError, match="simulated NVMe failure"):
            t.ensure(0, [0, 1])
        # nothing advertised, and the slots came back for reuse
        assert not t.resident(0, 0) and not t.resident(0, 1)
        assert t.stats()["resident_rows"] == 0
        monkeypatch.undo()
        # the tier is still usable afterwards — slots were reclaimed, not leaked
        t.ensure(0, [0, 1])
        assert t.resident(0, 0) and t.resident(0, 1)


def test_partial_batch_failure_publishes_only_what_landed(arena, monkeypatch):
    """One bad row in a batch must not poison the rows that read cleanly, and
    must not leave the failed slot advertised."""
    path, index, _ = arena
    import concurrent.futures as cf
    with _tier(path, index, E) as t:
        real = t.reader.read_row

        def flaky(layer, expert, dst):
            if expert == 1:
                fut = cf.Future(); fut.set_exception(RuntimeError("bad row"))
                return fut
            return real(layer, expert, dst)

        monkeypatch.setattr(t.reader, "read_row", flaky)
        with pytest.raises(RuntimeError, match="bad row"):
            t.ensure(0, [0, 1, 2])
        assert not t.resident(0, 1), "failed row must not be advertised"
        assert t.resident(0, 0) and t.resident(0, 2), "clean rows should survive"


# -------------------------------------- bit-identity back to the release --
def test_served_row_is_bit_identical_to_the_SHIPPED_bytes(arena):
    """The claim that matters: bytes served out of the tier are bit-identical to
    the bytes in the original checkpoint — not merely to the arena.

    Chain of custody: `nvme_arena.verify(--against-source)` proves arena ==
    shipped, and this proves tier == arena, so tier == shipped transitively. Here
    it is checked DIRECTLY against the per-expert source tensors instead, which
    needs no transitivity argument at all.
    """
    from nvme_arena import _seg_len, _seg_off
    from mxfp4_loader import EXPERT_SUFFIXES
    path, index, ground = arena
    with _tier(path, index, E) as t:
        for lay in range(L):
            t.ensure(lay, range(E))
            for e in range(E):
                row = t.row(lay, e)
                for suf in EXPERT_SUFFIXES:
                    off, ln = _seg_off(index, suf), _seg_len(index, suf)
                    served = bytes(row[off:off + ln])
                    shipped = ground[f"model.layers.{lay}.{suf}"][e]
                    assert served == shipped, (
                        f"layer {lay} expert {e} segment {suf}: served bytes "
                        f"differ from the shipped checkpoint")


def test_bit_identity_survives_eviction_and_refill(arena):
    """Re-reading an evicted row must reproduce the shipped bytes exactly — a
    stale or partially-overwritten slot would show up here."""
    from nvme_arena import _seg_len, _seg_off
    from mxfp4_loader import EXPERT_SUFFIXES
    path, index, ground = arena
    suf = EXPERT_SUFFIXES[0]
    off, ln = _seg_off(index, suf), _seg_len(index, suf)
    with _tier(path, index, 2) as t:            # tiny hot set forces churn
        for _round in range(3):
            for e in range(E):
                t.ensure(0, [e])
                served = bytes(t.row(0, e)[off:off + ln])
                assert served == ground[f"model.layers.0.{suf}"][e], (
                    f"expert {e} corrupted after eviction/refill")
        assert t.stats()["evictions"] > 0, "test did not actually force eviction"


# ------------- the last hop: reconstructed TENSOR == shipped TENSOR ---------
def test_segment_tensor_is_bit_identical_to_the_shipped_tensor(arena):
    """Closes the chain. Bytes being bit-identical is necessary but not
    sufficient: the engine consumes TENSORS, so reinterpreting arena bytes at the
    recorded dtype must reproduce the original tensor exactly — element for
    element, not merely byte for byte at some assumed layout.
    """
    torch = pytest.importorskip("torch")
    from mxfp4_loader import EXPERT_SUFFIXES
    from nvme_residency import segment_tensor
    path, index, ground = arena

    with _tier(path, index, E) as t:
        for lay in range(L):
            for suf in EXPERT_SUFFIXES:
                geo = next(g for g in index["segments"] if g["suffix"] == suf)
                dt = getattr(torch, {"U8": "uint8"}.get(geo["dtype"], "uint8"))
                got = segment_tensor(t, index, lay, range(E), suf)
                # rebuild the reference straight from the shipped per-expert bytes
                ref = torch.stack([
                    torch.frombuffer(bytearray(ground[f"model.layers.{lay}.{suf}"][e]),
                                     dtype=dt).view(tuple(geo["shape_per_expert"]))
                    for e in range(E)])
                assert got.shape == ref.shape
                assert got.dtype == ref.dtype
                assert torch.equal(got, ref), (
                    f"layer {lay} segment {suf}: reconstructed tensor differs "
                    f"from the shipped checkpoint")


def test_segment_tensor_cast_is_the_only_transform(arena):
    """`cast` must be the ONLY thing that changes values — and it must be applied,
    because the engine holds NF4 absmax as float32. Reinterpreting raw scale bytes
    as float32 instead would give plausible shapes and garbage numerics."""
    torch = pytest.importorskip("torch")
    from mxfp4_loader import EXPERT_SUFFIXES
    from nvme_residency import segment_tensor
    path, index, _ = arena
    suf = EXPERT_SUFFIXES[1]                       # a scales segment
    with _tier(path, index, E) as t:
        raw = segment_tensor(t, index, 0, range(E), suf)
        cast = segment_tensor(t, index, 0, range(E), suf, cast="float32")
        assert raw.dtype == torch.uint8 and cast.dtype == torch.float32
        # value-preserving: casting uint8 -> f32 must not reinterpret bits
        assert torch.equal(cast, raw.to(torch.float32))
        assert cast.max() <= 255.0


def test_segment_tensor_rejects_an_unknown_segment(arena):
    pytest.importorskip("torch")
    from nvme_residency import segment_tensor
    path, index, _ = arena
    with _tier(path, index, E) as t:
        with pytest.raises(KeyError, match="not in this arena"):
            segment_tensor(t, index, 0, [0], "no.such.segment")


def test_pinned_landing_is_aligned_even_when_the_allocator_suballocates():
    """Regression: PyTorch's caching host allocator suballocates, so
    `pin_memory()` is page-aligned only while the allocator is virgin. After
    other pinned blocks exist it can hand back an interior offset (measured
    1024 B off on 2026-07-30), which made O_DIRECT reads EINVAL far from the
    cause. `alloc_landing` must align defensively rather than trust it."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("pinned memory needs CUDA")
    from nvme_reader import alloc_landing, buffer_address
    # churn the pinned allocator with odd sizes first, to provoke suballocation
    keep = [torch.empty(n, dtype=torch.uint8).pin_memory()
            for n in (1234, 5678, 9012, 34567)]
    for n in (16384, 16384 * 6, 4096 * 3 + 17):
        mv, ka = alloc_landing(n, pinned=True)
        assert len(mv) == n, "aligned sub-view must still be the requested size"
        assert buffer_address(mv) % 4096 == 0, f"n={n} came back misaligned"
        del ka
    del keep
