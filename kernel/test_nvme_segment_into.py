# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""`segment_into` contract: caller-owned destinations, same bytes as
`segment_tensor`.

`segment_tensor` is the serving seam and allocates its own result. A *staging*
path cannot use that: it holds one reusable buffer (or writes straight to the
device), and it fills only the routed rows of a full-shaped destination. This
file pins that contract down — equivalence to `segment_tensor` first, since
that is the function whose bit-identity is already established, then the two
things only the new entry point can get wrong: which rows it touches, and the
slot arithmetic on the pinned path.

The pinned branch needs no CUDA here: it is exercised against a stand-in tier
whose `pinned_tensor()` is an ordinary CPU tensor, which is exactly what makes
a slot/offset skew visible. `ColdTier(pinned=True)` itself requires a CUDA host
(`pin_memory`), so the real pinned tier is covered on GPU boxes only — but the
byte math under test is the same code either way.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

torch = pytest.importorskip("torch")

from nvme_arena import bake, load_index  # noqa: E402
from nvme_residency import (ColdTier, segment_geometry,  # noqa: E402
                            segment_into, segment_tensor)
from mxfp4_loader import EXPERT_SUFFIXES  # noqa: E402
from test_nvme_arena import E, L, make_snapshot  # noqa: E402

# One expert subset used throughout: out of order and with a repeat, because
# `ensure` dedupes to one slot per unique row while still owing the caller one
# destination row per REQUESTED expert. A sorted, distinct list would let a
# slot-per-request bug pass.
PICK = [2, 0, 2, 1]


@pytest.fixture()
def arena(tmp_path):
    snap = tmp_path / "snap"
    make_snapshot(str(snap))
    path = str(tmp_path / "toy.arena")
    bake(str(snap), path, align=4096, log=lambda *a: None)
    return path, load_index(path)


def _tier(path, index, hot_rows=L * E):
    # pinned=False: mmap landing, so this runs on CPU CI with no CUDA.
    return ColdTier(path, hot_rows=hot_rows, pinned=False, index=index)


@pytest.mark.parametrize("suffix", EXPERT_SUFFIXES)
def test_matches_segment_tensor_bitwise(arena, suffix):
    """The claim everything else rests on: same bytes as the established seam."""
    path, index = arena
    dt, shape, _off, _ln = segment_geometry(index, suffix)
    with _tier(path, index) as t:
        want = segment_tensor(t, index, 1, PICK, suffix)
        out = torch.empty((len(PICK),) + shape, dtype=dt)
        got = segment_into(t, index, 1, PICK, suffix, out)
    assert got is out
    assert got.dtype == want.dtype and got.shape == want.shape
    # bitwise, not allclose: these are relocated bytes, so anything but exact
    # equality is a bug in the reinterpretation.
    assert torch.equal(got.view(torch.uint8), want.view(torch.uint8))


def test_fills_only_the_requested_rows(arena):
    """Routed staging writes into a full-shaped [E, ...] dest and must leave the
    unrouted rows exactly as it found them — the reroute guard upstream depends
    on untouched rows staying recognisably untouched."""
    path, index = arena
    suffix = EXPERT_SUFFIXES[0]
    dt, shape, _off, _ln = segment_geometry(index, suffix)
    routed = [3, 1]
    with _tier(path, index) as t:
        full = torch.full((E,) + shape, 0xA5, dtype=dt)
        segment_into(t, index, 0, routed, suffix, full, rows=routed)
        want = segment_tensor(t, index, 0, routed, suffix)
    for i, e in enumerate(routed):
        assert torch.equal(full[e], want[i]), f"routed row {e} wrong"
    for e in set(range(E)) - set(routed):
        assert torch.equal(full[e], torch.full(shape, 0xA5, dtype=dt)), \
            f"row {e} was not routed but got written"


def test_rows_maps_positionally_not_by_expert_id(arena):
    """`rows` is a destination index per requested expert, in order — not a
    filter. Passing them reversed must reverse where the bytes land, or a caller
    whose dest is compacted (row = position in the routed set, not the expert id)
    would silently get its experts transposed."""
    path, index = arena
    suffix = EXPERT_SUFFIXES[2]
    dt, shape, _off, _ln = segment_geometry(index, suffix)
    pick = [0, 1, 2]
    with _tier(path, index) as t:
        want = segment_tensor(t, index, 0, pick, suffix)
        out = torch.zeros((3,) + shape, dtype=dt)
        segment_into(t, index, 0, pick, suffix, out, rows=[2, 1, 0])
    for i in range(3):
        assert torch.equal(out[2 - i], want[i])


class _StandInTier:
    """A pinned tier without CUDA: `pinned_tensor()` is a plain CPU tensor whose
    slot rows carry the same bytes the real landing buffer would. Only the slot
    arithmetic is under test, and it is identical on a real pinned tier."""

    def __init__(self, real, index):
        self.pinned = True
        self.row_stride = real.row_stride
        self._real = real
        self._buf = torch.zeros(real.hot_rows, real.row_stride, dtype=torch.uint8)
        self._slots = {}
        self._next = 0

    def ensure(self, layer, experts):
        out = []
        for e in (int(x) for x in experts):
            slot = self._slots.get((layer, e))
            if slot is None:
                slot = self._slots[(layer, e)] = self._next
                self._next += 1
                # Fill the slot exactly as the reader would: the row's bytes at
                # the START of the slot, padding after.
                row = bytes(self._real.row(layer, e))
                self._buf[slot, :len(row)] = torch.frombuffer(
                    bytearray(row), dtype=torch.uint8)
            out.append(slot)
        return out

    def pinned_tensor(self):
        return self._buf


def test_pinned_path_reads_the_right_slot_bytes(arena):
    """The pinned branch indexes `pinned_tensor()[slot, off:off+len]`. A skew in
    either term — striding row_bytes instead of row_stride, or dropping the
    segment offset — still produces plausibly-shaped output, so compare against
    the unpinned path rather than eyeballing shapes."""
    path, index = arena
    for suffix in EXPERT_SUFFIXES:
        dt, shape, _off, _ln = segment_geometry(index, suffix)
        with _tier(path, index) as t:
            want = segment_tensor(t, index, 2, PICK, suffix)
            stand_in = _StandInTier(t, index)
            out = torch.empty((len(PICK),) + shape, dtype=dt)
            segment_into(stand_in, index, 2, PICK, suffix, out)
        assert torch.equal(out.view(torch.uint8), want.view(torch.uint8)), suffix


@pytest.mark.parametrize("bad,msg", [
    ("dtype", "dtype"),
    ("shape", "needs"),
    ("noncontig", "contiguous"),
    ("rows", "rows has"),
])
def test_refuses_a_destination_it_cannot_fill_correctly(arena, bad, msg):
    """Each of these would otherwise write real bytes to the wrong place: a
    mismatched dtype reinterprets them, a wrong trailing shape shifts every row,
    a non-contiguous dest makes `reshape(-1)` a copy that is silently discarded,
    and a short `rows` drops experts off the end."""
    path, index = arena
    suffix = EXPERT_SUFFIXES[0]
    dt, shape, _off, _ln = segment_geometry(index, suffix)
    kw = {}
    if bad == "dtype":
        other = torch.float32 if dt != torch.float32 else torch.uint8
        out = torch.empty((len(PICK),) + shape, dtype=other)
    elif bad == "shape":
        out = torch.empty((len(PICK), shape[0] + 1) + shape[1:], dtype=dt)
    elif bad == "noncontig":
        out = torch.empty((len(PICK),) + (shape[0] * 2,) + shape[1:],
                          dtype=dt)[:, ::2]
    else:
        out = torch.empty((len(PICK),) + shape, dtype=dt)
        kw["rows"] = [0, 1]
    with _tier(path, index) as t:
        with pytest.raises((TypeError, ValueError), match=msg):
            segment_into(t, index, 0, PICK, suffix, out, **kw)


def test_segment_geometry_agrees_with_the_index(arena):
    _path, index = arena
    for seg in index["segments"]:
        dt, shape, off, ln = segment_geometry(index, seg["suffix"])
        assert shape == tuple(seg["shape_per_expert"])
        assert (off, ln) == (seg["seg_off"], seg["length"])
        # length must be exactly the bytes the shape implies at that dtype, or
        # the flat-byte-run copy would over- or under-fill every row.
        n = 1
        for s in shape:
            n *= s
        assert ln == n * torch.empty(0, dtype=dt).element_size()


def test_unknown_segment_names_what_is_present(arena):
    _path, index = arena
    with pytest.raises(KeyError, match="not in this arena"):
        segment_geometry(index, "nf4.nope")
