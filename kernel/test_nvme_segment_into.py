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


# ------------------------------------------- V4's own dtype labels
# Lives here, not in test_nvme_residency.py, because that file is allowlisted
# "needs CUDA" in test_packaging_covers_kernel._NOT_IN_CI and so is never
# invoked by ci.yml — a regression test parked there could not fail. This file
# is in CI and is the staging seam that actually broke.
def _relabel(index, dtype, match="scale"):
    """A copy of `index` with the matching segments labelled `dtype`.

    Relabelling rather than re-baking is the point: the BYTES are identical and
    only the safetensors tag differs, which is exactly the difference between a
    DeepSeek-V4 checkpoint and a Kimi K3 one.
    """
    import copy
    out = copy.deepcopy(index)
    for g in out["segments"]:
        if match in g["suffix"]:
            g["dtype"] = dtype
    return out


def test_segment_geometry_reads_v4s_f8_e8m0_scale_label(arena):
    """DeepSeek-V4 labels its MXFP4 scales `F8_E8M0`; Kimi K3 labels them `U8`.

    Same bytes, different tag. `_ST_TO_TORCH` knew only the K3 spelling, so a
    real V4 arena raised `KeyError: 'F8_E8M0'` here — after the 149 GB download
    and the 147 GB bake that produced it. `nvme_bake_nf4` accepts the tag and
    `mxfp4_residency` serves from it, so this table was the only thing in the way.
    """
    path, index = arena
    v4 = _relabel(index, "F8_E8M0")
    assert any(g["dtype"] == "F8_E8M0" for g in v4["segments"]), "relabel did nothing"
    for g in v4["segments"]:
        dt, shape, off, ln = segment_geometry(v4, g["suffix"])
        if g["dtype"] == "F8_E8M0":
            assert dt is torch.uint8, f"{g['suffix']}: e8m0 must read back as bytes, got {dt}"
        assert shape == tuple(g["shape_per_expert"])
    # A relabel must not move a byte: everything but the dtype is unchanged.
    for g, h in zip(index["segments"], v4["segments"]):
        assert segment_geometry(index, g["suffix"])[1:] == segment_geometry(v4, h["suffix"])[1:]


def test_v4_labelled_segments_still_read_the_same_bytes(arena):
    """The tag must change the DTYPE and nothing else.

    `segment_tensor` on a relabelled index has to return the identical bytes it
    returns for the `U8` spelling — reinterpreted, never converted. A mapping to
    `float8_e8m0fnu` would satisfy the KeyError and fail this.
    """
    path, index = arena
    v4 = _relabel(index, "F8_E8M0")
    suffix = next(g["suffix"] for g in v4["segments"] if g["dtype"] == "F8_E8M0")
    with _tier(path, index) as t:
        a = segment_tensor(t, index, 0, PICK, suffix)
    with _tier(path, v4) as t:
        b = segment_tensor(t, v4, 0, PICK, suffix)
    assert a.dtype == b.dtype == torch.uint8
    assert torch.equal(a, b), "relabelling the dtype tag changed the bytes"


def test_the_byte_dtype_tables_do_not_drift_apart():
    """The root cause was three tables and only two of them kept current.

    `mxfp4_residency._PACKED_BYTE_DTYPES` and `nvme_bake_nf4._MXFP4_BYTE_DTYPES`
    both listed `F8_E8M0`; `_ST_TO_TORCH` did not — so an arena this package can
    bake and serve could not be staged. Containment, not a fixed set, so adding a
    tag in either of those places fails here until it is added here too.
    """
    from nvme_residency import _ST_TO_TORCH
    missing = []
    for mod, name in (("mxfp4_residency", "_PACKED_BYTE_DTYPES"),
                      ("nvme_bake_nf4", "_MXFP4_BYTE_DTYPES")):
        try:
            tags = getattr(__import__(mod), name)
        except Exception:                                  # pragma: no cover
            continue
        missing += [f"{mod}.{name}:{d}" for d in tags if d not in _ST_TO_TORCH]
    assert not missing, (
        "these dtype tags are accepted elsewhere in the package but cannot be read "
        f"back by segment_geometry/segment_tensor: {sorted(set(missing))}")
