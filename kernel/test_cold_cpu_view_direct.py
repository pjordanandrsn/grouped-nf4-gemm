# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Direct-scatter mode: the kernel DMAs each segment straight into the stack
row the CPU kernels index, with no arena-row stop and no `segment_into`
memcpy.

The whole claim is "same bytes, one fewer copy", so the load-bearing test is
byte equality against the copy path. The rest pin the refusals — a geometry
that cannot scatter, and a tier whose own buffer is now never filled.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from nvme_arena import load_index  # noqa: E402
from nvme_residency import ColdTier  # noqa: E402

torch = pytest.importorskip("torch")

from cold_cpu_view import ColdCpuView, scatter_layout  # noqa: E402


# An ALIGNED fixture of our own, following the precedent that scatter tests
# cannot use the shared toy arena: its segment lengths are not multiples of
# 4096, so a scattering read there is illegal by construction and every
# interesting assertion would skip. N=256, K=512 makes packed 256*256=65536 B
# and scale 256*16=4096 B — both whole multiples of the 4096 align.
AN, AK = 256, 512
AKINDS = ("w.weight_packed", "w.weight_scale")
ASHAPES = {"w.weight_packed": (AN, AK // 2), "w.weight_scale": (AN, AK // 32)}
ATEMPLATE = "model.layers.{layer}.experts.{expert}.{kind}"
AL, AE = 2, 3


def _st_bytes(tensors):
    import json
    import struct
    hdr, blobs, off = {}, [], 0
    for name, (data, shape) in tensors.items():
        hdr[name] = {"dtype": "U8", "shape": list(shape),
                     "data_offsets": [off, off + len(data)]}
        blobs.append(data)
        off += len(data)
    hj = json.dumps(hdr).encode()
    return struct.pack("<Q", len(hj)) + hj + b"".join(blobs)


@pytest.fixture()
def arena(tmp_path):
    import json

    from nvme_arena import bake_expert_tensors
    g = torch.Generator().manual_seed(7)
    shard, wm, ground = {}, {}, {}
    for lay in range(AL):
        for e in range(AE):
            for kind in AKINDS:
                t = torch.randint(0, 256, ASHAPES[kind], generator=g,
                                  dtype=torch.uint8)
                name = ATEMPLATE.format(layer=lay, expert=e, kind=kind)
                ground[name] = t
                shard[name] = (t.numpy().tobytes(), ASHAPES[kind])
                wm[name] = "a.safetensors"
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "a.safetensors").write_bytes(_st_bytes(shard))
    (snap / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": wm}))
    path = str(tmp_path / "aligned.arena")
    bake_expert_tensors(str(snap), path, name_template=ATEMPLATE,
                        kinds=AKINDS, align=4096, log=lambda *a: None)
    return path, load_index(path)


def _sufs(index, n=2):
    return [g["suffix"] for g in index["segments"]][:n]


def _copy_view(path, index, sufs, rows=AL * AE):
    t = ColdTier(path, hot_rows=rows, pinned=False, index=index)
    return t, ColdCpuView(t, index, sufs)


def _direct_view(path, index, sufs, rows=AL * AE):
    holder = {}

    def landing(layer, expert, slot):
        return holder["v"].landing(layer, expert, slot)

    t = ColdTier(path, hot_rows=rows, pinned=False, index=index,
                 landing=landing)
    holder["v"] = ColdCpuView(t, index, sufs, direct=True)
    return t, holder["v"]


def _scatterable(index):
    return scatter_layout(index, [g["suffix"] for g in index["segments"]]) is not None


# ------------------------------------------------------------ the claim --

def test_direct_scatter_bytes_equal_the_copy_path(arena):
    """Same bytes, one fewer copy. If this ever needs a tolerance, the
    scatter is landing segments in the wrong places."""
    path, index = arena
    if not _scatterable(index):
        pytest.skip("toy arena geometry is not scatterable (by design)")
    sufs = _sufs(index)
    tc, vc = _copy_view(path, index, sufs)
    td, vd = _direct_view(path, index, sufs)
    try:
        for lay in range(AL):
            sc = vc.ensure(lay, range(AE))
            sd = vd.ensure(lay, range(AE))
            assert sc == sd, "residency decisions must not depend on landing"
            for s in sufs:
                assert torch.equal(vc.stack(s)[list(sc)], vd.stack(s)[list(sd)]), (
                    f"layer {lay} segment {s}: direct scatter differs from copy")
    finally:
        tc.close()
        td.close()


def test_direct_does_one_read_and_no_relayout(arena):
    path, index = arena
    if not _scatterable(index):
        pytest.skip("geometry not scatterable")
    td, vd = _direct_view(path, index, _sufs(index), rows=4)
    try:
        vd.ensure(0, [0, 1])
        r1 = td.reader.traffic()["reads"]
        vd.ensure(0, [0, 1])                     # pure hit
        assert td.reader.traffic()["reads"] == r1
        assert vd.stats()["materializations"] == 2
        assert vd.stats()["view_hits"] == 2
    finally:
        td.close()


# --------------------------------------------------------- the refusals --

def test_external_landing_tier_refuses_to_serve_its_own_bytes(arena):
    """Its buffer is never filled, so row()/pinned_tensor()/buffer_ptr must
    refuse rather than hand back uninitialized memory."""
    path, index = arena
    if not _scatterable(index):
        pytest.skip("geometry not scatterable")
    td, vd = _direct_view(path, index, _sufs(index), rows=4)
    try:
        vd.ensure(0, [0])
        with pytest.raises(RuntimeError, match="EXTERNAL landing|external-landing"):
            td.row(0, 0)
        with pytest.raises(RuntimeError, match="external-landing"):
            td.buffer_ptr
    finally:
        td.close()


def test_landing_requires_direct(arena):
    path, index = arena
    t = ColdTier(path, hot_rows=4, pinned=False, index=index)
    try:
        v = ColdCpuView(t, index, _sufs(index))
        with pytest.raises(RuntimeError, match="requires direct=True"):
            v.landing(0, 0, 0)
    finally:
        t.close()


def test_direct_refuses_a_cast(arena):
    """The kernel DMAs the segment's own bytes; a widening conversion has
    nowhere to happen."""
    path, index = arena
    sufs = _sufs(index, 1)
    t = ColdTier(path, hot_rows=4, pinned=False, index=index)
    try:
        with pytest.raises(ValueError, match="cannot cast"):
            ColdCpuView(t, index, sufs, direct=True,
                        casts={sufs[0]: torch.float32})
    finally:
        t.close()


def test_unscatterable_geometry_is_a_named_refusal(arena):
    """A silent fallback to the copy path would hide a lost optimization;
    a silent scatter on bad geometry would EINVAL far from its cause."""
    path, index = arena
    bad = dict(index)
    bad["segments"] = [dict(g) for g in index["segments"]]
    bad["segments"][0] = dict(bad["segments"][0],
                              length=bad["segments"][0]["length"] + 1)
    assert scatter_layout(bad, [g["suffix"] for g in bad["segments"]]) is None
    t = ColdTier(path, hot_rows=4, pinned=False, index=index)
    try:
        with pytest.raises(ValueError, match="cannot scatter"):
            ColdCpuView(t, bad, _sufs(bad), direct=True)
    finally:
        t.close()


def test_layout_covers_exactly_one_row(arena):
    path, index = arena
    plan = scatter_layout(index, [g["suffix"] for g in index["segments"]])
    if plan is None:
        pytest.skip("geometry not scatterable")
    assert sum(ln for _, ln in plan) == index["row_stride"]


def test_unmaterialized_segments_become_scratch(arena):
    """A view holding only some segments still reads a whole row, so the
    others must be absorbed rather than leaving a hole."""
    path, index = arena
    allsuf = [g["suffix"] for g in index["segments"]]
    if scatter_layout(index, allsuf) is None:
        pytest.skip("geometry not scatterable")
    one = allsuf[:1]
    plan = scatter_layout(index, one)
    assert sum(ln for _, ln in plan) == index["row_stride"]
    assert {s for s, _ in plan if s is not None} == set(one)


# ----------------------------------------------------- late attachment --

def test_attach_landing_closes_the_mutual_reference(arena):
    """The tier needs the view's callback and the view needs the tier, so
    one is built first and the loop closes here."""
    path, index = arena
    t = ColdTier(path, hot_rows=AL * AE, pinned=False, index=index)
    v = ColdCpuView(t, index, _sufs(index), direct=True)
    t.attach_landing(v.landing)
    try:
        slots = v.ensure(0, range(AE))
        assert len(slots) == AE
        ref_t, ref_v = _copy_view(path, index, _sufs(index))
        try:
            rs = ref_v.ensure(0, range(AE))
            for s in _sufs(index):
                assert torch.equal(v.stack(s)[list(slots)],
                                   ref_v.stack(s)[list(rs)])
        finally:
            ref_t.close()
    finally:
        t.close()


def test_attach_landing_after_a_fill_is_refused(arena):
    """Rows already in the tier's own buffer would become unreachable the
    moment the landing redirects — two meanings of 'resident'."""
    path, index = arena
    t = ColdTier(path, hot_rows=4, pinned=False, index=index)
    v = ColdCpuView(t, index, _sufs(index), direct=True)
    try:
        t.ensure(0, [0])
        with pytest.raises(RuntimeError, match="after 1 request"):
            t.attach_landing(v.landing)
    finally:
        t.close()
