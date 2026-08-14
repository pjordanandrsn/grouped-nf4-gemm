# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""`preadv` scatter: the row lands in per-segment staging by DMA, no host copy.

Why this file has its OWN fixture. The shared toy arena in
`test_arena_experts.py` has 1024- and 64-byte segments, so the scatter layout is
REFUSED there and every test in that file exercises the copy fallback. Running
the existing suite against the scatter code proves nothing about it — the shapes
below are chosen so every segment length is a multiple of 4096 and the path is
actually taken. `test_scatter_is_the_path_taken` asserts that, so this cannot
quietly rot back into testing the fallback.

What the scatter is for: a CPU write to pinned memory makes the FOLLOWING H2D
~6x slower (70.5 ms vs 11.65 ms for the same 281 MB), so a host copy is charged
twice — once to make it, once as a penalty on the transfer. Letting the kernel
DMA each row straight into its segment slots removes both (#73).
"""
import json
import os
import struct
import sys

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, os.path.dirname(__file__))

from arena_experts import ArenaExpertSource, K3_KINDS, K3_TEMPLATE  # noqa: E402
from nvme_arena import bake_expert_tensors  # noqa: E402

# Chosen so EVERY segment length is a multiple of 4096 and the row needs no
# padding — the conditions the scatter requires. 128*2048/2 = 131,072 and
# 128*2048/32 = 8,192, both exact multiples of 4096.
E, L_ROUTED = 4, (1, 2)
N, K = 128, 2048
SHAPES = {"w1.weight_packed": [N, K // 2], "w1.weight_scale": [N, K // 32],
          "w3.weight_packed": [N, K // 2], "w3.weight_scale": [N, K // 32],
          "w2.weight_packed": [K, N // 2], "w2.weight_scale": [K, N // 32]}


def _st_bytes(tensors):
    hdr, blobs, off = {}, [], 0
    for name, t in tensors.items():
        raw = t.contiguous().view(torch.uint8).numpy().tobytes()
        hdr[name] = {"dtype": "U8", "shape": list(t.shape),
                     "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    hj = json.dumps(hdr).encode()
    return struct.pack("<Q", len(hj)) + hj + b"".join(blobs)


@pytest.fixture()
def aligned_arena(tmp_path):
    g = torch.Generator().manual_seed(5)
    tensors, ground = {}, {}
    for lay in L_ROUTED:
        for e in range(E):
            for kind in K3_KINDS:
                t = torch.randint(0, 255, SHAPES[kind], generator=g,
                                  dtype=torch.uint8)
                name = K3_TEMPLATE.format(layer=lay, expert=e, kind=kind)
                tensors[name] = t
                ground[name] = t
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "model.safetensors").write_bytes(_st_bytes(tensors))
    arena = str(tmp_path / "s.arena")
    bake_expert_tensors(str(snap), arena, name_template=K3_TEMPLATE,
                        kinds=K3_KINDS, align=4096, log=lambda *a: None)
    return arena, ground


def test_scatter_is_the_path_taken(aligned_arena):
    """Without this the whole file could be testing the copy fallback — which
    is exactly what the shared fixture does."""
    arena, _ = aligned_arena
    with ArenaExpertSource(arena) as src:
        assert src._scatter_layout() is not None, "layout refused on aligned segments"
        src.fetch_raw(1, [0, 2])
        assert src.last_fetch_path == "scatter"


def test_scatter_bytes_are_identical_to_the_release(aligned_arena):
    """Provenance, through the DMA path: what the caller gets is what shipped."""
    arena, ground = aligned_arena
    ids = [2, 0, 3]
    with ArenaExpertSource(arena) as src:
        raw = src.fetch_raw(1, ids)
        assert src.last_fetch_path == "scatter"
    for kind in K3_KINDS:
        got = raw[kind]
        assert tuple(got.shape) == (len(ids), *SHAPES[kind])
        for i, e in enumerate(ids):
            want = ground[K3_TEMPLATE.format(layer=1, expert=e, kind=kind)]
            assert torch.equal(got[i], want), (kind, e)


def test_scatter_and_copy_agree_bitwise(aligned_arena, monkeypatch):
    """The two paths must be indistinguishable. This is the gate that matters:
    scatter reorders WHERE the kernel writes, and an off-by-one in the iovec
    would produce a plausible tensor made of the wrong segment's bytes."""
    arena, _ = aligned_arena
    ids = [1, 3, 0]
    with ArenaExpertSource(arena) as src:
        a = {k: v.clone() for k, v in src.fetch_raw(2, ids).items()}
        assert src.last_fetch_path == "scatter"
    with ArenaExpertSource(arena) as src:
        monkeypatch.setattr(src, "_scatter_layout", lambda: None)  # force fallback
        b = src.fetch_raw(2, ids)
        assert src.last_fetch_path == "copy"
    assert set(a) == set(b)
    for k in a:
        assert torch.equal(a[k], b[k]), f"{k}: scatter and copy disagree"


def test_unaligned_segments_refuse_the_scatter(tmp_path):
    """The shared toy arena has 1024/64-byte segments. O_DIRECT would EINVAL on
    those, so the layout must REFUSE rather than try — and a refusal has to be
    visible, because a silent fallback here is a silent ~6x regression."""
    import test_arena_experts as T
    snap = tmp_path / "snap"
    T.make_snapshot(str(snap))
    arena = str(tmp_path / "u.arena")
    bake_expert_tensors(str(snap), arena, name_template=T.K3_TEMPLATE,
                        kinds=T.K3_KINDS, align=4096, log=lambda *a: None)
    with ArenaExpertSource(arena) as src:
        assert any(g["length"] % 4096 for g in src.segments.values()), \
            "fixture is aligned after all — this test proves nothing"
        assert src._scatter_layout() is None
        src.fetch_raw(1, [0])
        assert src.last_fetch_path == "copy"


def test_advance_resumes_inside_the_buffer_it_stopped_in():
    """A short `preadv` must resume mid-buffer. Resuming at the next buffer
    instead would drop or duplicate bytes and produce a plausible tensor."""
    from nvme_reader import ArenaReader
    bufs = [memoryview(bytearray(10)), memoryview(bytearray(6)),
            memoryview(bytearray(4))]
    assert [len(v) for v in ArenaReader._advance(bufs, 0)] == [10, 6, 4]
    assert [len(v) for v in ArenaReader._advance(bufs, 4)] == [6, 6, 4]   # mid first
    assert [len(v) for v in ArenaReader._advance(bufs, 10)] == [6, 4]     # exact edge
    assert [len(v) for v in ArenaReader._advance(bufs, 13)] == [3, 4]     # mid second
    assert [len(v) for v in ArenaReader._advance(bufs, 20)] == []         # all done


def test_scatter_views_must_cover_the_whole_row(aligned_arena):
    """Under-covering would leave the tail of the row unread and silently keep
    whatever was in staging from the previous fetch."""
    arena, _ = aligned_arena
    with ArenaExpertSource(arena) as src:
        short = [memoryview(bytearray(4096))]
        with pytest.raises(ValueError, match="cover"):
            src.reader.read_row_scatter(1, 0, short)
