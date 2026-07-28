# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""N1 contract tests: bake a toy checkpoint, verify every hash, corrupt one
byte and confirm the verifier names it, round-trip the index. Pure stdlib —
no torch, no GPU, runs on CPU CI (same class as test_gather_guard)."""
import json
import os
import random
import struct
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from nvme_arena import (bake, load_index, row_offset, verify,  # noqa: E402
                        resolve_weight_map, discover_layers)
from mxfp4_loader import EXPERT_SUFFIXES, _read_st_header  # noqa: E402

L, E = 3, 4
N1, HALF1, NB1 = 8, 16, 4          # gate_up: blocks [E,N1,HALF1], scales [E,N1,NB1]
N2, HALF2, NB2 = 6, 8, 2           # down


def _st_bytes(tensors: dict) -> bytes:
    """Minimal safetensors writer: u64 LE header length + JSON header +
    contiguous data section (the format _read_st_header parses)."""
    hdr, blobs, off = {}, [], 0
    for name, (shape, data) in tensors.items():
        hdr[name] = {"dtype": "U8", "shape": list(shape),
                     "data_offsets": [off, off + len(data)]}
        blobs.append(data)
        off += len(data)
    hj = json.dumps(hdr).encode()
    return struct.pack("<Q", len(hj)) + hj + b"".join(blobs)


def _expert_bytes(rng, e_shape):
    n = 1
    for s in e_shape:
        n *= s
    return bytes(rng.randrange(256) for _ in range(n))


def make_snapshot(root, sharded=True, seed=7):
    """Toy gpt-oss-shaped snapshot: L layers x 4 expert tensors, random bytes.
    Returns {tensor_name: raw_bytes} for independent re-derivation."""
    rng = random.Random(seed)
    shapes = {
        EXPERT_SUFFIXES[0]: (N1, HALF1),   # gate_up blocks
        EXPERT_SUFFIXES[1]: (N1, NB1),     # gate_up scales
        EXPERT_SUFFIXES[2]: (N2, HALF2),   # down blocks
        EXPERT_SUFFIXES[3]: (N2, NB2),     # down scales
    }
    ground = {}
    shards, weight_map = {}, {}
    for lay in range(L):
        shard = f"model-{lay % 2}.safetensors" if sharded else "model.safetensors"
        shards.setdefault(shard, {})
        for suf, es in shapes.items():
            name = f"model.layers.{lay}.{suf}"
            per = [_expert_bytes(rng, es) for _ in range(E)]
            ground[name] = per
            shards[shard][name] = ((E,) + es, b"".join(per))
            weight_map[name] = shard
    os.makedirs(root, exist_ok=True)
    for shard, tensors in shards.items():
        with open(os.path.join(root, shard), "wb") as f:
            f.write(_st_bytes(tensors))
    if sharded:
        with open(os.path.join(root, "model.safetensors.index.json"), "w") as f:
            json.dump({"weight_map": weight_map}, f)
    return ground


@pytest.fixture()
def baked(tmp_path):
    snap = tmp_path / "snap"
    ground = make_snapshot(str(snap))
    arena = str(tmp_path / "toy.arena")
    bake(str(snap), arena, align=4096, log=lambda *a: None)
    return snap, arena, ground


def test_bake_verify_ok(baked):
    snap, arena, _ = baked
    rep = verify(arena, log=lambda *a: None)
    assert rep["ok"] and rep["rows_checked"] == L * E
    rep2 = verify(arena, against_source=str(snap), log=lambda *a: None)
    assert rep2["ok"]


def test_corrupt_one_byte_caught(baked):
    _snap, arena, _ = baked
    idx = load_index(arena)
    # flip one byte inside layer 1, expert 2's down-blocks segment
    seg = next(g for g in idx["segments"] if g["suffix"] == EXPERT_SUFFIXES[2])
    pos = row_offset(idx, 1, 2) + seg["seg_off"] + 5
    with open(arena, "r+b") as f:
        f.seek(pos)
        b = f.read(1)
        f.seek(pos)
        f.write(bytes([b[0] ^ 0xFF]))
    rep = verify(arena, log=lambda *a: None)
    assert not rep["ok"]
    assert (1, 2, EXPERT_SUFFIXES[2], "arena") in [tuple(x) for x in rep["failures"]]
    # and ONLY that segment fails
    assert len(rep["failures"]) == 1


def test_source_edit_caught(baked):
    snap, arena, _ = baked
    # arena intact, source tampered: --against-source must fail
    idx = load_index(arena)
    man = json.load(open(arena + ".manifest.json"))
    row = next(r for r in man["rows"] if r["layer"] == 0 and r["expert"] == 0)
    seg = row["segments"][0]
    spath = os.path.join(str(snap), seg["source_file"])
    with open(spath, "r+b") as f:
        f.seek(seg["source_range"][0])
        b = f.read(1)
        f.seek(seg["source_range"][0])
        f.write(bytes([b[0] ^ 0x01]))
    assert verify(arena, log=lambda *a: None)["ok"]          # arena alone fine
    rep = verify(arena, against_source=str(snap), log=lambda *a: None)
    assert not rep["ok"]
    assert (0, 0, seg["suffix"], "source") in [tuple(x) for x in rep["failures"]]


def test_index_roundtrip_and_alignment(baked):
    _snap, arena, _ = baked
    idx = load_index(arena)
    assert idx["n_layers"] == L and idx["n_experts_per_layer"] == E
    seen = set()
    for lay, e, off in idx["rows"]:
        assert off % idx["align"] == 0, "row offset not block-aligned"
        assert off == row_offset(idx, lay, e)
        seen.add((lay, e))
    assert seen == {(lay, e) for lay in range(L) for e in range(E)}
    # geometry reconstructs the row: last segment must end within row_bytes
    last = idx["segments"][-1]
    assert last["seg_off"] + last["length"] <= idx["row_bytes"] <= idx["row_stride"]
    # arena file spans exactly rows * stride (padded, no EOF short-read zone)
    assert os.path.getsize(arena) == L * E * idx["row_stride"]
    assert idx["arena_bytes"] == L * E * idx["row_stride"]


def test_relocation_preserves_bytes(baked):
    """The claim itself: each arena segment is byte-identical to the expert's
    slice of the SOURCE tensor, independently re-derived from ground truth."""
    _snap, arena, ground = baked
    idx = load_index(arena)
    with open(arena, "rb") as f:
        for lay in range(L):
            for e in range(E):
                base = row_offset(idx, lay, e)
                for g in idx["segments"]:
                    f.seek(base + g["seg_off"])
                    got = f.read(g["length"])
                    want = ground[f"model.layers.{lay}.{g['suffix']}"][e]
                    assert got == want, (lay, e, g["suffix"])


def test_aligned_row_not_clobbered(tmp_path):
    """Regression: when row_bytes == row_stride (a row already a multiple of
    `align`), the old per-layer `seek(arena_off-1); write(0)` overwrote the
    last data byte of each layer's last expert. Baking with align=8 forces
    that condition (row_bytes is always 8-aligned), so self-verify must still
    pass and the byte must survive."""
    snap = tmp_path / "snap"
    ground = make_snapshot(str(snap))
    arena = str(tmp_path / "aligned.arena")
    bake(str(snap), arena, align=8, log=lambda *a: None)
    idx = load_index(arena)
    assert idx["row_bytes"] == idx["row_stride"]      # the clobber condition
    assert verify(arena, log=lambda *a: None)["ok"]
    assert verify(arena, against_source=str(snap), log=lambda *a: None)["ok"]
    # the last expert of the last layer, last segment, last byte, intact
    lay, e = L - 1, E - 1
    g = idx["segments"][-1]
    with open(arena, "rb") as f:
        f.seek(row_offset(idx, lay, e) + g["seg_off"] + g["length"] - 1)
        got = f.read(1)
    want = ground[f"model.layers.{lay}.{g['suffix']}"][e][-1:]
    assert got == want


def test_single_file_snapshot(tmp_path):
    snap = tmp_path / "snap1"
    make_snapshot(str(snap), sharded=False)
    wm, files = resolve_weight_map(str(snap))
    assert files == ["model.safetensors"]
    assert discover_layers(wm, "model.layers", EXPERT_SUFFIXES) == list(range(L))
    arena = str(tmp_path / "toy1.arena")
    bake(str(snap), arena, log=lambda *a: None)
    assert verify(arena, against_source=str(snap), log=lambda *a: None)["ok"]


def test_limit_experts_partial_bake(tmp_path):
    snap = tmp_path / "snap2"
    make_snapshot(str(snap))
    arena = str(tmp_path / "toy2.arena")
    bake(str(snap), arena, limit_experts=2, log=lambda *a: None)
    idx = load_index(arena)
    assert len(idx["rows"]) == L * 2
    assert verify(arena, log=lambda *a: None)["ok"]


def test_loader_importable_without_torch():
    """The bake path must not require torch: simulate its absence and
    re-import the two modules the CLI touches."""
    import importlib
    import mxfp4_loader
    import nvme_arena
    saved = {k: sys.modules.pop(k) for k in list(sys.modules)
             if k == "torch" or k.startswith("torch.")}
    sys.modules["torch"] = None  # any import attempt raises ImportError
    try:
        importlib.reload(mxfp4_loader)
        importlib.reload(nvme_arena)
    finally:
        del sys.modules["torch"]
        sys.modules.update(saved)
        importlib.reload(mxfp4_loader)
        importlib.reload(nvme_arena)
