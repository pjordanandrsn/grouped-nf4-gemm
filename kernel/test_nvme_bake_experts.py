# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Per-expert-tensor relocation bake (K3/DeepSeek lineage) contract tests:
K3-style naming, single-source segments, byte-identity, both verify hops,
corruption naming, parallel-worker equivalence. Pure stdlib."""
import json
import os
import random
import struct
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from nvme_arena import (bake_expert_tensors, load_index, row_offset,  # noqa: E402
                        verify)

L_ROUTED = [1, 2]            # layer 0 dense, like the real thing
E, KINDS = 3, ("w1.weight_packed", "w1.weight_scale", "w3.weight_packed",
               "w3.weight_scale", "w2.weight_packed", "w2.weight_scale")
TPL = "language_model.model.layers.{layer}.block_sparse_moe.experts.{expert}.{kind}"
SIZES = {"w1.weight_packed": 96, "w1.weight_scale": 8,
         "w3.weight_packed": 96, "w3.weight_scale": 8,
         "w2.weight_packed": 80, "w2.weight_scale": 6}


def _st_bytes(tensors):
    hdr, blobs, off = {}, [], 0
    for name, data in tensors.items():
        hdr[name] = {"dtype": "U8", "shape": [len(data)],
                     "data_offsets": [off, off + len(data)]}
        blobs.append(data)
        off += len(data)
    hj = json.dumps(hdr).encode()
    return struct.pack("<Q", len(hj)) + hj + b"".join(blobs)


def make_snapshot(root, seed=3):
    rng = random.Random(seed)
    ground, shards, wm = {}, {"a.safetensors": {}, "b.safetensors": {}}, {}
    for li, lay in enumerate(L_ROUTED):
        shard = ["a.safetensors", "b.safetensors"][li % 2]
        for e in range(E):
            for kind in KINDS:
                name = TPL.format(layer=lay, expert=e, kind=kind)
                data = bytes(rng.randrange(256) for _ in range(SIZES[kind]))
                ground[name] = data
                shards[shard][name] = data
                wm[name] = shard
    # non-expert noise the discovery must ignore
    shards["a.safetensors"]["language_model.model.layers.0.mlp.down_proj.weight"] = b"\x01" * 32
    wm["language_model.model.layers.0.mlp.down_proj.weight"] = "a.safetensors"
    os.makedirs(root, exist_ok=True)
    for s, tensors in shards.items():
        with open(os.path.join(root, s), "wb") as f:
            f.write(_st_bytes(tensors))
    with open(os.path.join(root, "model.safetensors.index.json"), "w") as f:
        json.dump({"weight_map": wm}, f)
    return ground


@pytest.fixture(params=[1, 4])
def baked(tmp_path, request):
    snap = tmp_path / "snap"
    ground = make_snapshot(str(snap))
    arena = str(tmp_path / "k3toy.arena")
    bake_expert_tensors(str(snap), arena, name_template=TPL, kinds=KINDS,
                        workers=request.param, log=lambda *a: None)
    return snap, arena, ground


def test_discovery_geometry_bytes(baked):
    snap, arena, ground = baked
    idx = load_index(arena)
    assert idx["bake_mode"] == "relocate-expert-tensors"
    assert idx["moe_layers"] == L_ROUTED and idx["n_experts_per_layer"] == E
    assert [g["suffix"] for g in idx["segments"]] == list(KINDS)
    with open(arena, "rb") as f:
        for lay in L_ROUTED:
            for e in range(E):
                base = row_offset(idx, lay, e)
                assert base % idx["align"] == 0
                for g in idx["segments"]:
                    f.seek(base + g["seg_off"])
                    got = f.read(g["length"])
                    want = ground[TPL.format(layer=lay, expert=e,
                                             kind=g["suffix"])]
                    assert got == want, (lay, e, g["suffix"])


def test_verify_both_hops(baked):
    snap, arena, _ = baked
    assert verify(arena, log=lambda *a: None)["ok"]
    assert verify(arena, against_source=str(snap), log=lambda *a: None)["ok"]


def test_corruption_named(baked):
    _snap, arena, _ = baked
    idx = load_index(arena)
    seg = idx["segments"][4]           # w2.weight_packed
    pos = row_offset(idx, 2, 1) + seg["seg_off"] + 2
    with open(arena, "r+b") as f:
        f.seek(pos)
        b = f.read(1)
        f.seek(pos)
        f.write(bytes([b[0] ^ 0x80]))
    rep = verify(arena, log=lambda *a: None)
    assert not rep["ok"]
    assert [tuple(x) for x in rep["failures"]] == [(2, 1, seg["suffix"], "arena")]


def _make_snapshot_one_shard_per_layer(root, seed=5):
    """Each routed layer's experts in a SEPARATE shard -> many distinct
    shards, so a per-shard fd leak accumulates one fd per layer."""
    rng = random.Random(seed)
    ground, wm = {}, {}
    os.makedirs(root, exist_ok=True)
    layers = list(range(8))            # 8 shards
    for lay in layers:
        shard = f"shard-{lay}.safetensors"
        tensors = {}
        for e in range(E):
            for kind in KINDS:
                name = TPL.format(layer=lay, expert=e, kind=kind)
                data = bytes(rng.randrange(256) for _ in range(SIZES[kind]))
                ground[name] = data
                tensors[name] = data
                wm[name] = shard
        with open(os.path.join(root, shard), "wb") as f:
            f.write(_st_bytes(tensors))
    with open(os.path.join(root, "model.safetensors.index.json"), "w") as f:
        json.dump({"weight_map": wm}, f)
    return ground, len(layers)


def test_fds_bounded_no_leak(tmp_path, monkeypatch):
    """Regression for Errno 24 (K3 96-shard bake): concurrent open fds must
    stay bounded regardless of shard count. workers=1 makes the count
    deterministic — the old per-thread fd cache would hold one fd per shard
    (== n_layers), the fix holds <= distinct shards per expert (1)."""
    snap = tmp_path / "snap"
    ground, n_shards = _make_snapshot_one_shard_per_layer(str(snap))
    arena = str(tmp_path / "manyshard.arena")

    real_open, real_close = os.open, os.close
    state = {"cur": 0, "max": 0}

    def tracked_open(*a, **k):
        fd = real_open(*a, **k)
        state["cur"] += 1
        state["max"] = max(state["max"], state["cur"])
        return fd

    def tracked_close(fd):
        state["cur"] -= 1
        return real_close(fd)

    monkeypatch.setattr(os, "open", tracked_open)
    monkeypatch.setattr(os, "close", tracked_close)
    bake_expert_tensors(str(snap), arena, name_template=TPL, kinds=KINDS,
                        workers=1, log=lambda *a: None)
    monkeypatch.undo()
    # dst_fd (1) + at most one source shard open at a time. The pre-fix leak
    # would peak at n_shards+1. Bound well below that.
    assert state["max"] <= 3, (state["max"], n_shards)
    assert verify(arena, against_source=str(snap), log=lambda *a: None)["ok"]


def test_workers_produce_identical_arena(tmp_path):
    snap = tmp_path / "snap"
    make_snapshot(str(snap))
    import hashlib
    digests = []
    for w in (1, 4):
        arena = str(tmp_path / f"w{w}.arena")
        bake_expert_tensors(str(snap), arena, name_template=TPL, kinds=KINDS,
                            workers=w, log=lambda *a: None)
        digests.append(hashlib.sha256(open(arena, "rb").read()).hexdigest())
    assert digests[0] == digests[1]
