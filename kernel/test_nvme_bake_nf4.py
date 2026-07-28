# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""nf4 quantize-bake contract tests, CPU-only via an injected mock
quantizer: geometry, two-hop manifest (arena self-hash + source hashes),
verifier semantics for both hops, corruption naming, and tier readability
of a quantize-baked arena. The REAL bnb quantizer is exercised on-pod (a
smoke precedes the big bake there)."""
import hashlib
import json
import os
import struct
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))

from nvme_arena import load_index, row_offset, verify  # noqa: E402
from nvme_bake_nf4 import bake_nf4  # noqa: E402

L, E, I, H = 2, 3, 64, 128   # H%64==0, I%2==0 — nf4 geometry constraints


def _st_bytes(tensors):
    hdr, blobs, off = {}, [], 0
    for name, t in tensors.items():
        raw = t.contiguous().view(torch.uint8).numpy().tobytes()
        hdr[name] = {"dtype": "BF16", "shape": list(t.shape),
                     "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    hj = json.dumps(hdr).encode()
    return struct.pack("<Q", len(hj)) + hj + b"".join(blobs)


def make_snapshot(root, seed=11):
    g = torch.Generator().manual_seed(seed)
    tensors, weight_map = {}, {}
    for lay in range(L):
        for e in range(E):
            base = f"model.layers.{lay}.mlp.experts.{e}."
            tensors[base + "gate_proj.weight"] = torch.randn(I, H, generator=g).bfloat16()
            tensors[base + "up_proj.weight"] = torch.randn(I, H, generator=g).bfloat16()
            tensors[base + "down_proj.weight"] = torch.randn(H, I, generator=g).bfloat16()
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "model.safetensors"), "wb") as f:
        f.write(_st_bytes(tensors))
    with open(os.path.join(root, "model.safetensors.index.json"), "w") as f:
        json.dump({"weight_map": {k: "model.safetensors" for k in tensors}}, f)
    return tensors


def mock_quantize(w):
    """Deterministic stand-in with the real output SHAPES: packed u8
    [N, K/2] derived from the bytes, absmax f32 [N, K/64]."""
    N, K = w.shape
    b = w.contiguous().view(torch.uint8).reshape(N, K * 2)
    packed = (b[:, ::4] ^ b[:, 1::4])[:, : K // 2].contiguous()
    absmax = w.float().abs().reshape(N, K // 64, 64).amax(-1).contiguous()
    return packed, absmax


@pytest.fixture()
def baked(tmp_path):
    snap = tmp_path / "snap"
    tensors = make_snapshot(str(snap))
    arena = str(tmp_path / "toy_nf4.arena")
    bake_nf4(str(snap), arena, quantize_fn=mock_quantize,
             log=lambda *a: None)
    return snap, arena, tensors


def test_geometry_and_roundtrip(baked):
    _snap, arena, _ = baked
    idx = load_index(arena)
    assert idx["bake_mode"] == "nf4-quantize"
    assert idx["n_layers"] == L and idx["n_experts_per_layer"] == E
    want = [2 * I * (H // 2), 2 * I * (H // 64) * 4, H * (I // 2),
            H * (I // 64) * 4]
    assert [g["length"] for g in idx["segments"]] == want
    for lay, e, off in idx["rows"]:
        assert off % idx["align"] == 0 and off == row_offset(idx, lay, e)
    assert os.path.getsize(arena) == L * E * idx["row_stride"]


def test_arena_bytes_match_mock_quantizer(baked):
    """The rows hold exactly what the (mock) quantizer produced from the
    exact shipped tensors — re-derived independently."""
    _snap, arena, tensors = baked
    idx = load_index(arena)
    with open(arena, "rb") as f:
        for lay in range(L):
            for e in range(E):
                base = f"model.layers.{lay}.mlp.experts.{e}."
                gu_b, gu_a = mock_quantize(torch.cat(
                    [tensors[base + "gate_proj.weight"],
                     tensors[base + "up_proj.weight"]], 0))
                dn_b, dn_a = mock_quantize(tensors[base + "down_proj.weight"])
                expect = [gu_b.view(torch.uint8), gu_a.view(torch.uint8),
                          dn_b.view(torch.uint8), dn_a.view(torch.uint8)]
                for g, exp in zip(idx["segments"], expect):
                    f.seek(row_offset(idx, lay, e) + g["seg_off"])
                    got = f.read(g["length"])
                    assert got == exp.contiguous().numpy().tobytes(), \
                        (lay, e, g["suffix"])


def test_two_hop_verify(baked):
    snap, arena, _ = baked
    assert verify(arena, log=lambda *a: None)["ok"]
    assert verify(arena, against_source=str(snap), log=lambda *a: None)["ok"]


def test_arena_corruption_named(baked):
    _snap, arena, _ = baked
    idx = load_index(arena)
    seg = idx["segments"][1]           # gu_absmax
    pos = row_offset(idx, 1, 2) + seg["seg_off"] + 3
    with open(arena, "r+b") as f:
        f.seek(pos)
        b = f.read(1)
        f.seek(pos)
        f.write(bytes([b[0] ^ 0x40]))
    rep = verify(arena, log=lambda *a: None)
    assert not rep["ok"]
    assert [tuple(x) for x in rep["failures"]] == [(1, 2, seg["suffix"], "arena")]


def test_source_tamper_caught_via_sources(baked):
    snap, arena, _ = baked
    man = json.load(open(arena + ".manifest.json"))
    src = man["rows"][0]["segments"][0]["sources"][0]
    spath = os.path.join(str(snap), src["source_file"])
    with open(spath, "r+b") as f:
        f.seek(src["source_range"][0] + 7)
        b = f.read(1)
        f.seek(src["source_range"][0] + 7)
        f.write(bytes([b[0] ^ 0x01]))
    assert verify(arena, log=lambda *a: None)["ok"]          # arena intact
    rep = verify(arena, against_source=str(snap), log=lambda *a: None)
    assert not rep["ok"] and rep["failures"][0][3] == "source"


def test_quantizer_record_present(baked):
    _snap, arena, _ = baked
    man = json.load(open(arena + ".manifest.json"))
    q = man["quantizer"]
    assert q["quant_type"] == "nf4" and q["blocksize"] == 64


def test_aligned_row_not_clobbered(tmp_path):
    """Regression (the on-pod 235B smoke catch): Qwen3-235B's NF4 row is
    exactly 4096-aligned, so row_bytes == row_stride and the old per-layer
    pad write clobbered expert E-1's down_absmax last byte. align=8 forces
    the same condition on the toy."""
    snap = tmp_path / "snap"
    make_snapshot(str(snap))
    arena = str(tmp_path / "aligned_nf4.arena")
    bake_nf4(str(snap), arena, align=8, quantize_fn=mock_quantize,
             log=lambda *a: None)
    idx = load_index(arena)
    assert idx["row_bytes"] == idx["row_stride"]
    assert verify(arena, against_source=str(snap), log=lambda *a: None)["ok"]
