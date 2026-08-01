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


# --------------- source="mxfp4" on a K3-spelled checkpoint -------------------
# 0.5.0 parameterized read_mxfp4's suffixes but left DISCOVERY and the geometry probe
# hardcoded to `.weight`, so a checkpoint spelling it `.weight_packed` (Kimi K3) matched
# zero keys and died on `max()` of an empty sequence one line into the bake. The signature
# tests passed; only running it on real K3 bytes found this.
K3_SUF = (".weight_packed", ".weight_scale")
MI, MH = 64, 128          # I, H for the synthetic expert


def _st_bytes_typed(tensors):
    """`_st_bytes` above hardcodes BF16; MXFP4 rows are U8 blocks + U8 e8m0 scales."""
    hdr, blobs, off = {}, [], 0
    for name, (t, dt) in tensors.items():
        raw = t.contiguous().view(torch.uint8).numpy().tobytes()
        hdr[name] = {"dtype": dt, "shape": list(t.shape),
                     "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    hj = json.dumps(hdr).encode()
    return struct.pack("<Q", len(hj)) + hj + b"".join(blobs)


def make_mxfp4_snapshot(root, spelling=K3_SUF, seed=5):
    """One layer, two experts, K3's key layout: a `language_model.` prefix, experts under
    `block_sparse_moe`, w1/w3/w2, and MXFP4 stored as packed nibbles + e8m0 scales."""
    g = torch.Generator().manual_seed(seed)
    t = {}
    for e in range(2):
        base = f"language_model.model.layers.1.block_sparse_moe.experts.{e}."
        for proj, (rows, k) in (("w1", (MI, MH)), ("w3", (MI, MH)), ("w2", (MH, MI))):
            blocks = torch.randint(0, 256, (rows, k // 2), generator=g, dtype=torch.uint8)
            # e8m0 exponent 127 == 2**0, so the decoded values stay in a sane range
            scales = torch.full((rows, k // 32), 127, dtype=torch.uint8)
            t[base + proj + spelling[0]] = (blocks, "U8")
            t[base + proj + spelling[1]] = (scales, "U8")
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "model.safetensors"), "wb") as f:
        f.write(_st_bytes_typed(t))
    with open(os.path.join(root, "model.safetensors.index.json"), "w") as f:
        json.dump({"weight_map": {k: "model.safetensors" for k in t}}, f)
    return t


def _bake_k3(tmp_path, **kw):
    from nvme_bake_nf4 import PROJ_W123
    snap = tmp_path / "k3snap"
    make_mxfp4_snapshot(str(snap))
    arena = str(tmp_path / "k3.arena")
    bake_nf4(str(snap), arena, quantize_fn=mock_quantize, log=lambda *a: None,
             prefix="language_model.model.layers", moe="block_sparse_moe",
             proj=PROJ_W123, source="mxfp4", **kw)
    return snap, arena


def test_mxfp4_source_bakes_a_k3_spelled_checkpoint(tmp_path):
    """Discovery, geometry and the read must all follow the source's own suffix."""
    snap, arena = _bake_k3(tmp_path, mxfp4_suffixes=K3_SUF)
    idx = json.load(open(arena + ".index.json"))
    assert idx["n_layers"] == 1 and idx["n_experts_per_layer"] == 2
    # geometry is derived from the PACKED shape, doubled in K
    seg = {s["suffix"]: s for s in idx["segments"]}
    assert seg["nf4.gate_up_blocks"]["shape_per_expert"] == [2 * MI, MH // 2], seg["nf4.gate_up_blocks"]
    assert seg["nf4.down_blocks"]["shape_per_expert"] == [MH, MI // 2], seg["nf4.down_blocks"]
    # and the provenance chain still closes against the source
    assert verify(arena, against_source=str(snap), log=lambda *a: None)["ok"]


def test_mxfp4_source_with_the_wrong_suffix_fails_loudly(tmp_path):
    """The default `.weight`/`.scale` pair is V4's. Against a K3-spelled checkpoint it
    matches nothing — that must raise, not bake an empty or half-built arena."""
    with pytest.raises(Exception):
        _bake_k3(tmp_path)          # defaults to V4's suffixes
