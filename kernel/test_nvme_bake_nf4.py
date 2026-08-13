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


# ----------------------- source="fp8": block-scaled FP8 ----------------------
# DeepSeek-V4-Flash ships MXFP4 experts (137 GiB); V4-Flash-Base ships block-scaled FP8
# (258 GiB) under the SAME tensor names. `source="mxfp4"` cannot read the latter, so
# V4-Base could not be baked at all. Two things differ and both are silent if crossed:
# FP8's on-disk shape is already logical (no nibble packing), and its scale is an F32
# per [128,128] tile rather than an e8m0 byte per 32 elements.
FP8_I, FP8_H = 64, 128
FP8_BLOCK = 32          # small tile so the synthetic expert stays tiny


def make_fp8_snapshot(root, seed=7, scale_dtype="F32"):
    g = torch.Generator().manual_seed(seed)
    t = {}
    for e in range(2):
        base = f"model.layers.0.mlp.experts.{e}."
        for proj, (rows, k) in (("w1", (FP8_I, FP8_H)), ("w3", (FP8_I, FP8_H)),
                                ("w2", (FP8_H, FP8_I))):
            w = (torch.randn(rows, k, generator=g) * 0.1).to(torch.float8_e4m3fn)
            sc = torch.rand(rows // FP8_BLOCK, k // FP8_BLOCK, generator=g) * 0.5 + 0.5
            t[base + proj + ".weight"] = (w.view(torch.uint8), "F8_E4M3")
            t[base + proj + ".scale"] = (sc, scale_dtype)
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "model.safetensors"), "wb") as f:
        f.write(_st_bytes_typed(t))
    with open(os.path.join(root, "model.safetensors.index.json"), "w") as f:
        json.dump({"weight_map": {k: "model.safetensors" for k in t}}, f)
    return t


def _bake_fp8(tmp_path, **kw):
    from nvme_bake_nf4 import PROJ_W123
    snap = tmp_path / "fp8snap"
    make_fp8_snapshot(str(snap), **kw)
    arena = str(tmp_path / "fp8.arena")
    bake_nf4(str(snap), arena, quantize_fn=mock_quantize, log=lambda *a: None,
             proj=PROJ_W123, source="fp8")
    return snap, arena


def test_fp8_source_bakes_and_geometry_is_not_doubled(tmp_path):
    """The on-disk FP8 shape IS the logical shape. MXFP4 halves K by packing two nibbles
    per byte, so the mxfp4 path doubles it back; doing that here would describe a matrix
    twice as wide as the model has, and every downstream shape would still 'look' valid."""
    snap, arena = _bake_fp8(tmp_path)
    idx = json.load(open(arena + ".index.json"))
    seg = {s["suffix"]: s for s in idx["segments"]}
    assert seg["nf4.gate_up_blocks"]["shape_per_expert"] == [2 * FP8_I, FP8_H // 2]
    assert seg["nf4.down_blocks"]["shape_per_expert"] == [FP8_H, FP8_I // 2]
    assert idx["n_experts_per_layer"] == 2
    assert verify(arena, against_source=str(snap), log=lambda *a: None)["ok"]


def test_fp8_reader_applies_block_scales(tmp_path):
    """The F32 scale is the multiplier for a whole [block, block] tile and is applied
    directly -- no 2**(x-127). Reconstruct independently and compare."""
    from nvme_bake_nf4 import _Shards
    snap = tmp_path / "fp8snap"
    t = make_fp8_snapshot(str(snap))          # compare against what we WROTE: the test
    sh = _Shards(str(snap))                   # owns the bytes, so it needs no reader
    got, _src = sh.read_fp8("model.layers.0.mlp.experts.0.w1", block=(FP8_BLOCK, FP8_BLOCK))
    w = t["model.layers.0.mlp.experts.0.w1.weight"][0].view(torch.float8_e4m3fn).float()
    sc = t["model.layers.0.mlp.experts.0.w1.scale"][0]
    want = w * sc.repeat_interleave(FP8_BLOCK, 0).repeat_interleave(FP8_BLOCK, 1)
    assert got.shape == want.shape == (FP8_I, FP8_H)
    assert torch.allclose(got.float(), want.to(torch.bfloat16).float(), atol=0, rtol=0)


def test_fp8_reader_rejects_an_mxfp4_scale(tmp_path):
    """An e8m0 BYTE scale means the checkpoint is MXFP4, not FP8. Crossing the two reads
    correct-shaped nonsense, so it must raise."""
    from nvme_bake_nf4 import _Shards
    snap = tmp_path / "badsnap"
    make_fp8_snapshot(str(snap), scale_dtype="F8_E8M0")
    with pytest.raises(ValueError, match="F32 block scales|expected F32"):
        _Shards(str(snap)).read_fp8("model.layers.0.mlp.experts.0.w1",
                                    block=(FP8_BLOCK, FP8_BLOCK))


def test_unknown_source_is_rejected(tmp_path):
    snap = tmp_path / "s"
    make_fp8_snapshot(str(snap))
    with pytest.raises(ValueError, match="source must be"):
        bake_nf4(str(snap), str(tmp_path / "a.arena"), source="int4",
                 quantize_fn=mock_quantize, log=lambda *a: None)


def test_mxfp4_reader_rejects_an_fp8_checkpoint(tmp_path):
    """The mirror of `test_fp8_reader_rejects_an_mxfp4_scale`. Both formats spell their
    tensors `.weight`/`.scale` on DeepSeek-V4, so `source=` is the ONLY thing separating
    them. Without a guard this died later on an opaque reshape -- the F32 scale carries
    4x the bytes its shape implies -- which reads as a corrupt checkpoint rather than the
    wrong flag. The message must name the fix."""
    from nvme_bake_nf4 import _Shards
    snap = tmp_path / "fp8forxmx"
    make_fp8_snapshot(str(snap))
    with pytest.raises(ValueError, match="source='fp8'"):
        _Shards(str(snap)).read_mxfp4("model.layers.0.mlp.experts.0.w1")


def test_mxfp4_reader_accepts_both_dtype_SPELLINGS(tmp_path):
    """V4 labels these I8/F8_E8M0; K3 labels both U8 for byte-identical content. The
    guard admits a SET for exactly this reason -- an equality check like read_fp8's
    would reject whichever family it was not written against."""
    from nvme_bake_nf4 import _MXFP4_BYTE_DTYPES
    assert {"U8", "I8", "F8_E8M0"} <= set(_MXFP4_BYTE_DTYPES)


# ------------------------------------------------- discovery diagnostics --
def _wm(names):
    """Minimal stand-in for _Shards: discovery only reads `.wm`."""
    class S:
        wm = {n: "shard0" for n in names}
    return S()


def test_no_experts_names_what_it_searched_for():
    """`max() arg is an empty sequence` names none of the three things that
    decide the match. This exact miss has cost a diagnosis twice -- Kimi K3
    spelling weights `.weight_packed`, Gemma-4 nesting under
    `model.language_model.layers` -- so the error has to be self-explaining."""
    from nvme_bake_nf4 import _explain_no_experts
    sh = _wm(["model.layers.0.mlp.experts.0.gate_proj.weight_packed"])
    with pytest.raises(ValueError) as ei:
        _explain_no_experts(sh, "model.layers", ".mlp.experts.", "gate_proj.weight")
    msg = str(ei.value)
    assert "model.layers." in msg and ".mlp.experts." in msg and "gate_proj.weight" in msg
    assert "weight_packed" in msg, "must show a near-miss key from the checkpoint"


def test_fused_expert_layout_is_reported_THROUGH_bake_nf4(tmp_path):
    """Route test, not a fixture test.

    The first version of this called _explain_no_experts directly with a
    hand-built marker and a pre-filled `unindexed` list -- and so passed while
    the branch was UNREACHABLE from bake_nf4, because the marker was always
    built as `.{moe}.experts.` and Gemma-4 hangs experts straight off the layer.
    A test that constructs the state under test cannot see that the production
    path never produces it. This one goes in through the real entry point with a
    real (if tiny) index file.
    """
    import json as _json
    names = {f"model.language_model.layers.{l}.experts.{p}": "model-00001.safetensors"
             for l in range(2) for p in ("gate_up_proj", "down_proj")}
    (tmp_path / "model.safetensors.index.json").write_text(_json.dumps({"weight_map": names}))
    from nvme_bake_nf4 import bake_nf4
    with pytest.raises(ValueError, match="FUSED"):
        bake_nf4(str(tmp_path), str(tmp_path / "out.arena"))   # DEFAULT knobs


def test_no_expert_keys_at_all_is_distinguished():
    from nvme_bake_nf4 import _explain_no_experts
    with pytest.raises(ValueError, match="NO keys containing 'expert'"):
        _explain_no_experts(_wm(["model.layers.0.self_attn.q_proj.weight"]),
                            "model.layers", ".mlp.experts.", "gate_proj.weight")


# --------------------------------------------------- fused expert layouts --
def make_fused_snapshot(root, seed=17, prefix="model.language_model.layers",
                        moe="", i=I, h=H):
    """Gemma-4 / GraniteMoe shape: ONE 3-D tensor per layer, no per-expert index.

    gate_up is [E, 2I, H] -- already concatenated, which is exactly the matrix
    the per-expert path builds with torch.cat -- and down is [E, H, I].
    """
    g = torch.Generator().manual_seed(seed)
    mid = f"{moe}." if moe else ""
    tensors = {}
    for lay in range(L):
        base = f"{prefix}.{lay}.{mid}experts."
        tensors[base + "gate_up_proj"] = torch.randn(E, 2 * i, h, generator=g).bfloat16()
        tensors[base + "down_proj"] = torch.randn(E, h, i, generator=g).bfloat16()
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "model.safetensors"), "wb") as f:
        f.write(_st_bytes(tensors))
    with open(os.path.join(root, "model.safetensors.index.json"), "w") as f:
        json.dump({"weight_map": {k: "model.safetensors" for k in tensors}}, f)
    return tensors


def test_fused_layout_bakes_through_bake_nf4(tmp_path):
    """Route test: in through bake_nf4, geometry derived from the SHAPE.

    E cannot come from the name here -- there is no per-expert index -- so this
    also pins that discovery reads it off the 3-D tensor.
    """
    snap = tmp_path / "snap"
    make_fused_snapshot(str(snap))
    arena = str(tmp_path / "fused.arena")
    bake_nf4(str(snap), arena, prefix="model.language_model.layers",
             quantize_fn=mock_quantize, log=lambda *a: None)
    idx = load_index(arena)
    assert idx["n_layers"] == L and idx["n_experts_per_layer"] == E
    assert {s["suffix"] for s in idx["segments"]} == {
        "nf4.gate_up_blocks", "nf4.gate_up_absmax",
        "nf4.down_blocks", "nf4.down_absmax"}
    # same row geometry the per-expert path produces for the same I/H
    assert dict(zip([s["suffix"] for s in idx["segments"]],
                    [s["shape_per_expert"] for s in idx["segments"]]))[
        "nf4.gate_up_blocks"] == [2 * I, H // 2]


def test_fused_rows_carry_the_right_expert(tmp_path):
    """Slab e must be expert e -- an off-by-one in the byte range would bake a
    self-consistent arena of the WRONG weights, which no hash check would catch."""
    snap = tmp_path / "snap"
    tensors = make_fused_snapshot(str(snap))
    arena = str(tmp_path / "fused.arena")
    bake_nf4(str(snap), arena, prefix="model.language_model.layers",
             quantize_fn=mock_quantize, log=lambda *a: None)
    idx = load_index(arena)
    segs = {s["suffix"]: s for s in idx["segments"]}
    raw = open(arena, "rb").read()
    gu = tensors["model.language_model.layers.0.experts.gate_up_proj"]
    for e in range(E):
        want, _ = mock_quantize(gu[e])
        s = segs["nf4.gate_up_blocks"]
        base = e * idx["row_stride"] + s["seg_off"]
        got = raw[base: base + s["length"]]
        assert got == want.contiguous().view(torch.uint8).numpy().tobytes(), f"expert {e}"


def test_fused_provenance_ranges_are_the_slab_not_the_whole_tensor(tmp_path):
    """Each source_range must cover ONE expert's slab and re-read to its sha256.

    Recording the parent tensor's whole range would still hash-verify while
    describing 128x the bytes the row actually consumed.
    """
    snap = tmp_path / "snap"
    make_fused_snapshot(str(snap))
    arena = str(tmp_path / "fused.arena")
    bake_nf4(str(snap), arena, prefix="model.language_model.layers",
             quantize_fn=mock_quantize, log=lambda *a: None)
    man = json.load(open(arena + ".manifest.json"))
    slab_gu, slab_dn = 2 * I * H * 2, H * I * 2
    seen = 0
    for row in man["rows"]:
        for seg in row["segments"]:
            for src in seg["sources"]:
                lo, hi = src["source_range"]
                assert (hi - lo) in (slab_gu, slab_dn), (seg["suffix"], hi - lo)
                with open(os.path.join(str(snap), src["source_file"]), "rb") as f:
                    f.seek(lo)
                    assert hashlib.sha256(f.read(hi - lo)).hexdigest() == src["sha256"]
                seen += 1
    assert seen == L * E * 4


def test_fused_slab_not_block_aligned_is_refused(tmp_path):
    """Per-slab and whole-stack quantization coincide only when each expert's
    numel is a multiple of the 64-element block. A checkpoint that breaks that
    must be refused, not baked into rows the loader will not reproduce."""
    snap = tmp_path / "snap"
    # 2I*H = 2*32*32 = 2048 is fine; make down's slab H*I = 32*2 = 64... use an
    # I that leaves gate_up divisible but down NOT: I=2 -> down slab 32*2=64 ok.
    # Instead shrink H so H*I is not a multiple of 64.
    make_fused_snapshot(str(snap), i=2, h=64)   # down slab = 64*2 = 128, ok
    # force the failure explicitly with a hand-rolled odd geometry
    g = torch.Generator().manual_seed(3)
    t = {"model.language_model.layers.0.experts.gate_up_proj":
         torch.randn(E, 2, 33, generator=g).bfloat16(),
         "model.language_model.layers.0.experts.down_proj":
         torch.randn(E, 33, 1, generator=g).bfloat16()}
    with open(os.path.join(str(snap), "model.safetensors"), "wb") as f:
        f.write(_st_bytes(t))
    with open(os.path.join(str(snap), "model.safetensors.index.json"), "w") as f:
        json.dump({"weight_map": {k: "model.safetensors" for k in t}}, f)
    with pytest.raises(ValueError, match="multiple of"):
        bake_nf4(str(snap), str(tmp_path / "bad.arena"),
                 prefix="model.language_model.layers",
                 quantize_fn=mock_quantize, log=lambda *a: None)
