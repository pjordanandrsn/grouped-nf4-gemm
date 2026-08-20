# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The gate for MXFP4-from-NVMe: serving the cold tail off disk must produce the
SAME answer as pinning every row in host DRAM.

Three levels, weakest assumption first, so a failure says which layer broke:

1. **Layout, pure python.** Geometry recovered from a bake's index equals the
   engine's, K3's split w1/w3 fuses byte-exactly, and a non-contiguous or
   wrong-dtype arena is REFUSED rather than served as noise. No GPU.
2. **Address arithmetic.** Cold addresses stride by ``row_stride`` and hot ones by
   ``row_bytes``, and the tier's pinned tensor starts where its buffer does.
   Getting either wrong reads mid-row and never raises. No GPU.
3. **Forward equivalence, bitwise.** A tier-backed engine's forward must be
   ``torch.equal`` to a fully-resident one — with ``hot_rows`` small enough to
   force real eviction and re-reads, asserted via the tier's own miss counter.
   Needs CUDA + triton>=3.4 (``tl.gather``); the skip is loud.

The arena is baked by RELOCATION from the same packed tensors the resident engine
is constructed from, so the bytes on disk are the bytes it would have held. Any
divergence below is the tiering, not a re-quantization artifact.
"""
import json
import os
import struct
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))

from mxfp4_residency import (  # noqa: E402
    fuse_gate_up_segments, mxfp4_geometry_from_arena)
from nvme_arena import bake_expert_tensors, load_index, verify  # noqa: E402
from nvme_residency import ColdTier  # noqa: E402

E, H, INTER, K_SLOTS = 8, 128, 128, 4
LAYER = 0
MX_BLOCK = 32
KINDS = ("mx.gate_up_blocks", "mx.gate_up_scales",
         "mx.down_blocks", "mx.down_scales")
ALPHA, LIMIT = 1.702, 7.0


def _st_bytes(tensors: dict) -> bytes:
    """Minimal safetensors writer (the format nvme_arena's header reader parses)."""
    hdr, blobs, off = {}, [], 0
    for name, (shape, dtype, data) in tensors.items():
        hdr[name] = {"dtype": dtype, "shape": list(shape),
                     "data_offsets": [off, off + len(data)]}
        blobs.append(data)
        off += len(data)
    hj = json.dumps(hdr).encode()
    return struct.pack("<Q", len(hj)) + hj + b"".join(blobs)


def _packed(seed=1689):
    """Synthetic native-MXFP4 experts: gate_up [E,2I,H], down [E,H,I]."""
    from mxfp4_pack_ref import quantize_pack_mxfp4
    g = torch.Generator().manual_seed(seed)

    def pack(w):
        E_, N_, K_ = w.shape
        B = torch.empty(E_, N_, K_ // 2, dtype=torch.uint8)
        S = torch.empty(E_, N_, K_ // MX_BLOCK, dtype=torch.uint8)
        for e in range(E_):
            b, s = quantize_pack_mxfp4(w[e])
            B[e], S[e] = b.reshape(N_, K_ // 2), s
        return B, S

    gu_b, gu_s = pack(torch.randn(E, 2 * INTER, H, generator=g) * 0.1)
    dn_b, dn_s = pack(torch.randn(E, H, INTER, generator=g) * 0.1)
    return gu_b, gu_s, dn_b, dn_s


def _bake(tmp_path, stacks, kinds=KINDS, split_gate_up=False, interleave=False,
          name="m.arena"):
    """Relocate the very tensors the resident engine will hold into an arena.

    ``interleave`` reproduces ``arena_experts.K3_KINDS`` ordering — per
    projection (w1 blocks, w1 scales, w3 blocks, ...) rather than blocks-then-
    scales. Correct for ArenaExpertSource, wrong for this engine.
    """
    gu_b, gu_s, dn_b, dn_s = stacks
    if split_gate_up:                       # K3 shape: w1 and w3 kept apart
        w1b, w3b = gu_b[:, :INTER], gu_b[:, INTER:]
        w1s, w3s = gu_s[:, :INTER], gu_s[:, INTER:]
        payload = ([w1b, w1s, w3b, w3s, dn_b, dn_s] if interleave
                   else [w1b, w3b, w1s, w3s, dn_b, dn_s])
    else:
        payload = [gu_b, gu_s, dn_b, dn_s]
    tensors = {}
    for kind, stack in zip(kinds, payload):
        for e in range(E):
            t = stack[e].contiguous().cpu()
            tensors[f"model.layers.{LAYER}.mlp.experts.{e}.{kind}"] = (
                tuple(t.shape), "U8", t.numpy().tobytes())
    snap = tmp_path / "snap"
    snap.mkdir(exist_ok=True)
    (snap / "model.safetensors").write_bytes(_st_bytes(tensors))
    path = str(tmp_path / name)
    bake_expert_tensors(
        str(snap), path,
        name_template="model.layers.{layer}.mlp.experts.{expert}.{kind}",
        kinds=kinds, align=4096, log=lambda *a: None)
    return path, load_index(path)


@pytest.fixture()
def arena(tmp_path):
    stacks = _packed()
    path, index = _bake(tmp_path, stacks)
    return stacks, path, index


# ------------------------------------------------------ 1. layout, no GPU ----
def test_arena_relocation_is_verifiable(arena):
    """The bake's own gate: arena bytes match the source byte ranges."""
    _s, path, _i = arena
    rep = verify(path, log=lambda *a: None)
    assert rep["ok"], rep["failures"]


def test_geometry_recovered_from_the_arena(arena):
    _s, _p, index = arena
    got = mxfp4_geometry_from_arena(index)
    assert got == (E, 2 * INTER, H // 2, H // MX_BLOCK,
                   H, INTER // 2, INTER // MX_BLOCK), got


def _seg(suffix, shape, length, off=0):
    return {"suffix": suffix, "shape_per_expert": list(shape),
            "length": length, "seg_off": off, "dtype": "U8"}


def _gptoss_index(three_d=True):
    """A gpt-oss-shaped index: blocks [n, nblocks, 16] as the checkpoint (and
    therefore the bake) records them, scales [n, nblocks]."""
    from mxfp4_residency import engine_segment_map          # noqa: F401
    nb, gu_n, dn_n = 90, 5760, 2880
    gb = list((gu_n, nb, 16)) if three_d else [gu_n, nb * 16]
    db = list((dn_n, nb, 16)) if three_d else [dn_n, nb * 16]
    segs, off = [], 0
    for suf, shp, ln in (("mlp.experts.gate_up_proj_blocks", gb, gu_n * nb * 16),
                         ("mlp.experts.gate_up_proj_scales", [gu_n, nb], gu_n * nb),
                         ("mlp.experts.down_proj_blocks", db, dn_n * nb * 16),
                         ("mlp.experts.down_proj_scales", [dn_n, nb], dn_n * nb)):
        segs.append(_seg(suf, shp, ln, off))
        off += ln
    return {"segments": segs, "align": 4096, "row_bytes": off,
            "n_experts_per_layer": 32, "n_layers": 24}


def test_a_gpt_oss_arena_is_accepted_with_its_checkpoint_shape():
    """bake records [n, nblocks, 16] because that is what the checkpoint says,
    and that fidelity is what makes sha256(arena)==sha256(source) mean
    anything. The engine must therefore accept it: on the unflattened shape
    the blocks-vs-scales discriminator sees nblocks on BOTH sides and the 16x
    signal disappears, which is why this used to raise 'is not [n, k]'."""
    from mxfp4_residency import engine_segment_map
    groups_3d, geo_3d = engine_segment_map(_gptoss_index(three_d=True))
    groups_2d, geo_2d = engine_segment_map(_gptoss_index(three_d=False))
    assert geo_3d == geo_2d, (geo_3d, geo_2d)
    assert groups_3d == groups_2d


def test_the_index_is_not_mutated_by_being_read():
    """The flattening happens on a copy. An arena index is provenance; a
    consumer that rewrites it in passing makes the next reader's view depend
    on who looked first."""
    from mxfp4_residency import engine_segment_map
    idx = _gptoss_index(three_d=True)
    before = [list(g["shape_per_expert"]) for g in idx["segments"]]
    engine_segment_map(idx)
    after = [list(g["shape_per_expert"]) for g in idx["segments"]]
    assert before == after, (before, after)


def test_a_shape_that_would_reinterpret_the_bytes_is_refused():
    """The flatten is checked, not assumed. Every dtype here is a packed BYTE
    dtype, so n*k must equal the segment's own per-expert length; a shape that
    fails that is not a reshape of these bytes."""
    from mxfp4_residency import engine_segment_map
    idx = _gptoss_index(three_d=True)
    idx["segments"][0]["shape_per_expert"] = [5760, 91, 16]     # one block too many
    with pytest.raises(ValueError, match="reinterpret"):
        engine_segment_map(idx)


def test_geometry_matches_the_engine_that_will_read_it(arena):
    """The runtime layout gate: offsets/lengths/row_bytes the bake wrote must
    equal what Mxfp4PipelinedGptOss computes, or every segment is misread."""
    from mxfp4_pipelined import Mxfp4PipelinedGptOss
    _s, _p, index = arena
    eng = Mxfp4PipelinedGptOss.__new__(Mxfp4PipelinedGptOss)
    _E, n1, half1, nb1, n2, half2, nb2 = mxfp4_geometry_from_arena(index)
    eng._init_geometry(E, n1, half1, nb1, n2, half2, nb2, k_slots=K_SLOTS,
                       device="cpu", alpha=ALPHA, limit=LIMIT,
                       compute_dtype=torch.bfloat16)
    assert eng.row_bytes == index["row_bytes"]
    assert eng.off == [g["seg_off"] for g in index["segments"]]
    assert eng.seg == [g["length"] for g in index["segments"]]


def test_split_gate_up_fuses_byte_exactly(tmp_path):
    """K3 ships w1/w3 apart; a 6-segment arena must present as the engine's 4."""
    stacks = _packed()
    kinds = ("w1.b", "w3.b", "w1.s", "w3.s", "w2.b", "w2.s")
    path, index = _bake(tmp_path, stacks, kinds=kinds, split_gate_up=True)
    assert len(index["segments"]) == 6
    assert mxfp4_geometry_from_arena(index) == (
        E, 2 * INTER, H // 2, H // MX_BLOCK, H, INTER // 2, INTER // MX_BLOCK)
    # and the fused range really is w1's bytes followed by w3's
    tier = ColdTier(path, hot_rows=2, pinned=False)
    try:
        g = fuse_gate_up_segments(index)["segments"][0]
        tier.ensure(LAYER, [3])
        row = bytes(tier.row(LAYER, 3)[g["seg_off"]:g["seg_off"] + g["length"]])
        assert row == bytes(stacks[0][3].contiguous().numpy().tobytes())
    finally:
        tier.close()


GEO = (E, 2 * INTER, H // 2, H // MX_BLOCK, H, INTER // 2, INTER // MX_BLOCK)


def test_both_k3_kinds_orders_give_the_same_geometry(tmp_path):
    """`arena_experts.K3_KINDS` (per-projection interleave — what the real
    1.446 TB arena on disk was baked in) and `K3_RESIDENCY_KINDS`
    (blocks-then-scales — what the engine reads) hold the same six tensors in a
    different order. Both must resolve to the same geometry; the difference shows
    up as a permutation to apply on gather, not as a refusal.
    """
    from arena_experts import K3_KINDS
    from mxfp4_residency import K3_RESIDENCY_KINDS, engine_segment_map

    assert set(K3_KINDS) == set(K3_RESIDENCY_KINDS), (
        "same six tensors, different order — if these sets ever diverge one of "
        "the two constants has the wrong NAMES, which is a separate bug")
    assert K3_KINDS != K3_RESIDENCY_KINDS, "the orders must actually differ"

    stacks = _packed()
    _p1, inter = _bake(tmp_path, stacks, kinds=K3_KINDS, split_gate_up=True,
                       interleave=True, name="a.arena")
    _p2, resid = _bake(tmp_path, stacks, kinds=K3_RESIDENCY_KINDS,
                       split_gate_up=True, interleave=False, name="b.arena")
    assert [g["suffix"] for g in inter["segments"]] == list(K3_KINDS)
    assert [g["suffix"] for g in resid["segments"]] == list(K3_RESIDENCY_KINDS)
    assert mxfp4_geometry_from_arena(inter) == GEO
    assert mxfp4_geometry_from_arena(resid) == GEO

    # residency order needs no permutation; the interleaved one does
    for idx, want_identity in ((resid, True), (inter, False)):
        groups, _geo = engine_segment_map(idx)
        dst, identity = 0, True
        for grp in groups:
            for s_off, ln in grp:
                identity &= (s_off == dst)
                dst += ln
        assert identity is want_identity, (want_identity, groups)


def test_shape_grouping_does_not_depend_on_tensor_names(tmp_path):
    """Grouping is by SHAPE. A release that renames its tensors again — three
    times since K2 — still maps correctly."""
    from mxfp4_residency import engine_segment_map
    stacks = _packed()
    weird = ("zz.packed", "aa.packed", "mm.scale", "bb.scale",
             "qq.packed", "cc.scale")
    _p, index = _bake(tmp_path, stacks, kinds=weird, split_gate_up=True)
    _groups, geo = engine_segment_map(index)
    assert geo == GEO


def test_noncontiguous_split_segments_are_refused():
    """An odd-length w1 segment leaves an 8-byte-padding hole between w1 and w3;
    fusing across it would read the padding as weights."""
    idx = {"row_bytes": 0, "n_experts_per_layer": E, "segments": [
        {"suffix": "w1.b", "seg_off": 0, "length": 5, "dtype": "U8",
         "shape_per_expert": [1, 5]},
        {"suffix": "w3.b", "seg_off": 8, "length": 5, "dtype": "U8",
         "shape_per_expert": [1, 5]},
        {"suffix": "w1.s", "seg_off": 16, "length": 1, "dtype": "U8",
         "shape_per_expert": [1, 1]},
        {"suffix": "w3.s", "seg_off": 24, "length": 1, "dtype": "U8",
         "shape_per_expert": [1, 1]},
        {"suffix": "w2.b", "seg_off": 32, "length": 1, "dtype": "U8",
         "shape_per_expert": [1, 1]},
        {"suffix": "w2.s", "seg_off": 40, "length": 1, "dtype": "U8",
         "shape_per_expert": [1, 1]}]}
    with pytest.raises(ValueError, match="not contiguous"):
        fuse_gate_up_segments(idx)


def test_nf4_arena_is_refused_by_dtype(arena):
    """An F32 segment means fp32 absmax — an NF4 arena. Serving it as MXFP4
    would reinterpret scale floats as packed nibbles."""
    _s, _p, index = arena
    bad = dict(index)
    bad["segments"] = [dict(g) for g in index["segments"]]
    bad["segments"][1]["dtype"] = "F32"
    with pytest.raises(ValueError, match="NF4 arena"):
        mxfp4_geometry_from_arena(bad)


def test_shifted_offsets_are_followed_not_refused(arena):
    """A shifted seg_off used to be refused as "not the engine's layout". It is
    now simply where the bytes are, and the gather follows it — that is the whole
    point of permuting on gather. Geometry is unchanged because geometry comes
    from SHAPES, and the piece table picks up the shift."""
    from mxfp4_residency import engine_segment_map
    _s, _p, index = arena
    moved = dict(index)
    moved["segments"] = [dict(g) for g in index["segments"]]
    moved["segments"][2]["seg_off"] += 8
    assert mxfp4_geometry_from_arena(moved) == mxfp4_geometry_from_arena(index)
    groups, _geo = engine_segment_map(moved)
    assert groups[2][0][0] == index["segments"][2]["seg_off"] + 8


def test_misaligned_segment_offset_is_refused(arena):
    """The gather moves int64 words, so an offset that is not 8-byte aligned
    cannot be expressed as a word offset — refuse rather than silently truncate
    to the nearest word and read shifted nibbles."""
    from mxfp4_residency import _chunk_table
    with pytest.raises(ValueError, match="not 8-byte aligned"):
        _chunk_table([(4, 0, 64)], 2048)
    with pytest.raises(ValueError, match="not 8-byte aligned"):
        _chunk_table([(0, 0, 60)], 2048)


# ------------------------------------------- 2. address arithmetic, no GPU ----
def test_pinned_tensor_starts_where_the_buffer_does(arena):
    """alloc_landing over-allocates by one alignment unit and returns an interior
    view, so a base-relative tensor would describe different bytes than the
    buffer the reader fills — and the engine adds slot offsets to its data_ptr."""
    if not torch.cuda.is_available():
        pytest.skip("pinned memory needs CUDA")
    _s, path, index = arena
    with ColdTier(path, hot_rows=4, pinned=True) as tier:
        t = tier.pinned_tensor()
        assert t.shape == (4, tier.row_stride)
        assert t.data_ptr() == tier.buffer_ptr
        assert t.data_ptr() % index["align"] == 0


def test_cold_slots_stride_by_row_stride_not_row_bytes(arena):
    """The trap this whole module is written around: the arena pads rows to
    `align` for O_DIRECT, so slot n starts at n*row_stride. An engine striding
    its own row_bytes reads mid-row from slot 1 onward and never errors."""
    _s, path, index = arena
    assert index["row_stride"] > index["row_bytes"], "fixture must exercise padding"
    with ColdTier(path, hot_rows=4, pinned=False) as tier:
        slots = tier.ensure(LAYER, [1, 2])
        a0 = tier.buffer_ptr + slots[0] * tier.row_stride
        a1 = tier.buffer_ptr + slots[1] * tier.row_stride
        assert abs(a1 - a0) == tier.row_stride
        assert abs(a1 - a0) != tier.row_bytes


# ---------------------------------------- 3. forward equivalence, on CUDA ----
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _needs_gather_kernel():
    pytest.importorskip("triton")
    import triton.language as tl
    if not hasattr(tl, "gather"):
        import triton
        pytest.skip(f"grouped-mxfp4 gather needs triton>=3.4 for tl.gather; "
                    f"have {triton.__version__}")


def _resident(stacks, hot_ids, bias=True):
    from mxfp4_pipelined import Mxfp4PipelinedGptOss
    gu_b, gu_s, dn_b, dn_s = stacks
    g = torch.Generator().manual_seed(7)
    gub = (torch.randn(E, 2 * INTER, generator=g) * 0.05).to(torch.bfloat16)
    dnb = (torch.randn(E, H, generator=g) * 0.05).to(torch.bfloat16)
    return Mxfp4PipelinedGptOss(
        gu_b, gu_s, dn_b, dn_s, gub if bias else None, dnb if bias else None,
        hot_ids=torch.tensor(hot_ids, dtype=torch.long), k_slots=K_SLOTS,
        device="cuda", alpha=ALPHA, limit=LIMIT), (gub, dnb)


def _tiered(path, index, hot_ids, hot_rows, biases):
    from mxfp4_residency import Mxfp4NvmeResidency
    return Mxfp4NvmeResidency(
        path, LAYER, hot_ids=hot_ids, k_slots=K_SLOTS, hot_rows=hot_rows,
        gate_up_bias=biases[0] if biases else None,
        down_bias=biases[1] if biases else None,
        device="cuda", alpha=ALPHA, limit=LIMIT, index=index)


def _route(n, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(n, 1, H, dtype=torch.bfloat16, device="cuda", generator=g)
    logits = torch.randn(n, E, device="cuda", generator=g)
    sc, idx = torch.topk(torch.softmax(logits, -1), k=K_SLOTS, dim=-1)
    return x, idx, sc.to(torch.bfloat16)


@cuda
@pytest.mark.parametrize("hot_ids", [(), (0, 1, 2, 3)])
def test_forward_matches_fully_resident(arena, hot_ids):
    """The headline: identical answers, cold rows off disk. hot_rows=K_SLOTS is
    the FLOOR (one fetch's distinct cold experts), which also guarantees the
    tier thrashes and every fetch re-reads."""
    _needs_gather_kernel()
    stacks, path, index = arena
    res, biases = _resident(stacks, hot_ids)
    tie = _tiered(path, index, hot_ids, K_SLOTS, biases)
    try:
        x, idx, sc = _route(6, seed=3)
        for t in range(x.shape[0]):
            a = res.forward(x[t], idx[t:t + 1], sc[t:t + 1])
            b = tie.forward(x[t], idx[t:t + 1], sc[t:t + 1])
            assert torch.equal(a, b), (t, (a - b).abs().max().item())
        # derive what the tier MUST have done from the routes actually taken, so
        # this cannot pass by never touching disk
        routed = [[int(e) for e in idx[t]] for t in range(idx.shape[0])]
        cold = [[e for e in r if e not in hot_ids] for r in routed]
        st = tie.traffic()["tier"]
        if any(cold):
            assert st["misses"] > 0, st
        if max((len(set(c)) for c in cold), default=0) and len(
                {e for c in cold for e in c}) > K_SLOTS:
            assert st["evictions"] > 0, st
    finally:
        tie.tier.close()


@cuda
def test_forward_matches_with_no_biases(arena):
    """K3's experts are bias-free: `None` must SKIP the add, not add zeros."""
    _needs_gather_kernel()
    stacks, path, index = arena
    res, _b = _resident(stacks, (), bias=False)
    tie = _tiered(path, index, (), K_SLOTS, None)
    try:
        x, idx, sc = _route(3, seed=11)
        for t in range(x.shape[0]):
            assert torch.equal(res.forward(x[t], idx[t:t + 1], sc[t:t + 1]),
                               tie.forward(x[t], idx[t:t + 1], sc[t:t + 1]))
    finally:
        tie.tier.close()


@cuda
def test_slot_reuse_by_a_different_expert_is_not_skipped(arena):
    """THE tiering-specific correctness bug. The gather skips a device slot whose
    source ADDRESS is unchanged — sound when address <-> expert is a bijection,
    wrong under a tier, where a new expert can land at the address the previous
    one just vacated. Route to disjoint expert sets on alternating steps so tier
    slots are reused, and demand the resident answer every time."""
    _needs_gather_kernel()
    stacks, path, index = arena
    res, biases = _resident(stacks, ())
    tie = _tiered(path, index, (), K_SLOTS, biases)
    try:
        g = torch.Generator(device="cuda").manual_seed(5)
        x = torch.randn(1, H, dtype=torch.bfloat16, device="cuda", generator=g)
        sc = torch.full((1, K_SLOTS), 1.0 / K_SLOTS, dtype=torch.bfloat16,
                        device="cuda")
        halves = [torch.tensor([[0, 1, 2, 3]], device="cuda"),
                  torch.tensor([[4, 5, 6, 7]], device="cuda")]
        for step in range(6):
            idx = halves[step % 2]
            assert torch.equal(res.forward(x, idx, sc),
                               tie.forward(x, idx, sc)), f"step {step}"
        assert tie.traffic()["tier"]["evictions"] > 0
    finally:
        tie.tier.close()


@cuda
def test_repeated_ids_hit_without_a_reread(arena):
    """The have-skip must still WORK: an unchanged route re-reads nothing."""
    _needs_gather_kernel()
    stacks, path, index = arena
    res, biases = _resident(stacks, ())
    tie = _tiered(path, index, (), E, biases)
    try:
        idx = torch.tensor([[2, 3, 4, 5]], device="cuda")
        sc = torch.full((1, K_SLOTS), 0.25, dtype=torch.bfloat16, device="cuda")
        g = torch.Generator(device="cuda").manual_seed(9)
        x = torch.randn(1, H, dtype=torch.bfloat16, device="cuda", generator=g)
        tie.forward(x, idx, sc)
        reads = tie.tier.stats()["disk_reads"]
        assert torch.equal(res.forward(x, idx, sc), tie.forward(x, idx, sc))
        assert tie.tier.stats()["disk_reads"] == reads, "resident rows re-read"
    finally:
        tie.tier.close()


@cuda
def test_hot_rows_below_the_floor_raises(arena):
    """Undersizing must be a named error, not silent thrash-to-wrong-answer."""
    _needs_gather_kernel()
    stacks, path, index = arena
    res, biases = _resident(stacks, ())
    tie = _tiered(path, index, (), 2, biases)     # < K_SLOTS distinct cold rows
    try:
        idx = torch.tensor([[0, 1, 2, 3]], device="cuda")
        sc = torch.full((1, K_SLOTS), 0.25, dtype=torch.bfloat16, device="cuda")
        x = torch.zeros(1, H, dtype=torch.bfloat16, device="cuda")
        with pytest.raises(ValueError, match="exceeds hot_rows"):
            tie.forward(x, idx, sc)
    finally:
        tie.tier.close()


@cuda
def test_unpinned_tier_is_refused(arena):
    """An mmap buffer is not in the GPU's address space; the gather would fault
    or read unrelated memory rather than fail."""
    from mxfp4_residency import Mxfp4NvmeResidency
    _s, path, index = arena
    with ColdTier(path, hot_rows=4, pinned=False) as tier:
        with pytest.raises(ValueError, match="pinned=True"):
            Mxfp4NvmeResidency(path, LAYER, k_slots=K_SLOTS, tier=tier,
                               index=index)


@cuda
def test_no_all_expert_arena_is_allocated(arena):
    """The whole point: expert storage must not scale with E. The base engine
    holds [E, row_bytes] pinned; this one holds nothing but hot rows."""
    _needs_gather_kernel()
    _s, path, index = arena
    tie = _tiered(path, index, (0,), K_SLOTS, None)
    try:
        assert tie.arena is None
        assert tie.hot_stack.shape[0] == 1
        with pytest.raises(TypeError, match="1.446 TB"):
            tie._build_source(None, None, None, None, ())
    finally:
        tie.tier.close()


# ------------------------------------------------- K3's epilogue, on CUDA ----
SITU_BETA, SITU_LINEAR_BETA = 4.0, 25.0


def _situ_ref(gate, up):
    """SiTU as transcribed from K3's own modeling_kimi_linear.py::SituAndMul,
    written out here rather than imported, so this test can disagree with
    moonshot_gather instead of inheriting its answer."""
    a = SITU_BETA * torch.tanh(gate / SITU_BETA) * torch.sigmoid(gate)
    return a * (SITU_LINEAR_BETA * torch.tanh(up / SITU_LINEAR_BETA))


@cuda
def test_k3_epilogue_is_situ_on_a_clean_concat(arena):
    """K3 differs from gpt-oss in BOTH respects: clean-concat halves (not
    interleaved columns) and SiTU (not clamped-GLU). Getting the split wrong
    still produces finite numbers of the right shape."""
    _needs_gather_kernel()
    from mxfp4_residency import Mxfp4NvmeResidencyK3
    _s, path, index = arena
    tie = Mxfp4NvmeResidencyK3(path, LAYER, k_slots=K_SLOTS, hot_rows=K_SLOTS,
                               device="cuda", index=index)
    try:
        g = torch.Generator(device="cuda").manual_seed(17)
        gu = torch.randn(K_SLOTS, 2 * INTER, device="cuda", generator=g) * 3
        got = tie._glu(gu)
        want = _situ_ref(gu[..., :INTER], gu[..., INTER:])
        assert got.shape == (K_SLOTS, INTER), got.shape
        assert got.dtype == torch.bfloat16, got.dtype
        # bounded at the precision the epilogue COMPUTES in: bfloat16 has an
        # 8-bit mantissa (eps ~3.9e-3), so a tighter bound would fail on
        # rounding alone. A wrong split or wrong formula is O(1) wrong, not
        # O(eps), so this still discriminates.
        rel = ((got.float() - want).abs().max() / want.abs().max()).item()
        assert rel < 1e-2, rel
        # and it is NOT the gpt-oss epilogue, on the same input
        base = super(Mxfp4NvmeResidencyK3, tie)._glu(gu)
        assert not torch.allclose(got.float(), base.float(), atol=1e-2)
    finally:
        tie.tier.close()


@cuda
def test_k3_forward_matches_a_dequant_reference(arena):
    """End to end for the K3 shape: NVMe-served MXFP4 experts + bias-free
    projections + SiTU must reproduce a dequantize-and-matmul reference. Not
    torch.equal — the reference accumulates in a different order — so the bound
    is the same 3e-2 relative the pipelined engine's own gates use."""
    _needs_gather_kernel()
    from mxfp4_pack_ref import MX_BLOCK as MXB, dequant_mxfp4
    from mxfp4_residency import Mxfp4NvmeResidencyK3
    (gu_b, gu_s, dn_b, dn_s), path, index = arena
    tie = Mxfp4NvmeResidencyK3(path, LAYER, k_slots=K_SLOTS, hot_rows=K_SLOTS,
                               device="cuda", index=index)
    try:
        g = torch.Generator(device="cuda").manual_seed(23)
        x = torch.randn(1, H, dtype=torch.bfloat16, device="cuda", generator=g)
        sc, idx = torch.topk(torch.softmax(
            torch.randn(1, E, device="cuda", generator=g), -1), k=K_SLOTS, dim=-1)
        got = tie.forward(x, idx, sc.to(torch.bfloat16))

        ref = torch.zeros(1, H, dtype=torch.float32)
        xc = x[0].float().cpu()
        for j in range(K_SLOTS):
            e, w = int(idx[0, j]), float(sc[0, j])
            gW = dequant_mxfp4(gu_b[e].reshape(2 * INTER, H // MXB, 16), gu_s[e])
            dW = dequant_mxfp4(dn_b[e].reshape(H, INTER // MXB, 16), dn_s[e])
            gu = xc @ gW.t()                       # no bias: K3 experts are bias-free
            h = _situ_ref(gu[..., :INTER], gu[..., INTER:])
            ref[0] += w * (h @ dW.t())
        rel = ((got.float().cpu() - ref).abs().max() / ref.abs().max()).item()
        assert rel < 3e-2, rel
        assert tie.traffic()["tier"]["misses"] > 0
    finally:
        tie.tier.close()


@cuda
def test_interleaved_arena_forwards_identically_to_residency_order(tmp_path):
    """THE gate for permuting on gather, and the reason the real 1.446 TB arena
    does not have to be re-baked.

    Two arenas, same packed tensors, segment orders differing: `K3_KINDS` (what
    the arena on disk actually is) and `K3_RESIDENCY_KINDS` (what the engine
    reads natively). The first takes the permuting kernel, the second the
    original contiguous one. Their forwards must be `torch.equal` — not close,
    equal, since both are the same bytes multiplied in the same order.
    """
    _needs_gather_kernel()
    from arena_experts import K3_KINDS
    from mxfp4_residency import K3_RESIDENCY_KINDS, Mxfp4NvmeResidency
    stacks = _packed()
    p_int, i_int = _bake(tmp_path, stacks, kinds=K3_KINDS, split_gate_up=True,
                         interleave=True, name="int.arena")
    p_res, i_res = _bake(tmp_path, stacks, kinds=K3_RESIDENCY_KINDS,
                         split_gate_up=True, interleave=False, name="res.arena")
    a = Mxfp4NvmeResidency(p_int, LAYER, k_slots=K_SLOTS, hot_rows=K_SLOTS,
                           device="cuda", index=i_int)
    b = Mxfp4NvmeResidency(p_res, LAYER, k_slots=K_SLOTS, hot_rows=K_SLOTS,
                           device="cuda", index=i_res)
    try:
        assert a.permuted is True, "the interleaved arena must need a permutation"
        assert b.permuted is False, "the residency order must take the fast path"
        x, idx, sc = _route(5, seed=31)
        for t in range(x.shape[0]):
            ra = a.forward(x[t], idx[t:t + 1], sc[t:t + 1])
            rb = b.forward(x[t], idx[t:t + 1], sc[t:t + 1])
            assert torch.equal(ra, rb), (t, (ra - rb).abs().max().item())
        assert a.traffic()["tier"]["misses"] > 0
    finally:
        a.tier.close()
        b.tier.close()


@cuda
def test_permuted_slots_hold_the_engine_layout_byte_exactly(tmp_path):
    """One level below the forward: after a gather from the interleaved arena, the
    device slot's bytes must equal the row a residency-order bake would have
    produced. A forward comparison could in principle pass while both engines read
    the same wrong thing; this cannot.

    Needs triton and CUDA but NOT ``tl.gather`` — the permuting gather moves int64
    words with load/store/arange/cast only. ``tl.gather`` is a
    ``gemm_mxfp4_grouped`` requirement, and no GEMM runs here, so this gate is
    reachable on a triton-3.2 box where the forward tests must skip.
    """
    pytest.importorskip("triton")
    from arena_experts import K3_KINDS
    from mxfp4_residency import K3_RESIDENCY_KINDS, Mxfp4NvmeResidency
    stacks = _packed()
    p_int, i_int = _bake(tmp_path, stacks, kinds=K3_KINDS, split_gate_up=True,
                         interleave=True, name="int2.arena")
    p_res, i_res = _bake(tmp_path, stacks, kinds=K3_RESIDENCY_KINDS,
                         split_gate_up=True, interleave=False, name="res2.arena")
    a = Mxfp4NvmeResidency(p_int, LAYER, k_slots=K_SLOTS, hot_rows=K_SLOTS,
                           device="cuda", index=i_int)
    b = Mxfp4NvmeResidency(p_res, LAYER, k_slots=K_SLOTS, hot_rows=K_SLOTS,
                           device="cuda", index=i_res)
    try:
        want = torch.tensor([2, 3, 5, 6], device="cuda")
        a._fetch(want)
        b._fetch(want)
        assert torch.equal(a.slots, b.slots), (
            "permuted gather did not reproduce the residency-order row: "
            f"{(a.slots.int() - b.slots.int()).abs().sum().item()} bytes differ")
        # and the engine's four views agree, which is what the kernel consumes
        for name in ("gu_p_v", "gu_a_v", "dn_p_v", "dn_a_v"):
            assert torch.equal(getattr(a, name), getattr(b, name)), name
    finally:
        a.tier.close()
        b.tier.close()


@cuda
def test_cuda_graph_capture_is_refused(arena, monkeypatch):
    """Capturing would freeze one fetch's addresses into the graph and never run
    another disk read — silently serving whatever those slots happened to hold.

    The capture flag is simulated rather than a real ``torch.cuda.graph`` block:
    aborting a live capture with an exception can leave the stream in capture
    mode and poison every later test in the process. What is under test is the
    guard, and this exercises exactly the branch that fires.
    """
    _needs_gather_kernel()
    stacks, path, index = arena
    _res, biases = _resident(stacks, ())
    tie = _tiered(path, index, (), K_SLOTS, biases)
    try:
        idx = torch.tensor([[0, 1, 2, 3]], device="cuda")
        sc = torch.full((1, K_SLOTS), 0.25, dtype=torch.bfloat16, device="cuda")
        x = torch.zeros(1, H, dtype=torch.bfloat16, device="cuda")
        tie.forward(x, idx, sc)                    # works eagerly
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing",
                            lambda: True)
        with pytest.raises(RuntimeError, match="CUDA graph"):
            tie.forward(x, idx, sc)
    finally:
        tie.tier.close()


# ------------------------------------- 4. sharing the k slots across layers ----
# Slots are the engine's only large device allocation: k x row_bytes, which is 281 MB
# at K3's geometry. One store per layer would spend 25.8 GB of VRAM on 92 buffers of
# which exactly one is ever live. These gate sharing them -- the mechanics on CPU, the
# correctness consequence (a handover must not serve the previous layer's rows) on CUDA.

def _bare(k_slots=K_SLOTS, device="cpu", n1=2 * INTER):
    """An engine with geometry and nothing else -- enough to build/attach a store."""
    from mxfp4_pipelined import Mxfp4PipelinedGptOss
    eng = Mxfp4PipelinedGptOss.__new__(Mxfp4PipelinedGptOss)
    eng._init_geometry(E, n1, H // 2, H // MX_BLOCK, H, INTER // 2,
                       INTER // MX_BLOCK, k_slots=k_slots, device=device,
                       alpha=ALPHA, limit=LIMIT, compute_dtype=torch.bfloat16)
    return eng


def _bake_two_layers(tmp_path, seeds=(1689, 4242)):
    """A 2-layer arena whose layers hold DIFFERENT experts.

    Same-weights layers would make every assertion below pass regardless of which
    layer's bytes the shared buffer actually held.
    """
    per_layer = [_packed(seed=s) for s in seeds]
    tensors = {}
    for lay, stacks in enumerate(per_layer):
        for kind, stack in zip(KINDS, stacks):
            for e in range(E):
                t = stack[e].contiguous().cpu()
                tensors[f"model.layers.{lay}.mlp.experts.{e}.{kind}"] = (
                    tuple(t.shape), "U8", t.numpy().tobytes())
    snap = tmp_path / "snap2"
    snap.mkdir(exist_ok=True)
    (snap / "model.safetensors").write_bytes(_st_bytes(tensors))
    path = str(tmp_path / "two.arena")
    bake_expert_tensors(
        str(snap), path,
        name_template="model.layers.{layer}.mlp.experts.{expert}.{kind}",
        kinds=KINDS, align=4096, log=lambda *a: None)
    return per_layer, path, load_index(path)


def test_a_shared_store_is_one_allocation():
    """The point of the whole exercise: N engines, one slot buffer."""
    from mxfp4_pipelined import SlotStore
    a, b = _bare(), _bare()
    a._init_slots()
    b._init_slots(a.store)
    assert b.store is a.store
    assert a.slots.data_ptr() == b.slots.data_ptr()
    assert a.store.users == 2 and a.store.bytes == K_SLOTS * a.row_bytes
    # every view the GEMM consumes must be the shared buffer's, not a private copy
    for name in ("slots64", "gu_p_v", "gu_a_v", "dn_p_v", "dn_a_v", "have",
                 "want_buf", "slot_eids"):
        assert getattr(a, name).data_ptr() == getattr(b, name).data_ptr(), name
    # and a private store is still the default, for callers that never opt in
    c = _bare()
    c._init_slots()
    assert isinstance(c.store, SlotStore) and c.store is not a.store
    assert c.slots.data_ptr() != a.slots.data_ptr()


def test_a_store_built_for_another_geometry_is_refused():
    """Slots are raw bytes read at fixed segment offsets, so a store built for a
    different layout is not merely wasteful -- it would be read as this one."""
    a = _bare()
    a._init_slots()
    for other in (_bare(k_slots=K_SLOTS + 1), _bare(n1=INTER)):
        with pytest.raises(ValueError, match="does not match this engine"):
            other._init_slots(a.store)
    assert a.store.users == 1, "a refused engine must not register as a user"


def test_taking_the_buffer_over_poisons_have():
    """`have` describes the BUFFER, so it moves with it. An engine that finds
    another layer holding the slots must not believe anything is resident."""
    a, b = _bare(), _bare()
    a._init_slots()
    b._init_slots(a.store)
    a._claim()
    a.have.fill_(1234)                    # pretend a's rows are resident
    assert a.store.claims == 1
    a._claim()                            # already the owner
    assert a.store.claims == 1, "re-claiming must not poison, or the skip never hits"
    assert int(a.have[0]) == 1234
    b._claim()
    assert a.store.claims == 2 and a.store.owner is b
    assert torch.equal(b.have, torch.full((K_SLOTS,), -1, dtype=torch.long)), (
        "the slots now hold another layer's rows; every one must re-gather")


def test_the_tiered_engine_forgets_its_residency_mirror():
    """`_invalidate` rebuilds `have` from this mirror, so a stale mirror would
    reconstruct a `have` that claims another layer's bytes are already resident."""
    from mxfp4_residency import Mxfp4NvmeResidency

    class Stub:
        k = 3
    s = Stub()
    s._have_eid, s._have_addr = [7, 7, 7], [0xF00D, 0xF00D, 0xF00D]
    Mxfp4NvmeResidency._forget(s)
    assert s._have_eid == [-1] * 3 and s._have_addr == [-1] * 3


@cuda
def test_layers_sharing_slots_answer_as_if_each_had_its_own(tmp_path):
    """THE cross-layer correctness gate, and the reason `_forget` exists.

    Two layers, one slot buffer, one tier, and the SAME expert ids on both -- the
    worst case, because `_invalidate`'s test is `want_eid != have_eid` and the ids
    match, while `hot_rows == K_SLOTS` forces the tier to hand back the very slots
    (hence the very addresses) it just gave the other layer. Without the handover
    poison, layer 1's fetch is skipped as already-resident and layer 1 computes with
    layer 0's experts, finitely and plausibly.
    """
    _needs_gather_kernel()
    from mxfp4_residency import Mxfp4NvmeResidency
    _per_layer, path, index = _bake_two_layers(tmp_path)
    tier = ColdTier(path, hot_rows=K_SLOTS, pinned=True, index=index)

    def engine(layer, store=None):
        return Mxfp4NvmeResidency(path, layer, k_slots=K_SLOTS, tier=tier,
                                  index=index, device="cuda", alpha=ALPHA,
                                  limit=LIMIT, store=store)
    try:
        shared0 = engine(0)
        shared1 = engine(1, store=shared0.store)
        ref0, ref1 = engine(0), engine(1)          # private stores: the reference
        assert shared0.store is shared1.store and shared0.store.users == 2
        assert ref0.store is not shared0.store and ref1.store is not ref0.store

        g = torch.Generator(device="cuda").manual_seed(41)
        x = torch.randn(1, H, dtype=torch.bfloat16, device="cuda", generator=g)
        idx = torch.tensor([[0, 1, 2, 3]], device="cuda")
        sc = torch.full((1, K_SLOTS), 0.25, dtype=torch.bfloat16, device="cuda")

        want0, want1 = ref0.forward(x, idx, sc), ref1.forward(x, idx, sc)
        assert not torch.equal(want0, want1), (
            "the two layers must hold different experts or this proves nothing")

        for step in range(4):
            assert torch.equal(shared0.forward(x, idx, sc), want0), f"L0 step {step}"
            assert torch.equal(shared1.forward(x, idx, sc), want1), f"L1 step {step}"

        # pin that the collision this test is about actually happened: both engines
        # recorded the SAME tier addresses, so `have` could not have discriminated
        assert set(shared0._have_addr) == set(shared1._have_addr), (
            "the tier did not reuse addresses across layers; the test lost its teeth")
        assert shared0.store.claims >= 8, shared0.store.claims
    finally:
        tier.close()


@cuda
def test_a_sole_owner_still_skips_re_reading(tmp_path):
    """Sharing must not cost the have-skip. One engine holding the buffer across
    consecutive decode steps re-reads nothing and hands over nothing."""
    _needs_gather_kernel()
    from mxfp4_residency import Mxfp4NvmeResidency
    _per_layer, path, index = _bake_two_layers(tmp_path)
    tier = ColdTier(path, hot_rows=E, pinned=True, index=index)
    try:
        eng = Mxfp4NvmeResidency(path, 0, k_slots=K_SLOTS, tier=tier, index=index,
                                 device="cuda", alpha=ALPHA, limit=LIMIT)
        idx = torch.tensor([[2, 3, 4, 5]], device="cuda")
        sc = torch.full((1, K_SLOTS), 0.25, dtype=torch.bfloat16, device="cuda")
        x = torch.zeros(1, H, dtype=torch.bfloat16, device="cuda")
        first = eng.forward(x, idx, sc)
        claims, reads = eng.store.claims, tier.stats()["disk_reads"]
        assert torch.equal(eng.forward(x, idx, sc), first)
        assert eng.store.claims == claims, "no handover happened; do not poison"
        assert tier.stats()["disk_reads"] == reads, "resident rows were re-read"
        assert eng.traffic()["slots"] == {"bytes": K_SLOTS * eng.row_bytes,
                                          "users": 1, "claims": claims}
    finally:
        tier.close()


# ------------------------------------------- 7. the device row cache ----
def _tiered_cached(path, index, hot_ids, hot_rows, biases, cache):
    from mxfp4_residency import Mxfp4NvmeResidency
    return Mxfp4NvmeResidency(
        path, LAYER, hot_ids=hot_ids, k_slots=K_SLOTS, hot_rows=hot_rows,
        gate_up_bias=biases[0] if biases else None,
        down_bias=biases[1] if biases else None,
        device="cuda", alpha=ALPHA, limit=LIMIT, index=index, dev_cache=cache)


@cuda
def test_dev_row_cache_answers_bit_for_bit_like_no_cache(arena):
    """The cache may not change a single output bit.

    It relocates WHERE a row is read from, never what the row is. Anything
    else is the cache reinterpreting packed bytes, which is the one thing the
    whole cold path is forbidden to do.
    """
    _needs_gather_kernel()
    from dev_row_cache import DevRowCache
    stacks, path, index = arena
    _res, biases = _resident(stacks, ())
    x, idx, sc = _route(24, seed=11)

    plain = _tiered(path, index, (), K_SLOTS, biases)
    try:
        want = [plain.forward(x[t], idx[t:t + 1], sc[t:t + 1]).clone()
                for t in range(x.shape[0])]
        plain_pcie = plain.traffic()["cold_pcie_bytes"]
    finally:
        plain.tier.close()

    # 2*k is the floor (see _init_dev_cache): the previous step's k rows are
    # still ACTIVE while this step claims its own k. Eviction is exercised by
    # the k_slots=2 test below, where 8 experts contend for 4 rows.
    cache = DevRowCache(2 * K_SLOTS, plain.row_stride, device="cuda",
                        protected=K_SLOTS)
    cached = _tiered_cached(path, index, (), K_SLOTS, biases, cache)
    try:
        for t in range(x.shape[0]):
            got = cached.forward(x[t], idx[t:t + 1], sc[t:t + 1])
            assert torch.equal(got, want[t]), (
                t, (got - want[t]).abs().max().item())
        tr = cached.traffic()
    finally:
        cached.tier.close()

    dc = tr["dev_cache"]
    routed_cold = sum(len({int(e) for e in idx[t]}) for t in range(idx.shape[0]))
    assert dc["gathers"] < routed_cold, (
        f"the cache filled {dc['gathers']} rows for {routed_cold} routed cold "
        f"experts -- nothing was reused, so it is pure overhead here")
    assert tr["host_to_device_bytes"] < plain_pcie, (
        tr["host_to_device_bytes"], plain_pcie)
    assert dc["rows"] == 2 * K_SLOTS and dc["protected"] == K_SLOTS


@cuda
def test_dev_row_cache_reuses_a_row_the_router_moved(arena):
    """The property the positional cache does not have. Route the SAME expert
    set in a different order and the second step must fill nothing."""
    _needs_gather_kernel()
    from dev_row_cache import DevRowCache
    stacks, path, index = arena
    _res, biases = _resident(stacks, ())
    x, idx, sc = _route(2, seed=5)
    idx[1] = idx[0].flip(0)                      # same experts, reversed

    cache = DevRowCache(8, _probe_stride(path, index), device="cuda",
                        protected=4)
    eng = _tiered_cached(path, index, (), K_SLOTS, biases, cache)
    try:
        eng.forward(x[0], idx[0:1], sc[0:1])
        after_first = eng.traffic()["dev_cache"]["gathers"]
        eng.forward(x[1], idx[1:2], sc[1:2])
        after_second = eng.traffic()["dev_cache"]["gathers"]
    finally:
        eng.tier.close()
    assert after_second == after_first, (
        f"a re-routed expert was re-fetched: {after_first} -> {after_second}")


def _probe_stride(path, index):
    """The tier's padded row size, without building an engine to ask."""
    t = ColdTier(path, hot_rows=2, pinned=False, index=index)
    try:
        return t.row_stride
    finally:
        t.close()


@cuda
def test_dev_row_cache_still_matches_when_it_must_evict(arena):
    """The cache is only interesting when it cannot hold everything.

    k_slots=2 puts all E=8 experts through a 4-row arena, so rows are
    logically evicted, resurrected, and overwritten during the trace -- and
    the answers still have to be bit-for-bit what the uncached engine gives.
    """
    _needs_gather_kernel()
    from dev_row_cache import DevRowCache
    from mxfp4_residency import Mxfp4NvmeResidency
    stacks, path, index = arena
    _res, biases = _resident(stacks, ())

    k = 2
    g = torch.Generator(device="cuda").manual_seed(21)
    x = torch.randn(40, 1, H, dtype=torch.bfloat16, device="cuda", generator=g)
    sc, idx = torch.topk(torch.softmax(
        torch.randn(40, E, device="cuda", generator=g), -1), k=k, dim=-1)
    sc = sc.to(torch.bfloat16)

    def _eng(cache):
        return Mxfp4NvmeResidency(
            path, LAYER, hot_ids=(), k_slots=k, hot_rows=k,
            gate_up_bias=biases[0], down_bias=biases[1], device="cuda",
            alpha=ALPHA, limit=LIMIT, index=index, dev_cache=cache)

    plain = _eng(None)
    try:
        want = [plain.forward(x[t], idx[t:t + 1], sc[t:t + 1]).clone()
                for t in range(x.shape[0])]
        stride = plain.row_stride
    finally:
        plain.tier.close()

    cache = DevRowCache(2 * k, stride, device="cuda", protected=k)
    eng = _eng(cache)
    try:
        for t in range(x.shape[0]):
            got = eng.forward(x[t], idx[t:t + 1], sc[t:t + 1])
            assert torch.equal(got, want[t]), (
                t, (got - want[t]).abs().max().item())
        dc = eng.traffic()["dev_cache"]
    finally:
        eng.tier.close()

    assert dc["logical_evictions"] > 0, ("8 experts through 4 rows evicted "
                                         f"nothing: {dc}")
    assert dc["overwritten"] > 0, f"nothing was ever actually reused-then-lost: {dc}"
