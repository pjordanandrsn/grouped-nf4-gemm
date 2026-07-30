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


def _bake(tmp_path, stacks, kinds=KINDS, split_gate_up=False):
    """Relocate the very tensors the resident engine will hold into an arena."""
    gu_b, gu_s, dn_b, dn_s = stacks
    if split_gate_up:                       # K3 shape: w1 and w3 kept apart
        payload = [gu_b[:, :INTER], gu_b[:, INTER:], gu_s[:, :INTER],
                   gu_s[:, INTER:], dn_b, dn_s]
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
    path = str(tmp_path / "m.arena")
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


def test_mismatched_layout_is_refused(arena):
    _s, _p, index = arena
    bad = dict(index)
    bad["segments"] = [dict(g) for g in index["segments"]]
    bad["segments"][2]["seg_off"] += 8
    with pytest.raises(ValueError, match="does not match the engine"):
        mxfp4_geometry_from_arena(bad)


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
