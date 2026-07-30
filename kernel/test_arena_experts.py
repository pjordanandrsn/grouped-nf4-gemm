"""Gates for the arena -> kernel link.

The claim under test is narrow and checkable: bytes that went into the arena
come back out of `ArenaExpertSource` **identical**, in the caller's expert
order, shaped the way `gemm_mxfp4_grouped` wants them. Every positive gate here
has a paired negative control — a byte-identity check with no demonstrated
failure mode is a constant function.
"""
import json
import os
import struct
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))

from arena_experts import (ArenaExpertSource, K3_KINDS, K3_TEMPLATE,  # noqa: E402
                           expert_bytes_per_token)
from nvme_arena import bake_expert_tensors, load_index, row_offset  # noqa: E402

E, L_ROUTED = 6, (1, 2)
# A faithful MoE geometry, not three identically-shaped tensors: gate/up map
# hidden->intermediate and down maps intermediate->hidden, so down's contracted
# dim is I and must itself be a multiple of MX_BLOCK=32. Getting this wrong
# makes down's scales [.., I//32] == [.., 0] and the kernel rejects it.
H, I_ = 64, 32                    # hidden, intermediate
N, K = I_, H                      # gate/up output rows, contracted dim
SHAPES = {"w1.weight_packed": [I_, H // 2], "w1.weight_scale": [I_, H // 32],
          "w3.weight_packed": [I_, H // 2], "w3.weight_scale": [I_, H // 32],
          "w2.weight_packed": [H, I_ // 2], "w2.weight_scale": [H, I_ // 32]}


def _st_bytes(tensors):
    hdr, blobs, off = {}, [], 0
    for name, (data, shape) in tensors.items():
        hdr[name] = {"dtype": "U8", "shape": shape,
                     "data_offsets": [off, off + len(data)]}
        blobs.append(data)
        off += len(data)
    hj = json.dumps(hdr).encode()
    return struct.pack("<Q", len(hj)) + hj + b"".join(blobs)


def make_snapshot(root, seed=11):
    """A toy released-K3-shaped checkpoint: per-expert MXFP4 packed weights
    and e8m0 scales, spelled the way the real release spells them."""
    g = torch.Generator().manual_seed(seed)
    ground, shard, wm = {}, {}, {}
    for lay in L_ROUTED:
        for e in range(E):
            for kind in K3_KINDS:
                shape = SHAPES[kind]
                # Packed nibbles: fully random, so byte-identity is a strong
                # claim. e8m0 SCALES: a sane exponent band. Random 0-255 scales
                # reach 2**128 and the dequantized product overflows to inf/nan
                # -- which then silently breaks any torch.equal comparison,
                # because NaN != NaN even bit-for-bit.
                lo, hi = (120, 135) if kind.endswith("weight_scale") else (0, 256)
                t = torch.randint(lo, hi, shape, generator=g, dtype=torch.uint8)
                name = K3_TEMPLATE.format(layer=lay, expert=e, kind=kind)
                ground[name] = t
                shard[name] = (t.numpy().tobytes(), shape)
                wm[name] = "a.safetensors"
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "a.safetensors"), "wb") as f:
        f.write(_st_bytes(shard))
    with open(os.path.join(root, "model.safetensors.index.json"), "w") as f:
        json.dump({"weight_map": wm}, f)
    return ground


@pytest.fixture
def baked(tmp_path):
    snap = tmp_path / "snap"
    ground = make_snapshot(str(snap))
    arena = str(tmp_path / "k3toy.arena")
    bake_expert_tensors(str(snap), arena, name_template=K3_TEMPLATE,
                        kinds=K3_KINDS, log=lambda *a: None)
    return arena, ground


def test_geometry_comes_from_the_index_not_assumption(baked):
    arena, _ = baked
    with ArenaExpertSource(arena) as src:
        assert src.n_experts == E
        assert src.layers == list(L_ROUTED)
        assert src.segment_shape("w1.weight_packed") == (N, K // 2)
        assert src.segment_shape("w1.weight_scale") == (N, K // 32)


def test_fetched_bytes_are_identical_to_the_release(baked):
    """The provenance claim, end to end: what the kernel gets is what shipped."""
    arena, ground = baked
    ids = [4, 0, 3]
    with ArenaExpertSource(arena) as src:
        raw = src.fetch_raw(1, ids)
    for suffix in K3_KINDS:
        got = raw[suffix]
        assert got.shape == (len(ids), *SHAPES[suffix])
        for i, e in enumerate(ids):
            want = ground[K3_TEMPLATE.format(layer=1, expert=e, kind=suffix)]
            assert torch.equal(got[i], want), (suffix, e)


def test_corrupting_one_arena_byte_is_detected(baked):
    """Negative control: without this, the identity test above could be
    comparing something to itself and would never fail."""
    arena, ground = baked
    idx = load_index(arena)
    off = row_offset(idx, 1, 3)                      # expert 3, layer 1
    with open(arena, "r+b") as f:
        f.seek(off)
        b = f.read(1)
        f.seek(off)
        f.write(bytes([b[0] ^ 0xFF]))
    with ArenaExpertSource(arena) as src:
        raw = src.fetch_raw(1, [3])
    want = ground[K3_TEMPLATE.format(layer=1, expert=3,
                                     kind="w1.weight_packed")]
    assert not torch.equal(raw["w1.weight_packed"][0], want)


def test_stack_order_follows_the_caller_not_sorted(baked):
    """Routing hands experts in selection order; a sorted stack would pair
    rows with the wrong tokens and still look plausible."""
    arena, ground = baked
    ids = [5, 1, 4, 0]
    with ArenaExpertSource(arena) as src:
        blocks, _ = src.fused_stacks(2, ids, proj="gate")
    for i, e in enumerate(ids):
        want = ground[K3_TEMPLATE.format(layer=2, expert=e,
                                         kind="w1.weight_packed")]
        assert torch.equal(blocks[i], want)


@pytest.mark.parametrize("proj,kind", [("gate", "w1"), ("up", "w3"),
                                       ("down", "w2")])
def test_canonical_projection_maps_to_released_spelling(baked, proj, kind):
    arena, ground = baked
    with ArenaExpertSource(arena) as src:
        blocks, scales = src.fused_stacks(1, [2], proj=proj)
    assert torch.equal(blocks[0], ground[K3_TEMPLATE.format(
        layer=1, expert=2, kind=f"{kind}.weight_packed")])
    assert torch.equal(scales[0], ground[K3_TEMPLATE.format(
        layer=1, expert=2, kind=f"{kind}.weight_scale")])


def test_fused_stacks_match_the_kernel_input_contract(baked):
    """`gemm_mxfp4_grouped` wants blocks [E, N, K//2] and scales [E, N, K//32]
    uint8. If these ever drift apart the kernel call fails far from here."""
    arena, _ = baked
    ids = [0, 1, 2]
    with ArenaExpertSource(arena) as src:
        blocks, scales = src.fused_stacks(1, ids)
    assert blocks.dtype is torch.uint8 and scales.dtype is torch.uint8
    assert blocks.shape == (len(ids), N, K // 2)
    assert scales.shape == (len(ids), N, K // 32)
    assert blocks.shape[-1] * 2 == scales.shape[-1] * 32 == K


def test_unbaked_projection_raises_a_useful_error(baked):
    arena, _ = baked
    with ArenaExpertSource(arena) as src:
        with pytest.raises(KeyError, match="was it baked with K3_KINDS"):
            src.fused_stacks(1, [0], proj="w9")


def test_only_the_routed_rows_are_read(baked):
    """A top-k fetch must cost k rows of traffic, not the whole layer — the
    entire economic premise of the tier."""
    arena, _ = baked
    with ArenaExpertSource(arena) as src:
        before = src.traffic()
        src.fetch_raw(1, [0, 2])
        after = src.traffic()
        assert after["reads"] - before["reads"] == 2, after
        read = after["bytes_read"] - before["bytes_read"]
        assert read == 2 * src.row_stride, read


def test_bytes_per_token_is_topk_x_layers_x_row(baked):
    arena, _ = baked
    idx = load_index(arena)
    row_useful = sum(g["length"] for g in idx["segments"])
    assert expert_bytes_per_token(idx, 3) == 3 * len(L_ROUTED) * row_useful


# --------------------------------------------------- arena -> kernel, on GPU --
# The gates above prove the BYTES arrive intact and correctly shaped. This one
# proves they compute: the same GEMM fed from the arena and fed from tensors
# held in memory must agree exactly. Requires CUDA+triton, so it skips on the
# control machine and runs where the kernel does.
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@cuda
def test_arena_fed_gemm_equals_memory_fed_gemm(baked):
    from mxfp4_grouped import gemm_mxfp4_grouped
    arena, ground = baked
    ids = [3, 0]
    T = 8
    a = torch.randn(T, K, device="cuda", dtype=torch.bfloat16)
    sizes = torch.tensor([4, 4], device="cuda", dtype=torch.int32)
    eids = torch.tensor(ids, device="cuda", dtype=torch.int32)

    with ArenaExpertSource(arena, device="cuda") as src:
        ab, as_ = src.fused_stacks(1, ids, "gate")
    mb = torch.stack([ground[K3_TEMPLATE.format(layer=1, expert=e,
                                                kind="w1.weight_packed")]
                      for e in ids]).cuda()
    ms = torch.stack([ground[K3_TEMPLATE.format(layer=1, expert=e,
                                                kind="w1.weight_scale")]
                      for e in ids]).cuda()
    assert torch.equal(ab, mb) and torch.equal(as_, ms)
    from_arena = gemm_mxfp4_grouped(a, ab, as_, sizes, eids, block_m=16)
    from_mem = gemm_mxfp4_grouped(a, mb, ms, sizes, eids, block_m=16)
    # Compare BITS, not values: torch.equal reports False for identical NaNs,
    # so a value compare can fail on outputs that are byte-for-byte the same.
    assert torch.equal(from_arena.view(torch.int16), from_mem.view(torch.int16))
    assert torch.isfinite(from_arena).all(), "fixture produced inf/nan"
    # and sizes must be reusable across the two calls
    assert sizes.tolist() == [4, 4], sizes.tolist()


@cuda
def test_moe_layer_forward_runs_off_the_arena(baked):
    arena, _ = baked
    from arena_experts import moe_layer_forward
    ids = [1, 4]
    a = torch.randn(8, K, device="cuda", dtype=torch.bfloat16)
    sizes = torch.tensor([4, 4], device="cuda", dtype=torch.int32)
    with ArenaExpertSource(arena, device="cuda") as src:
        out = moe_layer_forward(src, 1, a, sizes,
                                torch.tensor(ids, device="cuda",
                                             dtype=torch.int32))
    assert out.shape[0] == a.shape[0]
    assert torch.isfinite(out).all()
