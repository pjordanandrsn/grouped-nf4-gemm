"""#1 — does the arena path compute what upstream computes?

Everything else gates that the patch is *called* with the right inputs and can
be restored. None of it proves it produces the same answer, which is the only
claim a user cares about.

The reference here is deliberately not "some other kernel". It is upstream's own
`moe_infer` loop, running dequantized copies of **the exact packed bytes the
arena holds**, through the same GLU. So the only difference between the two arms
is where the weights came from and whether they were unpacked -- which is
precisely the thing under test.

Also covers, from the lined-up slate:
  #4  K3-shaped routing: many experts, top-k, and the T=1 decode path (which
      takes a different branch in the kernel: `max(sizes)==1` -> GEMV)
  #5  multi-layer sequencing through one source (landing-buffer reuse)
  #6  traffic vs the bytes/token prediction
"""
import json
import os
import struct
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))

from arena_experts import (ArenaExpertSource, K3_KINDS, K3_TEMPLATE,  # noqa: E402
                           expert_bytes_per_token, moe_layer_forward)
from nvme_arena import bake_expert_tensors, load_index  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

H, I_, E, LAYERS = 128, 64, 8, (1, 2, 3)     # hidden(latent), intermediate
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


@pytest.fixture(scope="module")
def baked(tmp_path_factory):
    root = tmp_path_factory.mktemp("snap")
    from mxfp4_pack_ref import quantize_pack_mxfp4
    g = torch.Generator().manual_seed(4242)
    ground, shard, wm = {}, {}, {}
    # Quantize REALISTIC weights rather than drawing random bytes. Random
    # nibbles paired with random e8m0 scales encode magnitudes ~1e3, and the
    # GLU squares them -- the product overflows and the comparison degenerates
    # to NaN != NaN. quantize_pack_mxfp4 exists for exactly this.
    for lay in LAYERS:
        for e in range(E):
            packed = {}
            for proj, (rows, k) in (("w1", (I_, H)), ("w3", (I_, H)),
                                    ("w2", (H, I_))):
                w = torch.randn(rows, k, generator=g) * 0.02
                blk, scl = quantize_pack_mxfp4(w)
                packed[f"{proj}.weight_packed"] = blk.reshape(rows, k // 2)
                packed[f"{proj}.weight_scale"] = scl.reshape(rows, k // 32)
            for kind in K3_KINDS:
                t = packed[kind].contiguous().to(torch.uint8)
                name = K3_TEMPLATE.format(layer=lay, expert=e, kind=kind)
                ground[name] = t
                shard[name] = (t.numpy().tobytes(), list(t.shape))
                wm[name] = "a.safetensors"
    with open(os.path.join(root, "a.safetensors"), "wb") as f:
        f.write(_st_bytes(shard))
    with open(os.path.join(root, "model.safetensors.index.json"), "w") as f:
        json.dump({"weight_map": wm}, f)
    arena = str(root / "eq.arena")
    bake_expert_tensors(str(root), arena, name_template=K3_TEMPLATE,
                        kinds=K3_KINDS, log=lambda *a: None)
    return arena, ground


def _deq(blocks, scales, dtype=torch.float32):
    """Reference dequantize, independent of the kernel: [rows, K//2] packed +
    [rows, K//32] e8m0 -> [rows, K]."""
    from mxfp4_pack_ref import dequant_mxfp4
    rows, half = blocks.shape
    K = half * 2
    return dequant_mxfp4(blocks.reshape(rows, K // 32, 16), scales, dtype=dtype)


def _upstream_expert(x, gb, gs, ub, us, db, ds):
    """Upstream's expert: dequantized weights, explicit GLU, plain matmuls."""
    W1 = _deq(gb, gs).to(x.dtype)          # [I, H]
    W3 = _deq(ub, us).to(x.dtype)
    W2 = _deq(db, ds).to(x.dtype)          # [H, I]
    return (torch.nn.functional.silu(x @ W1.T) * (x @ W3.T)) @ W2.T


@cuda
def test_arena_path_matches_upstream_dequantized_loop(baked):
    """#1 — the equivalence claim."""
    arena, ground = baked
    ids, sizes = [5, 1, 6], [3, 2, 4]
    T = sum(sizes)
    torch.manual_seed(0)
    a = torch.randn(T, H, device="cuda", dtype=torch.bfloat16) * 0.1

    with ArenaExpertSource(arena, device="cuda") as src:
        got = moe_layer_forward(src, 2, a, torch.tensor(sizes, device="cuda",
                                                        dtype=torch.int32), ids)

    # reference: upstream's per-expert loop over the SAME bytes, dequantized
    ref, start = [], 0
    for e, n in zip(ids, sizes):
        w = {k: ground[K3_TEMPLATE.format(layer=2, expert=e, kind=k)].cuda()
             for k in K3_KINDS}
        ref.append(_upstream_expert(
            a[start:start + n].float(),
            w["w1.weight_packed"], w["w1.weight_scale"],
            w["w3.weight_packed"], w["w3.weight_scale"],
            w["w2.weight_packed"], w["w2.weight_scale"]))
        start += n
    ref = torch.cat(ref).to(torch.bfloat16)

    assert got.shape == ref.shape, (got.shape, ref.shape)
    assert torch.isfinite(ref).all(), "reference overflowed — fixture magnitudes"
    assert torch.isfinite(got).all(), "arena path produced inf/nan"
    # bf16 over a K-term dot product: ~sqrt(K)*eps, eps=2**-8
    tol = (H ** 0.5) * (2 ** -8) * 4
    rel = (got.float() - ref.float()).abs().max() / ref.float().abs().max()
    assert rel < tol, f"rel {rel:.4f} >= tol {tol:.4f}"


@cuda
def test_decode_path_t1_matches_reference(baked):
    """#4 — T=1 per group takes the kernel's GEMV branch (max(sizes)==1),
    a different code path from prefill. Decode is the common case."""
    arena, ground = baked
    ids = [0, 3, 7]
    sizes = [1, 1, 1]
    a = torch.randn(3, H, device="cuda", dtype=torch.bfloat16) * 0.1
    with ArenaExpertSource(arena, device="cuda") as src:
        got = moe_layer_forward(src, 1, a, torch.tensor(sizes, device="cuda",
                                                        dtype=torch.int32), ids)
    ref = []
    for i, e in enumerate(ids):
        w = {k: ground[K3_TEMPLATE.format(layer=1, expert=e, kind=k)].cuda()
             for k in K3_KINDS}
        ref.append(_upstream_expert(
            a[i:i + 1].float(), w["w1.weight_packed"], w["w1.weight_scale"],
            w["w3.weight_packed"], w["w3.weight_scale"],
            w["w2.weight_packed"], w["w2.weight_scale"]))
    ref = torch.cat(ref).to(torch.bfloat16)
    assert torch.isfinite(ref).all() and torch.isfinite(got).all()
    rel = (got.float() - ref.float()).abs().max() / ref.float().abs().max()
    assert rel < (H ** 0.5) * (2 ** -8) * 4, rel


def test_many_layers_through_one_source_reuse_buffers(baked):
    """#5 — a real forward walks every routed layer through ONE source. Landing
    buffers are reused per fetch; a stale buffer would surface as one layer
    returning another layer's bytes."""
    arena, ground = baked
    with ArenaExpertSource(arena, qd=2) as src:      # qd < len(ids): forces reuse
        for lay in LAYERS:
            ids = [0, 1, 2, 3, 4]
            raw = src.fetch_raw(lay, ids)
            for i, e in enumerate(ids):
                want = ground[K3_TEMPLATE.format(layer=lay, expert=e,
                                                 kind="w1.weight_packed")]
                assert torch.equal(raw["w1.weight_packed"][i], want), (lay, e)


def test_repeated_fetch_is_stable(baked):
    """Same request twice must give the same bytes -- a reader that leaked
    state across calls would drift."""
    arena, _ = baked
    with ArenaExpertSource(arena, qd=2) as src:
        a = src.fetch_raw(3, [7, 0, 4])
        b = src.fetch_raw(3, [7, 0, 4])
    for k in a:
        assert torch.equal(a[k], b[k]), k


def test_traffic_matches_the_bytes_per_token_model(baked):
    """#6 — the tier's economics are an arithmetic claim; measure it. A top-k
    fetch across every routed layer must read exactly k*layers rows."""
    arena, _ = baked
    idx = load_index(arena)
    top_k = 4
    with ArenaExpertSource(arena, qd=8) as src:
        before = src.traffic()
        for lay in LAYERS:
            src.fetch_raw(lay, list(range(top_k)))
        after = src.traffic()
    reads = after["reads"] - before["reads"]
    read_bytes = after["bytes_read"] - before["bytes_read"]
    assert reads == top_k * len(LAYERS)
    assert read_bytes == reads * idx["row_stride"]
    # predicted USEFUL bytes/token (segments, excluding stride padding)
    predicted = expert_bytes_per_token(idx, top_k)
    useful = sum(g["length"] for g in idx["segments"]) * reads
    assert useful == predicted, (useful, predicted)
    # padding overhead is explicit, not hidden
    assert read_bytes >= useful
