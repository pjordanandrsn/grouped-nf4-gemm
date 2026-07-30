"""#2 — the Phase-1 oracle: our MXFP4 decode vs compressed-tensors' OWN.

This is the gate every K3 number rests on, and the reference matters more than
usual. K3's config declares `quant_method: "compressed-tensors"`, format
`mxfp4-pack-quantized`. Our decode was validated bit-exactly against
*transformers' gpt-oss* path (`convert_moe_packed_tensors`) -- a different
implementation of the same nominal format. If compressed-tensors reads e2m1
nibble order, the e8m0 bias, or the group axis differently by even one
convention, every downstream K3 number is wrong while every existing gate stays
green.

So this compares against compressed-tensors itself, on REAL released bytes
(one expert extracted verbatim from moonshotai/Kimi-K3, layer 1 expert 0).

Set K3_EXPERT to the extracted file; skips otherwise.
"""
import json
import os
import struct
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))

K3_EXPERT = os.environ.get("K3_EXPERT", "")
needs_bytes = pytest.mark.skipif(
    not (K3_EXPERT and os.path.exists(K3_EXPERT)),
    reason="set K3_EXPERT to a real extracted Kimi-K3 expert .safetensors")


def _load(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for name, m in hdr.items():
            lo, hi = m["data_offsets"]
            f.seek(base + lo)
            buf = bytearray(f.read(hi - lo))
            out[name] = torch.frombuffer(buf, dtype=torch.uint8).reshape(*m["shape"])
    return out


@needs_bytes
def test_real_k3_expert_shapes_are_what_the_release_ships():
    t = _load(K3_EXPERT)
    assert t["w1.weight_packed"].shape == (3072, 1792)     # [moe_inter, latent/2]
    assert t["w1.weight_scale"].shape == (3072, 112)       # latent/32
    assert t["w2.weight_packed"].shape == (3584, 1536)     # [latent, moe_inter/2]
    assert t["w2.weight_scale"].shape == (3584, 96)        # moe_inter/32
    assert t["w1.weight_packed"].dtype is torch.uint8


@needs_bytes
def test_our_decode_matches_compressed_tensors_own_dequant():
    """The oracle. Exact equality is the bar: both claim to implement the same
    packed format over the same bytes, so any difference is a convention gap,
    not noise."""
    ct = pytest.importorskip("compressed_tensors",
                             reason="pip install compressed-tensors")
    from mxfp4_pack_ref import dequant_mxfp4

    t = _load(K3_EXPERT)
    blocks, scales = t["w1.weight_packed"], t["w1.weight_scale"]
    rows, half = blocks.shape
    K = half * 2

    ours = dequant_mxfp4(blocks.reshape(rows, K // 32, 16), scales,
                         dtype=torch.float32)

    # compressed-tensors' own path. The import surface has moved between
    # releases; try the documented entry points and report what was used.
    theirs = used = None
    for modpath, fn in (
        ("compressed_tensors.quantization.lifecycle.forward", "dequantize"),
        ("compressed_tensors.quantization.utils", "dequantize"),
        ("compressed_tensors.compressors.quantized_compressors.mxfp4_quantized",
         "unpack_mxfp4"),
        ("compressed_tensors.utils", "unpack_fp4_from_uint8"),
    ):
        try:
            mod = __import__(modpath, fromlist=[fn])
            f = getattr(mod, fn)
        except (ImportError, AttributeError):
            continue
        try:
            theirs = f(blocks, scales)
            used = f"{modpath}.{fn}"
            break
        except TypeError:
            continue
    if theirs is None:
        pytest.skip(f"no usable compressed-tensors dequant entry point "
                    f"(version {getattr(ct, '__version__', '?')}) — resolve the "
                    f"API before treating the oracle as run")

    theirs = theirs.to(torch.float32).reshape(ours.shape)
    same = torch.equal(ours, theirs)
    if not same:
        d = (ours - theirs).abs()
        pytest.fail(f"decode differs from {used}: max|delta|={d.max():.6g}, "
                    f"mismatched={int((d > 0).sum())}/{d.numel()}")


@needs_bytes
def test_decode_is_finite_and_uses_the_full_e2m1_codebook():
    """A weak but independent sanity check that runs without
    compressed-tensors: a real expert should exercise the codebook and produce
    no inf/nan. Catches a scale-axis slip that silently overflows."""
    from mxfp4_pack_ref import dequant_mxfp4
    t = _load(K3_EXPERT)
    blocks, scales = t["w1.weight_packed"], t["w1.weight_scale"]
    rows, half = blocks.shape
    out = dequant_mxfp4(blocks.reshape(rows, half * 2 // 32, 16), scales,
                        dtype=torch.float32)
    assert torch.isfinite(out).all(), "real bytes decoded to inf/nan"
    assert out.abs().max() > 0, "everything decoded to zero"
    # e2m1 magnitudes are {0,.5,1,1.5,2,3,4,6} x 2**e; a correct decode of a
    # trained expert should show many distinct magnitudes, not a couple.
    mags = torch.unique(out.abs())
    assert mags.numel() > 8, f"only {mags.numel()} distinct magnitudes"
