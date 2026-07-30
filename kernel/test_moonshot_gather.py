# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Gates for the DeepSeek-V3-lineage per-expert -> fused gather (Kimi K2 today,
K3 when it lands). Format-agnostic gather is exercised on fp8-shaped bytes
(K2's real dtype) AND mxfp4-shaped bytes (K3's expected dtype) — same code,
proving one loader serves both. Real K2-index discovery is a network-optional
gate. No large download."""
import hashlib
import struct

import pytest
import torch

from moonshot_gather import (  # noqa: E402
    GLU_VARIANTS, K2_SCHEME, K3_SCHEME, apply_glu, detect_scheme,
    discover_layer, expert_tensor_name, file_sha256_map, gather_layer,
    moonshot_apply_gate, register_glu_variant, resolve_scheme,
    verify_gather_provenance,
)


# ------------------------------------------------------------ gather logic --
def _synth_layer(E, N, Kw, *, dtype, scale_shape, layer=1, prefix="model",
                 scale_suffix="weight_scale_inv", seed=0):
    """A dict of per-expert tensors mimicking the on-disk layout: gate/up
    `[N, Kw]`, down `[H=Kw_out, N]`, + per-proj scale of `scale_shape`."""
    g = torch.Generator().manual_seed(seed)
    d = {}
    def mk(shape):
        return (torch.randint(0, 255, shape, generator=g, dtype=torch.uint8).view(dtype)
                if dtype == torch.uint8 else
                torch.randn(shape, generator=g).to(dtype))
    for e in range(E):
        for proj, wshape in (("gate_proj", (N, Kw)), ("up_proj", (N, Kw)),
                             ("down_proj", (Kw, N))):
            d[expert_tensor_name(layer, e, proj, "weight", prefix)] = mk(wshape)
            d[expert_tensor_name(layer, e, proj, scale_suffix, prefix)] = \
                torch.randn(scale_shape, generator=g).float()
    return d


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
def test_gather_fused_shapes_and_exactness(dtype):
    """gate+up concat -> [E, 2N, Kw]; down -> [E, H, N]; every expert slice is
    the exact source tensor (bit-identical stack, no decode)."""
    E, N, Kw = 6, 32, 64
    layer = _synth_layer(E, N, Kw, dtype=dtype, scale_shape=(N // 16, Kw // 16))
    get = layer.__getitem__
    out = gather_layer(get, 1, E, scale_suffix="weight_scale_inv")

    assert out["gate_up"].shape == (E, 2 * N, Kw)
    assert out["down"].shape == (E, Kw, N)
    assert out["gate_up"].dtype == dtype
    for e in range(E):
        g = layer[expert_tensor_name(1, e, "gate_proj", "weight")]
        u = layer[expert_tensor_name(1, e, "up_proj", "weight")]
        # first N rows = gate, next N = up (clean concat, not interleaved)
        assert torch.equal(out["gate_up"][e, :N], g)
        assert torch.equal(out["gate_up"][e, N:], u)
        assert torch.equal(out["down"][e], layer[expert_tensor_name(1, e, "down_proj", "weight")])


def test_scales_gathered_format_agnostic():
    E, N, Kw = 4, 32, 64
    layer = _synth_layer(E, N, Kw, dtype=torch.uint8, scale_shape=(N // 16, Kw // 16))
    out = gather_layer(layer.__getitem__, 1, E, scale_suffix="weight_scale_inv")
    assert out["gate_up_scale"].shape == (E, 2 * (N // 16), Kw // 16)
    assert out["down_scale"].shape == (E, N // 16, Kw // 16)


def test_clean_split_glu_recovers_gate_up():
    """moonshot_apply_gate splits the fused [gate;up] GEMM output cleanly —
    NOT interleaved. silu(gate)*up on the right halves."""
    T, N = 5, 16
    gate = torch.randn(T, N)
    up = torch.randn(T, N)
    fused_out = torch.cat([gate, up], dim=-1)            # what the fused GEMM yields
    got = moonshot_apply_gate(fused_out)
    want = torch.nn.functional.silu(gate) * up
    torch.testing.assert_close(got, want)
    # an interleaved reading would be WRONG here — guard against regressing to it
    inter_gate, inter_up = fused_out[..., ::2], fused_out[..., 1::2]
    assert not torch.allclose(got, torch.nn.functional.silu(inter_gate) * inter_up)


def test_glu_registry_swiglu_matches_apply_gate():
    """apply_glu('swiglu') == moonshot_apply_gate default (silu) — the verified
    K2/DeepSeek baseline, addressable by name."""
    fused = torch.randn(5, 32)
    torch.testing.assert_close(apply_glu(fused, "swiglu"), moonshot_apply_gate(fused))


def test_situ_is_guarded_not_guessed():
    """K3's SiTU formula is unsourced — apply_glu('situ') must RAISE, never
    silently substitute a guess (R6). The error names the candidate readings."""
    fused = torch.randn(3, 16)
    with pytest.raises(NotImplementedError, match="SiTU: formula UNVERIFIED"):
        apply_glu(fused, "situ")


def test_register_glu_variant_activates_confirmed_formula():
    """Once a formula is sourced, registration makes it a one-line swap."""
    saved = GLU_VARIANTS.get("situ")
    try:
        register_glu_variant("situ", lambda g, u: torch.tanh(g) * u)  # stand-in
        fused = torch.randn(4, 20)
        n = 10
        want = torch.tanh(fused[..., :n]) * fused[..., n:]
        torch.testing.assert_close(apply_glu(fused, "situ"), want)
    finally:
        GLU_VARIANTS["situ"] = saved  # restore the guard


def test_unknown_glu_variant_raises():
    with pytest.raises(KeyError, match="unknown GLU variant"):
        apply_glu(torch.randn(2, 8), "gelu_banana")


def test_concat_gate_up_false_keeps_separate():
    E, N, Kw = 3, 32, 64
    layer = _synth_layer(E, N, Kw, dtype=torch.uint8, scale_shape=(2, 2))
    out = gather_layer(layer.__getitem__, 1, E, concat_gate_up=False)
    assert out["gate"].shape == (E, N, Kw) and out["up"].shape == (E, N, Kw)
    assert "gate_up" not in out


# --------------------------------------------------------------- discovery --
def test_discover_layer_counts_experts():
    E = 8
    layer = _synth_layer(E, 32, 64, dtype=torch.uint8, scale_shape=(2, 2), layer=3)
    wm = {name: "shard-x.safetensors" for name in layer}
    info = discover_layer(wm, 3)
    assert info["n_experts"] == E
    assert info["has_scale"] and info["scale_suffix"] == "weight_scale_inv"
    # a dense layer (no experts.* namespace) reports 0
    assert discover_layer({"model.layers.0.mlp.gate_proj.weight": "s"}, 0)["n_experts"] == 0


def test_discover_rejects_noncontiguous():
    wm = {expert_tensor_name(1, 0, "gate_proj", "weight"): "s",
          expert_tensor_name(1, 5, "gate_proj", "weight"): "s"}
    with pytest.raises(ValueError, match="non-contiguous"):
        discover_layer(wm, 1)


# ------------------------------------------------------------- provenance ---
def test_gather_provenance_per_source_tensor(tmp_path):
    """Write per-expert tensors to a real safetensors file; the gather's
    per-source-tensor hashes must equal the file byte-range hashes, and a
    flipped byte must break it."""
    pytest.importorskip("safetensors")
    from safetensors.torch import save_file
    E, N, Kw = 3, 32, 64
    layer = _synth_layer(E, N, Kw, dtype=torch.uint8, scale_shape=(2, 2))
    p = str(tmp_path / "k.safetensors")
    save_file(layer, p)

    fh = file_sha256_map(p, 1, E, scale_suffix="weight_scale_inv")
    assert len(fh) == E * 3 * 2  # 3 projs x (weight + scale)
    rep = verify_gather_provenance(layer.__getitem__, fh, 1, E,
                                   scale_suffix="weight_scale_inv")
    assert all(rep.values())

    # tamper one loaded tensor -> provenance must fail loudly
    bad = dict(layer)
    nm = expert_tensor_name(1, 0, "gate_proj", "weight")
    t = bad[nm].clone(); t[0] ^= 0xFF; bad[nm] = t
    with pytest.raises(ValueError, match="PROVENANCE FAIL"):
        verify_gather_provenance(bad.__getitem__, fh, 1, E, scale_suffix="weight_scale_inv")


# ---------------------------------------- real K2 checkpoint (network gate) --
@pytest.mark.network
def test_real_k2_index_discovery():
    """Against the LIVE Kimi-K2-Instruct index: layer 1 has 384 experts, and
    the scale suffix is the fp8 `weight_scale_inv`. Proves discovery on the
    real DeepSeek-lineage naming (K3's expected parent layout)."""
    import json
    import urllib.request
    url = ("https://huggingface.co/moonshotai/Kimi-K2-Instruct/resolve/main/"
           "model.safetensors.index.json")
    try:
        wm = json.loads(urllib.request.urlopen(url, timeout=60).read())["weight_map"]
    except Exception as e:
        pytest.skip(f"network/repo unavailable: {e}")
    info = discover_layer(wm, 1)
    assert info["n_experts"] == 384, info
    assert info["scale_suffix"] == "weight_scale_inv", info
    # layer 0 is dense (first_k_dense_replace=1) -> no routed experts
    assert discover_layer(wm, 0)["n_experts"] == 0


# ------------------------------------------------- released-K3 name scheme --
# Regression gates for the 2026-07-30 seam: released K3 renamed the container,
# the projections, and the weight kind. With the historical (K2) default those
# renames made discovery return n_experts=0 *silently* — a gather of nothing.
def _synth_k3_layer(E, N, Kw, *, layer=1, prefix="language_model.model", seed=0):
    """Per-expert tensors spelled the way the real K3 checkpoint spells them."""
    g = torch.Generator().manual_seed(seed)
    d = {}
    for e in range(E):
        for proj, wshape in (("gate_proj", (N, Kw)), ("up_proj", (N, Kw)),
                             ("down_proj", (Kw, N))):
            d[expert_tensor_name(layer, e, proj, "weight", prefix, "k3")] = \
                torch.randint(0, 255, wshape, generator=g, dtype=torch.uint8)
            d[expert_tensor_name(layer, e, proj, "weight_scale", prefix, "k3")] = \
                torch.randint(112, 123, (wshape[0], Kw // 32 or 1),
                              generator=g, dtype=torch.uint8)
    return d


def test_k3_scheme_spells_the_released_names():
    """Exact strings observed in moonshotai/Kimi-K3's safetensors index."""
    assert (expert_tensor_name(1, 0, "gate_proj", "weight",
                               "language_model.model", "k3")
            == "language_model.model.layers.1.block_sparse_moe.experts.0"
               ".w1.weight_packed")
    assert (expert_tensor_name(7, 42, "down_proj", "weight_scale",
                               "language_model.model", "k3")
            == "language_model.model.layers.7.block_sparse_moe.experts.42"
               ".w2.weight_scale")
    # w1=gate, w3=up, w2=down — confirmed by shapes, not by convention.
    assert K3_SCHEME.proj == {"gate_proj": "w1", "up_proj": "w3",
                              "down_proj": "w2"}


def test_k2_naming_is_unchanged_by_the_scheme_refactor():
    """The historical default must be byte-identical to the old hardcoded form."""
    assert (expert_tensor_name(3, 5, "up_proj", "weight")
            == "model.layers.3.mlp.experts.5.up_proj.weight")
    assert K2_SCHEME.default_scale_suffix == "weight_scale_inv"


def test_default_scheme_discovers_nothing_on_k3():
    """THE BUG: silent n_experts=0, so gather_layer would stack zero experts."""
    wm = dict.fromkeys(_synth_k3_layer(4, 32, 64), "shard-00001.safetensors")
    assert discover_layer(wm, 1, prefix="language_model.model")["n_experts"] == 0
    got = discover_layer(wm, 1, prefix="language_model.model", scheme="k3")
    assert got["n_experts"] == 4
    assert got["scale_suffix"] == "weight_scale"


def test_detect_scheme_picks_the_right_family():
    k3 = dict.fromkeys(_synth_k3_layer(3, 32, 64), "s.safetensors")
    assert detect_scheme(k3, 1) is K3_SCHEME
    k2 = dict.fromkeys(_synth_layer(3, 32, 64, dtype=torch.uint8,
                                    scale_shape=(2, 4)), "s.safetensors")
    assert detect_scheme(k2, 1) is K2_SCHEME
    assert detect_scheme({"unrelated.tensor": "s.safetensors"}, 1) is None


def test_k3_gather_fused_shapes_and_exactness():
    """Same fusion logic, K3 spelling: bit-identical slices, mxfp4 bytes."""
    E, N, Kw = 5, 32, 64
    layer = _synth_k3_layer(E, N, Kw)
    out = gather_layer(layer.__getitem__, 1, E,
                       prefix="language_model.model", scheme="k3")
    assert out["gate_up"].shape == (E, 2 * N, Kw)
    assert out["down"].shape == (E, Kw, N)
    assert out["gate_up"].dtype == torch.uint8
    # scale_suffix defaults to the SCHEME's, so K3 scales come along unasked
    assert "gate_up_scale" in out and "down_scale" in out
    for e in range(E):
        for i, proj in enumerate(("gate_proj", "up_proj")):
            src = layer[expert_tensor_name(1, e, proj, "weight",
                                           "language_model.model", "k3")]
            assert torch.equal(out["gate_up"][e, i * N:(i + 1) * N], src)
        assert torch.equal(out["down"][e], layer[expert_tensor_name(
            1, e, "down_proj", "weight", "language_model.model", "k3")])


def test_unknown_scheme_raises_and_namescheme_passes_through():
    with pytest.raises(ValueError, match="unknown scheme"):
        expert_tensor_name(1, 0, "gate_proj", scheme="k9")
    assert resolve_scheme(K3_SCHEME) is K3_SCHEME
    assert resolve_scheme(None) is K2_SCHEME


@pytest.mark.network
def test_real_k3_index_discovery():
    """Optional gate against a real released snapshot: set
    ``K3_SNAPSHOT=/path/to/Kimi-K3``. Asserts the scheme is auto-detected and
    the release's own geometry (93 layers, layer 0 dense, 896 experts) is what
    discovery reports. Verified 2026-07-30 on the real 1.5 TB checkpoint."""
    import json
    import os
    snap = os.environ.get("K3_SNAPSHOT", "")
    ix = os.path.join(snap, "model.safetensors.index.json") if snap else ""
    if not ix or not os.path.exists(ix):
        pytest.skip("set K3_SNAPSHOT to a released Kimi-K3 snapshot dir")
    wm = json.load(open(ix))["weight_map"]
    assert detect_scheme(wm, 1) is K3_SCHEME
    d = discover_layer(wm, 1, prefix="language_model.model", scheme="k3")
    assert d["n_experts"] == 896, d
    assert d["scale_suffix"] == "weight_scale", d
    # layer 0 is the dense `first_k_dense_replace` layer in the release
    assert discover_layer(wm, 0, prefix="language_model.model",
                          scheme="k3")["n_experts"] == 0
    # and the historical default must find nothing (this is the reported bug)
    assert discover_layer(wm, 1, prefix="language_model.model")["n_experts"] == 0
