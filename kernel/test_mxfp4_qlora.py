# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Phase-7 gates: QLoRA over native MXFP4 experts.

Reference numerics = transformers v5 ``GptOssExperts`` fed the SAME native
bytes through its own dequant orientation ([E, in, out], ``x @ W[e]``); ours is
decode + ``F.linear`` ([out, in]). GEMM orientation may differ at ulp level, so
forward/grad parity uses the single-module band b_rel < 2e-2 (the Phase-5 slip
calibration — correct at this scale; the P1 lesson was about FULL-MODEL drift,
not module gates). Decode itself is bit-exact (Phase 1); bf16 decode == fp32
decode downcast because every e2m1 value and power-of-two scale is exactly
representable in bf16 — gated here, not assumed.

The recompute gate is DIFFERENTIAL: the same stack with a plain
decode-then-linear (weights saved for backward) must retain >> what the
recompute path retains, on identical bytes and shapes — mirroring the
2026-07-12 dequant-retention methodology.
"""
import hashlib

import pytest
import torch
import torch.nn.functional as F

from mxfp4_pack_ref import MX_BLOCK, dequant_mxfp4, quantize_pack_mxfp4  # noqa: E402
from mxfp4_qlora import (  # noqa: E402
    ExpertsMxfp4, ExpertsMxfp4LoRA, apply_gate_gptoss, lora_parameters,
)

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


# ---------------------------------------------------------------- fixtures --
def _make(E=8, H=128, I=128, seed=0):
    """Synthetic native-mxfp4 gpt-oss expert stack (kernel shapes)."""
    g = torch.Generator().manual_seed(seed)
    gu_w = torch.randn(E, 2 * I, H, generator=g) * 0.1     # [E, out, in]
    dn_w = torch.randn(E, H, I, generator=g) * 0.1
    gub = (torch.randn(E, 2 * I, generator=g) * 0.05).to(torch.bfloat16)
    dnb = (torch.randn(E, H, generator=g) * 0.05).to(torch.bfloat16)

    def pack(w):
        E_, N_, K_ = w.shape
        B = torch.empty(E_, N_, K_ // 2, dtype=torch.uint8)
        S = torch.empty(E_, N_, K_ // MX_BLOCK, dtype=torch.uint8)
        for e in range(E_):
            b, s = quantize_pack_mxfp4(w[e])
            B[e], S[e] = b.reshape(N_, K_ // 2), s
        return B, S

    gu_b, gu_s = pack(gu_w)
    dn_b, dn_s = pack(dn_w)
    return dict(gu_b=gu_b, gu_s=gu_s, dn_b=dn_b, dn_s=dn_s, gub=gub, dnb=dnb,
                E=E, H=H, I=I)


def _base(m, device):
    return ExpertsMxfp4(
        m["gu_b"].to(device), m["gu_s"].to(device),
        m["dn_b"].to(device), m["dn_s"].to(device),
        m["gub"].to(device), m["dnb"].to(device))


def _lora(m, device, r=8, mode="loop", seed=11):
    torch.manual_seed(seed)
    return ExpertsMxfp4LoRA(_base(m, device), r=r, alpha=16, mode=mode).to(device)


def _dense_bf16(m, which, e):
    """Decode one expert to the dense bf16 [out, in] weight (fp32 path, cast)."""
    n, k = ((2 * m["I"], m["H"]) if which == "gu" else (m["H"], m["I"]))
    b, s = (m["gu_b"], m["gu_s"]) if which == "gu" else (m["dn_b"], m["dn_s"])
    return dequant_mxfp4(b[e].reshape(n, k // MX_BLOCK, 16), s[e],
                         dtype=torch.float32).to(torch.bfloat16)


def _transformers_ref(m, device):
    """A real transformers v5 GptOssExperts holding the SAME weights, in ITS
    orientation ([E, in, out], x @ W[e])."""
    from types import SimpleNamespace
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts
    # _experts_implementation: the v5 use_experts_implementation wrapper reads
    # it at call time; "eager" dispatches to the original python forward (the
    # reference numerics we are gating against).
    cfg = SimpleNamespace(hidden_size=m["H"], intermediate_size=m["I"],
                          num_local_experts=m["E"], _experts_implementation="eager")
    ref = GptOssExperts(cfg)
    with torch.no_grad():
        gu = torch.stack([_dense_bf16(m, "gu", e) for e in range(m["E"])])
        dn = torch.stack([_dense_bf16(m, "dn", e) for e in range(m["E"])])
        ref.gate_up_proj.copy_(gu.transpose(1, 2).float())     # [E, in, out]
        ref.down_proj.copy_(dn.transpose(1, 2).float())
        ref.gate_up_proj_bias.copy_(m["gub"].float())
        ref.down_proj_bias.copy_(m["dnb"].float())
    return ref.to(device=device, dtype=torch.bfloat16)


def _route(m, T, k=4, seed=5, device="cuda"):
    g = torch.Generator().manual_seed(seed)
    x = (torch.randn(T, m["H"], generator=g) * 0.5).to(torch.bfloat16)
    logits = torch.randn(T, m["E"], generator=g)
    val, idx = torch.topk(logits, k, dim=-1)
    sc = torch.softmax(val, dim=-1).to(torch.bfloat16)
    return x.to(device), idx.to(device), sc.to(device)


def _b_rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-12)).item()


def _sha_all(module):
    return module.base.expert_bytes_sha256() if hasattr(module, "base") \
        else module.expert_bytes_sha256()


# ------------------------------------------------------------------- gates --
def test_bf16_decode_equals_fp32_downcast():
    """Every e2m1 codebook value scaled by 2^e is exactly representable in bf16
    (in-range), so decoding straight to bf16 must equal the fp32 decode cast."""
    m = _make(seed=2)
    for which in ("gu", "dn"):
        b, s = (m["gu_b"], m["gu_s"]) if which == "gu" else (m["dn_b"], m["dn_s"])
        n, k = ((2 * m["I"], m["H"]) if which == "gu" else (m["H"], m["I"]))
        for e in (0, m["E"] - 1):
            d32 = dequant_mxfp4(b[e].reshape(n, k // MX_BLOCK, 16), s[e],
                                dtype=torch.float32).to(torch.bfloat16)
            d16 = dequant_mxfp4(b[e].reshape(n, k // MX_BLOCK, 16), s[e],
                                dtype=torch.bfloat16)
            torch.testing.assert_close(d16, d32, rtol=0, atol=0)


def test_apply_gate_matches_transformers():
    """Epilogue parity on shared inputs — interleaved split, clamps, alpha."""
    from types import SimpleNamespace
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts
    cfg = SimpleNamespace(hidden_size=64, intermediate_size=64, num_local_experts=2)
    ref = GptOssExperts(cfg)
    g = torch.Generator().manual_seed(3)
    gu = torch.randn(5, 128, generator=g) * 4.0            # exercise both clamps
    torch.testing.assert_close(apply_gate_gptoss(gu), ref._apply_gate(gu),
                               rtol=0, atol=0)


@cuda
@pytest.mark.parametrize("T", [1, 64])
def test_forward_matches_transformers_reference(T):
    """Loop path (LoRA at zero-init) vs the real v5 module on the same bytes."""
    m = _make(seed=1)
    x, idx, sc = _route(m, T=T, seed=9)
    ref = _transformers_ref(m, "cuda")
    with torch.no_grad():
        want = ref(x, idx, sc)
    ours = _lora(m, "cuda")
    with torch.no_grad():
        got = ours(x, idx, sc)
    assert got.shape == want.shape and got.dtype == want.dtype
    assert _b_rel(got, want) < 2e-2, _b_rel(got, want)


@cuda
def test_lora_delta_exactly_zero_at_init():
    m = _make(seed=4)
    ours = _lora(m, "cuda")
    x = torch.randn(7, m["H"], dtype=torch.bfloat16, device="cuda")
    d = ours._lora(x, ours.gate_up_lora_A[0], ours.gate_up_lora_B[0])
    assert (d == 0).all()


@cuda
def test_grads_match_dense_autograd():
    """grad_x through the frozen recompute path vs the dense reference; LoRA
    A-grads exactly zero at B=0; B-grads finite and nonzero."""
    m = _make(seed=6)
    x, idx, sc = _route(m, T=32, seed=13)

    ref = _transformers_ref(m, "cuda")
    for p in ref.parameters():
        p.requires_grad_(False)
    x_ref = x.clone().requires_grad_(True)
    ref(x_ref, idx, sc).float().pow(2).sum().backward()

    ours = _lora(m, "cuda")
    x_ours = x.clone().requires_grad_(True)
    ours(x_ours, idx, sc).float().pow(2).sum().backward()

    assert _b_rel(x_ours.grad, x_ref.grad) < 2e-2, _b_rel(x_ours.grad, x_ref.grad)
    assert (ours.gate_up_lora_A.grad == 0).all() and (ours.down_lora_A.grad == 0).all()
    for g in (ours.gate_up_lora_B.grad, ours.down_lora_B.grad):
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


@cuda
def test_recompute_retains_no_dense_weights():
    """DIFFERENTIAL: a stack of layers under recompute must hold ~no dense
    expert weights between forward and backward; the same stack with plain
    decode-then-linear (weights saved by autograd) must retain them all."""
    m = _make(E=8, H=512, I=512, seed=7)
    layers = 6
    x0, idx, sc = _route(m, T=16, seed=17)

    class PlainProject(torch.nn.Module):
        """Same math, NO recompute Function — autograd saves the decoded weights."""

        def __init__(self, base):
            super().__init__()
            self.base = base

        def forward(self, x, ridx, rsc):
            b = self.base
            out = torch.zeros_like(x)
            mask = F.one_hot(ridx, num_classes=b.num_experts).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
            for e in hit:
                e = e[0]
                pos, tok = torch.where(mask[e])
                cs = x[tok]
                w1 = b._dequantize_expert(b.gate_up_blocks, b.gate_up_scales,
                                          b.n1, b.k1, e, x.device, x.dtype)
                gu = F.linear(cs, w1) + b.gate_up_bias[e].to(cs.dtype)
                h = apply_gate_gptoss(gu, b.alpha, b.limit)
                w2 = b._dequantize_expert(b.down_blocks, b.down_scales,
                                          b.n2, b.k2, e, x.device, x.dtype)
                dn = F.linear(h, w2) + b.down_bias[e].to(h.dtype)
                out.index_add_(0, tok, (dn * rsc[tok, pos, None]).to(x.dtype))
            return out

    def peak_bwd_minus_fwd(make_layer):
        torch.manual_seed(11)
        mods = [make_layer() for _ in range(layers)]
        torch.cuda.synchronize(); torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        x = x0.clone().requires_grad_(True)
        h = x
        for mod in mods:
            h = mod(h, idx, sc) + h
        peak_fwd = torch.cuda.max_memory_allocated()
        h.float().pow(2).sum().backward()
        peak_all = torch.cuda.max_memory_allocated()
        del mods, h, x
        return peak_fwd, peak_all

    fwd_r, all_r = peak_bwd_minus_fwd(lambda: _lora(m, "cuda", seed=11))
    fwd_p, all_p = peak_bwd_minus_fwd(lambda: PlainProject(_base(m, "cuda")))

    # Plain decode-then-linear's saved-for-backward dense weights accumulate
    # DURING forward (that is the retention), so the differential is total
    # peak arm-vs-arm: one dense gu+dn pair (bf16) per hit expert per layer.
    dense_per_expert = (2 * m["I"] * m["H"] + m["H"] * m["I"]) * 2  # bytes, bf16
    retained_layer = m["E"] * dense_per_expert
    assert all_p > all_r + 3 * retained_layer, (all_p, all_r, retained_layer)
    # Recompute's own fwd->bwd growth stays under one layer's dense worth
    # (backward decodes one expert at a time, then frees it).
    assert all_r - fwd_r < retained_layer, (all_r - fwd_r, retained_layer)


@cuda
def test_bytes_bitidentical_after_training_steps():
    m = _make(seed=8)
    ours = _lora(m, "cuda")
    pre = _sha_all(ours)
    b_pre = ours.gate_up_lora_B.detach().clone()
    opt = torch.optim.AdamW(lora_parameters(ours), lr=1e-2)
    x, idx, sc = _route(m, T=16, seed=19)
    y = torch.randn_like(x)
    for _ in range(5):
        opt.zero_grad(set_to_none=True)
        loss = (ours(x, idx, sc).float() - y.float()).pow(2).mean()
        loss.backward()
        opt.step()
    post = _sha_all(ours)
    assert pre == post, "frozen native bytes changed under training"
    assert not torch.equal(b_pre, ours.gate_up_lora_B.detach()), "adapters did not train"


@cuda
def test_loss_descends():
    m = _make(seed=9)
    ours = _lora(m, "cuda")
    opt = torch.optim.Adam(lora_parameters(ours), lr=5e-3)
    x, idx, sc = _route(m, T=32, seed=23)
    y = torch.randn_like(x) * 0.1
    losses = []
    for _ in range(40):
        opt.zero_grad(set_to_none=True)
        loss = (ours(x, idx, sc).float() - y.float()).pow(2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.6 * losses[0], (losses[0], losses[-1])


@cuda
@pytest.mark.parametrize("T", [1, 48])
def test_fused_matches_loop(T):
    """Fused grouped forward (Phase-2 kernel) + recompute backward vs the loop
    path: same module state, same inputs — outputs and grads in band."""
    pytest.importorskip("triton")
    m = _make(seed=10)
    loop = _lora(m, "cuda", mode="loop", seed=31)
    fused = _lora(m, "cuda", mode="fused", seed=31)
    with torch.no_grad():  # same adapter state (seeded identically; assert anyway)
        for pl, pf in zip(loop.parameters(), fused.parameters()):
            assert torch.equal(pl, pf)

    x, idx, sc = _route(m, T=T, seed=29)

    xl = x.clone().requires_grad_(True)
    ol = loop(xl, idx, sc)
    ol.float().pow(2).sum().backward()

    xf = x.clone().requires_grad_(True)
    of = fused(xf, idx, sc)
    of.float().pow(2).sum().backward()

    assert _b_rel(of, ol) < 2e-2, _b_rel(of, ol)
    assert _b_rel(xf.grad, xl.grad) < 3e-2, _b_rel(xf.grad, xl.grad)
    for a, b in ((fused.gate_up_lora_B.grad, loop.gate_up_lora_B.grad),
                 (fused.down_lora_B.grad, loop.down_lora_B.grad)):
        assert _b_rel(a, b) < 3e-2, _b_rel(a, b)


def test_provenance_module_hashes_match_file_ranges(tmp_path):
    """Module-level hash table == the safetensors file's data-section byte
    ranges for the same tensors (the Phase-3 primitive, at the training
    module's boundary)."""
    pytest.importorskip("safetensors")
    from safetensors.torch import save_file
    from mxfp4_loader import file_tensor_sha256

    m = _make(seed=12)
    names = {
        "gate_up_blocks": m["gu_b"], "gate_up_scales": m["gu_s"],
        "down_blocks": m["dn_b"], "down_scales": m["dn_s"],
        "gate_up_bias": m["gub"], "down_bias": m["dnb"],
    }
    p = str(tmp_path / "experts.safetensors")
    save_file(names, p)
    module = _base(m, "cpu")
    table = module.expert_bytes_sha256()
    for name in names:
        assert table[name] == file_tensor_sha256(p, name), name
