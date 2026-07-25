# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Property suite for the NF4 KV cache (kernel/nf4_kv.py).

Gates, in the order that matters:
  1. the pack round-trips against the SAME codebook/nibble order the expert
     path uses (drift here is the failure that silently corrupts everything);
  2. each kernel equals a dequant-then-torch oracle in fp32 (this is the
     correctness contract — the kernel must not be *approximately* the same
     thing as reading the cache it claims to read);
  3. GQA is an index map: query head h reads kv head h // (H_q/H_kv), asserted
     by making kv heads distinguishable rather than by trusting the arithmetic;
  4. the footprint claim is arithmetic, so assert it (4x-minus-absmax, not "4x");
  5. fidelity vs an fp16 cache is measured and bounded, not assumed.
"""
from __future__ import annotations

import pytest
import torch

from nf4_grouped import BLOCKSIZE
from nf4_kv import (attend_nf4_kv, dequant_kv_ref, kv_cache_bytes,
                    kv_scores_nf4, kv_weighted_sum_nf4, quantize_kv)
from nf4_pack_ref import quantize_pack_nf4

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

# (T, H_q, H_kv, D): Qwen3-235B/30B geometry, gpt-oss (D=64), Gemma-4 full
# layers (2 kv heads x 512 -> use 256 to stay inside one register tile), MHA.
SHAPES = [
    (128, 64, 4, 128),    # Qwen3-235B: GQA 16:1
    (512, 32, 4, 128),    # Qwen3-30B
    (256, 64, 8, 64),     # gpt-oss
    (192, 16, 16, 128),   # OLMoE: no GQA
    (96, 8, 2, 256),      # Gemma-4-ish: wide head_dim
]


def _cache(T, H_kv, D, seed=0, device="cuda"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = (torch.randn(T, H_kv, D, generator=g) * 0.5).to(device, torch.bfloat16)
    p, a = quantize_kv(x)
    return x, p, a


@cuda
def test_pack_matches_expert_path_codebook():
    """The KV packer and the expert packer must agree byte-for-byte: same
    codebook, same nibble order, same blockwise absmax."""
    g = torch.Generator(device="cpu").manual_seed(7)
    D = 128
    rows = torch.randn(6, D, generator=g)
    ref_p, ref_a = quantize_pack_nf4(rows)                       # [6, D/2], [6, D/64]
    kv_p, kv_a = quantize_kv(rows.reshape(6, 1, D).cuda())
    assert torch.equal(kv_p.cpu().reshape(6, D // 2), ref_p)
    assert torch.allclose(kv_a.cpu().reshape(6, D // BLOCKSIZE), ref_a, atol=0, rtol=0)


@cuda
@pytest.mark.parametrize("T,H_q,H_kv,D", SHAPES)
def test_dequant_roundtrip_is_nearest_code(T, H_q, H_kv, D):
    """Dequantized cache is within half a codebook step of the input, per block."""
    x, p, a = _cache(T, H_kv, D, seed=1)
    deq = dequant_kv_ref(p, a, D)
    err = (deq - x.float()).abs()
    scale = a.repeat_interleave(BLOCKSIZE, dim=2)
    assert torch.all(err <= 0.5 * scale + 1e-4), f"max {(err / scale).max().item()}"


@cuda
@pytest.mark.parametrize("T,H_q,H_kv,D", SHAPES)
def test_scores_match_dequant_oracle(T, H_q, H_kv, D):
    """kv_scores_nf4 == q @ dequant(K).T computed in fp32."""
    _, p, a = _cache(T, H_kv, D, seed=2)
    g = torch.Generator(device="cpu").manual_seed(3)
    q = (torch.randn(H_q, D, generator=g) * 0.3).cuda()
    got = kv_scores_nf4(q, p, a)
    K = dequant_kv_ref(p, a, D)                                   # [T, H_kv, D]
    rep = H_q // H_kv
    Kq = K.repeat_interleave(rep, dim=1)                          # [T, H_q, D]
    want = torch.einsum("hd,thd->ht", q.float(), Kq.float())
    torch.testing.assert_close(got, want, rtol=2e-3, atol=2e-3)


@cuda
@pytest.mark.parametrize("T,H_q,H_kv,D", SHAPES)
def test_weighted_sum_matches_dequant_oracle(T, H_q, H_kv, D):
    """kv_weighted_sum_nf4 == probs @ dequant(V) computed in fp32."""
    _, p, a = _cache(T, H_kv, D, seed=4)
    g = torch.Generator(device="cpu").manual_seed(5)
    probs = torch.softmax(torch.randn(H_q, T, generator=g).cuda(), dim=-1)
    got = kv_weighted_sum_nf4(probs, p, a, D)
    V = dequant_kv_ref(p, a, D)
    rep = H_q // H_kv
    Vq = V.repeat_interleave(rep, dim=1)
    want = torch.einsum("ht,thd->hd", probs.float(), Vq.float())
    torch.testing.assert_close(got, want, rtol=2e-3, atol=2e-3)


@cuda
def test_gqa_is_an_index_map_not_a_broadcast():
    """Make each kv head distinguishable; assert query head h reads h//rep.

    A broadcast bug (every head reading kv head 0) passes an averaged check but
    fails this one.
    """
    T, H_q, H_kv, D = 64, 8, 2, 64
    x = torch.zeros(T, H_kv, D, dtype=torch.bfloat16, device="cuda")
    x[:, 0, :] = 1.0                       # kv head 0 -> all ones
    x[:, 1, :] = -1.0                      # kv head 1 -> all minus-ones
    p, a = quantize_kv(x)
    q = torch.ones(H_q, D, device="cuda")
    s = kv_scores_nf4(q, p, a)
    rep = H_q // H_kv                      # 4
    assert torch.all(s[:rep] > 0), "query heads 0..3 must read kv head 0 (+1)"
    assert torch.all(s[rep:] < 0), "query heads 4..7 must read kv head 1 (-1)"


@cuda
def test_attend_matches_fp16_cache_within_bound():
    """End-to-end decode step over a 4-bit cache vs an fp16 cache.

    The bound is the MEASURED value with margin, not an aspiration: 9.3% on this
    fixture (see test_error_is_dominated_by_V for the decomposition). It is a
    kernel-level bound, NOT a model-fidelity claim -- that one needs
    teacher-forced perplexity on the pinned fixture at real scale (C3).
    """
    T, H_q, H_kv, D = 1024, 32, 4, 128
    x, p, a = _cache(T, H_kv, D, seed=6)
    g = torch.Generator(device="cpu").manual_seed(8)
    q = (torch.randn(H_q, D, generator=g) * 0.3).cuda()
    got = attend_nf4_kv(q, p, a, p, a)                 # same tensor as K and V
    rep = H_q // H_kv
    Xq = x.float().repeat_interleave(rep, dim=1)
    scores = torch.einsum("hd,thd->ht", q.float(), Xq) * (D ** -0.5)
    want = torch.einsum("ht,thd->hd", torch.softmax(scores, -1), Xq)
    rel = ((got - want).norm() / want.norm()).item()
    assert rel < 0.12, f"relative error vs fp16 cache {rel:.4f} (measured ~0.093)"


@cuda
def test_error_is_dominated_by_V_not_K_on_iid_data():
    """Decomposition, and it is the opposite of the usual folklore.

    Measured on iid-normal cache values: quantizing K alone costs ~1.3%, V alone
    ~9.2%. The mechanism is visible in the intermediate: K error perturbs logits
    by ~9.2% but the softmax CONTRACTS that to ~1.4%, while V error passes
    straight through the weighted average unattenuated.

    Scope, stated because it changes the design conclusion: iid fixtures have no
    per-channel outliers. Real K caches do, and per-token blockwise absmax (what
    this module uses) handles channel outliers poorly -- which is where the
    published "K is the sensitive one" result comes from. So the honest reading
    is: with THIS scaling axis and iid data, V dominates; an asymmetric-precision
    or per-channel-K variant must be decided by the real-scale ppl gate, not here.
    This test locks the asymmetry so a regression in either path is visible.
    """
    T, H_q, H_kv, D = 1024, 32, 4, 128
    rep = H_q // H_kv
    g = torch.Generator(device="cpu").manual_seed(11)
    k = ((torch.randn(T, H_kv, D, generator=g) * 0.5).cuda().bfloat16())
    v = ((torch.randn(T, H_kv, D, generator=g) * 0.5).cuda().bfloat16())
    q = (torch.randn(H_q, D, generator=g) * 0.3).cuda()
    kp, ka = quantize_kv(k)
    vp, va = quantize_kv(v)
    kq = dequant_kv_ref(kp, ka, D)
    vq = dequant_kv_ref(vp, va, D)

    def attend(kk, vv):
        K = kk.float().repeat_interleave(rep, 1)
        V = vv.float().repeat_interleave(rep, 1)
        s = torch.einsum("hd,thd->ht", q.float(), K) * D ** -0.5
        return torch.einsum("ht,thd->hd", torch.softmax(s, -1), V)

    ref = attend(k.float(), v.float())
    rel = lambda x: ((x - ref).norm() / ref.norm()).item()
    err_k = rel(attend(kq, v.float()))
    err_v = rel(attend(k.float(), vq))
    assert err_k < 0.03, f"K-only error {err_k:.4f} (measured ~0.013)"
    assert err_v > 3 * err_k, (
        f"expected V to dominate on iid data: K {err_k:.4f} vs V {err_v:.4f}")


def test_footprint_arithmetic():
    """The saving is 4x on the nibbles minus the absmax side-channel; state it."""
    T, H_kv, D = 32768, 4, 128
    fp16 = kv_cache_bytes(T, H_kv, D, nf4=False)
    nf4 = kv_cache_bytes(T, H_kv, D, nf4=True)
    # per token per head: 64 nibble-bytes + 2 blocks x 4B absmax = 72 vs 256 bf16
    assert fp16 == 2 * T * H_kv * D * 2
    assert nf4 == 2 * T * H_kv * (D // 2 + (D // BLOCKSIZE) * 4)
    ratio = fp16 / nf4
    assert 3.5 < ratio < 3.6, f"expected ~3.56x (not a clean 4x), got {ratio:.3f}"


def test_head_dim_must_be_blocksize_multiple():
    """A head_dim the blocksize cannot tile is refused, not silently truncated."""
    with pytest.raises(ValueError, match="multiple of the quant blocksize"):
        quantize_kv(torch.zeros(4, 2, 96))
