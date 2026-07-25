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
from nf4_kv import (PERCHANNEL_GROUP, attend_nf4_kv, attend_nf4_kv_fused,
                    attend_nf4_kv_split, attend_nf4_kv_gqa,
                    dequant_kv_ref,
                    kv_cache_bytes, kv_cache_bytes_perchannel, kv_scores_nf4,
                    kv_weighted_sum_nf4, quantize_kv, quantize_kv_perchannel)


def _rel(got, want):
    return ((got.float() - want.float()).norm() / want.float().norm()).item()
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


@cuda
def test_strided_inner_dim_is_rejected_not_silently_wrong():
    """Bugbot #14: the kernels assume a packed innermost dim. A view that breaks
    that assumption must fail loudly — silent wrong scores are the worst outcome
    for a cache, since nothing downstream can detect them."""
    T, H, D = 256, 4, 128
    _, kp, ka = _cache(T, H, D, seed=1)
    bad_p = kp[:, :, ::2]                             # strided packed dim
    q = torch.randn(H, D, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="contiguous innermost dim"):
        kv_scores_nf4(q, bad_p, ka)
    with pytest.raises(ValueError, match="contiguous innermost dim"):
        kv_weighted_sum_nf4(torch.softmax(torch.randn(H, T, device="cuda"), -1),
                            bad_p, ka, D)


@cuda
def test_outer_slicing_still_works():
    """The complement of the guard: token-axis slicing (what a sliding window or
    an evicted prefix produces) keeps stride(-1)==1 and must stay supported —
    the guard would be useless if it also blocked the legitimate case."""
    T, H, D, keep = 256, 4, 128, 128
    _, kp, ka = _cache(T, H, D, seed=3)
    _, vp, va = _cache(T, H, D, seed=4)
    q = torch.randn(H, D, device="cuda", dtype=torch.float32)
    got = attend_nf4_kv(q, kp[keep:], ka[keep:], vp[keep:], va[keep:])
    k = dequant_kv_ref(kp[keep:], ka[keep:], D).float()
    v = dequant_kv_ref(vp[keep:], va[keep:], D).float()
    probs = torch.softmax(torch.einsum("hd,thd->ht", q, k) * D ** -0.5, dim=-1)
    ref = torch.einsum("ht,thd->hd", probs, v)
    rel = ((got - ref).norm() / ref.norm()).item()
    assert rel < 1e-4, f"sliced-cache attention diverged: {rel}"


@cuda
def test_reference_and_kernel_share_a_validity_domain():
    """The oracle must reject exactly what the kernels reject. If dequant_kv_ref
    silently accepted a strided view (reshape would copy it) while the kernels
    raised, the two would disagree about which inputs are legal -- and any test
    comparing them on such an input compares a value against an exception."""
    _, kp, ka = _cache(128, 4, 128, seed=11)
    bad = kp[:, :, ::2]
    with pytest.raises(ValueError, match="contiguous innermost dim"):
        dequant_kv_ref(bad, ka, 128)


def _outlier_cache(T, H, D, seed=0, outlier_channels=(3, 17, 90), gain=12.0):
    """A cache with loud CHANNELS whose magnitude is CONSTANT across tokens.

    Read the caveat before trusting any result built on this. Real keys are not
    like this: the gain here is token-invariant, which is the single regime
    where per-channel scaling is guaranteed to win. On a real model it LOSES
    (measured +0.275 ppl vs +0.083 for per-token, docs/context-budgets.md
    finding #9), because real key magnitude also varies strongly across tokens,
    so grouping a channel's scale over 64 tokens lets one loud token spoil the
    other 63 — the same failure this fixture was built to show per-token
    scaling having, just on the other axis.

    Kept as a MECHANISM test only: it pins that per-channel scaling does what
    it claims when its precondition holds. It is not evidence the precondition
    holds anywhere real."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(T, H, D, generator=g) * 0.5
    for c in outlier_channels:
        if c < D:                       # fixture must work at head_dim 64 too
            x[:, :, c] *= gain
    return x.cuda().bfloat16()


@cuda
@pytest.mark.parametrize("T,H,D", [(256, 4, 128), (192, 2, 64), (130, 8, 256)])
def test_perchannel_packs_bytes_identically_to_per_token(T, H, D):
    """Only the SCALES are grouped differently; the nibble stream layout is the
    same, which is what lets one kernel read both. (Values differ, shapes must
    not.) Also covers T not a multiple of the group (130)."""
    x = _outlier_cache(T, H, D, seed=1)
    p_tok, a_tok = quantize_kv(x)
    p_ch, a_ch = quantize_kv_perchannel(x)
    assert p_ch.shape == p_tok.shape and p_ch.dtype == p_tok.dtype
    n_grp = (T + PERCHANNEL_GROUP - 1) // PERCHANNEL_GROUP
    assert tuple(a_ch.shape) == (n_grp, H, D)


@cuda
def test_perchannel_absmax_is_free_at_any_head_dim():
    """The 'free fidelity' claim, asserted rather than left as prose. Both
    schemes store one fp32 scale per 64 quantized values (64 channels within a
    token vs 64 tokens within a channel), so the cost is identical for EVERY
    head_dim — not just the 128 case that motivated it."""
    T, H = 4096, 8
    for D in (64, 128, 256):
        assert kv_cache_bytes_perchannel(T, H, D) == kv_cache_bytes(T, H, D) // 2, D
    # the equality is a property of group == BLOCKSIZE, not a coincidence
    assert kv_cache_bytes_perchannel(T, H, 128, group=32) > \
        kv_cache_bytes_perchannel(T, H, 128, group=64)
    assert kv_cache_bytes_perchannel(T, H, 128, group=128) < \
        kv_cache_bytes_perchannel(T, H, 128, group=64)


@cuda
@pytest.mark.parametrize("T,H_q,H_kv,D", [(256, 32, 4, 128), (130, 8, 8, 128)])
def test_perchannel_scores_match_dequant_oracle(T, H_q, H_kv, D):
    x = _outlier_cache(T, H_kv, D, seed=2)
    p, a = quantize_kv_perchannel(x)
    q = torch.randn(H_q, D, device="cuda", dtype=torch.float32)
    got = kv_scores_nf4(q, p, a, token_group=PERCHANNEL_GROUP)
    k = dequant_kv_ref(p, a, D, token_group=PERCHANNEL_GROUP).float()
    rep = H_q // H_kv
    ref = torch.einsum("hd,thd->ht", q, k.repeat_interleave(rep, dim=1))
    assert ((got - ref).norm() / ref.norm()).item() < 1e-5


@cuda
def test_perchannel_beats_per_token_when_its_precondition_holds():
    """Mechanism check, NOT a recommendation. The fixture is token-invariant by
    construction, which is exactly the precondition per-channel scaling needs;
    real keys violate it and per-channel loses there. See _outlier_cache."""
    x = _outlier_cache(1024, 4, 128, seed=3)
    e_tok = _rel(dequant_kv_ref(*quantize_kv(x), 128), x)
    e_ch = _rel(dequant_kv_ref(*quantize_kv_perchannel(x), 128,
                               token_group=PERCHANNEL_GROUP), x)
    # measured 0.092 vs 0.163 on this fixture; bound set from that, not guessed
    assert e_ch < 0.75 * e_tok, f"per-channel {e_ch:.4f} vs per-token {e_tok:.4f}"


@cuda
def test_asymmetric_grouping_per_channel_K_per_token_V():
    """The configuration finding #7 actually recommends: keys per-channel,
    values per-token, in one attend call."""
    T, H, D = 256, 4, 128
    kx, vx = _outlier_cache(T, H, D, 4), _outlier_cache(T, H, D, 5)
    kp, ka = quantize_kv_perchannel(kx)
    vp, va = quantize_kv(vx)
    q = torch.randn(H, D, device="cuda", dtype=torch.float32)
    got = attend_nf4_kv(q, kp, ka, vp, va, k_token_group=PERCHANNEL_GROUP)
    k = dequant_kv_ref(kp, ka, D, token_group=PERCHANNEL_GROUP).float()
    v = dequant_kv_ref(vp, va, D).float()
    probs = torch.softmax(torch.einsum("hd,thd->ht", q, k) * D ** -0.5, dim=-1)
    ref = torch.einsum("ht,thd->hd", probs, v)
    assert ((got - ref).norm() / ref.norm()).item() < 1e-4


@cuda
def test_mismatched_absmax_layout_is_rejected():
    """Two legal absmax layouts now exist for identical packed bytes, so using
    one while configured for the other must fail loudly — on the per-token path
    a per-channel absmax is also short, i.e. an out-of-bounds read."""
    x = _outlier_cache(256, 4, 128, seed=6)
    p_ch, a_ch = quantize_kv_perchannel(x)
    _, a_tok = quantize_kv(x)
    q = torch.randn(4, 128, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="does not match per-token blockwise"):
        kv_scores_nf4(q, p_ch, a_ch)                     # per-channel scales, no flag
    with pytest.raises(ValueError, match="does not match per-channel"):
        kv_scores_nf4(q, p_ch, a_tok, token_group=PERCHANNEL_GROUP)


@cuda
def test_k_and_v_token_counts_must_agree():
    T, H, D = 256, 4, 128
    kp, ka = quantize_kv(_outlier_cache(T, H, D, 7))
    vp, va = quantize_kv(_outlier_cache(T - 8, H, D, 8))
    q = torch.randn(H, D, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="different token counts"):
        attend_nf4_kv(q, kp, ka, vp, va)


@cuda
def test_kv_head_count_mismatch_is_rejected():
    """Each kernel derives GQA from its own tensor, so mismatched K/V kv-head
    counts silently map one query head onto different K and V rows."""
    T, D = 128, 128
    kp, ka = quantize_kv(_outlier_cache(T, 4, D, 20))
    vp, va = quantize_kv(_outlier_cache(T, 8, D, 21))
    q = torch.randn(8, D, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="different kv-head counts"):
        attend_nf4_kv(q, kp, ka, vp, va)


@cuda
def test_reference_rejects_mismatched_absmax_layout_like_the_kernels():
    """Domain agreement again, now for SHAPE rather than stride: the oracle must
    not dequantize per-channel scales as if they were per-token."""
    x = _outlier_cache(256, 4, 128, seed=22)
    p_ch, a_ch = quantize_kv_perchannel(x)
    with pytest.raises(ValueError, match="does not match per-token blockwise"):
        dequant_kv_ref(p_ch, a_ch, 128)


@cuda
@pytest.mark.parametrize("T,H_q,H_kv,D", SHAPES)
def test_fused_matches_two_pass(T, H_q, H_kv, D):
    """The fused kernel must agree with the two-pass path it replaces. Online
    softmax reorders the reduction, so this is a numerics check as well as a
    correctness one — a wrong rescale still produces a valid distribution."""
    _, kp, ka = _cache(T, H_kv, D, seed=30)
    _, vp, va = _cache(T, H_kv, D, seed=31)
    q = torch.randn(H_q, D, device="cuda", dtype=torch.float32) * 0.5
    two = attend_nf4_kv(q, kp, ka, vp, va)
    one = attend_nf4_kv_fused(q, kp, ka, vp, va)
    assert ((one - two).norm() / two.norm()).item() < 2e-3


@cuda
def test_fused_survives_extreme_logits():
    """Online softmax exists to be numerically stable; a large scale makes the
    running-max rescale load-bearing rather than incidental."""
    T, H, D = 512, 4, 128
    _, kp, ka = _cache(T, H, D, seed=32)
    _, vp, va = _cache(T, H, D, seed=33)
    q = torch.randn(H, D, device="cuda", dtype=torch.float32) * 40.0
    one = attend_nf4_kv_fused(q, kp, ka, vp, va, scale=1.0)
    two = attend_nf4_kv(q, kp, ka, vp, va, scale=1.0)
    assert torch.isfinite(one).all()
    assert ((one - two).norm() / two.norm()).item() < 2e-3


@cuda
def test_fused_rejects_mixed_scaling_modes():
    T, H, D = 256, 4, 128
    kp, ka = quantize_kv_perchannel(_outlier_cache(T, H, D, 34))
    _, vp, va = _cache(T, H, D, seed=35)
    q = torch.randn(H, D, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="one scaling mode"):
        attend_nf4_kv_fused(q, kp, ka, vp, va, k_token_group=PERCHANNEL_GROUP)


@cuda
@pytest.mark.parametrize("T,H_q,H_kv,D", SHAPES)
def test_split_matches_two_pass(T, H_q, H_kv, D):
    """B5d: the combine step is a second place for the softmax rescale to be
    wrong, and a wrong merge still yields a plausible vector."""
    _, kp, ka = _cache(T, H_kv, D, seed=40)
    _, vp, va = _cache(T, H_kv, D, seed=41)
    q = torch.randn(H_q, D, device="cuda", dtype=torch.float32) * 0.5
    two = attend_nf4_kv(q, kp, ka, vp, va)
    got = attend_nf4_kv_split(q, kp, ka, vp, va)
    assert ((got - two).norm() / two.norm()).item() < 2e-3


@cuda
@pytest.mark.parametrize("splits", [1, 2, 3, 8, 64])
def test_split_count_does_not_change_the_answer(splits):
    """Partitioning is an implementation detail; any split count must agree.
    Includes splits > blocks available, and a non-divisor (3)."""
    T, H, D = 512, 4, 128
    _, kp, ka = _cache(T, H, D, seed=42)
    _, vp, va = _cache(T, H, D, seed=43)
    q = torch.randn(H, D, device="cuda", dtype=torch.float32) * 0.5
    ref = attend_nf4_kv(q, kp, ka, vp, va)
    got = attend_nf4_kv_split(q, kp, ka, vp, va, splits=splits)
    assert ((got - ref).norm() / ref.norm()).item() < 2e-3, splits


@cuda
def test_split_survives_extreme_logits():
    T, H, D = 4096, 4, 128
    _, kp, ka = _cache(T, H, D, seed=44)
    _, vp, va = _cache(T, H, D, seed=45)
    q = torch.randn(H, D, device="cuda", dtype=torch.float32) * 40.0
    got = attend_nf4_kv_split(q, kp, ka, vp, va, scale=1.0)
    ref = attend_nf4_kv(q, kp, ka, vp, va, scale=1.0)
    assert torch.isfinite(got).all()
    assert ((got - ref).norm() / ref.norm()).item() < 2e-3


@cuda
@pytest.mark.parametrize("T,H_q,H_kv,D", SHAPES)
def test_gqa_batched_matches_two_pass(T, H_q, H_kv, D):
    """B6d. Covers GQA 1:1 (BLOCK_M padded to the tl.dot minimum of 16) as well
    as 16:1, since the padding mask is where a batched kernel goes wrong."""
    _, kp, ka = _cache(T, H_kv, D, seed=50)
    _, vp, va = _cache(T, H_kv, D, seed=51)
    q = torch.randn(H_q, D, device="cuda", dtype=torch.float32) * 0.5
    ref = attend_nf4_kv(q, kp, ka, vp, va)
    got = attend_nf4_kv_gqa(q, kp, ka, vp, va)
    assert ((got - ref).norm() / ref.norm()).item() < 2e-3


@cuda
def test_gqa_batched_survives_extreme_logits():
    T, H_q, H_kv, D = 4096, 32, 4, 128
    _, kp, ka = _cache(T, H_kv, D, seed=52)
    _, vp, va = _cache(T, H_kv, D, seed=53)
    q = torch.randn(H_q, D, device="cuda", dtype=torch.float32) * 40.0
    got = attend_nf4_kv_gqa(q, kp, ka, vp, va, scale=1.0)
    ref = attend_nf4_kv(q, kp, ka, vp, va, scale=1.0)
    assert torch.isfinite(got).all()
    assert ((got - ref).norm() / ref.norm()).item() < 2e-3


@cuda
def test_gqa_tf32_is_measurably_worse_than_ieee():
    """Records WHY ieee is the default: tf32's ~10-bit mantissa is a second
    error source on top of quantization, and at extreme logits it exceeds the
    tolerance the ieee path meets. Kept as a test so the default cannot be
    flipped for speed without the number showing up."""
    T, H_q, H_kv, D = 4096, 32, 4, 128
    _, kp, ka = _cache(T, H_kv, D, seed=52)
    _, vp, va = _cache(T, H_kv, D, seed=53)
    q = torch.randn(H_q, D, device="cuda", dtype=torch.float32) * 40.0
    ref = attend_nf4_kv(q, kp, ka, vp, va, scale=1.0)
    e_ieee = _rel(attend_nf4_kv_gqa(q, kp, ka, vp, va, scale=1.0, precision="ieee"), ref)
    e_tf32 = _rel(attend_nf4_kv_gqa(q, kp, ka, vp, va, scale=1.0, precision="tf32"), ref)
    assert e_ieee < 2e-3, e_ieee
    assert e_tf32 > e_ieee, (e_tf32, e_ieee)
