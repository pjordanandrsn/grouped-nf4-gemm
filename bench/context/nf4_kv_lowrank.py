# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).

"""Low-rank KV codes — BUILT, MEASURED, AND REJECTED. Not a shipping tier.

This lives in ``bench/`` rather than ``kernel/`` on purpose: the algebra below
is correct and tested, but the compression premise it rests on does not hold on
real caches, and code in ``kernel/`` inherits the repo's bit-accurate framing by
adjacency. Keeping it here records the negative result so the idea is not
re-derived, without implying it is usable.

THE IDEA. NF4 compresses each value's precision; it cannot exploit a cache
living in a low-dimensional subspace. Cache rank-r codes ``C`` against a pinned
per-head basis ``B`` (``K ~= C @ B``) and the up-projection never touches the
cache::

    scores = q @ K.T = q @ (C @ B).T = (q @ B.T) @ C.T
    out    = p @ V   = p @ (C @ B)   = (p @ C) @ B

Exact algebra — only the rank truncation approximates. The cache read becomes
r-dimensional, so it is smaller *and* cheaper to read. Codes are just a narrower
``[T, H, r]`` tensor, so they feed the existing ``nf4_kv`` kernels unchanged.

WHY IT DOES NOT WORK. Measured on OLMoE-1B-7B (1024 wikitext tokens, per-head
SVD, D=128), basis fit on the first 512 tokens and scored on the next 512 —
see ``lowrank_probe.py`` and ``receipts-lowrank-20260724/``:

* rank 64 (2x): held-out error **K 48.2% / V 43.0%**, against NF4's 3.56x at a
  model-level cost of +0.124 ppl. Dominated on both axes at once.
* The rank sweep finds no operating point. V reaches NF4-parity error only at
  **rank 124 — a 1.03x saving**. Ranks the 64-element blocksize cannot pack were
  measured too, so the packer is demonstrably not the binding constraint; the
  data is.
* Even an *oracle* basis fit to the very tokens it is scored on costs 21-23% at
  rank 64. So this is not merely a generalization failure: post-hoc, KV is not
  low-rank enough at any rank that saves anything.

WHAT THE MEASUREMENT DID FIND. Keys are strongly low-rank **before** RoPE:
16.9% held-out at rank 64, against 48.2% after. Rotary embedding spreads
identical content across directions by position, inflating apparent rank ~3x.
That independently reproduces, from the outside, why MLA carries a *decoupled*
RoPE key — rotation does not commute with the projection. It is still not
cashable post-hoc: storing pre-RoPE codes forces an r->D lift before rotating,
which forfeits the absorption (the memory saving would survive, but V has no
RoPE excuse at 43%, so halving K alone is ~1.33x for real machinery).

The general lesson, since it generalizes past this module: rank works when a
model is *trained* with the bottleneck (MLA), and post-hoc SVD does not recover
a structure the model was never trained to have.

WHAT IS KEPT. The absorption implementation and its tests
(``test_nf4_kv_lowrank.py``, 11/11) are correct and stay useful the day a
trained bottleneck exists — at that point the codes are the model's own and the
truncation error is zero by construction.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "kernel"))

import torch

from nf4_grouped import BLOCKSIZE
from nf4_kv import kv_scores_nf4, kv_weighted_sum_nf4, quantize_kv

__all__ = [
    "calibrate_basis",
    "project_to_codes",
    "pack_lowrank_kv",
    "reconstruct_ref",
    "attend_lowrank_nf4",
    "lowrank_cache_bytes",
    "energy_retained",
]


def _check_rank(rank: int, head_dim: int) -> None:
    if rank % BLOCKSIZE != 0:
        raise ValueError(
            f"rank={rank} must be a multiple of the quant blocksize {BLOCKSIZE} "
            "so the codes can be NF4-packed; use 64/128/... (a sub-block packer "
            "would be needed for finer ranks)."
        )
    if rank > head_dim:
        raise ValueError(f"rank={rank} exceeds head_dim={head_dim}")


def calibrate_basis(sample: torch.Tensor, rank: int) -> torch.Tensor:
    """Per-head orthonormal basis from sample cache values.

    ``sample [T, H, D]`` -> ``B [H, rank, D]``, the top-`rank` right singular
    vectors per head. Calibrated once, offline, then pinned: the basis is part
    of the engine's configuration, not per-token state.
    """
    if sample.dim() != 3:
        raise ValueError(f"expected [T, H, D] sample; got {tuple(sample.shape)}")
    T, H, D = sample.shape
    _check_rank(rank, D)
    B = torch.empty(H, rank, D, dtype=torch.float32, device=sample.device)
    for h in range(H):
        # economy SVD of [T, D]; rows of Vh are the principal directions
        _, _, Vh = torch.linalg.svd(sample[:, h, :].float(), full_matrices=False)
        B[h] = Vh[:rank]
    return B


def energy_retained(sample: torch.Tensor, rank: int) -> float:
    """Fraction of squared energy the rank-`rank` subspace keeps (mean over heads).

    This is the dial the fidelity gate reads: it predicts the truncation error
    before any model is run.
    """
    T, H, D = sample.shape
    tot = 0.0
    for h in range(H):
        s = torch.linalg.svdvals(sample[:, h, :].float())
        e = (s ** 2)
        tot += float(e[:rank].sum() / e.sum())
    return tot / H


def project_to_codes(x: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """``x [T, H, D]`` and ``B [H, r, D]`` -> codes ``[T, H, r]`` (x @ B.T per head)."""
    return torch.einsum("thd,hrd->thr", x.float(), B.float())


def pack_lowrank_kv(x: torch.Tensor, B: torch.Tensor):
    """Project then NF4-pack: the two compression axes applied in order."""
    return quantize_kv(project_to_codes(x, B).to(x.dtype))


def reconstruct_ref(codes: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """codes ``[T, H, r]`` @ basis -> ``[T, H, D]``. Reference only; the attention
    path never calls this, which is the whole point of the absorption."""
    return torch.einsum("thr,hrd->thd", codes.float(), B.float())


def lowrank_cache_bytes(n_tokens: int, kv_heads: int, rank: int,
                        head_dim: int, n_layers: int = 1) -> dict:
    """Footprint accounting for the composed scheme, including the pinned basis.

    The basis is a per-layer constant (``H * r * D`` fp32), so it amortizes over
    context; reported separately rather than folded in, because at short context
    it is not negligible.
    """
    per_tok = 2 * kv_heads * (rank // 2 + (rank // BLOCKSIZE) * 4)   # K and V codes
    basis = 2 * kv_heads * rank * head_dim * 4                        # K and V bases
    fp16 = 2 * n_tokens * kv_heads * head_dim * 2
    return {
        "fp16_bytes": fp16 * n_layers,
        "codes_bytes": per_tok * n_tokens * n_layers,
        "basis_bytes": basis * n_layers,
        "total_bytes": (per_tok * n_tokens + basis) * n_layers,
        "ratio_vs_fp16": fp16 * n_layers / max((per_tok * n_tokens + basis) * n_layers, 1),
    }


def attend_lowrank_nf4(q: torch.Tensor,
                       k_basis: torch.Tensor, k_codes_p: torch.Tensor, k_codes_a: torch.Tensor,
                       v_basis: torch.Tensor, v_codes_p: torch.Tensor, v_codes_a: torch.Tensor,
                       scale: float | None = None, block_t: int = 128) -> torch.Tensor:
    """One decode step over a low-rank, 4-bit cache. ``q [H_q, D]`` -> ``[H_q, D]``.

    The cache is never reconstructed: the query is projected into the code space
    on the way in, and the basis is applied to the r-dimensional result on the
    way out. Scaling uses the ORIGINAL head_dim, because the scores are
    mathematically q @ K.T — the projection is a change of basis, not of scale.
    """
    H_q, D = q.shape
    H_kv, r, D_b = k_basis.shape
    if D_b != D:
        raise ValueError(f"basis last dim {D_b} != head_dim {D}")
    _check_rank(r, D)
    rep = H_q // H_kv
    scale = D ** -0.5 if scale is None else scale

    # absorb the K up-projection into the query: q' [H_q, r] = q @ B_k[h//rep].T
    q_proj = torch.einsum("hd,hrd->hr", q.float(),
                          k_basis.repeat_interleave(rep, dim=0)).contiguous()
    scores = kv_scores_nf4(q_proj, k_codes_p, k_codes_a, block_t=block_t) * scale
    probs = torch.softmax(scores, dim=-1)
    # weighted sum stays in code space, then one small [r, D] lift
    c_out = kv_weighted_sum_nf4(probs, v_codes_p, v_codes_a, r, block_t=block_t)
    return torch.einsum("hr,hrd->hd", c_out, v_basis.repeat_interleave(rep, dim=0).float())
