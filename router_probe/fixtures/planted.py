# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Planted + null synthetic routers with analytically-controlled Bayes-optimal
set-agreement (CHARTER §3.4).

DESIGN (v2 — the one that passed the learnability shakedown):
  stream 2 (router_logits_l) = the true score vector u  (planted map = identity
    on a contract stream — faithful to why layer-l logits are in the contract)
  streams 1,3               = distractor noise
  labels                    = top-k(u + sigma * g)   (temperature noise)

Bayes-optimal predictor is top-k(u) (membership probability is monotone in u_i
under symmetric noise); its set-agreement is a deterministic function H(sigma)
of the generator, computed by high-n Monte Carlo and calibrated by bisection to
the frozen levels. Recorded per fixture: sigma, the MC analytic target, and the
realized-Bayes on the fixture's own draw.

DESIGN HISTORY (kept so it isn't retried): v1 used whole-set replacement noise
with a planted low-rank linear map. Two failures, both diagnosed empirically:
(a) BCE/CE on set-membership cannot recover a planted ranking through a
generic linear map — the CE-optimal fit is a per-sample two-level step, and the
best in-class approximation loses ~5-6 points of H at the k-th-place boundary;
(b) under replacement noise the CE landscape is nearly flat (0.0035 nats
between rank-perfect and rank-mangled solutions at eps=0.34), so even with the
identity solution IN CLASS, first-order training converges elsewhere
(identity had strictly lower loss AND H=0.70 vs trained 0.64 — verified).
Temperature noise keeps every sample informative; signal-as-stream keeps the
Bayes map inside every rung's hypothesis class. Combined: linear rung recovers
the target within 0.006 by epoch 2.
"""
from __future__ import annotations

import numpy as np

from capture.streams import write_capture


def _mc_h(sigma: float, E: int, k: int, m: int, rng) -> float:
    """Monte-Carlo Bayes set-agreement for iid N(0,1) scores + N(0,sigma) noise."""
    h, done, chunk = 0.0, 0, 200_000
    while done < m:
        c = min(chunk, m - done)
        u = rng.standard_normal((c, E), dtype=np.float32)
        lab = np.argsort(-(u + sigma * rng.standard_normal((c, E), dtype=np.float32)), axis=1)[:, :k]
        prd = np.argsort(-u, axis=1)[:, :k]
        h += float(((prd[:, :, None] == lab[:, None, :]).any(2).sum(1) / k).sum())
        done += c
    return h / m


def calibrate_sigma(h_target: float, E: int, k: int, seed: int,
                    iters: int = 14, m_search: int = 300_000, m_final: int = 2_000_000):
    """Bisection on the monotone map sigma -> H; returns (sigma, analytic_h)."""
    rng = np.random.default_rng(seed)
    lo, hi = 1e-4, 4.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if _mc_h(mid, E, k, m_search, rng) > h_target:
            lo = mid
        else:
            hi = mid
    sigma = 0.5 * (lo + hi)
    return sigma, _mc_h(sigma, E, k, m_final, rng)


def make_planted(dir_path, h_target, dims, n, seed):
    E, k = dims["E"], dims["k"]
    dh, de = dims["d_hidden"], dims["d_embed"]
    assert dims["d_logits"] == E, "stream 2 carries the E-dim score vector"
    rng = np.random.default_rng(seed)
    sigma, analytic = calibrate_sigma(h_target, E, k, seed + 101)
    u = rng.standard_normal((n, E)).astype(np.float32)
    hidden = rng.standard_normal((n, dh)).astype(np.float32)
    embed = rng.standard_normal((n, de)).astype(np.float32)
    lab = np.argsort(-(u + sigma * rng.standard_normal((n, E)).astype(np.float32)), axis=1)[:, :k]
    prd = np.argsort(-u, axis=1)[:, :k]
    realized_bayes = float(((prd[:, :, None] == lab[:, None, :]).any(2).sum(1) / k).mean())
    # loader pairs features[t] with labels[t+delta]; serialize pre-shifted:
    # L[t+1] = lab[t]  (L[0] is a dummy, excluded by y = labels[delta:])
    L = np.vstack([lab[:1], lab[:-1]]).astype(np.int32)
    write_capture(
        dir_path, hidden, u, embed, L,
        {"E": E, "k": k, "layer": "fixture", "decode_only": True,
         "kind": "planted_temperature", "h_target": h_target, "sigma": sigma,
         "analytic_h": analytic, "realized_bayes": realized_bayes, "seed": seed},
    )
    return analytic


def make_null(dir_path, dims, n, seed):
    E, k = dims["E"], dims["k"]
    dh, de = dims["d_hidden"], dims["d_embed"]
    rng = np.random.default_rng(seed)
    hidden = rng.standard_normal((n, dh)).astype(np.float32)
    logits = rng.standard_normal((n, E)).astype(np.float32)
    embed = rng.standard_normal((n, de)).astype(np.float32)
    top = np.argsort(rng.random((n, E)), axis=1)[:, :k].astype(np.int32)
    L = np.vstack([top[:1], top[:-1]]).astype(np.int32)
    write_capture(
        dir_path, hidden, logits, embed, L,
        {"E": E, "k": k, "layer": "fixture", "decode_only": True,
         "kind": "null", "chance": k / E, "seed": seed},
    )
    return k / E
