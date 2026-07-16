# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Stream serialization + the contract-enforcing dataloader (CHARTER §3.2).

Features and labels live in SEPARATELY serialized streams; the probe-side code
never touches raw activations. The lead distance Δ is applied HERE, at join
time, by index arithmetic over per-token records — never as an eval-time slice.

Stream store layout (one directory per capture):
    <dir>/hidden_post_block_l.npy    [T, d_hidden]  fp16/fp32
    <dir>/router_logits_l.npy        [T, E]
    <dir>/token_embedding.npy        [T, d_embed]
    <dir>/topk_set.npy               [T, k] int32   (realized top-k at EVERY layer step)
    <dir>/meta.json                  {"E":…, "k":…, "layer":…, "decode_only":true}

A record t in the feature streams is what a runtime predictor could see after
layer l at decode step t; topk_set[t + Δ_effective] is the label. For the
per-layer capture used in Phase 1, consecutive records within one token are
consecutive layers, so Δ layers ahead == Δ records ahead within the same token;
fixtures emulate the same shape with a flat index.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FEATURE_STREAMS = ("hidden_post_block_l", "router_logits_l", "token_embedding")
LABEL_STREAM = "topk_set"


def write_capture(dir_path, hidden, logits, embed, topk, meta,
                  record_token=None, record_layer=None):
    d = Path(dir_path)
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "hidden_post_block_l.npy", hidden)
    np.save(d / "router_logits_l.npy", logits)
    np.save(d / "token_embedding.npy", embed)
    np.save(d / "topk_set.npy", topk.astype(np.int32))
    # join metadata (NOT features, law 11): token index masks cross-token Δ
    # pairs; layer index enables per-band grouping in analysis.
    if record_token is not None:
        np.save(d / "record_token.npy", np.asarray(record_token, np.int64))
    if record_layer is not None:
        np.save(d / "record_layer.npy", np.asarray(record_layer, np.int32))
    meta = dict(meta)
    meta.setdefault("decode_only", True)
    (d / "meta.json").write_text(json.dumps(meta, indent=1))


class ContractLoader:
    """Joins feature streams to labels at lead Δ. The ONLY place Δ exists."""

    def __init__(self, dir_path, delta: int = 1):
        d = Path(dir_path)
        self.meta = json.loads((d / "meta.json").read_text())
        if not self.meta.get("decode_only", False):
            raise ValueError("contract violation: capture is not decode-only")
        feats = [np.load(d / f"{s}.npy") for s in FEATURE_STREAMS]
        labels = np.load(d / f"{LABEL_STREAM}.npy")
        T = labels.shape[0]
        if delta < 1:
            raise ValueError("lead must be >= 1 (a 0-lead probe would see its own label's router)")
        # features at t predict the realized set at t + delta
        X = [f[: T - delta] for f in feats]
        y = labels[delta:]
        # real captures are layer-major within a token: a Δ shift must never
        # pair records from different tokens (the last Δ layers of each token
        # have no in-token successor). Fixtures have no record_token file.
        tok_path = d / "record_token.npy"
        if tok_path.exists():
            tok = np.load(tok_path)
            keep = tok[: T - delta] == tok[delta:]
            X = [x[keep] for x in X]
            y = y[keep]
            self.masked_fraction = round(float(1.0 - keep.mean()), 4)
        else:
            self.masked_fraction = 0.0
        self.X = X
        self.y = y
        self.delta = delta
        self.E = int(self.meta["E"])
        self.k = int(self.meta["k"])

    def arrays(self):
        """(features list [T', d_i], labels [T', k], E, k)"""
        return self.X, self.y, self.E, self.k

    def split(self, heldout: int, seed: int):
        """Deterministic shuffled split -> (train_X, train_y, held_X, held_y)."""
        T = self.y.shape[0]
        rng = np.random.default_rng(seed)
        idx = rng.permutation(T)
        hi, ti = idx[:heldout], idx[heldout:]
        tX = [x[ti] for x in self.X]
        hX = [x[hi] for x in self.X]
        return tX, self.y[ti], hX, self.y[hi]
