# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The capacity ladder (CHARTER §3.3): linear -> MLP(d) -> MLP(4d) -> 2-layer
attention probe. One training path for all rungs: scores over E experts trained
with BCE against the realized top-k as a multi-hot target; H = mean
|top-k(scores) ∩ realized| / k.

The attention rung consumes the three contract streams as three tokens (each
projected to a common width) so stream-level structure is available to it;
linear/MLP rungs see the flat concatenation. No rung ever sees the label stream
except through the loss.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0)


def _flat(X):
    return np.concatenate(X, axis=1)


class LinearProbe(nn.Module):
    def __init__(self, d_in, E):
        super().__init__()
        self.f = nn.Linear(d_in, E)

    def forward(self, xs):
        return self.f(xs)


class MLPProbe(nn.Module):
    def __init__(self, d_in, E, width):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d_in, width), nn.GELU(), nn.Linear(width, E))

    def forward(self, xs):
        return self.f(xs)


class AttnProbe(nn.Module):
    """2 transformer encoder layers over the 3 stream-tokens + learned CLS."""

    def __init__(self, dims, E, width=256, heads=4, layers=2):
        super().__init__()
        self.proj = nn.ModuleList([nn.Linear(d, width) for d in dims])
        self.cls = nn.Parameter(torch.zeros(1, 1, width))
        enc = nn.TransformerEncoderLayer(width, heads, dim_feedforward=2 * width,
                                         batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, num_layers=layers)
        self.head = nn.Linear(width, E)

    def forward(self, xs_list):
        toks = [p(x) for p, x in zip(self.proj, xs_list)]
        seq = torch.stack(toks, dim=1)
        seq = torch.cat([self.cls.expand(seq.shape[0], -1, -1), seq], dim=1)
        return self.head(self.enc(seq)[:, 0])


def set_agreement(scores: torch.Tensor, y: torch.Tensor, k: int) -> float:
    pred = torch.topk(scores, k, dim=1).indices
    # membership match via broadcast compare (E small enough)
    agree = (pred.unsqueeze(2) == y.unsqueeze(1)).any(dim=2).float().sum(dim=1) / k
    return float(agree.mean())


def _multihot(y, E):
    mh = torch.zeros(y.shape[0], E, device=y.device)
    mh.scatter_(1, y.long(), 1.0)
    return mh


def train_eval_rung(rung, tX, ty, hX, hy, E, k, cfg):
    """Returns {'train_h':…, 'heldout_h':…} for one ladder rung."""
    torch.manual_seed(int(cfg.get("seed", 0)))
    # Probes are tiny torch modules; on CPU the MLP(4d) rung is >1h each, on the
    # GPU ~1 min. cfg["device"] (default cpu) selects it; features + model + index
    # perm all live on the same device so the training loop never touches the host.
    dev = torch.device(cfg.get("device", "cpu"))
    attn = rung["kind"] == "attn"
    dims = [x.shape[1] for x in tX]
    if rung["kind"] == "linear":
        model = LinearProbe(sum(dims), E)
    elif rung["kind"] == "mlp":
        model = MLPProbe(sum(dims), E, rung["width_mult"] * sum(dims))
    else:
        model = AttnProbe(dims, E, width=rung.get("width", 256),
                          heads=rung.get("heads", 4), layers=rung.get("layers", 2))
    model = model.to(dev)

    def tensors(X):
        return [torch.from_numpy(np.ascontiguousarray(x)).float().to(dev) for x in X] if attn \
            else torch.from_numpy(_flat(X)).float().to(dev)

    Xt, Xh = tensors(tX), tensors(hX)
    yt = torch.from_numpy(np.ascontiguousarray(ty)).long().to(dev)
    yh = torch.from_numpy(np.ascontiguousarray(hy)).long().to(dev)
    mh = _multihot(yt, E)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]),
                            weight_decay=float(cfg.get("weight_decay", 0.0)))
    epochs = int(cfg["epochs"])
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
             if cfg.get("cosine", False) else None)
    # Multi-positive softmax CE: -(1/k) sum_{i in realized set} log softmax(s)_i.
    # Top-k MEMBERSHIP is not linear in x even when scores are (the k-th-place
    # cut is sample-dependent), so BCE-on-multihot cannot recover a planted
    # ranking (linear rung capped ~0.065 under target even on train). Softmax CE
    # matches what a router is and recovers score directions up to a monotone
    # transform — exactly what set-agreement needs.
    def loss_fn(scores, mh_b):
        return -(torch.log_softmax(scores, dim=1) * mh_b).sum(1).mean() / k
    n = yt.shape[0]
    bs = int(cfg["batch"])
    for _ in range(epochs):
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb = [x[idx] for x in Xt] if attn else Xt[idx]
            opt.zero_grad()
            loss = loss_fn(model(xb), mh[idx])
            loss.backward()
            opt.step()
        if sched is not None:
            sched.step()
    model.eval()
    with torch.no_grad():
        def batched_h(X, y):
            hs, tot = 0.0, 0
            m = y.shape[0]
            for i in range(0, m, 8192):
                xb = [x[i:i + 8192] for x in X] if attn else X[i:i + 8192]
                s = model(xb)
                hs += set_agreement(s, y[i:i + 8192], k) * (min(i + 8192, m) - i)
                tot += min(i + 8192, m) - i
            return hs / tot
        return {"train_h": round(batched_h(Xt, yt), 5),
                "heldout_h": round(batched_h(Xh, yh), 5)}
