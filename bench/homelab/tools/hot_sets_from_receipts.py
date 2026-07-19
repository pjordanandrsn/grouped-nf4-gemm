#!/usr/bin/env python3
"""Committed reducer: per-layer hot-expert rankings + held-out capture-vs-K
from the stamped router-probe stream (gpt_oss_120b ``topk_set.npy`` — the same
input the specstream reanalysis pinned: sha256[:16]=45c071e689b9f173).

Split is BY PROMPT (records per prompt are contiguous; within a prompt the
record index is token*L + layer per the stream layout): first 2/3 of prompts
train, rest held-out. Ranking = per-layer expert frequency on TRAIN.
capture(K) = P(routed expert lands in its layer's train-derived top-K),
scored per-assignment on HELD-OUT — the same definition as the reanalysis's
per-layer top-16 capture (~30%).

Usage: hot_sets_from_receipts.py <streams_dir> <out_dir>
Writes hot_full.npy [L, E] (rank order), capture.json; prints input sha +
capture table for RESULTS citation.
"""
import hashlib, json, os, sys
import numpy as np

D, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
meta = json.load(open(f"{D}/meta.json"))
E, k = meta["E"], meta["k"]
L = len(meta["layers"]) if "layers" in meta else meta["n_layers"]
P = meta.get("prompts") or meta.get("n_prompts")
topk = np.load(f"{D}/topk_set.npy")
rl = np.load(f"{D}/record_layer.npy")
sha = hashlib.sha256(open(f"{D}/topk_set.npy", "rb").read()).hexdigest()
N = len(topk)
assert N % P == 0, (N, P)
per = N // P
prom = np.arange(N) // per
n_train = max(1, int(round(P * 2 / 3)))
tr = prom < n_train
ho = ~tr
print(f"stream sha256[:16]={sha[:16]} records={N} prompts={P} (train {n_train}) L={L} E={E} k={k}", flush=True)

counts = np.zeros((L, E), dtype=np.int64)
for l in range(L):
    m = tr & (rl == l)
    counts[l] = np.bincount(topk[m].reshape(-1), minlength=E)
rank = np.argsort(-counts, axis=1, kind="stable")
np.save(f"{OUT}/hot_full.npy", rank)
pos = np.empty((L, E), dtype=np.int64)
for l in range(L):
    pos[l, rank[l]] = np.arange(E)

cap = {}
for K in (0, 8, 16, 32, 64, 128):
    if K == 0:
        cap[K] = 0.0
        continue
    hit = tot = 0
    for l in range(L):
        m = ho & (rl == l)
        r = topk[m].reshape(-1)
        hit += int((pos[l][r] < K).sum())
        tot += r.size
    cap[K] = hit / tot
json.dump({"sha256": sha, "capture": cap, "n_train_prompts": n_train}, open(f"{OUT}/capture.json", "w"), indent=1)
print("held-out capture: " + "  ".join(f"K={K}:{v:.3f}" for K, v in cap.items()), flush=True)
