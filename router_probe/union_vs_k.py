#!/usr/bin/env python3
"""Exploratory union(w): distinct routed experts over w consecutive decode
tokens, per layer, from router-probe capture dirs (topk_set + record_token/
record_layer). Windows never span non-consecutive token indices (prompt
boundaries after combine_streams offsetting). $0, existing data.

This is the spec-dec economics number: a verify forward over w drafted tokens
streams union(w) experts instead of w*k. union(w) << w*k  <=>  spec-dec pays.
"""
import json, sys
from pathlib import Path
import numpy as np

WINDOWS = (1, 2, 4, 8, 16)

def capture_dirs(root):
    for meta in Path(root).rglob("meta.json"):
        d = meta.parent
        if (d / "topk_set.npy").exists():
            yield d

def unions_for_dir(d):
    meta = json.loads((d / "meta.json").read_text())
    topk = np.load(d / "topk_set.npy")                     # [T, k]
    tok = np.load(d / "record_token.npy") if (d / "record_token.npy").exists() else None
    lay = np.load(d / "record_layer.npy") if (d / "record_layer.npy").exists() else None
    E, k = int(meta["E"]), int(meta["k"])
    out = {}
    layers = [None] if lay is None else np.unique(lay)
    for L in layers:
        m = slice(None) if L is None else (lay == L)
        tk, tt = topk[m], (None if tok is None else tok[m])
        if tt is not None:
            o = np.argsort(tt, kind="stable"); tk, tt = tk[o], tt[o]
        for w in WINDOWS:
            sizes = []
            for i in range(0, len(tk) - w + 1):
                if tt is not None:
                    seg = tt[i:i+w]
                    if seg[-1] - seg[0] != w - 1:          # boundary/jump: skip window
                        continue
                sizes.append(len(np.unique(tk[i:i+w])))
            if sizes:
                out.setdefault(w, []).extend(sizes)
    return E, k, out

def main(root):
    E = k = None
    agg = {}
    ndirs = 0
    for d in capture_dirs(root):
        e, kk, u = unions_for_dir(d)
        E, k = e, kk
        ndirs += 1
        for w, s in u.items():
            agg.setdefault(w, []).extend(s)
    if not agg:
        print(json.dumps({"error": "no captures found"})); return
    rep = {"root": str(root), "capture_dirs": ndirs, "E": E, "k": k, "windows": {}}
    for w in sorted(agg):
        a = np.array(agg[w])
        # uniform null: expected distinct of w*k draws over E without replacement structure
        null = E * (1 - (1 - k / E) ** w)
        rep["windows"][str(w)] = {
            "n_windows": int(a.size),
            "union_mean": round(float(a.mean()), 2),
            "union_p50": float(np.percentile(a, 50)),
            "union_p90": float(np.percentile(a, 90)),
            "naive_wk": w * k,
            "uniform_null": round(null, 2),
            "bytes_ratio_vs_serial": round(float(a.mean()) / (w * k), 4),
        }
    print(json.dumps(rep, indent=1))

if __name__ == "__main__":
    main(sys.argv[1])
