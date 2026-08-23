"""The multi-expert interaction-term hunt. Registered in
bench/cold-engine/PREREG-interaction-hunt.md (bars frozen first).

Grid: G x rows x N kernel-only grouped NF4 calls; least-squares fit of
med = F + a*(G*N) + b*(rows*N) + c*G + d*rows; the serving-shape points
(G=29) are measured but HELD OUT of the fit.
"""
import argparse
import json
import os
import statistics
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "kernel"))

import cpu_grouped as cg                                    # noqa: E402
import gnf4_native                                          # noqa: E402

K = 2048
E = 64
GRID_G = (1, 2, 4, 8, 16, 32, 64)
GRID_ROWS = (32, 64, 128)
GRID_N = (256, 768)
HOLDOUT = ((29, 64, 768), (29, 128, 768))


def build(n, seed=7):
    g = np.random.default_rng(seed)
    pk = torch.from_numpy(g.integers(0, 256, size=(E, n, K // 2),
                                     dtype=np.uint8))
    am = torch.from_numpy((g.random((E, n, K // 64), dtype=np.float32)
                           * 0.02 + 1e-3))
    return pk, am


def call(pk, am, G, rows, threads, reps=50):
    g = np.random.default_rng(G * 1000 + rows)
    base, extra = divmod(rows, G)
    sizes = [base + (1 if i < extra else 0) for i in range(G)]
    sizes = [s for s in sizes if s > 0]
    eids = list(range(len(sizes)))
    a = torch.from_numpy(g.standard_normal((rows, K), dtype=np.float32))
    cg.gemv_nf4_grouped_cpu(a, pk, am, sizes, eids, threads=threads)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        cg.gemv_nf4_grouped_cpu(a, pk, am, sizes, eids, threads=threads)
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--out", required=True)
    a_ = ap.parse_args()
    gnf4_native.load()
    stacks = {n: build(n) for n in GRID_N}
    cells, hold = [], []
    for n in GRID_N:
        pk, am = stacks[n]
        for G in GRID_G:
            for rows in GRID_ROWS:
                if G > rows:
                    continue
                med = call(pk, am, G, rows, a_.threads)
                cells.append({"G": G, "rows": rows, "N": n, "med_us": med})
                print("G=%2d rows=%3d N=%4d  med %8.1f us"
                      % (G, rows, n, med), flush=True)
    for (G, rows, n) in HOLDOUT:
        pk, am = stacks[n]
        med = call(pk, am, G, rows, a_.threads)
        hold.append({"G": G, "rows": rows, "N": n, "med_us": med})
        print("HOLDOUT G=%2d rows=%3d N=%4d  med %8.1f us"
              % (G, rows, n, med), flush=True)

    X = np.array([[1, c["G"] * c["N"], c["rows"] * c["N"], c["G"],
                   c["rows"]] for c in cells], float)
    y = np.array([c["med_us"] for c in cells])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    rms = float(np.sqrt(np.mean((pred - y) ** 2)))
    med_cell = float(np.median(y))
    h1 = rms <= 0.08 * med_cell
    names = ("F", "a_GN", "b_rowsN", "c_G", "d_rows")
    fit = dict(zip(names, (float(b) for b in beta)))
    hp = []
    for h in hold:
        p = (fit["F"] + fit["a_GN"] * h["G"] * h["N"]
             + fit["b_rowsN"] * h["rows"] * h["N"] + fit["c_G"] * h["G"]
             + fit["d_rows"] * h["rows"])
        hp.append({"cell": h, "pred_us": p,
                   "err_pct": 100.0 * (p - h["med_us"]) / h["med_us"]})
    h2 = all(abs(x["err_pct"]) <= 10.0 for x in hp)
    Gs, Rs, Ns = 29, 128, 768
    terms = {"b_rowsN": fit["b_rowsN"] * Rs * Ns,
             "c_G": fit["c_G"] * Gs, "d_rows": fit["d_rows"] * Rs}
    ranked = sorted(terms.items(), key=lambda kv: -abs(kv[1]))
    attribution = None
    if h1 and h2:
        top, second = ranked[0], ranked[1]
        if abs(second[1]) < 1e-9 or abs(top[1]) / max(abs(second[1]), 1e-9) >= 1.5:
            attribution = top[0]
        else:
            attribution = "mixed:" + "+".join(k for k, _ in ranked[:2])
    verdict = ("ATTRIBUTED:" + attribution if attribution and h1 and h2
               else "MODEL-REFUTED")
    out = {"fit": fit, "rms_us": rms, "median_cell_us": med_cell,
           "h1_fit_ok": h1, "holdout": hp, "h2_predict_ok": h2,
           "serving_terms_us": terms, "verdict": verdict,
           "cells": cells}
    print(json.dumps({k: v for k, v in out.items() if k != "cells"},
                     indent=1))
    print("HUNT:", verdict)
    with open(a_.out, "w") as f:
        json.dump(out, f, indent=1)
    print("receipt ->", a_.out)


if __name__ == "__main__":
    main()
