"""The low-G split re-certification harness. Registered in
bench/cold-engine/PREREG-lowg-split2.md (A/A gate, noise-derived bars,
scored cells only). Arms are selected by checking out commits around this
script; it only measures the CURRENT build.
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
B1_CELLS = ((1, 128, 256), (1, 128, 768))
B2_CELLS = ((16, 128, 256), (32, 64, 256), (8, 128, 768), (16, 128, 768),
            (32, 128, 768), (64, 128, 768), (29, 64, 768), (29, 128, 768))


def build(n, seed=7):
    g = np.random.default_rng(seed)
    pk = torch.from_numpy(g.integers(0, 256, size=(E, n, K // 2),
                                     dtype=np.uint8))
    am = torch.from_numpy((g.random((E, n, K // 64), dtype=np.float32)
                           * 0.02 + 1e-3))
    return pk, am


def call(stacks, G, rows, n, threads, reps=50):
    pk, am = stacks[n]
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
    stacks = {n: build(n) for n in (256, 768)}
    out = {}
    for (G, rows, n) in B1_CELLS + B2_CELLS:
        out[f"{G},{rows},{n}"] = call(stacks, G, rows, n, a_.threads)
        print("G=%2d rows=%3d N=%4d  med %8.1f us"
              % (G, rows, n, out[f"{G},{rows},{n}"]), flush=True)
    with open(a_.out, "w") as f:
        json.dump(out, f, indent=1)
    print("receipt ->", a_.out)


if __name__ == "__main__":
    main()
