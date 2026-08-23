"""P4 rows-scaling microbench: the b16close kernel-only measurement, in-repo.

Registered in bench/cold-engine/PREREG-p4-rowblock.md. Matched call shapes
(e4b instrument law 5): one grouped NF4 call at the Qwen3-30B serving
geometry — G unique experts, rows split across them as serving routes —
at rows = 64 (B=8-class) and 128 (B=16-class). Reports med us and achieved
weight-GB/s. The A/B is old-vs-new BINARY on the same box (checkout the
parent commit for the old arm); the harness is identical either way.
"""
import argparse
import json
import statistics
import sys
import time

import numpy as np
import torch

import os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "kernel"))

import cpu_grouped as cg                                    # noqa: E402
import gnf4_native                                          # noqa: E402

N, K = 768, 2048             # Qwen3-30B-class expert geometry
E, G = 64, 29                # arena experts; uniques per call (b16close: 29)


def one_call(a, pk, am, sizes, eids, threads, reps=50):
    cg.gemv_nf4_grouped_cpu(a, pk, am, sizes, eids, threads=threads)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        cg.gemv_nf4_grouped_cpu(a, pk, am, sizes, eids, threads=threads)
        ts.append(time.perf_counter() - t0)
    med = statistics.median(ts)
    wbytes = len(set(eids)) * (N * K // 2 + N * (K // 64) * 4)
    return med, wbytes / med / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--out", required=True)
    a_ = ap.parse_args()
    gnf4_native.load()
    g = np.random.default_rng(7)
    pk = torch.from_numpy(g.integers(0, 256, size=(E, N, K // 2),
                                     dtype=np.uint8))
    am = torch.from_numpy((g.random((E, N, K // 64), dtype=np.float32)
                           * 0.02 + 1e-3))
    out = {"n": N, "k": K, "uniques": G, "threads": a_.threads, "arms": {}}
    for rows in (64, 128):
        base, extra = divmod(rows, G)
        sizes = [base + (1 if i < extra else 0) for i in range(G)]
        sizes = [s for s in sizes if s > 0]
        eids = list(range(len(sizes)))
        a = torch.from_numpy(g.standard_normal((rows, K), dtype=np.float32))
        med, gbs = one_call(a, pk, am, sizes, eids, a_.threads)
        out["arms"][str(rows)] = {"med_us": med * 1e6, "achieved_gbs": gbs}
        print("rows=%3d  med %8.1f us  achieved %6.1f GB/s"
              % (rows, med * 1e6, gbs))
    r64 = out["arms"]["64"]["achieved_gbs"]
    r128 = out["arms"]["128"]["achieved_gbs"]
    out["rows_ratio_128_over_64"] = r128 / r64
    print("rows-scaling ratio (128/64): %.3f" % (r128 / r64))
    with open(a_.out, "w") as f:
        json.dump(out, f, indent=1)
    print("receipt ->", a_.out)


if __name__ == "__main__":
    main()
