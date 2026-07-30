# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Residency decision curve at Kimi-K3 scale.

The question N3 exists to answer: a 503 GB host cannot hold K3's 1.446 TB of
experts, so what does the miss traffic actually cost? Routing is heavy-tailed,
so hit rate — not capacity — sets throughput.

Measured inputs (2026-07-30, real checkpoint + real gather):
  92 MoE layers x 896 experts, top-16 routed
  17.55 MB per expert  ->  280.8 MB per layer  ->  25.83 GB per token
  pinned-DRAM -> VRAM gather: 19 GB/s  (9.25 ms per 176 MB layer slice)
  NVMe arena read: ~2 GB/s assumed (N0 measured 99% of device link; substitute
                   your device's figure with --nvme-gbs)

The skew of K3's real router is NOT known until the model loads, so this sweeps
it. Read the row matching your belief about the router, not the best row.

    python bench/nvme_residency_curve.py [--host-ram-gb 503] [--nvme-gbs 2.0]
"""
from __future__ import annotations

import argparse
import random

N_LAYERS = 92
N_EXPERTS = 896
TOP_K = 16
MB_PER_EXPERT = 17.55
DRAM_GBS = 19.0          # measured host_gather throughput


def simulate(hot_rows: int, zipf: float, steps: int, seed: int = 1689) -> float:
    """Steady-state hit rate for one layer under LFU + a Zipf(zipf) router."""
    rng = random.Random(seed)
    weights = [1.0 / (i + 1) ** zipf for i in range(N_EXPERTS)]
    freq: dict[int, int] = {}
    resident: set[int] = set()
    hits = total = 0
    warmup = steps // 5
    for step in range(steps):
        picks, seen = [], set()
        while len(picks) < TOP_K:
            e = rng.choices(range(N_EXPERTS), weights=weights, k=1)[0]
            if e not in seen:
                seen.add(e); picks.append(e)
        protected = set()
        for e in picks:
            freq[e] = freq.get(e, 0) + 1
            if step >= warmup:
                total += 1
            if e in resident:
                if step >= warmup:
                    hits += 1
                protected.add(e)
                continue
            if len(resident) >= hot_rows:
                victim = min((x for x in resident if x not in protected),
                             key=lambda x: freq.get(x, 0), default=None)
                if victim is not None:
                    resident.discard(victim)
            resident.add(e)
            protected.add(e)
    return hits / total if total else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host-ram-gb", type=float, default=503.0)
    ap.add_argument("--nvme-gbs", type=float, default=2.0)
    ap.add_argument("--steps", type=int, default=4000)
    a = ap.parse_args()

    total_gb = N_LAYERS * N_EXPERTS * MB_PER_EXPERT / 1024
    per_token_gb = N_LAYERS * TOP_K * MB_PER_EXPERT / 1024
    # leave headroom for non-expert weights (~115 GB) + activations
    usable = max(0.0, a.host_ram_gb - 140.0)
    frac = min(1.0, usable / total_gb)
    hot_rows = max(TOP_K, int(N_EXPERTS * frac))

    print(f"experts total     : {total_gb:.0f} GB across {N_LAYERS}x{N_EXPERTS}")
    print(f"routed per token  : {per_token_gb:.2f} GB (top-{TOP_K})")
    print(f"host RAM          : {a.host_ram_gb:.0f} GB "
          f"-> {usable:.0f} GB usable after ~140 GB non-expert/activations")
    print(f"resident fraction : {frac*100:.1f}%  ({hot_rows} of {N_EXPERTS} per layer)")
    print(f"NVMe assumed      : {a.nvme_gbs:.1f} GB/s | DRAM gather: {DRAM_GBS:.0f} GB/s")
    print()
    print(f"{'zipf':>5}{'hit%':>8}{'disk GB/tok':>13}{'s/tok':>9}{'tok/s':>8}")
    print("-" * 44)
    for z in (0.0, 0.4, 0.8, 1.0, 1.2, 1.5, 2.0):
        hr = simulate(hot_rows, z, a.steps)
        disk_gb = per_token_gb * (1.0 - hr)
        dram_gb = per_token_gb * hr
        secs = disk_gb / a.nvme_gbs + dram_gb / DRAM_GBS
        print(f"{z:>5.1f}{hr*100:>8.1f}{disk_gb:>13.2f}{secs:>9.2f}"
              f"{(1.0/secs if secs else 0):>8.2f}")
    print()
    print("Weight movement only — excludes MLA/Kimi-Linear attention, the MXFP4")
    print("GEMM, the 2 always-on shared experts, and the router. Treat tok/s as a")
    print("CEILING. Prefetch changes this picture again: miss-correction measured")
    print("0.05x cost at 15/16 predicted, so an accurate predictor collapses the")
    print("DRAM term and leaves disk misses as the only real cost.")


if __name__ == "__main__":
    main()
