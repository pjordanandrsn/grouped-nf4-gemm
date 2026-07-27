# Copyright (c) 2026 Cerin Amroth LLC. MIT.
"""Does merging pinned H2D copies buy anything at routed-staging granularity?

`PREREG-expert-major-coalescer` predicts 1.06-1.18x from cutting the routed
stage's copy count 32 -> 16 per layer. That prediction rests on the premise that
~2.7 MB is "below where an H2D reaches asymptotic bandwidth". This prices the
premise directly, and it is the whole reason to run it BEFORE renting an A100
for the 235B fixture: if 2.7 MB is already at ceiling, the coalescer's ceiling
is 1.00x and the expensive run is answering a question with a known answer.

The gnf4 arc already paid for this lesson once -- months aimed at a "4.9x
headroom" that was distance to a coalesced read the kernel could not perform.
Before optimising toward a bound, check the bound is reachable.

WHAT IS MEASURED
    Same total bytes, same pinned source, same device, moved as N copies of
    (total/N). Sweeping N sweeps per-copy size. If GB/s is flat across the sweep
    at and above the routed-staging size, copy COUNT is not the constraint.

    Arms are paired and re-timed against a self-pair, because an unpaired sweep
    on a drifting box once reported a config 1.283x faster than itself (#46).

NOT MEASURED
    Duty cycle. `routed_gbps` divides bytes by WALL STEP time, so 0.59x of
    ceiling includes every microsecond the link sat idle during attention, the
    router and the expert GEMM. That fraction is not recoverable by ANY change
    to copy granularity, and this harness deliberately does not conflate them:
    it reports achieved bandwidth INSIDE the copy window only.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys

import torch

# (label, bytes) -- the two granularities the coalescer moves between, plus the
# span around them needed to locate the knee if there is one.
SIZES_MB = [0.25, 0.5, 1.0, 2.0, 2.7, 4.0, 5.4, 8.0, 16.0, 32.0]

# 2.7 MB == the measured per-copy size of a routed stage at 235B shapes (#52);
# 5.4 MB == what one expert x dtype becomes after the coalescer merges 4 -> 2.
ROUTED_MB, COALESCED_MB = 2.7, 5.4


def time_copies(src, dst, nbytes_total, chunk, iters):
    """Move `nbytes_total` as ceil(total/chunk) copies; return median ms."""
    n = max(1, nbytes_total // chunk)
    for _ in range(3):                                  # warm
        for i in range(n):
            dst[i * chunk : i * chunk + chunk].copy_(
                src[i * chunk : i * chunk + chunk], non_blocking=True)
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for i in range(n):
            dst[i * chunk : i * chunk + chunk].copy_(
                src[i * chunk : i * chunk + chunk], non_blocking=True)
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total-mb", type=float, default=256.0,
                    help="bytes moved per arm; constant across the sweep")
    ap.add_argument("--iters", type=int, default=9)
    ap.add_argument("--pairs", type=int, default=3)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("needs CUDA")
    p = torch.cuda.get_device_properties(0)
    total = int(args.total_mb * (1 << 20))
    total -= total % (1 << 20)

    src = torch.empty(total, dtype=torch.uint8, device="cpu").pin_memory()
    dst = torch.empty(total, dtype=torch.uint8, device="cuda")
    print(f"# {p.name}  torch {torch.__version__}")
    print(f"# {args.total_mb:.0f} MB per arm, pinned -> device, "
          f"median of {args.iters}, {args.pairs} pairs\n")
    print(f"  {'per-copy':>10}  {'copies':>7}  {'ms':>8}  {'GB/s':>8}")

    rows = []
    for mb in SIZES_MB:
        chunk = int(mb * (1 << 20))
        if chunk > total:
            continue
        ms_list = []
        for _ in range(args.pairs):
            ms, n = time_copies(src, dst, total, chunk, args.iters)
            ms_list.append(ms)
        ms = statistics.median(ms_list)
        gbps = total / (ms * 1e-3) / 1e9
        tag = ""
        if abs(mb - ROUTED_MB) < 1e-6:
            tag = "  <- routed staging today"
        if abs(mb - COALESCED_MB) < 1e-6:
            tag = "  <- after the coalescer"
        rows.append({"mb": mb, "copies": n, "ms": ms, "gbps": gbps})
        print(f"  {mb:8.2f} MB  {n:7d}  {ms:8.3f}  {gbps:8.2f}{tag}")

    # self-pair: the largest size against itself, same protocol -- must read ~1.000x
    chunk = int(max(r["mb"] for r in rows) * (1 << 20))
    sp = []
    for _ in range(args.pairs):
        a, _ = time_copies(src, dst, total, chunk, args.iters)
        b, _ = time_copies(src, dst, total, chunk, args.iters)
        sp.append(a / b)
    self_pair = statistics.median(sp)

    by = {r["mb"]: r for r in rows}
    ceiling = max(r["gbps"] for r in rows)
    r_routed, r_coal = by.get(ROUTED_MB), by.get(COALESCED_MB)
    print(f"\n# self-pair: {self_pair:.3f}x  "
          f"{'OK' if 0.95 < self_pair < 1.05 else 'VOID -- too noisy'}")
    print(f"# best observed: {ceiling:.2f} GB/s")

    verdict = None
    if r_routed and r_coal:
        gain = r_coal["gbps"] / r_routed["gbps"]
        frac = r_routed["gbps"] / ceiling
        print(f"# routed 2.7 MB = {r_routed['gbps']:.2f} GB/s "
              f"({frac:.2%} of best observed)")
        print(f"# coalesced 5.4 MB = {r_coal['gbps']:.2f} GB/s")
        print(f"# granularity gain 2.7 -> 5.4 MB: {gain:.3f}x")
        if gain < 1.02:
            verdict = ("DEAD: copy granularity is not the constraint at these sizes. "
                       "The coalescer's ceiling is ~1.00x; the 0.59x in #52 is duty "
                       "cycle, not per-copy inefficiency. Do NOT rent the A100 run.")
        elif gain < 1.06:
            verdict = (f"MARGINAL: {gain:.3f}x at the copy layer is below the prereg's "
                       "1.06x landing bar even before the step dilutes it by the "
                       "non-transfer 28.7%. Landing is not reachable.")
        else:
            verdict = (f"LIVE: {gain:.3f}x headroom at the copy layer. The premise "
                       "holds; the 235B fixture is worth renting.")
        print(f"#\n# -> {verdict}")

    if args.out:
        json.dump({"device": p.name, "torch": torch.__version__,
                   "total_mb": args.total_mb, "rows": rows,
                   "self_pair": self_pair, "best_gbps": ceiling,
                   "verdict": verdict}, open(args.out, "w"), indent=2)
        print(f"# wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
