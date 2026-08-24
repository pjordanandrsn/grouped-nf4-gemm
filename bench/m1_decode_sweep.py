# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""PREREG-m1-decode-config (K1) sweep: time gemm_4bit_grouped's decode
GEMV over the ablation space at the collapsed M=1 census shapes.

Two shapes only — gate_up (N=1536, K=2048, T=8) and down (N=2048,
K=768, T=8) at Qwen3-30B-A3B geometry, 48x each per decode step.
Synthetic packed tensors (bandwidth is value-blind). The current
plan's own pick runs as the baseline row, re-measured at sweep START
and END: >5% drift between the two reads makes the whole sweep
NO-VERDICT (the registered noise gate).

Emits JSON: per-shape config table + plan baseline + the registered
winner (min summed median across shapes; tie -> fewer programs)."""

import argparse
import itertools
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "kernel"))
import nf4_grouped  # noqa: E402

SHAPES = (("gate_up", 1536, 2048, 8), ("down", 2048, 768, 8))
BNS = (32, 64, 128, 256)
WARPS = (2, 4, 8)
SKS = (1, 2, 4, 8, 16)


def _mk(N, K, T, E=8, device="cuda"):
    g = torch.Generator(device="cpu").manual_seed(0)
    packed = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8,
                           generator=g).to(device)
    absmax = (torch.rand(E, N, K // 64, generator=g) + 0.5).to(device)
    a = torch.randn(T, K, generator=g).to(device=device,
                                          dtype=torch.bfloat16)
    eids = torch.arange(E, dtype=torch.int32, device=device)[:T]
    return a, packed, absmax, [1] * T, eids


def _time(fn, iters, warmup, chunks=20):
    """MEDIAN of per-chunk means (iters split into `chunks` event-timed
    spans). The registered winner rule is a median; a single whole-loop
    span divided by iters is a MEAN and a few noisy launches can rank a
    different winner than the rule (Bugbot, gnf4#241). Chunked medians
    keep event overhead off the kernel while honoring the registration."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    per = max(1, iters // chunks)
    spans = []
    for _ in range(chunks):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(per):
            fn()
        e1.record()
        e1.synchronize()
        spans.append(e0.elapsed_time(e1) / per)
    spans.sort()
    return spans[len(spans) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--out", default="m1_sweep.json")
    args = ap.parse_args()
    rep = {"shapes": {}, "gpu": torch.cuda.get_device_name(0)}
    winner_sum = None
    plan_sum = 0.0
    per_shape_best = {}
    for name, N, K, T in SHAPES:
        a, p, ax, sizes, eids = _mk(N, K, T)
        run_plan = lambda: nf4_grouped.gemm_4bit_grouped(a, p, ax, sizes,
                                                         eids)
        plan_start = _time(run_plan, args.iters, args.warmup)
        rows = []
        for bn, w, sk in itertools.product(BNS, WARPS, SKS):
            if sk > max(K // 64, 1):
                continue
            fn = lambda: nf4_grouped.gemm_4bit_grouped(
                a, p, ax, sizes, eids, decode_config=(bn, w), split_k=sk)
            try:
                ms = _time(fn, args.iters, args.warmup)
            except Exception as e:                   # noqa: BLE001
                rows.append({"bn": bn, "warps": w, "sk": sk,
                             "error": str(e)[:80]})
                continue
            programs = T * -(-N // bn) * sk
            rows.append({"bn": bn, "warps": w, "sk": sk, "ms": ms,
                         "programs": programs})
        plan_end = _time(run_plan, args.iters, args.warmup)
        drift = abs(plan_end - plan_start) / min(plan_start, plan_end) * 100
        ok = [r for r in rows if "ms" in r]
        ok.sort(key=lambda r: (r["ms"], r["programs"]))
        rep["shapes"][name] = {
            "N": N, "K": K, "T": T,
            "plan_ms_start": plan_start, "plan_ms_end": plan_end,
            "plan_drift_pct": drift,
            "noise_gate_pass": drift <= 5.0,
            "best": ok[0] if ok else None,
            "table": rows,
        }
        plan_sum += min(plan_start, plan_end)
        per_shape_best[name] = ok[0] if ok else None
    if all(v is not None for v in per_shape_best.values()):
        winner_sum = sum(v["ms"] for v in per_shape_best.values())
    rep["summary"] = {
        "plan_sum_ms": plan_sum,
        "winner_sum_ms": winner_sum,
        "ratio_winner_over_plan": (winner_sum / plan_sum
                                   if winner_sum else None),
        "winners": per_shape_best,
        "noise_gate_pass": all(s["noise_gate_pass"]
                               for s in rep["shapes"].values()),
    }
    Path(args.out).write_text(json.dumps(rep, indent=1))
    s = rep["summary"]
    print(f"SWEEP plan_sum={s['plan_sum_ms']:.3f}ms "
          f"winner_sum={s['winner_sum_ms']:.3f}ms "
          f"ratio={s['ratio_winner_over_plan']:.3f} "
          f"noise_gate={'PASS' if s['noise_gate_pass'] else 'FAIL'}")
    for k, v in per_shape_best.items():
        print(f"  {k}: bn={v['bn']} warps={v['warps']} sk={v['sk']} "
              f"{v['ms']:.3f}ms")


if __name__ == "__main__":
    main()
