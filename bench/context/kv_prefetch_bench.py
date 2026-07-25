#!/usr/bin/env python3
# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).

"""B1 of PREREG-kv-stream-faster.md — prefetch. Stamped before it existed.

Under host residence the copy and the dequant serialize on the default stream,
so a step costs `compute + transfer`. Issuing layer L+1's copy on a side stream
while L's dequant runs should make it `max(compute, transfer)`. At 32K the
per-layer dequant is 10.3 ms against a 3.05 ms transfer, so there is room to
hide all of it — B1a predicts the streamed path lands within 1.00-1.15x of the
fully-resident one, i.e. the transfer becomes essentially free.

Three arms at the same context, same geometry, same cache contents:

  resident        residence="gpu"   — no transfer at all, the floor
  streamed        residence="host"  — copy and dequant serialized
  streamed+pre    residence="host"  — copy on a side stream, one layer ahead

Correctness is not re-checked here (the property suite pins that prefetch
changes timing and not values, with repeats, because a missing event-wait is a
race and fails intermittently rather than loudly).
"""
from __future__ import annotations

import json
import os
import statistics
import sys

import torch

sys.path.insert(0, "/root/g/kernel")
from experts4bit_qlora import NF4KVCache  # noqa: E402

OUT = os.environ.get("PF_OUT", "/root/g/bench/context/kv_prefetch_bench.json")
REPS = int(os.environ.get("PF_REPS", "8"))
LAYERS, H_KV, D = 94, 4, 128
CONTEXTS = [int(x) for x in os.environ.get("PF_CTX", "8192,32768").split(",")]
B_PER_TOKEN = 2 * H_KV * D * (0.5 + 4.0 / 64.0)


def build(ctx, residence):
    kw = dict(quantize_keys=True, quantize_values=True)
    if residence == "host":
        kw.update(residence="host", max_context=ctx + 64)
    c = NF4KVCache(**kw)
    g = torch.Generator(device="cpu").manual_seed(0)
    for lo in range(0, ctx, 2048):
        n = min(2048, ctx - lo)
        k = (torch.randn(1, H_KV, n, D, generator=g) * 0.5).bfloat16().cuda()
        v = (torch.randn(1, H_KV, n, D, generator=g) * 0.5).bfloat16().cuda()
        for li in range(LAYERS):
            c.update(k, v, li)
        del k, v
    torch.cuda.synchronize()
    return c


def measure(ctx, residence, prefetch):
    torch.cuda.empty_cache()
    c = build(ctx, residence)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    def run():
        if prefetch:
            c.prefetch(0)
        for li in range(LAYERS):
            if prefetch and li + 1 < LAYERS:
                # issue L+1's copy BEFORE L's dequant, so the side stream has
                # something in flight for the whole of it
                c.prefetch(li + 1)
            k = c._load_layer(li, "k", torch.bfloat16)
            v = c._load_layer(li, "v", torch.bfloat16)
            del k, v

    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    ts = []
    for i in range(REPS + 2):
        torch.cuda.synchronize()
        start.record()
        run()
        end.record()
        torch.cuda.synchronize()
        if i >= 2:
            ts.append(start.elapsed_time(end) / 1e3)
    r = dict(context=ctx, residence=residence, prefetch=prefetch,
             load_s=statistics.median(ts),
             peak_bytes=torch.cuda.max_memory_allocated(),
             streamed_bytes=(ctx * B_PER_TOKEN * LAYERS if residence == "host" else 0))
    del c
    torch.cuda.empty_cache()
    return r


def main():
    print(f"{LAYERS}L x {H_KV}kv x {D}d   reps {REPS}", flush=True)
    rows = []
    for ctx in CONTEXTS:
        for residence, pre in (("gpu", False), ("host", False), ("host", True)):
            r = measure(ctx, residence, pre)
            rows.append(r)
            tag = f"{residence}{'+pre' if pre else ''}"
            print(f"ctx={ctx:>6} {tag:<9} load={r['load_s'] * 1e3:8.1f} ms  "
                  f"peak={r['peak_bytes'] / 2**20:7.1f} MB", flush=True)
            json.dump(rows, open(OUT, "w"), indent=2)

    print("\n=== scoring (ctx 32768) ===")
    def g(ctx, res, pre):
        return next(r for r in rows if r["context"] == ctx
                    and r["residence"] == res and r["prefetch"] == pre)
    verdicts = {}
    if 32768 in CONTEXTS:
        base = g(32768, "gpu", False)["load_s"]
        plain = g(32768, "host", False)["load_s"]
        pre = g(32768, "host", True)["load_s"]
        a = pre / base
        b = plain / pre
        dmb = (g(32768, "host", True)["peak_bytes"]
               - g(32768, "host", False)["peak_bytes"]) / 2 ** 20
        verdicts = {
            "B1a": dict(measured=a, interval=[1.00, 1.15],
                        verdict="CONFIRMED" if a <= 1.25 else "FALSIFIED",
                        inside_interval=1.00 <= a <= 1.15),
            "B1b": dict(measured=b, interval=[1.20, 1.35],
                        verdict="CONFIRMED" if 1.20 <= b <= 1.35 else "FALSIFIED"),
            "B1c": dict(measured_mb=dmb, threshold_mb=100,
                        verdict="CONFIRMED" if dmb < 100 else "FALSIFIED"),
        }
        print(f"B1a prefetched/resident   = {a:.3f}   [1.00,1.15], falsify >1.25  "
              f"{verdicts['B1a']['verdict']}")
        print(f"B1b plain/prefetched      = {b:.3f}   [1.20,1.35]  "
              f"{verdicts['B1b']['verdict']}")
        print(f"B1c peak delta            = {dmb:+.1f} MB  < 100 MB  "
              f"{verdicts['B1c']['verdict']}")
    for ctx in CONTEXTS:
        base, plain, pre = (g(ctx, "gpu", False)["load_s"],
                            g(ctx, "host", False)["load_s"],
                            g(ctx, "host", True)["load_s"])
        print(f"  ctx {ctx}: transfer exposed {(plain - base) * 1e3:7.1f} ms -> "
              f"{(pre - base) * 1e3:7.1f} ms "
              f"({(1 - (pre - base) / max(plain - base, 1e-9)) * 100:5.1f}% hidden)")
    json.dump(dict(rows=rows, verdicts=verdicts), open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
