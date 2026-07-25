#!/usr/bin/env python3
# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).

"""A1 of PREREG-kv-stream-faster.md — split residency. Stamped before it existed.

Scores on a FIT across four points rather than a difference of two, which is the
methodology fix #15 forced: `t_host - t_gpu` carries ~15 ms of noise, fine
against a 286 ms overhead and useless against a 72 ms one.

  overhead(f) = t_split(f) - t_resident        (one common baseline, not pairwise)
  fit          overhead = c + streamed_bytes(f) / B

`B` should come back as the link speed (A1a) and `c` as the fixed cost of
splitting at all (A1b). Peak GPU is measured AFTER the prefill, over the timed
loop only -- #15's S4 was falsified partly because its peak included ~340 MB of
build transients that a ratio cannot cancel.
"""
from __future__ import annotations

import json
import os
import statistics
import sys

import torch

sys.path.insert(0, "/root/g/kernel")
from experts4bit_qlora import NF4KVCache  # noqa: E402

OUT = os.environ.get("SP_OUT", "/root/g/bench/context/kv_split_bench.json")
REPS = int(os.environ.get("SP_REPS", "8"))
LINK = float(os.environ.get("SP_LINK", "6.20"))
LAYERS, H_KV, D, CTX = 94, 4, 128, 32768
FRACTIONS = [0.0, 0.25, 0.5, 0.75]
B_PER_TOKEN = 2 * H_KV * D * (0.5 + 4.0 / 64.0)          # 576 B/layer/token


def build(resident_tokens, residence):
    kw = dict(quantize_keys=True, quantize_values=True)
    if residence == "host":
        kw.update(residence="host", max_context=CTX + 64,
                  resident_tokens=resident_tokens)
    c = NF4KVCache(**kw)
    g = torch.Generator(device="cpu").manual_seed(0)
    for lo in range(0, CTX, 2048):
        n = min(2048, CTX - lo)
        k = (torch.randn(1, H_KV, n, D, generator=g) * 0.5).bfloat16().cuda()
        v = (torch.randn(1, H_KV, n, D, generator=g) * 0.5).bfloat16().cuda()
        for li in range(LAYERS):
            c.update(k, v, li)
        del k, v
    torch.cuda.synchronize()
    return c


def measure(resident_tokens, residence):
    torch.cuda.empty_cache()
    c = build(resident_tokens, residence)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()          # AFTER the prefill, per #15

    def load_all():
        for li in range(LAYERS):
            if residence == "host" and c.resident_tokens:
                k = c._materialize(li, "k", torch.bfloat16)
                v = c._materialize(li, "v", torch.bfloat16)
            else:
                k = c._load(c._k[li], torch.bfloat16)
                v = c._load(c._v[li], torch.bfloat16)
            del k, v

    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    ts = []
    for i in range(REPS + 2):
        torch.cuda.synchronize()
        start.record()
        load_all()
        end.record()
        torch.cuda.synchronize()
        if i >= 2:
            ts.append(start.elapsed_time(end) / 1e3)
    streamed = (CTX - resident_tokens) * B_PER_TOKEN * LAYERS if residence == "host" else 0
    r = dict(residence=residence, resident_tokens=resident_tokens,
             fraction=resident_tokens / CTX, load_s=statistics.median(ts),
             streamed_bytes=streamed, cache_bytes=c.memory_bytes(),
             device_bytes=c.device_bytes(),
             steady_peak_bytes=torch.cuda.max_memory_allocated())
    del c
    torch.cuda.empty_cache()
    return r


def fit(points):
    """Least squares on t = c + n/B over (bytes, seconds)."""
    n = len(points)
    sx = sum(p[0] for p in points)
    sy = sum(p[1] for p in points)
    sxx = sum(p[0] * p[0] for p in points)
    sxy = sum(p[0] * p[1] for p in points)
    slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    return (sy - slope * sx) / n, 1.0 / slope


def main():
    print(f"{LAYERS}L x {H_KV}kv x {D}d @ ctx {CTX}   link {LINK} GB/s   "
          f"reps {REPS}", flush=True)
    base = measure(0, "gpu")
    print(f"resident baseline: {base['load_s'] * 1e3:8.1f} ms  "
          f"peak {base['steady_peak_bytes'] / 2**20:7.1f} MB  "
          f"device {base['device_bytes'] / 1e9:.3f} GB", flush=True)
    rows = [base]
    for f in FRACTIONS:
        r = measure(int(f * CTX), "host")
        r["overhead_s"] = r["load_s"] - base["load_s"]
        rows.append(r)
        print(f"f={f:<5} resident={r['resident_tokens']:>6}  "
              f"load={r['load_s'] * 1e3:8.1f} ms  over={r['overhead_s'] * 1e3:7.1f} ms  "
              f"streamed={r['streamed_bytes'] / 1e9:.3f} GB  "
              f"peak={r['steady_peak_bytes'] / 2**20:7.1f} MB  "
              f"device={r['device_bytes'] / 1e9:.3f} GB", flush=True)
        json.dump(rows, open(OUT, "w"), indent=2)

    hosts = [r for r in rows if r["residence"] == "host"]
    c, B = fit([(r["streamed_bytes"], r["overhead_s"]) for r in hosts])
    print("\n=== scoring ===")
    print(f"A1a fitted B = {B / 1e9:.2f} GB/s          [5.0, 7.5]   "
          f"{'CONFIRMED' if 5.0 <= B / 1e9 <= 7.5 else 'FALSIFIED'}")
    print(f"A1b fitted c = {c * 1e3:.1f} ms            < 25 ms      "
          f"{'CONFIRMED' if c * 1e3 < 25 else 'FALSIFIED'}")
    p0 = next(r for r in hosts if r["fraction"] == 0.0)["steady_peak_bytes"]
    p5 = next(r for r in hosts if r["fraction"] == 0.5)["steady_peak_bytes"]
    print(f"A1d peak(f=.5)/peak(f=0) = {p5 / p0:.3f}   [0.4, 0.65]  "
          f"{'CONFIRMED' if 0.4 <= p5 / p0 <= 0.65 else 'FALSIFIED'}")
    json.dump(dict(rows=rows, fit=dict(c_s=c, B_GBs=B / 1e9),
                   link_GBs=LINK, ctx=CTX,
                   geometry=dict(layers=LAYERS, h_kv=H_KV, head_dim=D)),
              open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
