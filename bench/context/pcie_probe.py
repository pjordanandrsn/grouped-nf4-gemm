#!/usr/bin/env python3
# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).

"""Host->device transfer characterization — the input a streamed KV tier needs.

C4 makes the KV cache a streamed tier, and every claim about whether that is
affordable divides bytes by a link speed. This measures the link instead of
assuming it, in the shape the project's own transfer work uses:

    t(n) = c + n / B

`B` is the asymptotic bandwidth and `c` is the fixed per-copy cost. `c` is the
term that gets forgotten and then dominates: a decode step issues one copy per
LAYER, so a 94-layer model pays `94 * c` per token no matter how small the
slices are. At 20 us that is 1.9 ms/token — invisible at 2 tok/s, a fifth of the
budget at 100 tok/s.

Three things get measured because they can disagree:

  * **pinned vs pageable** — only pinned memory makes `non_blocking` H2D
    actually asynchronous; pageable silently stages through a bounce buffer.
  * **one big copy vs L small ones** totalling the same bytes — this is what
    isolates `c`, and it is the realistic shape (per-layer slices, not one
    contiguous blob).
  * **the reported link vs the achieved one** — `nvidia-smi` reports gen/width
    that can differ from what a sustained transfer gets, and the current values
    downtrain at idle.

No hypothesis is being tested here, so there is no prereg: this is a device
characterization like `kv_verify.py`, not an experiment.
"""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys

import torch

OUT = os.environ.get("PCIE_OUT", "/root/g/bench/context/pcie_probe.json")
REPS = int(os.environ.get("PCIE_REPS", "20"))
# Spans a realistic per-layer packed-KV slice: Qwen3-235B holds 576 B/layer/token
# packed, so 4K ctx = 2.4 MB, 32K = 18.9 MB, 128K = 75.5 MB per layer.
SIZES_MB = [0.0625, 0.25, 1, 4, 16, 64, 256]


def _time_copy(src, dst, reps=REPS):
    """Median wall time of a single H2D copy, CUDA-event timed."""
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    ts = []
    for i in range(reps + 3):
        torch.cuda.synchronize()
        start.record()
        dst.copy_(src, non_blocking=True)
        end.record()
        torch.cuda.synchronize()
        if i >= 3:                                   # discard warmup
            ts.append(start.elapsed_time(end) / 1e3)
    return statistics.median(ts)


def _time_many(srcs, dsts, reps=REPS):
    """Median wall time of issuing N copies back to back — the decode shape."""
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    ts = []
    for i in range(reps + 3):
        torch.cuda.synchronize()
        start.record()
        for s, d in zip(srcs, dsts):
            d.copy_(s, non_blocking=True)
        end.record()
        torch.cuda.synchronize()
        if i >= 3:
            ts.append(start.elapsed_time(end) / 1e3)
    return statistics.median(ts)


def _fit(points):
    """Least-squares fit of t = c + n/B over (bytes, seconds). Returns (c, B)."""
    n = len(points)
    sx = sum(p[0] for p in points)
    sy = sum(p[1] for p in points)
    sxx = sum(p[0] * p[0] for p in points)
    sxy = sum(p[0] * p[1] for p in points)
    denom = n * sxx - sx * sx
    slope = (n * sxy - sx * sy) / denom            # seconds per byte
    c = (sy - slope * sx) / n
    return c, 1.0 / slope


def _link():
    try:
        q = ("pcie.link.gen.max,pcie.link.gen.current,"
             "pcie.link.width.max,pcie.link.width.current")
        out = subprocess.run(["nvidia-smi", f"--query-gpu={q}",
                              "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=20).stdout.strip()
        g_max, g_cur, w_max, w_cur = [x.strip() for x in out.split(",")]
        # PCIe per-lane throughput, GB/s, after encoding overhead
        per_lane = {"1": 0.250, "2": 0.500, "3": 0.985, "4": 1.969, "5": 3.938}
        theo = per_lane.get(g_max, 0) * int(w_max)
        return dict(gen_max=g_max, gen_current=g_cur, width_max=w_max,
                    width_current=w_cur, theoretical_GBs=round(theo, 2))
    except Exception as e:                            # noqa: BLE001
        return dict(error=str(e))


def main():
    dev = torch.device("cuda:0")
    res = dict(link=_link(), gpu=torch.cuda.get_device_name(0), sizes=[])
    print(f"{res['gpu']}  link={res['link']}", flush=True)
    print(f"\n{'MB':>8} {'pinned GB/s':>12} {'pageable GB/s':>14} "
          f"{'pinned ms':>10}", flush=True)

    fit_pts = []
    for mb in SIZES_MB:
        n = int(mb * 2 ** 20)
        pin = torch.empty(n, dtype=torch.uint8).pin_memory()
        page = torch.empty(n, dtype=torch.uint8)
        dst = torch.empty(n, dtype=torch.uint8, device=dev)
        t_pin = _time_copy(pin, dst)
        t_page = _time_copy(page, dst)
        fit_pts.append((n, t_pin))
        row = dict(mb=mb, bytes=n, pinned_s=t_pin, pageable_s=t_page,
                   pinned_GBs=n / t_pin / 1e9, pageable_GBs=n / t_page / 1e9)
        res["sizes"].append(row)
        print(f"{mb:>8.4g} {row['pinned_GBs']:>12.2f} {row['pageable_GBs']:>14.2f} "
              f"{t_pin * 1e3:>10.3f}", flush=True)
        del pin, page, dst
        torch.cuda.empty_cache()

    c, B = _fit(fit_pts)
    res["fit"] = dict(fixed_cost_s=c, asymptotic_GBs=B / 1e9)
    print(f"\nfit  t = c + n/B :  c = {c * 1e6:.1f} us   B = {B / 1e9:.2f} GB/s")

    # The decode shape: L per-layer slices vs one blob of the same total bytes.
    # This is where a per-copy cost that looks negligible turns into L * c.
    print(f"\n{'layers':>7} {'slice MB':>9} {'split ms':>9} {'blob ms':>8} "
          f"{'overhead':>9} {'implied c':>10}", flush=True)
    res["split"] = []
    for layers, slice_mb in ((94, 18.9), (94, 2.4), (48, 18.0), (30, 6.4), (16, 36.0)):
        n = int(slice_mb * 2 ** 20)
        srcs = [torch.empty(n, dtype=torch.uint8).pin_memory() for _ in range(layers)]
        dsts = [torch.empty(n, dtype=torch.uint8, device=dev) for _ in range(layers)]
        t_split = _time_many(srcs, dsts, reps=max(5, REPS // 4))
        blob_src = torch.empty(n * layers, dtype=torch.uint8).pin_memory()
        blob_dst = torch.empty(n * layers, dtype=torch.uint8, device=dev)
        t_blob = _time_copy(blob_src, blob_dst, reps=max(5, REPS // 4))
        row = dict(layers=layers, slice_mb=slice_mb, split_s=t_split, blob_s=t_blob,
                   overhead=t_split / t_blob,
                   implied_c_us=(t_split - t_blob) / layers * 1e6)
        res["split"].append(row)
        print(f"{layers:>7} {slice_mb:>9.1f} {t_split * 1e3:>9.2f} "
              f"{t_blob * 1e3:>8.2f} {row['overhead']:>8.2f}x "
              f"{row['implied_c_us']:>9.1f}us", flush=True)
        del srcs, dsts, blob_src, blob_dst
        torch.cuda.empty_cache()

    json.dump(res, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
