#!/usr/bin/env python3
# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Saturating pinned H2D stream for the N0 concurrent-with-PCIe leg — and,
as a byproduct, this box's measured L (the same 1 GiB pinned-copy microbench
the flagship harness uses). VRAM-gated: refuses to run into a contended card
(exit 3) rather than produce a number contaminated by eviction pressure.

  python h2d_loop.py <seconds> <out_json>
"""
import json
import statistics
import sys
import time

import torch

GIB = 1 << 30
NEED_FREE = int(2.5 * GIB)   # 1 GiB device buffer + slack


def main():
    seconds = float(sys.argv[1])
    out = sys.argv[2]
    free, total = torch.cuda.mem_get_info()
    if free < NEED_FREE:
        json.dump({"skipped": "vram", "free_gb": free / 1e9,
                   "total_gb": total / 1e9}, open(out, "w"))
        print(f"SKIP: {free/1e9:.1f} GB free < {NEED_FREE/1e9:.1f} needed")
        return 3
    pin = torch.empty(GIB, dtype=torch.uint8).pin_memory()
    dev = torch.empty(GIB, dtype=torch.uint8, device="cuda")
    for _ in range(2):
        dev.copy_(pin, non_blocking=True)
    torch.cuda.synchronize()
    ts = []
    t_end = time.time() + seconds
    while time.time() < t_end:
        a = time.perf_counter()
        dev.copy_(pin, non_blocking=True)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - a)
    med = statistics.median(ts)
    rep = {"h2d_gbps": round(GIB / med / 1e9, 2), "n_copies": len(ts),
           "seconds": seconds, "gpu": torch.cuda.get_device_name(0),
           "p90_gbps": round(GIB / sorted(ts)[int(0.9 * len(ts))] / 1e9, 2),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    json.dump(rep, open(out, "w"))
    print(json.dumps(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
