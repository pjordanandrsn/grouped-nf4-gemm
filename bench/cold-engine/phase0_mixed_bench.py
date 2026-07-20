# Phase-0 item 1, third measurement (ADDENDUM-1 A2): the MIXED bench —
# concurrent H2D DMA + CPU STREAM-class copies, sweeping the CPU-thread ratio.
# D_eff in the fusion theorem is read off THIS curve at the operating split,
# not off the solo STREAM number (contention is the theorem's named predator).
#
# Conventions (stated so rows are comparable): DMA rate counts bytes over the
# link once (each byte is also one DRAM read on the host side); CPU copy rate
# counts read+write (2x buffer bytes), same as phase0_ddr_bench. Rows report
# both raw rates; the aggregate-drain column is dma + cpu.
#
# Short + polite: each row is a ~3 s window; CPU side nice(10).
import argparse
import json
import os
import threading
import time

import torch


def dma_worker(stop, out, mb):
    src = torch.empty(mb << 20, dtype=torch.uint8).pin_memory()
    dst = torch.empty(mb << 20, dtype=torch.uint8, device="cuda")
    torch.cuda.synchronize()
    n = 0
    t0 = time.perf_counter()
    while not stop.is_set():
        dst.copy_(src, non_blocking=True)
        torch.cuda.synchronize()
        n += 1
    out["gbs"] = n * (mb << 20) / (time.perf_counter() - t0) / 1e9


def cpu_worker(stop, out, idx, mb, shape="read"):
    """ADDENDUM-2 A6.2: the CPU side is READ-shaped by default (sum-reduce,
    1x bytes) — the shape the cold kernel has; copy (2x) kept for reference."""
    src = torch.empty((mb << 20) // 4, dtype=torch.float32).uniform_()
    dst = torch.empty_like(src) if shape == "copy" else None
    n = 0
    t0 = time.perf_counter()
    while not stop.is_set():
        if shape == "copy":
            dst.copy_(src)
        else:
            _ = src.sum()
        n += 1
    mult = 2 if shape == "copy" else 1
    out[idx] = mult * n * (mb << 20) / (time.perf_counter() - t0) / 1e9


def row(cpu_threads, secs, dma_mb, cpu_mb):
    torch.set_num_threads(1)  # each cpu_worker is its own stream of copies
    stop = threading.Event()
    dma_out, cpu_out = {}, {}
    ts = [threading.Thread(target=dma_worker, args=(stop, dma_out, dma_mb))]
    ts += [threading.Thread(target=cpu_worker, args=(stop, cpu_out, i, cpu_mb))
           for i in range(cpu_threads)]
    for t in ts:
        t.start()
    time.sleep(secs)
    stop.set()
    for t in ts:
        t.join()
    dma = round(dma_out.get("gbs", 0.0), 2)
    cpu = round(sum(cpu_out.values()), 2)
    return dict(cpu_threads=cpu_threads, dma_gbs=dma, cpu_gbs=cpu,
                aggregate_gbs=round(dma + cpu, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", default="0,1,2,4,6,8")
    ap.add_argument("--secs", type=float, default=3.0)
    ap.add_argument("--dma-mb", type=int, default=256)
    ap.add_argument("--cpu-mb", type=int, default=256)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    try:
        os.nice(10)
    except OSError:
        pass
    assert torch.cuda.is_available(), "mixed bench needs the CUDA link"
    rows = []
    for t in (int(x) for x in args.threads.split(",")):
        r = row(t, args.secs, args.dma_mb, args.cpu_mb)
        rows.append(r)
        print(f"cpu_threads={t:2d}  dma={r['dma_gbs']:6.2f}  "
              f"cpu={r['cpu_gbs']:6.2f}  aggregate={r['aggregate_gbs']:6.2f} GB/s",
              flush=True)
    result = dict(tool="torch pinned-H2D + threaded cpu-copy (conventions in header)",
                  gpu=torch.cuda.get_device_name(0), rows=rows)
    print(json.dumps(result))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
