# Phase-0 premise test 1: DDR bandwidth + thread-scaling ceilings (STREAM-class,
# torch-based — not certified STREAM; the tool is named in the receipt).
# Short + polite on shared boxes: each point is ~0.3 s of traffic, nice(10).
#
# Usage: python3 phase0_ddr_bench.py [--threads 1,2,4,6,8,12] [--mb 512]
import argparse
import json
import os
import time

import torch


def bench_copy(n_bytes: int, threads: int, reps: int = 5) -> float:
    """STREAM 'copy': dst.copy_(src). Reports GB/s counting read+write."""
    torch.set_num_threads(threads)
    src = torch.empty(n_bytes // 4, dtype=torch.float32)
    src.uniform_()
    dst = torch.empty_like(src)
    dst.copy_(src)  # warm
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        dst.copy_(src)
        best = min(best, time.perf_counter() - t0)
    return 2 * n_bytes / best / 1e9


def bench_read(n_bytes: int, threads: int, reps: int = 5) -> float:
    """Read-shaped (ADDENDUM-2 A6.1): sum-reduce a large buffer — reads N
    bytes, writes a scalar. This is the shape the cold kernel's traffic has
    (packed weights in, tiny activations out); counted 1x bytes."""
    torch.set_num_threads(threads)
    src = torch.empty(n_bytes // 4, dtype=torch.float32).uniform_()
    _ = src.sum()  # warm
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        _ = src.sum()
        best = min(best, time.perf_counter() - t0)
    return n_bytes / best / 1e9


def bench_triad(n_bytes: int, threads: int, reps: int = 5) -> float:
    """STREAM 'triad': a = b + 3.0*c (3 streams)."""
    torch.set_num_threads(threads)
    n = n_bytes // 4
    b = torch.empty(n, dtype=torch.float32).uniform_()
    c = torch.empty(n, dtype=torch.float32).uniform_()
    a = torch.empty_like(b)
    torch.add(b, c, alpha=3.0, out=a)  # warm
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        torch.add(b, c, alpha=3.0, out=a)
        best = min(best, time.perf_counter() - t0)
    return 3 * n_bytes / best / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", default="1,2,4,6,8,12")
    ap.add_argument("--mb", type=int, default=512)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    try:
        os.nice(10)
    except OSError:
        pass
    n_bytes = args.mb * (1 << 20)
    rows = []
    for t in (int(x) for x in args.threads.split(",")):
        copy = bench_copy(n_bytes, t)
        triad = bench_triad(n_bytes, t)
        read = bench_read(n_bytes, t)
        rows.append(dict(threads=t, copy_gbs=round(copy, 2),
                         triad_gbs=round(triad, 2), read_gbs=round(read, 2)))
        print(f"threads={t:3d}  copy={copy:7.2f}  triad={triad:7.2f}  "
              f"read={read:7.2f} GB/s", flush=True)
    result = dict(tool="torch-copy/triad (not certified STREAM)",
                  torch=torch.__version__, mb=args.mb,
                  cpu_count=os.cpu_count(), rows=rows)
    print(json.dumps(result))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
