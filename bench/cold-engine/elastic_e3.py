"""Gate E3: do the phase-0 constants transfer to this box, and does H2D hide?

Registered in PREREG-elastic-promotion.md. Measures, at the 13.22 MB gpt-oss
row size:

  B_cpu   grouped expert GEMV, via the committed phase-2 bench (shape
          gptossish_gateup), best over the thread sweep;
  B_link  pinned H2D, evented, median over repeats;
  B_gpu   bf16 GEMV touching the same bytes on-GPU, evented, median --
          the bandwidth-bound proxy the 1572 GB/s constant assumed. The
          real NF4 GPU path adds dequant work; using the proxy is the
          same convention the model used and is disclosed in the receipt.

n*_direct = 1 + [(1/B_link + 1/B_gpu) - 1/B_cpu] / (1/B_cpu - 1/B_gpu)
(total invocations, including the promoted one). Gate: in [2, 5].

E3b (reported, not gated): hideability. Wall of a CPU-GEMV batch alone, an
H2D batch alone, and both concurrently -- hidden fraction
= 1 - (wall_joint - wall_cpu) / wall_h2d. Both engines pull DRAM, so this is
exactly the contention question the controller depends on.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROW_BYTES = 13_219_200               # gpt-oss expert row (both projections)


def parse_phase2_receipt(d):
    """Best GB/s from the phase-2 receipt's own schema: `sweep` + `best_gbs`.

    The first version read a `results` key that harness never writes, so the
    structured path never hit and every run silently fell through to scraping
    stdout for "GB/s" lines (Bugbot, gnf4#202). The number survived only
    because best_gbs is max(sweep gbs) BY CONSTRUCTION and stdout prints the
    same values -- but a measurement whose provenance is a print-format
    accident is not tied to the receipt the preregistration named. The
    fallback is now a hard error rather than a silent substitution.
    """
    if "best_gbs" not in d or "sweep" not in d:
        raise SystemExit("phase2 receipt missing best_gbs/sweep -- schema "
                         "changed? keys: %s" % sorted(d))
    best = max(r["gbs"] for r in d["sweep"])
    if abs(best - d["best_gbs"]) > 1e-6:
        raise SystemExit("phase2 receipt inconsistent: best_gbs=%s but "
                         "max(sweep)=%s" % (d["best_gbs"], best))
    return d["best_gbs"], d


def bench_cpu(threads):
    """Best GB/s from the committed phase-2 bench at the gpt-oss shape."""
    out = os.path.join("/tmp", "e3_phase2.json")
    cmd = [sys.executable, os.path.join(HERE, "phase2_gemv_bench.py"),
           "--fmt", "nf4", "--shape", "gptossish_gateup",
           "--threads", threads, "--tokens", "24", "--out", out]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [os.path.join(HERE, "..", "..", "kernel"),
         os.path.join(HERE, "..", ".."), env.get("PYTHONPATH", "")])
    r = subprocess.run(cmd, capture_output=True, text=True, env=env,
                       cwd=os.path.join(HERE, "..", ".."))
    if r.returncode != 0:
        sys.exit("phase2 bench failed:\n" + r.stdout[-1500:] + r.stderr[-1500:])
    return parse_phase2_receipt(json.load(open(out)))


def evented_ms(fn, reps, stream=None):
    s = stream or torch.cuda.current_stream()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    for i in range(reps):
        starts[i].record(s)
        fn(i)
        ends[i].record(s)
    torch.cuda.synchronize()
    return [starts[i].elapsed_time(ends[i]) for i in range(reps)]


def bench_link(reps=40, nbuf=4):
    src = [torch.empty(ROW_BYTES, dtype=torch.uint8).pin_memory()
           for _ in range(nbuf)]
    dst = [torch.empty(ROW_BYTES, dtype=torch.uint8, device="cuda")
           for _ in range(nbuf)]
    for i in range(4):
        dst[i % nbuf].copy_(src[i % nbuf], non_blocking=True)
    torch.cuda.synchronize()
    ms = evented_ms(lambda i: dst[i % nbuf].copy_(src[i % nbuf],
                                                  non_blocking=True), reps)
    med = statistics.median(ms)
    return ROW_BYTES / (med * 1e-3) / 1e9, ms


def bench_gpu(reps=60):
    n_elem = ROW_BYTES // 2                       # bf16
    K = 2880
    N = n_elem // K
    W = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    x = torch.randn(K, dtype=torch.bfloat16, device="cuda")
    y = torch.mv(W, x)                            # warm
    torch.cuda.synchronize()
    ms = evented_ms(lambda i: torch.mv(W, x), reps)
    med = statistics.median(ms)
    return (N * K * 2) / (med * 1e-3) / 1e9, ms


def bench_hide(seconds=3.0):
    """CPU GEMV batch alone, H2D batch alone, then both concurrently.

    The CPU side is the REAL phase-2 kernel via phase2_gemv_bench's own
    arena builder and entry point -- not a reimplementation -- at the
    gpt-oss shape, one layer, 32 threads (the phase-0 bandwidth budget).
    """
    sys.path.insert(0, os.path.join(HERE, "..", "..", "kernel"))
    sys.path.insert(0, os.path.join(HERE, "..", ".."))
    sys.path.insert(0, HERE)
    import numpy as np
    import cpu_grouped as cg
    import gnf4_native
    import phase2_gemv_bench as p2

    gnf4_native.load()
    N, K = p2.SHAPES["gptossish_gateup"]
    L, E, k = 1, 32, 4
    keep, packed, scales, arena_bytes = p2.build_arena(
        "nf4", L, E, N, K, 20260822)
    t_packed = torch.from_numpy(packed[0])
    t_scales = torch.from_numpy(scales[0])
    a_cat = torch.randn(k, K, dtype=torch.float32).contiguous()
    sizes = [1] * k
    rng = np.random.default_rng(7)
    per_call = k * N * (K // 2 + (K // 64) * 4)

    def one_call():
        eids = rng.choice(E, size=k, replace=False).tolist()
        cg.gemv_nf4_grouped_cpu(a_cat, t_packed, t_scales, sizes, eids,
                                threads=32)

    one_call()                                     # compile-at-first-use
    t0 = time.perf_counter()
    one_call()
    per_call_s = time.perf_counter() - t0
    calls = max(8, int(seconds / max(per_call_s, 1e-4)))

    def cpu_batch():
        for _ in range(calls):
            one_call()

    t0 = time.perf_counter(); cpu_batch(); wall_cpu = time.perf_counter() - t0

    stream = torch.cuda.Stream()
    src = [torch.empty(ROW_BYTES, dtype=torch.uint8).pin_memory()
           for _ in range(4)]
    dst = [torch.empty(ROW_BYTES, dtype=torch.uint8, device="cuda")
           for _ in range(4)]
    ncopy = max(8, int(wall_cpu * 0.7 * 52e9 / ROW_BYTES))

    def h2d_batch():
        with torch.cuda.stream(stream):
            for i in range(ncopy):
                dst[i % 4].copy_(src[i % 4], non_blocking=True)
        stream.synchronize()

    t0 = time.perf_counter(); h2d_batch(); wall_h2d = time.perf_counter() - t0

    th = threading.Thread(target=h2d_batch)
    t0 = time.perf_counter()
    th.start(); cpu_batch(); th.join()
    wall_joint = time.perf_counter() - t0

    hidden = 1.0 - max(0.0, wall_joint - wall_cpu) / wall_h2d
    return {"wall_cpu_s": wall_cpu, "wall_h2d_s": wall_h2d,
            "wall_joint_s": wall_joint, "ncopy": ncopy, "calls": calls,
            "cpu_gbs_alone": calls * per_call / wall_cpu / 1e9,
            "hidden_frac": hidden,
            "h2d_gbs_alone": ncopy * ROW_BYTES / wall_h2d / 1e9}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", default="16,32,64,96")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    assert torch.cuda.is_available()
    x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
    assert float((x @ x).float().abs().mean()) > 0    # a real kernel ran

    b_cpu, cpu_detail = bench_cpu(a.threads)
    b_link, link_ms = bench_link()
    b_gpu, gpu_ms = bench_gpu()
    save = 1.0 / b_cpu - 1.0 / b_gpu
    nstar = 1 + ((1.0 / b_link + 1.0 / b_gpu) - 1.0 / b_cpu) / save
    hide = bench_hide()

    rec = {"row_bytes": ROW_BYTES,
           "cpu_sweep": cpu_detail.get("sweep"),
           "B_cpu_gbs": b_cpu, "B_link_gbs": b_link, "B_gpu_gbs": b_gpu,
           "nstar_direct": nstar, "gate": [2, 5],
           "verdict": "PASS" if 2 <= nstar <= 5 else "FAIL",
           "e3b": hide,
           "gpu": torch.cuda.get_device_name(0),
           "note": "B_gpu is the bf16 bandwidth proxy the 1572 GB/s constant "
                   "assumed; the NF4 GPU path adds dequant work"}
    print(json.dumps({k: v for k, v in rec.items() if k != "e3b"}, indent=1))
    print("E3b:", json.dumps(hide, indent=1))
    with open(a.out, "w") as f:
        json.dump(rec, f, indent=1)
    print("receipt ->", a.out)


if __name__ == "__main__":
    main()
