# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
#
# Phase-2 gate G2: grouped expert GEMV at decode shapes, achieved GB/s vs
# the box's measured STREAM triad (>=70% passes). Trace-driven like the G0
# scatter bench: L layers x E experts, k=8 distinct experts per (token,
# layer) from a fixed-seed trace; the arena is sized >> total L3 so cache
# residency cannot inflate the number, and is first-touched parallel +
# MADV_HUGEPAGE'd (state recorded).
#
# Bytes counted: packed + scales actually touched (k rows' worth per call).
# Activations/outputs are not counted (they are ~1e-3 of the traffic).
#
# Usage: python3 phase2_gemv_bench.py [--fmt nf4|mxfp4] [--layers 24]
#          [--experts 96] [--tokens 40] [--threads 8,16,32,48] [--out r.json]

import argparse
import ctypes
import json
import mmap
import os
import time

import numpy as np
import torch

import cpu_grouped as cg
import gnf4_native

SHAPES = {                      # (N, K) per projection, decode-relevant
    "qwen3ish_gateup": (1536, 2048),
    "gptossish_gateup": (5760, 2880),
    "k3ish_gateup": (6144, 3584),
}


HUGETLB_STATE = {"mode": "thp-madvise"}


def huge_alloc(nbytes, explicit=False):
    """THP-madvise by default; --hugetlb asks for explicit 2 MiB pages
    (MAP_HUGETLB — needs vm.nr_hugepages reserved; bare-metal root can
    `sysctl vm.nr_hugepages=N`). Falls back with the state recorded."""
    if explicit and hasattr(mmap, "MAP_HUGETLB"):
        try:
            m = mmap.mmap(-1, nbytes,
                          flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS
                          | mmap.MAP_HUGETLB)
            HUGETLB_STATE["mode"] = "hugetlb-2m"
            return m
        except OSError as e:
            HUGETLB_STATE["mode"] = f"hugetlb-FAILED({e.errno})->thp"
    m = mmap.mmap(-1, nbytes)
    m.madvise(mmap.MADV_HUGEPAGE)
    return m


def build_arena(fmt, L, E, N, K, seed, explicit_huge=False):
    """One mmap'd arena per tensor kind, filled in parallel-ish (numpy)."""
    packed_b = L * E * N * (K // 2)
    m_packed = huge_alloc(packed_b, explicit_huge)
    packed = np.frombuffer(m_packed, dtype=np.uint8).reshape(L, E, N, K // 2)
    rng = np.random.default_rng(seed)
    # fill by layer to bound temp memory; contents irrelevant to bandwidth
    for layer in range(L):
        packed[layer] = rng.integers(0, 256, size=(E, N, K // 2), dtype=np.uint8)
    if fmt == "nf4":
        sc_b = L * E * N * (K // 64) * 4
        m_sc = huge_alloc(sc_b, explicit_huge)
        scales = np.frombuffer(m_sc, dtype=np.float32).reshape(L, E, N, K // 64)
        scales[:] = rng.random(size=scales.shape, dtype=np.float32) + 0.5
    else:
        sc_b = L * E * N * (K // 32)
        m_sc = huge_alloc(sc_b, explicit_huge)
        scales = np.frombuffer(m_sc, dtype=np.uint8).reshape(L, E, N, K // 32)
        scales[:] = rng.integers(110, 130, size=scales.shape, dtype=np.uint8)
    return (m_packed, m_sc), packed, scales, packed_b + sc_b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fmt", choices=["nf4", "mxfp4"], default="nf4")
    ap.add_argument("--shape", choices=list(SHAPES), default="qwen3ish_gateup")
    ap.add_argument("--layers", type=int, default=24)
    ap.add_argument("--experts", type=int, default=96)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--rows", type=int, default=1, help="tokens per group (T)")
    ap.add_argument("--tokens", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--threads", default="8,16,32,48")
    ap.add_argument("--seed", type=int, default=20260816)
    ap.add_argument("--pool", action="store_true",
                    help="dispatch on the persistent pinned pool per point")
    ap.add_argument("--hugetlb", action="store_true",
                    help="explicit MAP_HUGETLB arena (else THP madvise)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    N, K = SHAPES[args.shape]
    L, E, k = args.layers, args.experts, args.topk
    lib = gnf4_native.load()
    feats = gnf4_native.features()
    fn = (cg.gemv_nf4_grouped_cpu if args.fmt == "nf4"
          else cg.gemv_mxfp4_grouped_cpu)

    keep, packed, scales, arena_bytes = build_arena(
        args.fmt, L, E, N, K, args.seed, explicit_huge=args.hugetlb)
    print(f"arena: {arena_bytes / 2**30:.2f} GiB "
          f"({L}L x {E}E x [{N},{K}] {args.fmt})", flush=True)

    # fixed-seed routing trace: [tokens, L, k] distinct ids
    rng = np.random.default_rng(args.seed)
    trace = np.stack([
        np.stack([rng.choice(E, size=k, replace=False) for _ in range(L)])
        for _ in range(args.tokens + args.warmup)
    ])
    a = torch.randn(args.rows, K, dtype=torch.float32)
    sizes = [args.rows] * k          # k groups of T rows (T tokens sharing
    a_cat = a.repeat(k, 1).contiguous()  # the same k experts is the decode
    #                                  shape; rows are contiguous per group)
    t_packed = [torch.from_numpy(packed[layer]) for layer in range(L)]
    t_scales = [torch.from_numpy(scales[layer]) for layer in range(L)]

    per_call_bytes = k * N * (K // 2 + (K // 64) * 4 if args.fmt == "nf4"
                              else K // 2 + K // 32)
    results = []
    for th in [int(x) for x in args.threads.split(",")]:
        if args.pool:
            cg.pool_stop()
            got = cg.pool_start(th)
            assert got == th, f"pool_start({th}) -> {got}"
        times = []
        for tok in range(args.tokens + args.warmup):
            t0 = time.perf_counter()
            for layer in range(L):
                eids = trace[tok, layer].tolist()
                fn(a_cat, t_packed[layer], t_scales[layer], sizes, eids,
                   threads=th)
            dt = time.perf_counter() - t0
            if tok >= args.warmup:
                times.append(dt)
        times.sort()
        med = times[len(times) // 2]
        gbs = per_call_bytes * L / med / 1e9
        results.append({"threads": th, "gbs": round(gbs, 1),
                        "ms_per_token_layerset": round(med * 1e3, 2)})
        print(f"threads {th:3d}: {gbs:7.1f} GB/s "
              f"({med * 1e3:.2f} ms per {L}-layer token)", flush=True)

    if args.pool:
        cg.pool_stop()
    best = max(r["gbs"] for r in results)
    out = {
        "fmt": args.fmt, "shape": args.shape, "N": N, "K": K,
        "dispatch": "pool" if args.pool else "openmp",
        "arena_pages": HUGETLB_STATE["mode"],
        "layers": L, "experts": E, "topk": k, "rows": args.rows,
        "arena_gib": round(arena_bytes / 2**30, 2),
        "per_call_mib": round(per_call_bytes / 2**20, 2),
        "features": feats,
        "thp": open("/sys/kernel/mm/transparent_hugepage/enabled").read().strip()
        if os.path.exists("/sys/kernel/mm/transparent_hugepage/enabled") else "?",
        "sweep": results, "best_gbs": best,
        "seed": args.seed,
    }
    print(json.dumps({"best_gbs": best, "features": feats}))
    if args.out:
        json.dump(out, open(args.out, "w"), indent=2)
    del lib, keep


if __name__ == "__main__":
    main()
