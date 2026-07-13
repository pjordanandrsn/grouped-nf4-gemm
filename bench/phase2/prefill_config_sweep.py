#!/usr/bin/env python3
"""Prefill (M-tile) config sweep for the fused kernel.

v1 measured the M-tile path at 0.22-0.85x the dequant baseline at prefill and
v1.1 proved the gap is NOT a dtype problem (bf16-MMA regressed it). Before any
mainloop rewrite, this sweeps the free knobs — BLOCK_M x BLOCK_N x num_warps x
num_stages (BLOCK_K stays 64: one absmax scalar per (n, k-step)) — on REAL
bnb-quantized census stacks under the uniform prefill regime, with the dequant
baseline timed per cell for context. If a config-only win exists, it's
adopted; if the ceiling stays below parity, that's the honest answer and the
mainloop rewrite stays a separate project.

Usage: python bench/phase2/prefill_config_sweep.py --out sweep.json
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "kernel"))
sys.path.insert(0, str(REPO / "bench" / "phase1"))

import triton  # noqa: E402
from nf4_grouped import (  # noqa: E402
    BLOCKSIZE,
    _gemm_nf4_grouped,
    _lut,
    build_group_tiles,
)

CONFIGS = [
    (bm, bn, w, s)
    for bm in (16, 32, 64, 128)
    for bn in (32, 64, 128)
    for w in (4, 8)
    for s in (2, 3, 4)
]


def time_launch(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b))
    return statistics.median(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = "cuda"

    from harness import BACKENDS, QuantStack, census_specs, make_activations, time_backend

    specs = census_specs(REPO / "census" / "shape_census.json", None)
    p = torch.cuda.get_device_properties(0)
    out = {"env": {"gpu": p.name, "sm_count": p.multi_processor_count,
                   "torch": torch.__version__},
           "regime": "prefill_s2048 (uniform)", "cells": []}

    for spec in specs:
        stack = QuantStack(spec, dev)
        groups = make_activations(spec, "prefill_s2048", dev)
        base_ms = time_backend(BACKENDS["dequant_grouped"], stack, groups, 20, dev)
        cur_ms = time_backend(BACKENDS["fused_nf4"], stack, groups, 20, dev)

        B, A = stack.fusedpack()
        a_cat = torch.cat([a for _, a in groups])
        sizes = [a.shape[0] for _, a in groups]
        eids = torch.tensor([e for e, _ in groups], dtype=torch.int32, device=dev)
        T = a_cat.shape[0]
        o = torch.empty(T, spec.N, dtype=torch.bfloat16, device=dev)
        lut = _lut(dev)

        rows = []
        for bm, bn, w, s in CONFIGS:
            r = {"block_m": bm, "block_n": bn, "warps": w, "stages": s}
            try:
                t_row0, t_rows, t_group = build_group_tiles(sizes, bm, dev)
                grid = (t_row0.numel(), triton.cdiv(spec.N, bn))

                def fn():
                    _gemm_nf4_grouped[grid](
                        a_cat, B, A, o, lut, t_row0, t_rows, t_group, eids,
                        spec.K, spec.N,
                        B.stride(0), B.stride(1), A.stride(0), A.stride(1),
                        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=BLOCKSIZE,
                        num_warps=w, num_stages=s,
                    )
                r["ms"] = time_launch(fn)
                r["status"] = "ok"
            except Exception as e:
                r.update({"status": "failed", "reason": str(e)[:100]})
            rows.append(r)
        ok = [r for r in rows if r["status"] == "ok"]
        best = min(ok, key=lambda r: r["ms"]) if ok else None
        cell = {
            "model": spec.model, "proj": spec.proj,
            "N": spec.N, "K": spec.K, "E": spec.E, "top_k": spec.top_k,
            "tokens": T, "dequant_ms": base_ms, "current_default_ms": cur_ms,
            "configs": rows, "oracle": best,
            "oracle_vs_dequant": (base_ms / best["ms"]) if best else None,
            "oracle_vs_current": (cur_ms / best["ms"]) if best else None,
        }
        out["cells"].append(cell)
        print(f"[{spec.model[:24]:<24} {spec.proj:<8}] dequant {base_ms:7.2f} "
              f"current {cur_ms:7.2f} oracle {best['ms']:7.2f} "
              f"({best['block_m']}/{best['block_n']}/w{best['warps']}/s{best['stages']}) "
              f"-> vs-deq {cell['oracle_vs_dequant']:.2f}x  vs-cur {cell['oracle_vs_current']:.2f}x",
              flush=True)
        del stack
        torch.cuda.empty_cache()

    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
