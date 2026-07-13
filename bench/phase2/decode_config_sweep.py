#!/usr/bin/env python3
"""Decode-config sweep for the cost-model follow-on.

The Gate-2 blind confirmatory showed the exact-(N,K) census dict does not
transfer: off-census shapes run the default config at parity, and two census
cells missed on a different A5000 instance. This sweep produces the data a
shape-general selector is fit on.

Part A (grid): synthetic stacks (timing is data-independent for a fixed LUT
gather) over an (N, K, T) grid x launch configs, timed with CUDA events.
The grid deliberately excludes the confirmatory-v2 validation shapes.

Part B (dev shapes): real bnb-quantized stacks (harness QuantStack) for the
8 census + 8 v1 held-out shapes: dequant-path baseline + every config for the
fused gemv, so each dev shape gets an oracle config and an oracle ratio.
v1 held-out shapes are development data now — they were burned as validation
the moment the confirmatory measured them; v2 gets a fresh set.

Usage (on the bench box):
  python bench/phase2/decode_config_sweep.py --out sweep.json [--part a|b|all]
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
from nf4_grouped import BLOCKSIZE, _gemv_nf4_grouped, _lut  # noqa: E402

GRID_N = [1024, 1536, 2048, 2816, 4096, 5760, 8192, 12800, 16384, 28672]
GRID_K = [704, 1024, 1536, 2048, 2880, 4096, 6400, 8192, 14336]
GRID_T = [1, 2, 4, 8]
CONFIGS = [(bn, w) for bn in (32, 64, 128, 256) for w in (2, 4, 8)] + [
    (512, 4),
    (512, 8),
]


def time_launch(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return statistics.median(times)


def gemv_launcher(a_cat, B, amax, out, eids, N, K, bn, warps):
    lut = _lut(a_cat.device)
    grid = (a_cat.shape[0], triton.cdiv(N, bn))

    def fn():
        _gemv_nf4_grouped[grid](
            a_cat, B, amax, out, lut, eids, K, N,
            B.stride(0), B.stride(1), amax.stride(0), amax.stride(1),
            BLOCK_N=bn, BLOCK_K=BLOCKSIZE, num_warps=warps, num_stages=3,
        )
    return fn


def sweep_cell(N, K, T, dev, synthetic_stack=None):
    """Time every config for one (N, K, T); returns rows + frees the stack."""
    if synthetic_stack is None:
        E = max(T, 1)
        B = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, device=dev)
        amax = torch.rand(E, N, K // BLOCKSIZE, dtype=torch.float32, device=dev) * 0.1
    else:
        B, amax = synthetic_stack
    a_cat = torch.randn(T, K, dtype=torch.bfloat16, device=dev)
    eids = (torch.arange(T, dtype=torch.int32, device=dev) % B.shape[0]).contiguous()
    out = torch.empty(T, N, dtype=torch.bfloat16, device=dev)
    rows = []
    for bn, warps in CONFIGS:
        row = {"N": N, "K": K, "T": T, "block_n": bn, "warps": warps}
        try:
            row["ms"] = time_launch(gemv_launcher(a_cat, B, amax, out, eids, N, K, bn, warps))
            row["status"] = "ok"
        except Exception as e:  # compile failure / OOM for a config is data
            row.update({"status": "failed", "reason": str(e)[:120]})
        rows.append(row)
    if synthetic_stack is None:
        del B, amax
    del a_cat, out
    torch.cuda.empty_cache()
    return rows


def part_a(dev):
    rows, t0 = [], time.time()
    cells = [(n, k, t) for n in GRID_N for k in GRID_K for t in GRID_T]
    for i, (N, K, T) in enumerate(cells):
        need = max(T, 1) * N * (K // 2) + T * K * 2 + T * N * 2
        if need > 18e9:
            rows.append({"N": N, "K": K, "T": T, "status": "skipped-mem"})
            continue
        rows += sweep_cell(N, K, T, dev)
        if i % 20 == 0:
            print(f"[grid {i+1}/{len(cells)}] {time.time()-t0:.0f}s", flush=True)
    return rows


def part_b(dev):
    from harness import (  # noqa: E402
        BACKENDS, GemmSpec, QuantStack, make_activations, time_backend,
    )
    from harness import census_specs  # noqa: E402

    specs = census_specs(REPO / "census" / "shape_census.json", None)
    for s in json.loads((REPO / "bench/phase1/heldout_shapes.json").read_text()):
        specs.append(GemmSpec(s["model"], s["proj"], s["N"], s["K"], s["E"], s["top_k"]))
    rows = []
    for spec in specs:
        stack = QuantStack(spec, dev)
        groups = make_activations(spec, "decode_bs1", dev)
        base_ms = time_backend(BACKENDS["dequant_grouped"], stack, groups, 50, dev)
        B, A = stack.fusedpack()
        a_cat = torch.cat([a for _, a in groups])
        eids = torch.tensor([e for e, _ in groups], dtype=torch.int32, device=dev)
        out = torch.empty(a_cat.shape[0], spec.N, dtype=torch.bfloat16, device=dev)
        cfgs = []
        for bn, warps in CONFIGS:
            r = {"block_n": bn, "warps": warps}
            try:
                r["ms"] = time_launch(
                    gemv_launcher(a_cat, B, A, out, eids, spec.N, spec.K, bn, warps))
                r["status"] = "ok"
            except Exception as e:
                r.update({"status": "failed", "reason": str(e)[:120]})
            cfgs.append(r)
        ok = [r for r in cfgs if r["status"] == "ok"]
        best = min(ok, key=lambda r: r["ms"]) if ok else None
        rows.append({
            "model": spec.model, "proj": spec.proj,
            "N": spec.N, "K": spec.K, "E": spec.E, "T": spec.top_k,
            "dequant_ms": base_ms, "configs": cfgs,
            "oracle": best, "oracle_ratio": (base_ms / best["ms"]) if best else None,
        })
        print(f"[dev] {spec.model[:30]:<30} {spec.proj:<8} oracle "
              f"{best['block_n']}/{best['warps']} ratio {rows[-1]['oracle_ratio']:.2f}x",
              flush=True)
        del stack
        torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--part", default="all", choices=["a", "b", "all"])
    args = ap.parse_args()
    dev = "cuda"
    p = torch.cuda.get_device_properties(0)
    env = {"gpu": p.name, "sm_count": p.multi_processor_count,
           "cc": f"{p.major}.{p.minor}", "torch": torch.__version__,
           "driver": None}
    try:
        import pynvml
        pynvml.nvmlInit()
        env["driver"] = pynvml.nvmlSystemGetDriverVersion()
    except Exception:
        pass
    out = {"env": env, "blocksize": BLOCKSIZE, "configs": CONFIGS}
    if args.part in ("a", "all"):
        out["grid"] = part_a(dev)
    if args.part in ("b", "all"):
        out["dev_shapes"] = part_b(dev)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
