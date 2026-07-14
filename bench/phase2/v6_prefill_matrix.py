# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).

"""v6 prefill mainloop EXPLORATORY: isolated-variant matrix (v1.1 lesson).

RESULTS-phase2-v1.1-negative.md registered that (a) the fused P-fid edge is
dtype-bought (fp32/TF32 inputs) and load-bearing, (b) a bundled bf16-MMA +
interleave + autotune change regressed everything un-attributably, and (c)
the prefill gap hypothesis is the per-element codebook gather + K-pipelining,
not the dot dtype. This sweep measures each candidate ISOLATED against the
v5 control on real bnb-quantized census stacks under prefill_s2048:

  V0/G1  control (shipped v5 mainloop)
  V1/G1  register-resident codebook via tl.gather (kills the [BN,BK] L1 LDG
         gather per K-step) — fidelity-preserving
  V0/G2  two quant groups per K-step (BLOCK_K=128; halves loop trips) —
         fidelity-preserving, K % 128 == 0 cells only
  V1/G2  both fidelity-preserving changes
  V3/G1  OPT-IN bf16 MMA retested ISOLATED at v4 tiles (v1.1 measured it only
         inside the bundle at 64-row tiles); P-fid cost recorded per cell
  V3/G2  bf16 MMA + two-group step

Per combo x config: median launch ms; per combo (cell's best config): rel
err vs fp64 alongside the dequant path's, so fidelity cost is a first-class
output. Exploratory only — no adjudication; the confirmatory bars get set
after this, before any confirmatory data.

Usage: python bench/phase2/v6_prefill_matrix.py --out v6_matrix.json
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
import triton.language as tl  # noqa: E402
from nf4_grouped import (  # noqa: E402
    BLOCKSIZE,
    _gemm_nf4_grouped,
    _lut,
    build_group_tiles,
)
from test_nf4_grouped import err_vs_fp64  # noqa: E402

COMBOS = [  # (variant, groups, label)
    (0, 1, "V0G1_control"),
    (1, 1, "V1G1_regLUT"),
    (0, 2, "V0G2_k128"),
    (1, 2, "V1G2_regLUT_k128"),
    (3, 1, "V3G1_bf16"),
    (3, 2, "V3G2_bf16_k128"),
]

CONFIGS = [
    (bm, bn, w, s)
    for bm in (64, 128)
    for bn in (64, 128)
    for w in (4, 8)
    for s in (3, 4)
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
    if not hasattr(tl, "gather"):
        print("NOTE: tl.gather unavailable — V1 combos will be skipped")

    from harness import BACKENDS, QuantStack, census_specs, make_activations, time_backend

    specs = census_specs(REPO / "census" / "shape_census.json", None)
    p = torch.cuda.get_device_properties(0)
    out = {"env": {"gpu": p.name, "sm_count": p.multi_processor_count,
                   "torch": torch.__version__, "triton": triton.__version__},
           "regime": "prefill_s2048 (uniform)", "combos": [c[2] for c in COMBOS],
           "cells": []}

    for spec in specs:
        stack = QuantStack(spec, dev)
        groups = make_activations(spec, "prefill_s2048", dev)
        base_ms = time_backend(BACKENDS["dequant_grouped"], stack, groups, 20, dev)
        cur_ms = time_backend(BACKENDS["fused_nf4"], stack, groups, 20, dev)

        B, A = stack.fusedpack()
        a_cat = torch.cat([a for _, a in groups])
        sizes = [a.shape[0] for _, a in groups]
        ids = [e for e, _ in groups]
        eids = torch.tensor(ids, dtype=torch.int32, device=dev)
        T = a_cat.shape[0]
        o = torch.empty(T, spec.N, dtype=torch.bfloat16, device=dev)
        lut = _lut(dev)

        # dequant path fidelity for context (same helper the suite uses)
        packed, states = stack.packed, stack.states
        from test_nf4_grouped import dequant_path_err
        e_deq = dequant_path_err(a_cat, sizes, ids, packed, states, B, A,
                                 spec.N, spec.K)

        cell = {"model": spec.model, "proj": spec.proj,
                "N": spec.N, "K": spec.K, "E": spec.E, "top_k": spec.top_k,
                "tokens": T, "dequant_ms": base_ms, "current_default_ms": cur_ms,
                "dequant_err": e_deq, "combos": {}}

        for variant, gpk, label in COMBOS:
            if variant == 1 and not hasattr(tl, "gather"):
                cell["combos"][label] = {"status": "skipped", "reason": "no tl.gather"}
                continue
            if gpk == 2 and spec.K % (BLOCKSIZE * 2) != 0:
                cell["combos"][label] = {"status": "skipped", "reason": "K % 128 != 0"}
                continue
            block_k = BLOCKSIZE * gpk
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
                            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=block_k,
                            GROUPS=gpk, VARIANT=variant,
                            num_warps=w, num_stages=s,
                        )
                    r["ms"] = time_launch(fn)
                    r["status"] = "ok"
                except Exception as e:
                    r.update({"status": "failed", "reason": str(e)[:100]})
                rows.append(r)
            ok = [r for r in rows if r["status"] == "ok"]
            best = min(ok, key=lambda r: r["ms"]) if ok else None
            entry = {"status": "ok" if best else "all-failed", "configs": rows,
                     "best": best}
            if best:
                # fidelity at the combo's best config (one extra launch)
                t_row0, t_rows, t_group = build_group_tiles(sizes, best["block_m"], dev)
                grid = (t_row0.numel(), triton.cdiv(spec.N, best["block_n"]))
                o.zero_()
                _gemm_nf4_grouped[grid](
                    a_cat, B, A, o, lut, t_row0, t_rows, t_group, eids,
                    spec.K, spec.N,
                    B.stride(0), B.stride(1), A.stride(0), A.stride(1),
                    BLOCK_M=best["block_m"], BLOCK_N=best["block_n"],
                    BLOCK_K=block_k, GROUPS=gpk, VARIANT=variant,
                    num_warps=best["warps"], num_stages=best["stages"],
                )
                torch.cuda.synchronize()
                entry["err"] = err_vs_fp64(o, a_cat, sizes, ids, B, A,
                                           spec.N, spec.K)
                entry["err_vs_dequant"] = entry["err"] / e_deq if e_deq else None
                entry["vs_dequant"] = base_ms / best["ms"]
                entry["vs_control_default"] = cur_ms / best["ms"]
            cell["combos"][label] = entry
            b = entry.get("best")
            print(f"[{spec.model[:20]:<20} {spec.proj:<8}] {label:<18} "
                  + (f"{b['ms']:7.2f} ms ({b['block_m']}/{b['block_n']}/w{b['warps']}/s{b['stages']}) "
                     f"vs-deq {entry['vs_dequant']:.2f}x err/deq {entry['err_vs_dequant']:.2f}"
                     if b else entry.get("reason", "failed")), flush=True)
        out["cells"].append(cell)
        del stack
        torch.cuda.empty_cache()

    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
