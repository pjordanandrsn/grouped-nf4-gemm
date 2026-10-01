#!/usr/bin/env python3
"""k19_probe.py -- a $0 first signal for the B=16 expert-matmul lane (P86's next step), on any CUDA card.

Recorded B=16 routing (P60's eids_b16.int16.bin: [128 steps, 48 layers, 16 rows, 8]) through Qwen3-30B-A3B's expert
shapes (gate_up N=1536 K=2048; down N=2048 K=768) on synthetic int4-b32 weights, two arms per decode step, each a
CUDA graph:
  served     P60's served arm: gnf4's _gemv_int4_b32, R = 128 routed rows per call, split-K + separate reduce
  k16_each   for every DISTINCT expert of the call, one gemm_int4_b32_smallm (K16: bf16 tensor-core MMA, in-register
             int4 dequant) over that expert's 1-16 rows -- the arithmetic a grouped tensor-core kernel would do. Each
             call is its own small launch here (a grouped kernel would be one), so its summed kernel time is an UPPER
             bound for such a kernel's.
Reads per arm: graph-replay step time (median) and per-kernel GPU time per step (torch.profiler over replays).
Also checks k16_each against served on the rows: max |diff| / max |served| (different arithmetic -- int8 GEMV vs bf16
MMA -- so close, not bitwise).
"""
import argparse
import json
import statistics
import sys

import numpy as np
import torch

sys.path.insert(0, "/root/k19")
import replay_gemv as rg  # noqa: E402  (P60's harness: build, make_bufs, step_fn, time_graph, profile_graph)


def k16_step_fn(W, eids_layers, x_bf16, ws):
    from int4_smallm import gemm_int4_b32_smallm
    plans = []
    for eids in eids_layers:                       # precompute (expert, row index tensor) groups per layer, off-capture
        e = eids.cpu()
        groups = []
        for ex in torch.unique(e).tolist():
            rows = torch.nonzero(e == ex).flatten()
            for i in range(0, rows.numel(), 16):   # an expert with > 16 rows of 128 would split (never seen at B=16, k=8)
                groups.append((ex, rows[i:i + 16].to(eids.device)))
        plans.append(groups)

    def fn():
        for groups in plans:
            for name in ("gate_up", "down"):
                w = W[name]
                for ex, rows in groups:
                    gemm_int4_b32_smallm(x_bf16[name].index_select(0, rows), w["packed"][ex], w["scales"][ex],
                                         workspace=ws[name])
    return fn, sum(len(g) for g in plans)


def k19_step_fn(W, eids_layers, x_bf16):
    """K19: per layer, ONE tile-table build (build_group_tiles_fused) and one grouped launch per projection --
    gate_up gathers the unsorted rows in-kernel; down reads rows already in expert-major order."""
    from int4_b32 import build_group_tiles_fused
    from int4_smallm import gemm_int4_b32_grouped_smallm
    def fn():
        for eids in eids_layers:
            row0, rows, grp, order, _counts = build_group_tiles_fused(eids, 128, 16)
            wg, wd = W["gate_up"], W["down"]
            gemm_int4_b32_grouped_smallm(x_bf16["gate_up"], wg["packed"], wg["scales"], row0, rows, grp, order)
            gemm_int4_b32_grouped_smallm(x_bf16["down"], wd["packed"], wd["scales"], row0, rows, grp, None)
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eids", required=True)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda")
    eids_all = torch.from_numpy(np.fromfile(a.eids, dtype="<i2").reshape(128, 48, 16, 8).astype(np.int16))
    W = rg.build(128, dev)
    bufs = rg.make_bufs(W, dev)
    from int4_smallm import plan_smallm, smallm_workspace
    ws, x_bf16 = {}, {}
    g = torch.Generator(device="cpu").manual_seed(1)
    for name, w in W.items():
        bn, kc, sk = plan_smallm(w["N"], w["K"])
        ws[name] = smallm_workspace(w["N"], block_n=bn, sk=sk, device=dev)
        x_bf16[name] = (torch.randn(128, w["K"], generator=g) / 4).to(torch.bfloat16).to(dev)
    rec = {"device": torch.cuda.get_device_name(), "sm": torch.cuda.get_device_properties(dev).multi_processor_count,
           "torch": torch.__version__, "steps": a.steps, "arms": {}}
    for arm in ("served", "k16_each", "k19"):
        per_step, kern = [], {}
        for s in range(a.steps):
            el = rg.arm_eids(eids_all[s], "served", dev)
            if arm == "served":
                fn = rg.step_fn(W, el, "served", bufs)
                n_calls = 2 * len(el)
            elif arm == "k16_each":
                fn, n_groups = k16_step_fn(W, el, x_bf16, ws)
                n_calls = 2 * n_groups
            else:
                fn = k19_step_fn(W, el, x_bf16)
                n_calls = 3 * len(el)
            graph, med, _ = rg.time_graph(fn, a.iters)
            per_step.append(med)
            for k, ms in rg.profile_graph(graph, 2).items():
                kern[k] = kern.get(k, 0.0) + ms / a.steps
            del graph
            torch.cuda.empty_cache()
        rec["arms"][arm] = {"step_ms_median": statistics.median(per_step), "step_ms": per_step, "launches_per_step": n_calls,
                            "kernel_ms_per_step": dict(sorted(kern.items(), key=lambda kv: -kv[1])[:8]),
                            "kernel_total_ms": sum(kern.values())}
        print(arm, json.dumps({k: rec["arms"][arm][k] for k in ("step_ms_median", "kernel_total_ms", "launches_per_step")}), flush=True)
    # numerics on one call: served vs k16_each rows (layer 0, gate_up, step 0)
    from int4_b32 import gemv_int4_b32, quant_x_rows
    from int4_smallm import gemm_int4_b32_smallm
    w = W["gate_up"]
    e = eids_all[0, 0].reshape(-1).to(torch.int32).to(dev)
    xq, xs = quant_x_rows(x_bf16["gate_up"])
    ref = gemv_int4_b32(xq, xs, w["packed"], w["scales"], e, w["N"], w["K"]).float()
    got = torch.empty_like(ref)
    for ex in torch.unique(e).tolist():
        rows = torch.nonzero(e == ex).flatten()
        got[rows] = gemm_int4_b32_smallm(x_bf16["gate_up"][rows], w["packed"][ex], w["scales"][ex]).float()
    rec["numerics"] = {"max_abs_diff_over_max_ref": float((got - ref).abs().max() / ref.abs().max())}
    print("numerics", rec["numerics"], flush=True)
    json.dump(rec, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
