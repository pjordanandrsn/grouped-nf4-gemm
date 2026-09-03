# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""B=16 expert-path microbench: is the shipped decode GEMV anywhere near
the bandwidth floor at the batched shape, and which kernel/config gets
closest?

Same-box head-to-head (receipts INT4B16/P20-H2H) put our B=16 step at
13.2 ms against vLLM's 4.68 ms, and the census attributes 6.4 ms of ours
to ``_gemv_int4_b32`` alone -- ~96 launches/step, ~66 us each. The
unique expert bytes that step must read are ~215 MB (81 distinct experts
x 2.65 MB), a 0.12 ms floor at 1.79 TB/s. Being ~34x off that floor says
the kernel is NOT bandwidth bound at R=128; ``_plan`` was swept at R=1
(the qkv census shapes), where split-K=8 fills the grid, and is applied
unchanged at R=128 where it multiplies 1,536 programs to 12,288 and
writes 8x fp32 partials that a second launch then reduces.

Arms, all timed under CUDA-graph replay (the registered basis):
  ship    the wrapper exactly as the serving path calls it
  sweep   ``_gemv_int4_b32`` direct, over SK x BLOCK_N x KU x warps
  mtile   ``build_group_tiles_fused`` + ``gemm_int4_b32_grouped_captured``
          (the tensor-core path P7 measured as losing at ~1-2 rows/expert)
Timing only; outputs are discarded. A K8/parity cycle carries any
routing change this motivates.
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "kernel"))
import triton  # noqa: E402
from int4_b32 import (_gemv_int4_b32, _plan, build_group_tiles_fused,  # noqa: E402
                      gemm_int4_b32_grouped_captured, gemv_int4_b32,
                      quant_x_rows)

# Qwen3-30B-A3B expert cells: gate_up is the fused [2I, H]; down is [H, I]
CELLS = {"gate_up": (1536, 2048), "down": (2048, 768)}
E, TOPK, B = 128, 8, 16


def _workload(N, K, seed=0, device="cuda"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    packed = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8,
                           generator=g).to(device)
    scales = ((torch.rand(E, N, K // 32, generator=g) + 0.5) * 0.01
              ).to(device=device, dtype=torch.float16)
    # B tokens x top-8 routing -> R = 128 rows in INPUT order, expert ids
    # drawn uniformly: ~81 distinct experts in expectation, the same
    # multiplicity structure the serving path sees at B=16
    eids = torch.randint(0, E, (B * TOPK,), generator=g,
                         dtype=torch.int32).to(device)
    x = torch.randn(B * TOPK, K, generator=g).to(device=device,
                                                 dtype=torch.bfloat16)
    xq, xs = quant_x_rows(x)
    return packed, scales, eids, xq, xs


def _graph_us(fn, warmup=20, inner=16, replays=4, chunks=8):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(inner):
            fn()
    torch.cuda.synchronize()
    g.replay()
    torch.cuda.synchronize()
    spans = []
    for _ in range(chunks):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(replays):
            g.replay()
        e1.record()
        e1.synchronize()
        spans.append(e0.elapsed_time(e1) * 1e3 / (replays * inner))
    spans.sort()
    return spans[len(spans) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    rep = {"gpu": props.name, "cells": {}}
    for cell, (N, K) in CELLS.items():
        packed, scales, eids, xq, xs = _workload(N, K)
        R = eids.numel()
        uniq = int(eids.unique().numel())
        per_expert_bytes = N * (K // 2) + N * (K // 32) * 2
        uniq_mb = uniq * per_expert_bytes / 1e6
        pair_mb = R * per_expert_bytes / 1e6
        out = {"N": N, "K": K, "R": R, "distinct_experts": uniq,
               "unique_MB": round(uniq_mb, 2), "per_pair_MB": round(pair_mb, 2),
               "floor_us_at_1790GBs": round(uniq_mb / 1790 * 1e3, 2)}
        print(f"=== {cell}: N={N} K={K} R={R} distinct={uniq} "
              f"unique {uniq_mb:.1f} MB  floor {out['floor_us_at_1790GBs']} us")

        # ---- ship: exactly as served (wrapper + its reduce) ----
        bn, wp, sk, ku = _plan(N, K)
        part = torch.empty(sk * R, N, dtype=torch.float32, device=dev)
        us = _graph_us(lambda: gemv_int4_b32(xq, xs, packed, scales, eids,
                                             N, K, part=part))
        out["ship"] = {"plan": [bn, wp, sk, ku], "us": round(us, 2),
                       "GBs_unique": round(uniq_mb / us * 1e3, 1)}
        print(f"  ship  plan bn{bn}/w{wp}/sk{sk}/ku{ku}: {us:8.2f} us  "
              f"{out['ship']['GBs_unique']:7.1f} GB/s (unique)")

        # ---- sweep the kernel directly ----
        kb = K // 32
        grid_sk = [1, 2, 4, 8] if not a.quick else [1, 8]
        grid_bn = [64, 128, 256] if not a.quick else [64, 128]
        best = None
        out["sweep"] = []
        for SK, BN, KU, W in itertools.product(grid_sk, grid_bn, [1, 2, 4],
                                               [4, 8]):
            if kb % KU:
                continue
            if SK > max(1, kb // KU):
                continue
            p = torch.empty(SK * R, N, dtype=torch.float32, device=dev)

            def one(SK=SK, BN=BN, KU=KU, W=W, p=p):
                _gemv_int4_b32[(triton.cdiv(N, BN), R, SK)](
                    xq, xs, packed, scales, eids, p, N, K=K, R=R,
                    BLOCK_N=BN, SK=SK, KU=KU, num_warps=W)
                if SK > 1:
                    p.reshape(SK, R, N).sum(0)
            try:
                us = _graph_us(one)
            except Exception as ex:      # a config the compiler refuses
                out["sweep"].append({"sk": SK, "bn": BN, "ku": KU, "w": W,
                                     "error": str(ex)[:80]})
                continue
            row = {"sk": SK, "bn": BN, "ku": KU, "w": W, "us": round(us, 2),
                   "GBs_unique": round(uniq_mb / us * 1e3, 1)}
            out["sweep"].append(row)
            if best is None or us < best["us"]:
                best = row
        out["sweep_best"] = best
        print(f"  sweep best sk{best['sk']}/bn{best['bn']}/ku{best['ku']}"
              f"/w{best['w']}: {best['us']:8.2f} us  "
              f"{best['GBs_unique']:7.1f} GB/s  "
              f"({out['ship']['us'] / best['us']:.2f}x over ship)")

        # ---- mtile: tensor-core grouped GEMM over device tiles ----
        # rows must be expert-major: sort once (the serving path builds
        # this on device); timing covers ONLY the GEMM, matching the
        # kernel-vs-kernel question, with the tile build cost noted
        order = torch.argsort(eids, stable=True)
        xq_s, xs_s = xq[order].contiguous(), xs[order].contiguous()
        eids_s = eids[order].contiguous()
        out["mtile"] = []
        for BM in (16, 32):
            try:
                t_row0, t_rows, t_group, _ord, _cnt = build_group_tiles_fused(
                    eids_s, E, BM)
            except Exception as ex:
                out["mtile"].append({"bm": BM, "error": str(ex)[:80]})
                continue
            for BN, W in ((64, 8), (128, 4), (128, 8)):
                def mt(BM=BM, BN=BN, W=W):
                    gemm_int4_b32_grouped_captured(
                        xq_s, xs_s, packed, scales, t_row0, t_rows,
                        t_group, block_m=BM, block_n=BN, warps=W)
                try:
                    us = _graph_us(mt)
                except Exception as ex:
                    out["mtile"].append({"bm": BM, "bn": BN, "w": W,
                                         "error": str(ex)[:80]})
                    continue
                row = {"bm": BM, "bn": BN, "w": W, "us": round(us, 2),
                       "GBs_unique": round(uniq_mb / us * 1e3, 1)}
                out["mtile"].append(row)
                print(f"  mtile bm{BM}/bn{BN}/w{W}: {us:8.2f} us  "
                      f"{row['GBs_unique']:7.1f} GB/s")
        # tile-build cost, eager (one launch; charged once per layer)
        t0 = time.perf_counter()
        for _ in range(50):
            build_group_tiles_fused(eids_s, E, 16)
        torch.cuda.synchronize()
        out["tile_build_us_eager"] = round(
            (time.perf_counter() - t0) / 50 * 1e6, 1)
        rep["cells"][cell] = out
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print("B16SWEEP_DONE", a.out)


if __name__ == "__main__":
    main()
