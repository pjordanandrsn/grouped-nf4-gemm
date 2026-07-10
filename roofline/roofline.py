#!/usr/bin/env python3
"""Roofline pre-registration (Phase 0.3): predicted ceiling per (model, proj, regime, device),
registered BEFORE any kernel exists.

Device parameters are explicit inputs (datasheet values, recorded verbatim below) — the
registered claims are the RATIOS in memory-bound cells, which depend on bytes moved and the
bandwidth, not on peak-FLOPs fine print. sm_120 rows carry a verify-at-Phase-4 flag.

Byte model per expert-group GEMM (M tokens, weight N x K):
  fused w4 kernel : W4 = N*K/2 (packed) + N*K/64*4 (fp32 absmax)  + acts 2*M*(K+N)
  bf16-resident   : 2*N*K                                          + acts 2*M*(K+N)
  dequant path    : N*K*0.5625 read + 2*N*K write + 2*N*K re-read  + acts  (two-pass lower bound)
FLOPs = 2*M*N*K. time = max(bytes/BW, flops/peak). Speedup = time_baseline / time_fused.
"""

import json
import sys
from pathlib import Path

DEVICES = [
    # name, BW GB/s, bf16 tensor dense TFLOPS (fp32 acc), source note
    ("RTX_A2000_12GB_sm86", 288.0, 31.9, "datasheet 288 GB/s; 63.9 TF sparse -> 31.9 dense (pro card, full-rate fp32 acc)"),
    ("RTX_3090_sm86", 936.0, 17.8, "datasheet 936 GB/s; GeForce GA102 halves fp32-acc: 35.6 fp16-acc dense -> 17.8 bf16/fp32-acc dense"),
    ("RTX_PRO6000_Blackwell_sm120", 1792.0, 125.0, "PRELIMINARY - verify at Phase 4 pod session; 96GB GDDR7 ~1.79 TB/s; bf16 dense placeholder"),
]

ABSMAX_BYTES_PER_WEIGHT = 4.0 / 64.0  # fp32 per 64-block


def cell(M, N, K, bw_gbs, peak_tflops):
    flops = 2.0 * M * N * K
    acts = 2.0 * M * (K + N)
    w4 = N * K * (0.5 + ABSMAX_BYTES_PER_WEIGHT) + acts
    bf16 = N * K * 2.0 + acts
    deq = N * K * (0.5 + ABSMAX_BYTES_PER_WEIGHT) + N * K * 4.0 + acts  # write bf16 + re-read
    bw = bw_gbs * 1e9
    peak = peak_tflops * 1e12

    def t(bytes_):
        return max(bytes_ / bw, flops / peak)

    return {
        "arith_intensity_fused": round(flops / w4, 2),
        "bound_fused": "memory" if w4 / bw > flops / peak else "compute",
        "ceiling_speedup_vs_bf16_resident": round(t(bf16) / t(w4), 2),
        "ceiling_speedup_vs_dequant_path": round(t(deq) / t(w4), 2),
        "ceiling_tflops_fused": round(min(peak, flops / (w4 / bw)) / 1e12, 2),
    }


def main(census_path):
    census = json.loads(Path(census_path).read_text())
    rows = []
    for m in census["models"]:
        E, k = m["experts"], m["top_k"]
        prefill_M = m["regimes"]["prefill_s2048"]["M_p50"]
        for proj, g in m["per_expert_gemms"].items():
            for regime, M, groups in [
                ("decode_bs1", 1, k),
                ("prefill_s2048", prefill_M, E),
                ("train_microbatch_fwd", prefill_M, E),
            ]:
                for dev, bw, tf, src in DEVICES:
                    c = cell(M, g["N"], g["K"], bw, tf)
                    rows.append(
                        {"model": m["model"].split("/")[-1], "proj": proj, "regime": regime,
                         "M_per_expert": M, "active_groups": groups, "N": g["N"], "K": g["K"],
                         "device": dev, **c}
                    )
    out = {
        "generated_by": "roofline/roofline.py",
        "device_params": [{"device": d, "bw_gbs": b, "bf16_dense_tflops": t, "source": s}
                          for d, b, t, s in DEVICES],
        "byte_model": "see module docstring; dequant baseline is a two-pass LOWER bound (cache-friendly reality can be faster, making our measured speedup SMALLER — conservative direction for claims)",
        "cells": rows,
    }
    p = Path(__file__).parent / "ceilings.json"
    p.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {p} ({len(rows)} cells)")
    # headline summary: decode bs1 on the two dev cards
    for r in rows:
        if r["regime"] == "decode_bs1" and r["device"].endswith("sm86") and r["proj"] == "gate_up":
            print(f"{r['model']:>18} {r['device']:<22} decode bs1: fused AI={r['arith_intensity_fused']:>5} "
                  f"[{r['bound_fused']}] ceil vs bf16={r['ceiling_speedup_vs_bf16_resident']}x "
                  f"vs dequant={r['ceiling_speedup_vs_dequant_path']}x")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent.parent / "census" / "shape_census.json")
