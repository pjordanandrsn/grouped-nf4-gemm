#!/usr/bin/env python3
"""Grade `kernel/prereg_fused_roofline.json` from two R-scan receipts.

Written and committed BEFORE the discriminating card's data existed, for the
same reason the protocol was: a reducer authored after the numbers are in hand
is a reducer that can be talked into the answer you wanted.

Weight bytes are analytic, not measured: E experts x N x K params, NF4 at
0.5 B/param and bf16 at 2 B/param. Per replay the fused arm streams the packed
stack twice (forward, dgrad); the baseline's dequant reads it once and writes
the bf16 stack once, and its GEMMs read that bf16 stack twice. Achieved
bandwidth is those bytes over the profiler's per-family device self-time.

Usage:
  reduce_fused_roofline.py <h100_rscan.json> <discriminator_rscan.json>
"""
from __future__ import annotations

import json
import statistics as st
import sys

# (N, K) per projection, OLMoE-1B-7B census; E is read from the receipt path's
# scan rows via T_real/R so the reducer cannot silently assume the wrong stack.
SHAPES = {"gate_up": (2048, 2048), "down": (2048, 1024)}
PEAK_GBS = {"NVIDIA H100 80GB HBM3": 3350.0,
            "NVIDIA GeForce RTX 4090": 1008.0}
BANDS = {"issue_lo": 140.0, "issue_hi": 340.0, "bw_hi": 80.0,
         "flat_tol": 0.10, "control_lo": 0.20, "control_hi": 0.60}


def cells(receipt):
    """Achieved GB/s per profiled cell, per kernel family."""
    out = []
    for r in receipt["scan"]:
        if "profile_fused" not in r or "profile_base" not in r:
            continue
        N, K = SHAPES[r["proj"]]
        E = r["T_real"] // r["R"]
        params = E * N * K
        nf4_mb, bf16_mb = params * 0.5 / 1e6, params * 2.0 / 1e6
        f = r["profile_fused"]["families_per_replay"]
        b = r["profile_base"]["families_per_replay"]
        row = {"proj": r["proj"], "R": r["R"], "E": E,
               "d_over_g": r.get("d_over_g")}
        row["fused_gbs"] = (2 * nf4_mb) / f["fused_triton"]["self_us"] * 1e3
        row["dequant_gbs"] = (nf4_mb + bf16_mb) / b["dequant"]["self_us"] * 1e3
        row["cublas_gbs"] = (2 * bf16_mb) / b["gemm"]["self_us"] * 1e3
        row["traffic_adv"] = ((nf4_mb + bf16_mb) + 2 * bf16_mb) / (2 * nf4_mb)
        out.append(row)
    return out


def main() -> int:
    ref = json.load(open(sys.argv[1]))          # H100 (already public)
    new = json.load(open(sys.argv[2]))          # discriminator card
    ref_c, new_c = cells(ref), cells(new)
    if not ref_c or not new_c:
        print("REFUSAL: a receipt carries no profiled cells")
        return 1
    peak = PEAK_GBS.get(new["gpu"])
    report = {"ref_gpu": ref["gpu"], "new_gpu": new["gpu"], "cells": new_c,
              "verdicts": {}}

    for tag, rows, gpu in (("ref", ref_c, ref["gpu"]), ("new", new_c, new["gpu"])):
        print(f"\n=== {gpu} ===")
        for c in rows:
            frac = f"{c['fused_gbs'] / PEAK_GBS[gpu] * 100:.1f}%" \
                if gpu in PEAK_GBS else "?"
            print(f"  {c['proj']:8} R={c['R']:<4} fused {c['fused_gbs']:7.1f} GB/s "
                  f"({frac} of peak)  dequant {c['dequant_gbs']:7.1f}  "
                  f"cublas {c['cublas_gbs']:7.1f}")

    # --- P3 FIRST: the positive control gates whether P1 may be read at all.
    ref_deq = st.median(c["dequant_gbs"] for c in ref_c)
    new_deq = st.median(c["dequant_gbs"] for c in new_c)
    ratio = new_deq / ref_deq
    p3 = BANDS["control_lo"] <= ratio <= BANDS["control_hi"]
    report["verdicts"]["P3_control_dequant_tracks_bandwidth"] = {
        "ref_median_gbs": ref_deq, "new_median_gbs": new_deq,
        "ratio": ratio, "band": [BANDS["control_lo"], BANDS["control_hi"]],
        "pass": p3}

    # --- P2: R-flatness per projection on the new card. A projection that
    # fails is VOID for P1 (registered falsifier), so its cells are EXCLUDED
    # from P1's median rather than merely annotated — a note beside a verdict
    # is still a published verdict (Bugbot, PR #96).
    p2_all, live_projs = True, []
    for proj in sorted({c["proj"] for c in new_c}):
        vals = {c["R"]: c["fused_gbs"] for c in new_c if c["proj"] == proj}
        if len(vals) < 2:
            report["verdicts"][f"P2_flat_{proj}"] = {
                "refusal": "fewer than two profiled R values"}
            p2_all = False
            continue
        lo, hi = min(vals.values()), max(vals.values())
        spread = (hi - lo) / lo
        ok = spread <= BANDS["flat_tol"]
        report["verdicts"][f"P2_flat_{proj}"] = {
            "by_R": vals, "spread": spread, "pass": ok}
        p2_all &= ok
        if ok:
            live_projs.append(proj)

    # --- P1: the discriminator, read ONLY if the control held AND only over
    # projections that survived P2.
    graded = [c for c in new_c if c["proj"] in live_projs]
    med = st.median(c["fused_gbs"] for c in graded) if graded else float("nan")
    if not graded:
        p1 = {"verdict": "VOID — every projection failed P2; nothing to read",
              "excluded_projs": sorted({c["proj"] for c in new_c})}
    elif not p3:
        p1 = {"median_gbs": med,
              "verdict": "UNINTERPRETABLE — the positive control failed"}
    elif BANDS["issue_lo"] <= med <= BANDS["issue_hi"]:
        p1 = {"median_gbs": med, "verdict": "H_ISSUE CONFIRMED"}
    elif med <= BANDS["bw_hi"]:
        p1 = {"median_gbs": med, "verdict": "H_BW CONFIRMED"}
    else:
        p1 = {"median_gbs": med,
              "verdict": "UNRESOLVED — falsifies both hypotheses as registered"}
    if not p2_all and graded:
        p1["excluded_projs"] = sorted(
            {c["proj"] for c in new_c} - set(live_projs))
        p1["graded_over"] = live_projs
    report["verdicts"]["P1_binding_constraint"] = p1

    print("\n=== VERDICTS ===")
    for k, v in report["verdicts"].items():
        print(f"{k}: {json.dumps(v, default=str)}")
    if peak:
        print(f"\nfused median {med:.1f} GB/s = {med / peak * 100:.1f}% of "
              f"{new['gpu']} peak; on {ref['gpu']} it was "
              f"{st.median(c['fused_gbs'] for c in ref_c):.1f} GB/s")
    json.dump(report, open("roofline_reduced.json", "w"), indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
