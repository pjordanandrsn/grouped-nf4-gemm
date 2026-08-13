#!/usr/bin/env python3
"""Mechanical reducer for the dequant-on-forward training leg.

Reads the FROZEN prereg and the per-device receipts and adjudicates. The agent
that ran the experiment does not decide whether it passed; this file does, from
criteria written down before the data existed.

    python3 bench/phase1/reduce_dequant_forward.py \
        --prereg kernel/prereg_dequant_forward.json \
        --receipt H100=.../dequant_forward_h100.json \
        --receipt ADA=.../dequant_forward_ada.json \
        --out .../verdicts.json

Exit code is NOT the verdict. Read the verdicts JSON: a pipe swallows `$?`,
which has already cost this repo one misread adjudication.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _in(x, band):
    return x is not None and band[0] <= x <= band[1]


def _cellid(r):
    return f"{r['model'].split('/')[-1]}/{r['proj']}"


def grade_device(dev, rec, pre):
    p = pre["frozen_verdict_criteria_params"]
    rows = [r for r in rec["rows"] if r.get("status") == "ok"]
    skipped = [r for r in rec["rows"] if r.get("status") != "ok"]

    # ---- Q1/Q3: per-cell instrument gates. A void cell has no ratio. -------
    for r in rows:
        why = []
        if not _in(r.get("g_selfpair"), p["selfpair_band"]):
            why.append(f"g_selfpair={r.get('g_selfpair'):.4f}")
        if not _in(r.get("d_selfpair"), p["selfpair_band"]):
            why.append(f"d_selfpair={r.get('d_selfpair'):.4f}")
        if not _in(r.get("g_drift"), p["drift_band"]):
            why.append(f"g_drift={r.get('g_drift'):.4f}")
        r["_void"] = bool(why)
        r["_void_why"] = "; ".join(why)

    live = [r for r in rows if not r["_void"]]
    void = [r for r in rows if r["_void"]]
    void_frac = (len(void) / len(rows)) if rows else 1.0
    q1 = void_frac <= p["void_cell_fraction_that_voids_the_leg"]

    # ---- Q2: wiring, positive-controlled ----------------------------------
    q2_fail = []
    for r in rows:
        g = r.get("gate", {})
        if not g.get("deq_calls_ok"):
            q2_fail.append(f"{_cellid(r)}/{r['regime']}: deq_calls "
                           f"{g.get('deq_calls_D')} < {g.get('nonempty_groups')}")
        if not g.get("routed_matches_D"):
            q2_fail.append(f"{_cellid(r)}/{r['regime']}: routed probe differs on "
                           f"{g.get('routed_rows_differing')} rows")
        for arm in ("G", "D", "D_routed", "U"):
            gg = g.get(f"grad_{arm}")
            if gg is None:
                continue
            for name, st in gg.items():
                if st is None or not st.get("finite"):
                    q2_fail.append(f"{_cellid(r)}/{r['regime']}: {arm}.{name} "
                                   f"grad {st}")
                # lora_A is exactly zero at LoRA init (B is zero-init) and is
                # checked by its own positive control below, not here.
                elif name != "lora_A" and not st.get("nonzero"):
                    q2_fail.append(f"{_cellid(r)}/{r['regime']}: {arm}.{name} "
                                   f"grad {st}")
            ctl = g.get(f"gradA_at_nonzero_B_{arm}")
            if ctl is None or not (ctl.get("present") and ctl.get("finite")
                                   and ctl.get("nonzero")):
                q2_fail.append(f"{_cellid(r)}/{r['regime']}: {arm} lora_A "
                               f"positive control {ctl}")
    q2 = not q2_fail

    def by_regime(regime, key):
        return [(r, r[key]) for r in live
                if r["regime"] == regime and r.get(key) is not None]

    # ---- S1: speed at the small-batch regime ------------------------------
    s1_cells = by_regime(p["s1_regime"], "d_over_g")
    s1_pass_n = sum(1 for _, v in s1_cells if v >= p["s1_bar"])
    s1_med = statistics.median([v for _, v in s1_cells]) if s1_cells else None
    s1 = (len(s1_cells) > 0 and s1_pass_n >= p["s1_min_cells"])
    s1_band = _in(s1_med, p["s1_predicted_median_band"])

    # ---- M1: transient memory at the large token budget -------------------
    m1_cells = by_regime(p["m1_regime"], "mem_transient_d_over_g")
    m1_pass_n = sum(1 for _, v in m1_cells if v > p["m1_bar"])
    m1_med = statistics.median([v for _, v in m1_cells]) if m1_cells else None
    m1 = (len(m1_cells) > 0 and m1_pass_n >= p["m1_min_cells"])
    m1_band = _in(m1_med, p["m1_predicted_median_band"])

    # ---- F1: fidelity gate, over ALL cells including void ones ------------
    # A void cell's TIMING is not a measurement; its arithmetic still is.
    f1_cells = [(r, r["b_rel_G_over_D"]) for r in rows
                if r.get("b_rel_G_over_D") is not None]
    f1_fail = [(_cellid(r), r["regime"], v) for r, v in f1_cells
               if not v <= p["f1_bar"]]
    f1_med = statistics.median([v for _, v in f1_cells]) if f1_cells else None
    f1 = (len(f1_cells) > 0 and not f1_fail)
    f1_band = _in(f1_med, p["f1_predicted_ratio_band"])

    # ---- S2: the M-axis direction (report-only prediction) ----------------
    order = rec.get("regimes", [])
    s2_meds = []
    for regime in order:
        vals = [v for _, v in by_regime(regime, "d_over_g")]
        s2_meds.append({"regime": regime,
                        "median_d_over_g": statistics.median(vals) if vals else None,
                        "n": len(vals)})
    seq = [m["median_d_over_g"] for m in s2_meds if m["median_d_over_g"] is not None]
    s2_monotone_decay = all(a >= b for a, b in zip(seq, seq[1:])) if len(seq) > 1 else None

    def med_of(key):
        vals = [r[key] for r in live if r.get(key) is not None]
        return statistics.median(vals) if vals else None

    return {
        "device": dev,
        "gpu": rec.get("gpu"), "capability": rec.get("capability"),
        "cells_ok": len(rows), "cells_skipped": len(skipped),
        "skipped": [{"cell": _cellid(r), "regime": r["regime"],
                     "reason": r.get("reason")} for r in skipped],
        "void_cells": [{"cell": _cellid(r), "regime": r["regime"],
                        "why": r["_void_why"]} for r in void],
        "void_fraction": void_frac,
        "Q1_self_pair": q1,
        "Q2_wiring": q2, "Q2_failures": q2_fail,
        "Q3_drift": "folded into Q1 per-cell voiding",
        "S1_speed_small_batch": s1,
        "S1_detail": {"regime": p["s1_regime"], "cells_at_or_above_bar": s1_pass_n,
                      "of": len(s1_cells), "median": s1_med,
                      "predicted_band": p["s1_predicted_median_band"],
                      "median_in_predicted_band": s1_band,
                      "per_cell": [{"cell": _cellid(r), "d_over_g": v}
                                   for r, v in s1_cells]},
        "M1_memory": m1,
        "M1_detail": {"regime": p["m1_regime"], "cells_above_bar": m1_pass_n,
                      "of": len(m1_cells), "median": m1_med,
                      "predicted_band": p["m1_predicted_median_band"],
                      "median_in_predicted_band": m1_band,
                      "per_cell": [{"cell": _cellid(r), "mem_ratio": v}
                                   for r, v in m1_cells]},
        "F1_fidelity": f1,
        "F1_detail": {"median_b_rel_G_over_D": f1_med, "failures": f1_fail,
                      "predicted_band": p["f1_predicted_ratio_band"],
                      "median_in_predicted_band": f1_band},
        "S2_M_axis_report_only": {"per_regime": s2_meds,
                                  "predicted_monotone_decay": True,
                                  "observed_monotone_decay": s2_monotone_decay},
        "E1_energy_report_only": {"median_j_ratio_d_over_g": med_of("j_ratio_d_over_g")},
        "P1_plumbing_report_only": {"median_dr_over_d": med_of("dr_over_d"),
                                    "median_dr_over_g": med_of("dr_over_g")},
        "U1_unsloth_report_only": {"median_u_over_g": med_of("u_over_g"),
                                   "note": "separate comparator; never divided "
                                           "into the dequant_forward result"},
        "shared_floor_report_only": {
            "median_lora_floor_frac_of_g": med_of("lora_floor_frac_of_g"),
            "note": "identical in both arms; compresses every ratio toward 1.0"},
        "DEVICE_CONFIRMED": bool(q1 and q2 and s1 and m1 and f1),
    }


def markdown(verdicts, pre):
    L = []
    L.append("| device | cell | regime | d/g | self-pair G | self-pair D | "
             "mem D/G | b_rel G/D | J D/G | dr/d |")
    L.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for dev, rows in verdicts["_rows"].items():
        for r in rows:
            if r.get("status") != "ok":
                L.append(f"| {dev} | {_cellid(r)} | {r.get('regime')} | "
                         f"NOT-RUN | | | | | | |")
                continue
            v = r.get("_void")
            def f(k, fmt="{:.3f}"):
                x = r.get(k)
                return fmt.format(x) if isinstance(x, (int, float)) else "—"
            L.append(
                f"| {dev} | {_cellid(r)} | {r['regime']} | "
                f"{'VOID' if v else f('d_over_g')} | {f('g_selfpair')} | "
                f"{f('d_selfpair')} | {f('mem_transient_d_over_g', '{:.2f}')} | "
                f"{f('b_rel_G_over_D')} | {f('j_ratio_d_over_g')} | "
                f"{f('dr_over_d')} |")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prereg", required=True)
    ap.add_argument("--receipt", action="append", required=True,
                    help="LABEL=path/to/dequant_forward_*.json")
    ap.add_argument("--out", default="verdicts_dequant_forward.json")
    ap.add_argument("--md", default=None)
    args = ap.parse_args()

    pre = json.loads(Path(args.prereg).read_text())
    per_dev, rows_by_dev = {}, {}
    for spec in args.receipt:
        label, path = spec.split("=", 1)
        rec = json.loads(Path(path).read_text())
        per_dev[label] = grade_device(label, rec, pre)
        rows_by_dev[label] = rec["rows"]

    two_device = len(per_dev) >= 2
    confirmed = two_device and all(d["DEVICE_CONFIRMED"] for d in per_dev.values())
    out = {
        "prereg": args.prereg,
        "prereg_verdict_key": pre["frozen_verdict_criteria"]["verdict_key"],
        "devices": per_dev,
        "two_device_rule_met": two_device,
        "DQF_CONFIRMED": confirmed,
        "note": ("NOT CONFIRMED does not mean the numbers are wrong; it means "
                 "the registered criteria were not all met. Report at full "
                 "volume and narrow the claim."),
        "_rows": rows_by_dev,
    }
    Path(args.out).write_text(json.dumps(
        {k: v for k, v in out.items() if k != "_rows"}, indent=1, default=str))
    if args.md:
        Path(args.md).write_text(markdown(out, pre) + "\n")

    for dev, d in per_dev.items():
        print(f"[{dev}] {d['gpu']} sm_{d['capability']}  "
              f"S1={d['S1_speed_small_batch']} M1={d['M1_memory']} "
              f"F1={d['F1_fidelity']} Q1={d['Q1_self_pair']} Q2={d['Q2_wiring']} "
              f"=> {d['DEVICE_CONFIRMED']}")
        print(f"      S1 median {d['S1_detail']['median']} "
              f"({d['S1_detail']['cells_at_or_above_bar']}/{d['S1_detail']['of']} "
              f"at bar, band {d['S1_detail']['predicted_band']})")
        print(f"      S2 M-axis {[ (m['regime'], m['median_d_over_g']) for m in d['S2_M_axis_report_only']['per_regime'] ]}"
              f"  decay={d['S2_M_axis_report_only']['observed_monotone_decay']}")
        if d["void_cells"]:
            print(f"      VOID: {d['void_cells']}")
    print(f"TWO_DEVICE={two_device}  DQF_CONFIRMED={confirmed}")
    print("REDUCE_DONE — the verdict is the JSON, not this exit code")


if __name__ == "__main__":
    main()
