#!/usr/bin/env python3
"""Mechanical reducer for leg 3 (interleaved pairing).

    python3 bench/phase1/reduce_dequant_forward_interleaved.py \
        --prereg kernel/prereg_dequant_forward_interleaved.json \
        --receipt H100=.../interleaved_H100.json \
        --receipt ADA=.../interleaved_ADA.json \
        --out .../verdicts.json

Exit code is NOT the verdict. Read the verdicts JSON.
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

    for r in rows:
        why = []
        sp = (r.get("selfpair") or {}).get("ratio_median")
        hv = (r.get("primary") or {}).get("halves_ratio")
        if not _in(sp, p["selfpair_band"]):
            why.append(f"selfpair={sp}")
        if not _in(hv, p["halves_band"]):
            why.append(f"halves={hv}")
        r["_void"] = bool(why)
        r["_void_why"] = "; ".join(str(w) for w in why)

    live = [r for r in rows if not r["_void"]]
    void = [r for r in rows if r["_void"]]
    void_frac = (len(void) / len(rows)) if rows else 1.0
    q1 = void_frac <= p["void_cell_fraction_that_voids_the_leg"]

    q2_fail = []
    for r in rows:
        g = r.get("gate", {})
        if not g.get("deq_calls_ok"):
            q2_fail.append(f"{_cellid(r)}/{r['regime']}: deq_calls")
        if not g.get("base_arms_agree"):
            q2_fail.append(f"{_cellid(r)}/{r['regime']}: base arms differ")
        for arm in ("G_base", "D_base", "G_full", "D_full"):
            st_ = g.get(f"grad_{arm}")
            if not st_ or not st_.get("act_finite") or not st_.get("act_nonzero"):
                q2_fail.append(f"{_cellid(r)}/{r['regime']}: {arm} grad")
        if not g.get("gradA_at_nonzero_B"):
            q2_fail.append(f"{_cellid(r)}/{r['regime']}: lora_A control")
    q2 = not q2_fail

    f1_cells = [(r, r["dbase_over_gbase"]) for r in live
                if r["regime"] == p["f1_regime"] and r.get("dbase_over_gbase")]
    f1_n = sum(1 for _, v in f1_cells if v >= p["f1_bar"])
    f1_med = statistics.median([v for _, v in f1_cells]) if f1_cells else None
    f1 = (len(f1_cells) > 0 and f1_n >= p["f1_min_cells"])

    q4_cells = [(r, r["b_rel_G_over_D"]) for r in rows
                if r.get("b_rel_G_over_D") is not None]
    q4_fail = [(_cellid(r), r["regime"], v) for r, v in q4_cells
               if not v <= p["q4_fidelity_bar"]]
    q4 = (len(q4_cells) > 0 and not q4_fail)

    # ---- P1: what the pairing bought, from identical data -----------------
    p1 = []
    for r in rows:
        a, b = r.get("dbase_over_gbase"), r.get("dbase_over_gbase_blockstat")
        if a and b:
            p1.append({"cell": _cellid(r), "regime": r["regime"],
                       "interleaved": a, "block": b, "block_over_interleaved": b / a,
                       "void": r["_void"]})
    diverged = [x for x in p1 if not _in(x["block_over_interleaved"],
                                         p["p1_predicted_h100_agreement"])]

    def med(key, sub=None):
        vals = []
        for r in live:
            v = r.get(sub, {}).get(key) if sub else r.get(key)
            if v is not None:
                vals.append(v)
        return statistics.median(vals) if vals else None

    by_regime = []
    for reg in rec.get("regimes", []):
        v = [r["dbase_over_gbase"] for r in live if r["regime"] == reg
             and r.get("dbase_over_gbase")]
        by_regime.append({"regime": reg, "n": len(v),
                          "median": statistics.median(v) if v else None,
                          "at_bar": sum(1 for x in v if x >= 1.0)})

    return {
        "device": dev, "gpu": rec.get("gpu"), "capability": rec.get("capability"),
        "cells_ok": len(rows), "cells_skipped": len(skipped),
        "void_cells": [{"cell": _cellid(r), "regime": r["regime"],
                        "why": r["_void_why"]} for r in void],
        "void_fraction": void_frac,
        "Q1_self_pair": q1, "Q2_wiring": q2, "Q2_failures": q2_fail,
        "Q3_ratio_stability": "folded into Q1 per-cell voiding",
        "Q4_fidelity": q4,
        "Q4_detail": {"median": statistics.median([v for _, v in q4_cells])
                      if q4_cells else None, "failures": q4_fail},
        "F1_small_batch": f1,
        "F1_detail": {"regime": p["f1_regime"], "at_bar": f1_n, "of": len(f1_cells),
                      "median": f1_med,
                      "predicted_band": p["f1_predicted_median_band"],
                      "median_in_predicted_band": _in(
                          f1_med, p["f1_predicted_median_band"]),
                      "per_cell": [{"cell": _cellid(r), "d/g": v}
                                   for r, v in f1_cells]},
        "by_regime": by_regime,
        "P1_pairing_dividend_report_only": {
            "cells": len(p1), "diverged_outside_band": len(diverged),
            "median_block_over_interleaved": statistics.median(
                [x["block_over_interleaved"] for x in p1]) if p1 else None,
            "max_divergence": max((x for x in p1),
                                  key=lambda x: abs(1 - x["block_over_interleaved"]),
                                  default=None),
            "detail": p1},
        "Q5_order_bias_report_only": {"median": med("order_bias", "primary")},
        "instrument": {"median_selfpair": med("ratio_median", "selfpair"),
                       "median_pairs": med("pairs")},
        "DEVICE_CONFIRMED": bool(q1 and q2 and q4 and f1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prereg", required=True)
    ap.add_argument("--receipt", action="append", required=True)
    ap.add_argument("--out", default="verdicts_interleaved.json")
    args = ap.parse_args()
    pre = json.loads(Path(args.prereg).read_text())
    per = {}
    for spec in args.receipt:
        label, path = spec.split("=", 1)
        per[label] = grade_device(label, json.loads(Path(path).read_text()), pre)
    two = len(per) >= 2
    confirmed = two and all(d["DEVICE_CONFIRMED"] for d in per.values())
    Path(args.out).write_text(json.dumps(
        {"prereg": args.prereg, "devices": per, "two_device_rule_met": two,
         "INTERLEAVED_CONFIRMED": confirmed}, indent=1, default=str))
    for dev, d in per.items():
        print(f"[{dev}] {d['gpu']} sm_{d['capability']}  F1={d['F1_small_batch']} "
              f"Q1={d['Q1_self_pair']} Q2={d['Q2_wiring']} Q4={d['Q4_fidelity']} "
              f"=> {d['DEVICE_CONFIRMED']}")
        f = d["F1_detail"]
        print(f"      F1 median {f['median']} ({f['at_bar']}/{f['of']} at bar, "
              f"band {f['predicted_band']}, in-band={f['median_in_predicted_band']})")
        print(f"      live {d['cells_ok']-len(d['void_cells'])}/{d['cells_ok']}  "
              f"median selfpair {d['instrument']['median_selfpair']}  "
              f"median pairs {d['instrument']['median_pairs']}")
        pp = d["P1_pairing_dividend_report_only"]
        print(f"      P1 block/interleaved median {pp['median_block_over_interleaved']}, "
              f"{pp['diverged_outside_band']}/{pp['cells']} cells outside [0.97,1.03]")
        summary = [(x["regime"], x["median"], "%d/%d" % (x["at_bar"], x["n"]))
                   for x in d["by_regime"]]
        print(f"      by regime: {summary}")
        if d["void_cells"]:
            print(f"      VOID: {d['void_cells']}")
    print(f"TWO_DEVICE={two}  INTERLEAVED_CONFIRMED={confirmed}")
    print("REDUCE_DONE — the verdict is the JSON, not this exit code")


if __name__ == "__main__":
    main()
