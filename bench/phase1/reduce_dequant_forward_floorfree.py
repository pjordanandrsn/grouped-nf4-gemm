#!/usr/bin/env python3
"""Mechanical reducer for the floor-free small-batch leg (leg 2).

Reads the FROZEN prereg and the per-device receipts and adjudicates. The agent
that ran the experiment does not decide whether it passed.

    python3 bench/phase1/reduce_dequant_forward_floorfree.py \
        --prereg kernel/prereg_dequant_forward_floorfree.json \
        --receipt H100=.../floorfree_H100.json \
        --receipt ADA=.../floorfree_ADA.json \
        --leg1 ADA=.../dequant_forward_ADA.json \
        --out .../verdicts.json

`--leg1` supplies leg 1's receipt for the SAME device so B1 (the replication
gate on the full arms) can be adjudicated. Without it B1 reports NOT-ADJUDICATED
rather than silently passing.

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


def grade_device(dev, rec, pre, leg1=None):
    p = pre["frozen_verdict_criteria_params"]
    rows = [r for r in rec["rows"] if r.get("status") == "ok"]
    skipped = [r for r in rec["rows"] if r.get("status") != "ok"]

    for r in rows:
        why = []
        if not _in(r.get("gb_selfpair"), p["selfpair_band"]):
            why.append(f"gb_selfpair={r.get('gb_selfpair'):.4f}")
        if not _in(r.get("db_selfpair"), p["selfpair_band"]):
            why.append(f"db_selfpair={r.get('db_selfpair'):.4f}")
        if not _in(r.get("gb_drift"), p["drift_band"]):
            why.append(f"gb_drift={r.get('gb_drift'):.4f}")
        r["_void"] = bool(why)
        r["_void_why"] = "; ".join(why)

    live = [r for r in rows if not r["_void"]]
    void = [r for r in rows if r["_void"]]
    void_frac = (len(void) / len(rows)) if rows else 1.0
    q1 = void_frac <= p["void_cell_fraction_that_voids_the_leg"]

    q2_fail = []
    for r in rows:
        g = r.get("gate", {})
        if not g.get("deq_calls_ok"):
            q2_fail.append(f"{_cellid(r)}/{r['regime']}: deq_calls "
                           f"{g.get('deq_calls_D')} < {g.get('nonempty_groups')}")
        if not g.get("base_arms_agree"):
            q2_fail.append(f"{_cellid(r)}/{r['regime']}: base arms differ on "
                           f"{g.get('base_rows_differing')} rows")
        for arm in ("G_base", "D_base", "G_full", "D_full"):
            st = g.get(f"grad_{arm}")
            if not st or not st.get("act_finite") or not st.get("act_nonzero"):
                q2_fail.append(f"{_cellid(r)}/{r['regime']}: {arm} act grad {st}")
        if not g.get("gradA_at_nonzero_B"):
            q2_fail.append(f"{_cellid(r)}/{r['regime']}: lora_A control failed")
    q2 = not q2_fail

    def by_regime(regime, key, pool=None):
        return [(r, r[key]) for r in (pool if pool is not None else live)
                if r["regime"] == regime and r.get(key) is not None]

    # ---- F1: the primary, on the floor-free ratio --------------------------
    f1_cells = by_regime(p["f1_regime"], "dbase_over_gbase")
    f1_n = sum(1 for _, v in f1_cells if v >= p["f1_bar"])
    f1_med = statistics.median([v for _, v in f1_cells]) if f1_cells else None
    f1 = (len(f1_cells) > 0 and f1_n >= p["f1_min_cells"])
    f1_band = _in(f1_med, p["f1_predicted_median_band"])

    # ---- B1: does the FULL-arm ratio reproduce leg 1 on this device? -------
    b1, b1_detail = None, {"note": "NOT ADJUDICATED — no leg 1 receipt supplied"}
    if leg1:
        l1 = {(r["model"], r["proj"], r["regime"]): r
              for r in leg1.get("rows", []) if r.get("status") == "ok"}
        pairs, fails = [], []
        for r in rows:
            if r["regime"] != p["f1_regime"]:
                continue
            o = l1.get((r["model"], r["proj"], r["regime"]))
            if not o or o.get("d_over_g") is None:
                continue
            # leg 1's own void cells carry no measurement to replicate against
            if not (0.97 <= o["g_selfpair"] <= 1.03
                    and 0.97 <= o["d_selfpair"] <= 1.03
                    and 0.95 <= o["g_drift"] <= 1.05):
                continue
            if r["_void"] or r.get("dfull_over_gfull") is None:
                continue
            q = r["dfull_over_gfull"] / o["d_over_g"]
            pairs.append({"cell": _cellid(r), "leg2_full": r["dfull_over_gfull"],
                          "leg1": o["d_over_g"], "ratio": q})
            if not _in(q, p["b1_leg1_replication_band"]):
                fails.append(pairs[-1])
        b1 = bool(pairs) and not fails
        b1_detail = {"comparable_cells": len(pairs), "outside_band": fails,
                     "band": p["b1_leg1_replication_band"], "pairs": pairs}
        if not pairs:
            b1, b1_detail["note"] = None, "NOT ADJUDICATED — no comparable cells"

    # ---- Q4: fidelity, over ALL cells (arithmetic survives a void cell) ----
    q4_cells = [(r, r["b_rel_G_over_D"]) for r in rows
                if r.get("b_rel_G_over_D") is not None]
    q4_fail = [(_cellid(r), r["regime"], v) for r, v in q4_cells
               if not v <= p["q4_fidelity_bar"]]
    q4 = (len(q4_cells) > 0 and not q4_fail)

    def med_of(key, regime=None):
        vals = [r[key] for r in live if r.get(key) is not None
                and (regime is None or r["regime"] == regime)]
        return statistics.median(vals) if vals else None

    f2 = []
    for reg in rec.get("regimes", []):
        vals = [v for _, v in by_regime(reg, "dbase_over_gbase")]
        f2.append({"regime": reg, "n": len(vals),
                   "median_dbase_over_gbase": statistics.median(vals) if vals else None,
                   "min": min(vals) if vals else None,
                   "max": max(vals) if vals else None,
                   "median_dfull_over_gfull": med_of("dfull_over_gfull", reg)})

    floor = med_of("lora_floor_frac_of_gbase", p["f1_regime"])
    return {
        "device": dev, "gpu": rec.get("gpu"), "capability": rec.get("capability"),
        "eids_form": rec.get("eids_form"),
        "cells_ok": len(rows), "cells_skipped": len(skipped),
        "skipped": [{"cell": _cellid(r), "regime": r["regime"],
                     "reason": r.get("reason")} for r in skipped],
        "void_cells": [{"cell": _cellid(r), "regime": r["regime"],
                        "why": r["_void_why"]} for r in void],
        "void_fraction": void_frac,
        "Q1_self_pair": q1, "Q2_wiring": q2, "Q2_failures": q2_fail,
        "Q4_fidelity": q4,
        "Q4_detail": {"median_b_rel_G_over_D": (
            statistics.median([v for _, v in q4_cells]) if q4_cells else None),
            "failures": q4_fail},
        "F1_floorfree_small_batch": f1,
        "F1_detail": {"regime": p["f1_regime"], "cells_at_or_above_bar": f1_n,
                      "of": len(f1_cells), "median": f1_med,
                      "predicted_band": p["f1_predicted_median_band"],
                      "median_in_predicted_band": f1_band,
                      "per_cell": [{"cell": _cellid(r), "dbase_over_gbase": v}
                                   for r, v in f1_cells]},
        "B1_leg1_replication": b1, "B1_detail": b1_detail,
        "F2_small_batch_band_report_only": f2,
        "L1_floor_report_only": {
            "median_lora_floor_frac_of_gbase_at_" + p["f1_regime"]: floor,
            "expectation": "> 0.5", "met": (floor > 0.5) if floor else None},
        "E1_energy_report_only": {
            "median_j_ratio": med_of("j_ratio_dbase_over_gbase")},
        "DEVICE_CONFIRMED": bool(q1 and q2 and q4 and f1 and (b1 is not False)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prereg", required=True)
    ap.add_argument("--receipt", action="append", required=True)
    ap.add_argument("--leg1", action="append", default=[],
                    help="LABEL=path to leg 1's receipt for the same device")
    ap.add_argument("--out", default="verdicts_floorfree.json")
    args = ap.parse_args()

    pre = json.loads(Path(args.prereg).read_text())
    l1 = {}
    for spec in args.leg1:
        label, path = spec.split("=", 1)
        l1[label] = json.loads(Path(path).read_text())

    per = {}
    for spec in args.receipt:
        label, path = spec.split("=", 1)
        per[label] = grade_device(label, json.loads(Path(path).read_text()),
                                  pre, l1.get(label))

    two = len(per) >= 2
    confirmed = two and all(d["DEVICE_CONFIRMED"] for d in per.values())
    out = {"prereg": args.prereg,
           "prereg_verdict_key": pre["frozen_verdict_criteria"]["verdict_key"],
           "devices": per, "two_device_rule_met": two,
           "FLOORFREE_CONFIRMED": confirmed,
           "note": "NOT CONFIRMED means the registered criteria were not all "
                   "met, not that the numbers are wrong. Report at full volume."}
    Path(args.out).write_text(json.dumps(out, indent=1, default=str))

    for dev, d in per.items():
        print(f"[{dev}] {d['gpu']} sm_{d['capability']} eids={d['eids_form']}  "
              f"F1={d['F1_floorfree_small_batch']} B1={d['B1_leg1_replication']} "
              f"Q1={d['Q1_self_pair']} Q2={d['Q2_wiring']} Q4={d['Q4_fidelity']} "
              f"=> {d['DEVICE_CONFIRMED']}")
        f = d["F1_detail"]
        print(f"      F1 median {f['median']} ({f['cells_at_or_above_bar']}/{f['of']} "
              f"at bar, band {f['predicted_band']}, in-band={f['median_in_predicted_band']})")
        print(f"      floor-free by regime: "
              f"{[(x['regime'], x['median_dbase_over_gbase'], x['median_dfull_over_gfull']) for x in d['F2_small_batch_band_report_only']]}")
        if d["void_cells"]:
            print(f"      VOID: {d['void_cells']}")
        if d["B1_detail"].get("outside_band"):
            print(f"      B1 outside band: {d['B1_detail']['outside_band']}")
    print(f"TWO_DEVICE={two}  FLOORFREE_CONFIRMED={confirmed}")
    print("REDUCE_DONE — the verdict is the JSON, not this exit code")


if __name__ == "__main__":
    main()
