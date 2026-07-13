#!/usr/bin/env python3
"""Reduce the confirmatory-v2 reps and apply the criteria REGISTERED in
kernel/prereg_v2_confirmatory.json (passed via --spec) — the thresholds live
in the stamped file, not here, so the verdict pipeline is frozen with the
protocol.

Reduction per cell (as registered, worst rep over n reps):
  speed ratio vs dequant   = min over reps of dequant_ms / fused_ms
  median-rep ratio         = median over reps of dequant_ms / fused_ms
  paired config ratio      = min over reps of v1cfg_ms / fused_ms
  energy margin            = max over reps of fused_J / dequant_J
Cells skipped on every rep of a device are NOT-RUN: excluded from that
device's denominators (thresholds scale down one-for-one), never substituted.

Usage:
  python reduce_confirmatory_v2.py --spec kernel/prereg_v2_confirmatory.json \
      --device A5000 rep1.json rep2.json rep3.json \
      --device A2000 rep1.json rep2.json rep3.json \
      --suite A5000:35/35 A2000:35/35 [--out out.json]
"""
import argparse
import json
import statistics
import sys
from collections import defaultdict


def load_reps(paths):
    cells = defaultdict(lambda: defaultdict(list))
    for p in paths:
        for c in json.loads(open(p).read())["cells"]:
            if c["regime"] == "decode_bs1":
                cells[(c["model"], c["proj"])][c["backend"]].append(c)
    return cells


def reduce_device(cells):
    rows = []
    for key in sorted(cells):
        bk = cells[key]
        ok = lambda name: [c for c in bk.get(name, []) if c["status"] == "ok"]
        deq, fus, old = ok("dequant_grouped"), ok("fused_nf4"), ok("fused_nf4_v1cfg")
        row = {"model": key[0], "proj": key[1],
               "heldout": key[0].startswith("heldout2/")}
        if not deq or not fus:
            row.update({"status": "NOT-RUN", "reasons": sorted(
                {c.get("reason", "?")[:80] for b in bk.values()
                 for c in b if c["status"] != "ok"})})
            rows.append(row)
            continue
        n = min(len(deq), len(fus))
        ratios = [deq[i]["ms_median"] / fus[i]["ms_median"] for i in range(n)]
        row.update({
            "status": "ok", "n_reps": n,
            "worst_speed_ratio": min(ratios),
            "median_speed_ratio": statistics.median(ratios),
            "speed_ratios": [round(r, 3) for r in ratios],
            "b_rel_fused_max": max(c["b_rel_vs_fp64"] for c in fus[:n]),
            "b_rel_dequant_max": max(c["b_rel_vs_fp64"] for c in deq[:n]),
        })
        if old:
            m = min(len(old), n)
            paired = [old[i]["ms_median"] / fus[i]["ms_median"] for i in range(m)]
            row["worst_paired_ratio"] = min(paired)
            row["paired_ratios"] = [round(r, 3) for r in paired]
        ej = [(deq[i].get("j_per_token"), fus[i].get("j_per_token")) for i in range(n)]
        ej = [(d, f) for d, f in ej if d and f]
        if ej:
            row["worst_energy_margin"] = max(f / d for d, f in ej)
        rows.append(row)
    return rows


def _median_paired(row):
    pr = row.get("paired_ratios")
    return statistics.median(pr) if pr else 0


def verdicts(per_device, suites, P):
    v = {}
    okc = lambda rs: [r for r in rs if r["status"] == "ok"]
    per = {}
    for d, rs in per_device.items():
        census = [r for r in okc(rs) if not r["heldout"]]
        held = [r for r in okc(rs) if r["heldout"]]
        notrun_h = len([r for r in rs if r["status"] == "NOT-RUN" and r["heldout"]])
        notrun_c = len([r for r in rs if r["status"] == "NOT-RUN" and not r["heldout"]])
        allcells = census + held
        # P1a bounded loss (worst-rep, every run cell); P1b gains exist (median-rep)
        p1a = all(r.get("worst_paired_ratio", 0) >= P["p1a_worst_floor"]
                  for r in allcells)
        p1b = sum(_median_paired(r) >= P["p1b_min_gain"]
                  for r in allcells) >= P["p1b_min_cells"]
        # S1 census vs dequant: median>=1.3 on >=7/8; median>=1.0 on 8/8;
        # worst>=1.0 on >=7/8 (one home-card contention transient tolerated)
        s1 = (notrun_c == 0 and len(census) == 8
              and sum(r["median_speed_ratio"] >= P["s1_median_bar"] for r in census)
              >= P["s1_median_min_cells"]
              and all(r["median_speed_ratio"] >= 1.0 for r in census)
              and sum(r["worst_speed_ratio"] >= 1.0 for r in census)
              >= P["s1_worst_floor_min_cells"])
        # S2 held-out-v2 (NOT-RUN scales thresholds one-for-one)
        s2 = (len(held) > 0
              and sum(r["median_speed_ratio"] >= 1.0 for r in held)
              >= max(P["s2_median_floor_min_cells"] - notrun_h, 0)
              and sum(r["median_speed_ratio"] >= P["s1_median_bar"] for r in held)
              >= max(P["s2_median13_min_cells"] - notrun_h, 0)
              and sum(r["worst_speed_ratio"] >= P["s2_worst_floor"] for r in held)
              >= max(P["s2_median_floor_min_cells"] - notrun_h, 0))
        n_run = len(allcells)
        e1 = (sum(r.get("worst_energy_margin", 9) < 1.0 for r in allcells)
              >= max(P["e1_min_cells"] - (16 - n_run), 0))
        per[d] = {"P1a": p1a, "P1b": p1b, "S1": s1, "S2": s2, "E1": e1,
                  "notrun": notrun_c + notrun_h}
        v[f"_detail_{d}"] = per[d]
    for crit in ("P1a", "P1b", "S1", "S2", "E1"):
        v[crit] = all(per[d][crit] for d in per)
    v["Q1_suite"] = len(suites) == 2 and all(s.strip() == "35/35" for s in suites.values())
    v["V2_CONFIRMED"] = all(v[k] for k in ("P1a", "P1b", "S1", "S2", "E1", "Q1_suite"))
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--device", nargs="+", action="append", required=True)
    ap.add_argument("--suite", nargs="*", default=[])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    P = json.loads(open(args.spec).read())["frozen_verdict_criteria_params"]
    per_device = {g[0]: reduce_device(load_reps(g[1:])) for g in args.device}
    suites = dict(s.split(":", 1) for s in args.suite)
    v = verdicts(per_device, suites, P)
    out = {"reduction": per_device, "suites": suites, "params": P, "verdicts": v}
    txt = json.dumps(out, indent=1)
    if args.out:
        open(args.out, "w").write(txt)
    print(txt)
    return 0 if v["V2_CONFIRMED"] else 1


if __name__ == "__main__":
    sys.exit(main())
