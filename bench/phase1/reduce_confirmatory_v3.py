#!/usr/bin/env python3
"""Reduce the confirmatory-v3 reps and apply the criteria REGISTERED in
kernel/prereg_v3_confirmatory.json (--spec). Same discipline as v2's reducer:
thresholds live in the stamped file; cells skipped on every rep are NOT-RUN
and scale denominators one-for-one.

Per cell (n=3 fresh-process reps):
  vs dequant        : median-rep and worst-rep of dequant_ms / fused_ms
  vs v2 constant    : median-rep of v2cfg_ms / fused_ms   (paired)
  split-K           : median-rep of nosplit_ms / fused_ms (paired)
  energy            : worst-rep fused_J / dequant_J
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
        ok = lambda n: [c for c in bk.get(n, []) if c["status"] == "ok"]
        deq, fus = ok("dequant_grouped"), ok("fused_nf4")
        v2c, nsp = ok("fused_nf4_v2cfg"), ok("fused_nf4_nosplit")
        row = {"model": key[0], "proj": key[1]}
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
            "median_speed_ratio": statistics.median(ratios),
            "worst_speed_ratio": min(ratios),
            "speed_ratios": [round(r, 3) for r in ratios],
            "b_rel_fused_max": max(c["b_rel_vs_fp64"] for c in fus[:n]),
            "b_rel_dequant_max": max(c["b_rel_vs_fp64"] for c in deq[:n]),
        })
        if v2c:
            m = min(len(v2c), n)
            row["paired_v2cfg_median"] = statistics.median(
                v2c[i]["ms_median"] / fus[i]["ms_median"] for i in range(m))
        if nsp:
            m = min(len(nsp), n)
            row["paired_nosplit_median"] = statistics.median(
                nsp[i]["ms_median"] / fus[i]["ms_median"] for i in range(m))
        ej = [(deq[i].get("j_per_token"), fus[i].get("j_per_token")) for i in range(n)]
        ej = [(d, f) for d, f in ej if d and f]
        if ej:
            row["worst_energy_margin"] = max(f / d for d, f in ej)
        rows.append(row)
    return rows


def verdicts(per_device, suites, P):
    v = {}
    per = {}
    for d, rs in per_device.items():
        run = [r for r in rs if r["status"] == "ok"]
        notrun = len(rs) - len(run)
        total = len(rs)
        cell = {(r["model"], r["proj"]): r for r in run}

        def med_paired(names, field):
            return [(nm, cell[tuple(nm)].get(field)) for nm in names
                    if tuple(nm) in cell and cell[tuple(nm)].get(field)]

        det = {"notrun": notrun}
        if d == P["a2000_name"]:
            rec = med_paired(P["x1_named_recovery_cells"], "paired_v2cfg_median")
            det["X1a"] = sum(val >= P["x1_recovery_bar"] for _, val in rec) >= P["x1_recovery_min_cells"]
            allp = [r["paired_v2cfg_median"] for r in run if r.get("paired_v2cfg_median")]
            det["X1b"] = (all(p >= P["x1_global_floor"] for p in allp)
                          and statistics.median(allp) >= 1.0)
            sp = med_paired(P["split_cells_a2000"], "paired_nosplit_median")
            det["X2"] = all(val >= P["x2_a2000_floor"] for _, val in sp) and len(sp) > 0
        else:
            det["X1a"] = det["X1b"] = True  # A5000 plan == v2 constant off-split-cells
            sp = med_paired(P["split_cells_a5000"], "paired_nosplit_median")
            scout = cell.get(tuple(P["scout_down_cell"]))
            det["X2"] = (scout is not None
                         and scout.get("paired_nosplit_median", 0) >= P["x2_scout_bar"]
                         and sum(val >= P["x2_blind_bar"] for nm, val in sp
                                 if nm != P["scout_down_cell"]) >= P["x2_blind_min_cells"])
        scout = cell.get(tuple(P["scout_down_cell"]))
        det["X3"] = scout is not None and scout["median_speed_ratio"] >= P["x3_scout_dequant_bar"]
        e_ok = sum(r.get("worst_energy_margin", 9) < 1.0 for r in run)
        det["X4"] = (e_ok >= max(P["x4_min_cells"] - (P["x4_total_cells"] - total) - notrun, 0)
                     and (scout is None or scout.get("worst_energy_margin", 9) < 1.0))
        per[d] = det
        v[f"_detail_{d}"] = det
    for crit in ("X1a", "X1b", "X2", "X3", "X4"):
        v[crit] = all(per[d][crit] for d in per)
    v["X5_suite"] = len(suites) == 2 and all(
        s.strip() == P["suite_expect"] for s in suites.values())
    v["V3_CONFIRMED"] = all(v[k] for k in ("X1a", "X1b", "X2", "X3", "X4", "X5_suite"))
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
    return 0 if v["V3_CONFIRMED"] else 1


if __name__ == "__main__":
    sys.exit(main())
