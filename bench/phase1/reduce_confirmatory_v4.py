#!/usr/bin/env python3
"""Reduce confirmatory-v4 reps against the criteria REGISTERED in
kernel/prereg_v4_confirmatory.json (--spec). Two files per rep: the decode
run (census + floor/split set) and the prefill run (census).

Per decode cell: median/worst of dequant_ms/routed_ms; paired routed/fused.
Per prefill cell: paired v3prefill_ms/fused_ms (the config claim) and
dequant_ms/fused_ms (the absolute report).

Usage:
  python reduce_confirmatory_v4.py --spec kernel/prereg_v4_confirmatory.json \
    --device A5000 d1.json d2.json d3.json p1.json p2.json p3.json \
    --device A2000 ... --suite A5000:44/44 A2000:44/44 [--out out.json]
(decode and prefill rep files may be passed in any order; regime is read
from each cell.)
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
            cells[(c["regime"], c["model"], c["proj"])][c["backend"]].append(c)
    return cells


def med_ratio(num, den, n):
    return statistics.median(num[i]["ms_median"] / den[i]["ms_median"] for i in range(n))


def reduce_device(cells):
    rows = []
    for key in sorted(cells):
        regime, model, proj = key
        bk = cells[key]
        ok = lambda nm: [c for c in bk.get(nm, []) if c["status"] == "ok"]
        deq, fus = ok("dequant_grouped"), ok("fused_nf4")
        row = {"regime": regime, "model": model, "proj": proj}
        if not deq or not fus:
            row.update({"status": "NOT-RUN", "reasons": sorted(
                {c.get("reason", "?")[:80] for b in bk.values()
                 for c in b if c["status"] != "ok"})})
            rows.append(row)
            continue
        row["status"] = "ok"
        n = min(len(deq), len(fus))
        row["fused_vs_dequant_med"] = med_ratio(deq, fus, n)
        row["fused_vs_dequant_worst"] = min(
            deq[i]["ms_median"] / fus[i]["ms_median"] for i in range(n))
        if regime == "decode_bs1":
            ro = ok("fused_routed")
            if ro:
                m = min(len(ro), n)
                row["routed_vs_dequant_med"] = med_ratio(deq, ro, m)
                row["routed_vs_fused_med"] = med_ratio(fus, ro, m)
                ej = [(deq[i].get("j_per_token"), ro[i].get("j_per_token"))
                      for i in range(m)]
                ej = [(d, f) for d, f in ej if d and f]
                if ej:
                    row["routed_energy_margin_worst"] = max(f / d for d, f in ej)
        else:
            old = ok("fused_nf4_v3prefill")
            if old:
                m = min(len(old), n)
                row["newcfg_vs_oldcfg_med"] = med_ratio(old, fus, m)
        rows.append(row)
    return rows


def verdicts(per_device, suites, P):
    v, per = {}, {}
    for d, rs in per_device.items():
        run = [r for r in rs if r["status"] == "ok"]
        dec = [r for r in run if r["regime"] == "decode_bs1"]
        pre = [r for r in run if r["regime"] != "decode_bs1"]
        floor_cells = [tuple(x) for x in P["floor_cells"]]
        census = [r for r in dec if [r["model"], r["proj"]] in P["census_cells"]]
        floors = [r for r in dec if (r["model"], r["proj"]) in floor_cells]
        eligible = [r for r in dec if (r["model"], r["proj"]) not in floor_cells]
        det = {"notrun": len(rs) - len(run)}
        det["F1"] = all(r.get("routed_vs_dequant_med", 0) >= P["f1_floor"] for r in dec)
        det["F2"] = (d != P["a5000_name"]) or all(
            P["f2_lo"] <= r.get("routed_vs_dequant_med", 0) <= P["f2_hi"]
            for r in floors)
        det["F3"] = all(
            P["f3_lo"] <= r.get("routed_vs_fused_med", 0) <= P["f3_hi"]
            for r in eligible if r.get("routed_vs_fused_med"))
        det["P1"] = sum(r.get("newcfg_vs_oldcfg_med", 0) >= P["p1_bar"]
                        for r in pre) >= P["p1_min_cells"]
        p2_need = P["p2_min_cells_a5000"] if d == P["a5000_name"] else P["p2_min_cells_a2000"]
        det["P2"] = sum(r["fused_vs_dequant_med"] >= P["p2_floor"] for r in pre) >= p2_need
        det["P2b"] = (d != P["a5000_name"]) or sum(
            r["fused_vs_dequant_med"] >= P["p2b_bar"] for r in pre) >= P["p2b_min_cells"]
        det["E1"] = all(r.get("routed_energy_margin_worst", 9) < 1.0 for r in census)
        per[d] = det
        v[f"_detail_{d}"] = det
    for crit in ("F1", "F2", "F3", "P1", "P2", "P2b", "E1"):
        v[crit] = all(per[d][crit] for d in per)
    v["Q1_suite"] = len(suites) == 2 and all(
        s.strip() == P["suite_expect"] for s in suites.values())
    v["V4_CONFIRMED"] = all(v[k] for k in
                            ("F1", "F2", "F3", "P1", "P2", "P2b", "E1", "Q1_suite"))
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
    return 0 if v["V4_CONFIRMED"] else 1


if __name__ == "__main__":
    sys.exit(main())
