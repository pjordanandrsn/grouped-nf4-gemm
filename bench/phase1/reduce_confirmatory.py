#!/usr/bin/env python3
"""Reduce the Gate-2 blind-confirmatory reps to per-cell worst-rep numbers and
apply the frozen criteria in kernel/prereg_gate2_confirmatory.json.

Usage:
  python reduce_confirmatory.py --device A5000 rep1.json rep2.json rep3.json \
                                --device A2000 rep1.json rep2.json rep3.json \
                                [--suite A5000:35/35 A2000:35/35]

Reduction (as registered):
  per cell: worst-rep speed ratio  = min over reps of dequant_ms / fused_ms
            worst-rep energy margin = max over reps of fused_J / dequant_J
Census cells   = the 8 gemm_predictions.json cells (OLMoE/Qwen3-30B/gemma-4/gpt-oss x2).
Held-out cells = the 8 heldout_shapes.json cells (model prefixed "heldout/").
A cell skipped on every rep of a device (e.g. A2000 OOM) is NOT-RUN: excluded
from that device's C3/C4 denominator, never substituted.
"""
import argparse
import json
import sys
from collections import defaultdict

CENSUS_MODELS = {"OLMoE", "Qwen3-30B", "gemma-4", "gpt-oss"}
BAR = 1.3


def load_reps(paths):
    """-> {(model,proj): {backend: [cell per rep]}} for decode_bs1 rows."""
    cells = defaultdict(lambda: defaultdict(list))
    for p in paths:
        data = json.loads(open(p).read())
        for c in data["cells"]:
            if c["regime"] != "decode_bs1":
                continue
            cells[(c["model"], c["proj"])][c["backend"]].append(c)
    return cells


def reduce_device(cells):
    rows = []
    for key in sorted(cells):
        per_bk = cells[key]
        deq = [c for c in per_bk.get("dequant_grouped", []) if c["status"] == "ok"]
        fus = [c for c in per_bk.get("fused_nf4", []) if c["status"] == "ok"]
        gemv = [c for c in per_bk.get("gemv_4bit", []) if c["status"] == "ok"]
        row = {"model": key[0], "proj": key[1],
               "heldout": key[0].startswith("heldout/")}
        if not deq or not fus:
            reasons = {c.get("reason", "?")[:80] for bk in per_bk.values()
                       for c in bk if c["status"] != "ok"}
            row.update({"status": "NOT-RUN", "reasons": sorted(reasons),
                        "reps": {"dequant": len(deq), "fused": len(fus)}})
            rows.append(row)
            continue
        n = min(len(deq), len(fus))
        ratios = [deq[i]["ms_median"] / fus[i]["ms_median"] for i in range(n)]
        row.update({
            "status": "ok", "n_reps": n,
            "worst_speed_ratio": min(ratios),
            "speed_ratios": [round(r, 3) for r in ratios],
            "fused_ms_worst": max(c["ms_median"] for c in fus[:n]),
            "b_rel_fused_max": max(c["b_rel_vs_fp64"] for c in fus[:n]),
            "b_rel_dequant_max": max(c["b_rel_vs_fp64"] for c in deq[:n]),
        })
        if gemv:
            m = min(len(gemv), n)
            row["gemv_context_worst"] = min(
                gemv[i]["ms_median"] / fus[i]["ms_median"] for i in range(m))
        ej = [(deq[i].get("j_per_token"), fus[i].get("j_per_token"))
              for i in range(n)]
        ej = [(d, f) for d, f in ej if d and f]
        if ej:
            margins = [f / d for d, f in ej]
            row["worst_energy_margin"] = max(margins)
            row["energy_margins"] = [round(m, 3) for m in margins]
        rows.append(row)
    return rows


def verdicts(per_device, suites):
    v = {}
    ok = lambda rs: [r for r in rs if r["status"] == "ok"]
    census = {d: [r for r in ok(rs) if not r["heldout"]] for d, rs in per_device.items()}
    heldout = {d: [r for r in ok(rs) if r["heldout"]] for d, rs in per_device.items()}
    notrun = {d: [r for r in rs if r["status"] == "NOT-RUN"] for d, rs in per_device.items()}

    v["C1_census_speed"] = all(
        len(census[d]) == 8 and all(r["worst_speed_ratio"] >= BAR for r in census[d])
        for d in per_device)
    v["C2_census_energy"] = all(
        len(census[d]) == 8 and all(
            r.get("worst_energy_margin", 9) < 1.0 for r in census[d])
        for d in per_device)
    c3, c4 = {}, {}
    for d in per_device:
        hs = heldout[d]
        n_run = len(hs)
        hit = sum(r["worst_speed_ratio"] >= BAR for r in hs)
        never_slower = all(r["worst_speed_ratio"] >= 1.0 for r in hs)
        # >=6/8 scales to >=6/n_run only via NOT-RUN exclusion (registered)
        need = 6 - (8 - n_run) if n_run < 8 else 6
        c3[d] = (hit >= max(need, 0)) and never_slower and n_run > 0
        e_hit = sum(r.get("worst_energy_margin", 9) < 1.0 for r in hs)
        c4[d] = e_hit >= max(need, 0) and n_run > 0
        v[f"_c3_{d}"] = f"{hit}/{n_run} >= bar, never_slower={never_slower}, notrun={len(notrun[d])}"
        v[f"_c4_{d}"] = f"{e_hit}/{n_run} energy-below"
    v["C3_heldout_speed"] = all(c3.values())
    v["C4_heldout_energy"] = all(c4.values())
    v["C5_suite"] = all(s.strip() == "35/35" for s in suites.values()) and len(suites) == 2
    v["GATE2_CONFIRMED"] = all(v[k] for k in
                               ("C1_census_speed", "C2_census_energy",
                                "C3_heldout_speed", "C4_heldout_energy", "C5_suite"))
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", nargs="+", action="append", metavar=("NAME", "REP"),
                    required=True, help="device name followed by rep JSONs")
    ap.add_argument("--suite", nargs="*", default=[],
                    help="NAME:passed/total per device, e.g. A5000:35/35")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    per_device = {}
    for grp in args.device:
        name, paths = grp[0], grp[1:]
        per_device[name] = reduce_device(load_reps(paths))
    suites = dict(s.split(":", 1) for s in args.suite)

    v = verdicts(per_device, suites)
    out = {"reduction": per_device, "suites": suites, "verdicts": v}
    txt = json.dumps(out, indent=1)
    if args.out:
        open(args.out, "w").write(txt)
    print(txt)
    return 0 if v["GATE2_CONFIRMED"] else 1


if __name__ == "__main__":
    sys.exit(main())
