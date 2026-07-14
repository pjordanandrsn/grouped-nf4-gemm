# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).

"""Mechanical reducer for the v6 confirmatory (register-LUT M-tile mainloop).

Reads bars ONLY from the stamped prereg (--spec). Cell value = median across
reps of the per-rep paired ratio. Verdict keys (A5000 bars; A2000 rows are
REPORT-ONLY by registration):

  W1 rewrite effect   fused_v5loop/fused_nf4 >= w1_bar on ALL 8 prefill
                      cells AND median >= w1_median_bar (instance-robust:
                      both sides share the instance)
  W2 dequant floor    dequant/fused >= w2_floor on the 7 barred cells AND
                      >= w2_bar on >= w2_min_cells of them
  W3 gate_up          dequant/fused >= w3_floor on all 3 big gate_ups AND
                      >= w3_bar on >= w3_min_cells of them
  W4 decode guard     decode census cells >= 1.0 on >= w4_min_cells of 8
  Q1 suite            both devices == suite_expect

OLMoE gate_up dequant-ratio is REPORT-ONLY (R1). Usage:
  python bench/phase1/reduce_confirmatory_v6.py --spec kernel/prereg_v6_confirmatory.json \
      --a5000 d1.json d2.json d3.json [--a2000 ...] \
      --suite-a5000 "44/44" [--suite-a2000 "44/44"] --out reduction_v6.json
"""
import argparse
import json
import statistics
from collections import defaultdict


def load_reps(paths):
    cells = defaultdict(lambda: defaultdict(list))
    for p in paths:
        for c in json.loads(open(p).read())["cells"]:
            cells[(c["regime"], c["model"], c["proj"])][c["backend"]].append(c)
    return cells


def device_rows(cells, num_backend="dequant_grouped"):
    rows = []
    for (regime, model, proj), by_b in sorted(cells.items()):
        num, fus = by_b.get(num_backend, []), by_b.get("fused_nf4", [])
        n = min(len(num), len(fus))
        if n == 0:
            continue
        ratios = [num[i]["ms_median"] / fus[i]["ms_median"] for i in range(n)]
        rows.append({
            "regime": regime, "model": model, "proj": proj, "reps": n,
            "numerator": num_backend,
            "ratios": [round(r, 4) for r in ratios],
            "median": statistics.median(ratios),
            "worst": min(ratios),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--a5000", nargs="+", required=True)
    ap.add_argument("--a2000", nargs="*", default=[])
    ap.add_argument("--suite-a5000", required=True)
    ap.add_argument("--suite-a2000", default="not-run")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    spec = json.loads(open(args.spec).read())
    P = spec["frozen_verdict_criteria_params"]
    barred = [tuple(c) for c in P["barred_prefill_cells"]]
    gu3 = [tuple(c) for c in P["gate_up_parity_cells"]]
    census_decode = [tuple(c) for c in P["census_cells"]]
    report_only = [tuple(c) for c in P["report_only_cells"]]

    out = {"params": P, "devices": {}}
    reps5 = load_reps(args.a5000)
    a5000 = device_rows(reps5)
    a5000_rw = device_rows(reps5, num_backend="fused_v5loop")
    out["devices"]["A5000"] = a5000
    out["devices"]["A5000_rewrite"] = a5000_rw
    if args.a2000:
        reps2 = load_reps(args.a2000)
        out["devices"]["A2000_report_only"] = device_rows(reps2)
        out["devices"]["A2000_rewrite_report_only"] = device_rows(
            reps2, num_backend="fused_v5loop")

    pre = {(r["model"], r["proj"]): r for r in a5000
           if r["regime"] == "prefill_s2048"}
    pre_rw = {(r["model"], r["proj"]): r for r in a5000_rw
              if r["regime"] == "prefill_s2048"}
    dec = {(r["model"], r["proj"]): r for r in a5000
           if r["regime"] == "decode_bs1"}

    def med(cell):
        return pre[cell]["median"] if cell in pre else None

    def med_rw(cell):
        return pre_rw[cell]["median"] if cell in pre_rw else None

    w1_vals = {f"{m}/{p}": med_rw((m, p)) for m, p in census_decode}
    w1_present = [v for v in w1_vals.values() if v is not None]
    w1_med = (statistics.median(w1_present)
              if len(w1_present) == len(census_decode) else None)
    w1 = (all(v is not None and v >= P["w1_bar"] for v in w1_vals.values())
          and w1_med is not None and w1_med >= P["w1_median_bar"])
    w2_vals = {f"{m}/{p}": med((m, p)) for m, p in barred}
    w2 = (all(v is not None and v >= P["w2_floor"] for v in w2_vals.values())
          and sum(1 for v in w2_vals.values()
                  if v is not None and v >= P["w2_bar"]) >= P["w2_min_cells"])
    w3_vals = {f"{m}/{p}": med((m, p)) for m, p in gu3}
    w3 = (all(v is not None and v >= P["w3_floor"] for v in w3_vals.values())
          and sum(1 for v in w3_vals.values()
                  if v is not None and v >= P["w3_bar"]) >= P["w3_min_cells"])
    dec_vals = {f"{m}/{p}": dec[(m, p)]["median"] for m, p in census_decode
                if (m, p) in dec}
    w4_pass_cells = sum(1 for v in dec_vals.values() if v >= 1.0)
    w4 = (len(dec_vals) == len(census_decode)
          and w4_pass_cells >= P["w4_min_cells"])
    q1 = (args.suite_a5000 == P["suite_expect"]
          and (args.suite_a2000 in (P["suite_expect"], "not-run")))
    q1_note = "A2000 suite not run" if args.suite_a2000 == "not-run" else ""

    out["verdicts"] = {
        "W1_rewrite_effect": {"pass": w1, "bar": P["w1_bar"],
                              "median_bar": P["w1_median_bar"],
                              "median": w1_med, "values": w1_vals},
        "W2_dequant_floor": {"pass": w2, "floor": P["w2_floor"],
                             "bar": P["w2_bar"], "min_cells": P["w2_min_cells"],
                             "values": w2_vals},
        "W3_gate_up": {"pass": w3, "floor": P["w3_floor"], "bar": P["w3_bar"],
                       "min_cells": P["w3_min_cells"], "values": w3_vals},
        "W4_decode_guard": {"pass": w4, "min_cells": P["w4_min_cells"],
                            "pass_cells": w4_pass_cells, "values": dec_vals},
        "Q1_suite": {"pass": q1, "a5000": args.suite_a5000,
                     "a2000": args.suite_a2000, "note": q1_note},
        "R1_report_only": {f"{m}/{p}": med((m, p)) for m, p in report_only},
        "V6_CONFIRMED": w1 and w2 and w3 and w4 and q1,
    }
    open(args.out, "w").write(json.dumps(out, indent=1))
    for k, v in out["verdicts"].items():
        print(k, json.dumps(v) if not isinstance(v, dict) or "pass" not in v
              else ("PASS" if v["pass"] else "FAIL"), sep=": ")


if __name__ == "__main__":
    main()
