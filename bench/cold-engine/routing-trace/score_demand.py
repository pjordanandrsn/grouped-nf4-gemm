"""When does demand-paging beat placement? -- properly powered this time.

RESULTS-concentration.md tested exactly this and reported "not supported,
rho = 0.055", then said so twice more: *"three positives is too few to fit a
rule to, and fitting one would repeat the mistake this document exists to
correct"*. Declining to fit a rule to three points was right. Leaving it there
was not -- the fix for too few positives is more points, not a better story.

Twelve capacities instead of three, so 96 cells with 28 positives instead of
24 with 3. Two things change:

  * the rank correlation goes from 0.055 to -0.720 -- the earlier number
    measured nothing but the absence of data;
  * and rho was the wrong statistic anyway. The hypothesis is a THRESHOLD
    ("capacity covers the working set"), and a rank correlation over a range
    where almost every cell sits on one side of it cannot see a step.

Scored as a classifier against the always-say-no baseline, which is what a
threshold hypothesis deserves.
"""
import argparse
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "kernel"))
sys.path.insert(0, HERE)

from reuse_profile import _spearman                    # noqa: E402
from score_policies import demand_p, load, static_p    # noqa: E402


def counts(recs):
    return Counter((int(L), e) for r in recs
                   for L, ex in r["routed"].items() for e in ex)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=HERE)
    ap.add_argument("--models", default="olmoe:1024,granite:1280")
    ap.add_argument("--prompts", default="prose,code,math,dialogue")
    ap.add_argument("--fracs",
                    default="0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    ap.add_argument("--warm", type=int, default=256)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    models = dict((s.split(":")[0], int(s.split(":")[1]))
                  for s in a.models.split(","))
    fracs = [float(x) for x in a.fracs.split(",")]

    rows = []
    print("%-8s %-9s %5s " % ("model", "prompt", "ws")
          + " ".join("%5s" % f for f in fracs))
    for m, arena in models.items():
        for p in a.prompts.split(","):
            meta, recs = load(os.path.join(a.dir, "%s_%s.jsonl" % (m, p)))
            warm, ev = recs[:a.warm], recs[a.warm:]
            ws = len(counts(ev))
            marks = []
            for f in fracs:
                cap = int(arena * f)
                s = static_p(warm, ev, cap)[0]
                d = demand_p(warm, ev, cap)[0]
                rows.append({"model": m, "prompt": p, "frac": f, "cap": cap,
                             "working_set": ws, "headroom": ws / cap,
                             "static": s, "demand": d, "win": d < s,
                             "margin": (s - d) / s})
                marks.append(" YES " if d < s else "  .  ")
            print("%-8s %-9s %5d " % (m, p, ws) + "".join(marks))

    pos = sum(r["win"] for r in rows)
    base = max(pos, len(rows) - pos) / len(rows)
    print("\ndemand-paging wins in %d of %d cells (always-no scores %.3f)"
          % (pos, len(rows), base))

    def conf(t):
        tp = sum(1 for r in rows if r["headroom"] <= t and r["win"])
        fp = sum(1 for r in rows if r["headroom"] <= t and not r["win"])
        fn = sum(1 for r in rows if r["headroom"] > t and r["win"])
        tn = sum(1 for r in rows if r["headroom"] > t and not r["win"])
        return tp, fp, fn, tn

    print("\n%-22s %6s  %s" % ("rule", "acc", "TP / FP / FN / TN"))
    out = {"cells": rows, "positives": pos, "baseline": base, "splits": []}
    for t in (0.9, 1.0, 1.1, 1.5, 2.0):
        tp, fp, fn, tn = conf(t)
        acc = (tp + tn) / len(rows)
        out["splits"].append({"threshold": t, "acc": acc, "tp": tp, "fp": fp,
                              "fn": fn, "tn": tn})
        print("headroom <= %-10.1f %6.3f  %2d / %2d / %2d / %2d" % (
            t, acc, tp, fp, fn, tn))

    tp, fp, fn, tn = conf(1.0)
    out["rule"] = {"threshold": 1.0, "acc": (tp + tn) / len(rows),
                   "false_positives": fp,
                   "sufficient_not_necessary": fp == 0 and fn > 0}
    out["spearman_headroom_win"] = _spearman(
        [r["headroom"] for r in rows], [1.0 if r["win"] else 0.0 for r in rows])
    print("\nAt the mechanistic threshold (headroom <= 1.0): %d false positives."
          % fp)
    if fp == 0 and fn > 0:
        print("So it is SUFFICIENT, not necessary -- capacity covering the")
        print("working set always wins, and sometimes less than that does too.")
    print("\nmisses (demand wins above the threshold):")
    for r in rows:
        if r["headroom"] > 1.0 and r["win"]:
            print("  %-8s %-9s frac=%.2f headroom=%.2f margin=%+5.1f%%" % (
                r["model"], r["prompt"], r["frac"], r["headroom"],
                r["margin"] * 100))
    print("\nspearman(headroom, wins) over %d cells: %.3f "
          "(it was 0.055 over 24)" % (len(rows), out["spearman_headroom_win"]))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=2)
        print("\nreceipt ->", a.out)


if __name__ == "__main__":
    main()
