"""Which conclusions are about MoE routing, and which were about one prompt?

Every offline result in this campaign replayed a single captured trace, and
"one prompt" is the limit printed at the top of each of them. Four traces now
exist -- prose, code, mathematics, dialogue, same model, same decode shape --
so each conclusion can be re-run against generations that are unlike each
other, and the ones that do not survive can be said so.

Emits one receipt covering four claims:

  R4          long-run frequency beats short-window recurrence
  dev-cache   the expert-keyed device cache beats the positional one already
              in the engine
  gate-3      adaptive re-placement beats static placement
  policy      EWMA (decayed counts) is the better re-placement policy
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "kernel"))
sys.path.insert(0, HERE)

from replay_dev_cache import positional_transfers, replay      # noqa: E402
from score_policies import (adaptive_p, demand_p, hybrid_p,    # noqa: E402
                            load, oracle_p, static_p)
from score_r4 import score as r4_score                          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=HERE)
    ap.add_argument("--models", default="olmoe:1024,granite:1280",
                    help="name:arena_pairs. Capacity is taken as a FRACTION "
                         "of the arena so models with different expert counts "
                         "are compared at the same pressure, not the same "
                         "absolute row count.")
    ap.add_argument("--prompts", default="prose,code,math,dialogue")
    ap.add_argument("--fracs", default="0.125,0.375,0.5")
    ap.add_argument("--warm", type=int, default=256)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    prompts = a.prompts.split(",")
    fracs = [float(x) for x in a.fracs.split(",")]
    models = {}
    for spec in a.models.split(","):
        nm, tot = spec.split(":")
        models[nm] = int(tot)
    out = {"models": models, "prompts": prompts, "fracs": fracs,
           "warm": a.warm, "claims": {}}
    keys = [(m, p) for m in models for p in prompts]

    traces = {}
    print("%-9s %-9s %7s %9s  working set of steps %d-end" % (
        "model", "prompt", "steps", "distinct", a.warm))
    for m, p in keys:
        meta, recs = load(os.path.join(a.dir, "%s_%s.jsonl" % (m, p)))
        traces[(m, p)] = (meta, recs)
        ev = recs[a.warm:]
        ks = {(int(L), e) for r in ev for L, ex in r["routed"].items()
              for e in ex}
        print("%-9s %-9s %7d %9d  (of %d)" % (m, p, len(recs), len(ks),
                                              models[m]))
        out.setdefault("working_set", {})["%s/%s" % (m, p)] = len(ks)

    # ---- R4 -------------------------------------------------------------
    print("\nR4 -- frequency vs SHORT-window recurrence (w=4, w=8)")
    tally, rows = {"frequency": 0, "recency": 0}, []
    for m, p in keys:
        _meta, recs = traces[(m, p)]
        for w in (4, 8):
            for cap in [int(models[m] * f) for f in fracs]:
                sc, n, _ = r4_score(recs, cap, w)
                r, f = sc.get("recency"), sc.get("frequency")
                if r is None or f is None or max(abs(r), abs(f)) < 0.15:
                    who = "no signal"
                else:
                    who = "recency" if r > f else "frequency"
                    tally[who] += 1
                rows.append({"model": m, "prompt": p, "window": w, "rows": cap,
                             "recency": r, "frequency": f, "winner": who})
    out["claims"]["r4"] = {"cells": rows, "tally": tally,
                           "holds": tally["frequency"] > tally["recency"]}
    print("  signal-bearing cells: %s -> R4 %s" % (
        tally, "REFUTED on every prompt" if tally["frequency"] > tally["recency"]
        else "NOT settled"))

    # ---- device row cache ------------------------------------------------
    print("\ndev-cache -- expert-keyed vs the positional cache in the engine")
    dc = []
    for m, p in keys:
        meta, recs = traces[(m, p)]
        pos = positional_transfers(meta, recs)
        for cap in [int(models[m] * f) for f in fracs]:
            fills, _st = replay(meta, recs, cap)
            dc.append({"model": m, "prompt": p, "rows": cap, "positional": pos,
                       "cache": fills, "vs_positional": fills / pos})
            print("  %-8s %-9s rows=%-4d %6.1f%% of positional%s" % (
                m, p, cap, fills / pos * 100,
                "   <-- WORSE" if fills > pos else ""))
    out["claims"]["dev_cache"] = {"cells": dc,
                                  "holds": all(c["vs_positional"] < 1 for c in dc)}

    # ---- gate 3 and the policy choice ------------------------------------
    print("\ngate-3 / policy -- adaptive vs static, and which policy wins")
    g3, wins = [], {}
    for m, p in keys:
        _meta, recs = traces[(m, p)]
        warm, ev = recs[:a.warm], recs[a.warm:]
        for cap in [int(models[m] * f) for f in fracs]:
            s = static_p(warm, ev, cap)[0]
            ad = adaptive_p(warm, ev, cap, period=32)[0]
            ew = adaptive_p(warm, ev, cap, period=32, decay=0.5)[0]
            hy = min(hybrid_p(warm, ev, cap, period=32, pinned_frac=f)[0]
                     for f in (0.5, 0.75, 0.9, 1.0))
            de = demand_p(warm, ev, cap)[0]
            orc = oracle_p(warm, ev, cap)[0]
            cands = {"adaptive": ad, "ewma": ew, "hybrid": hy, "demand": de}
            w = min(cands, key=cands.get)
            wins[w] = wins.get(w, 0) + 1
            g3.append({"model": m, "prompt": p, "rows": cap, "static": s, "adaptive": ad,
                       "ewma": ew, "hybrid": hy, "demand": de, "oracle": orc,
                       "adaptive_vs_static_pct": (ad - s) / s * 100,
                       "winner": w})
            print("  %-8s %-9s rows=%-4d adaptive vs static %+6.1f%%  winner: %s%s" % (
                m, p, cap, (ad - s) / s * 100, w,
                "   <-- NOT BETTER" if ad >= s else ""))
    out["claims"]["gate3"] = {
        "cells": g3,
        "adaptive_beats_static_everywhere":
            all(c["adaptive_vs_static_pct"] < 0 for c in g3),
        "gain_range_pct": [min(c["adaptive_vs_static_pct"] for c in g3),
                           max(c["adaptive_vs_static_pct"] for c in g3)]}
    out["claims"]["policy"] = {"winners": wins,
                               "ewma_wins_everywhere": wins.get("ewma", 0) == len(g3)}
    print("\n  policy winners across %d cells: %s" % (len(g3), wins))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=2)
        print("\nreceipt ->", a.out)


if __name__ == "__main__":
    main()
