"""Is concentration actually the variable, or was that a story after the fact?

`RESULTS-generalization.md` asserts that every conclusion which broke, broke
the way *concentration* predicts -- how much of the arena a generation
actually touches. That is an explanation offered after seeing the results,
which is exactly the kind of claim that deserves to be scored rather than
believed.

So: define concentration WITHOUT reference to any outcome, then ask whether
it predicts the outcomes across all eight traces.

Three measures, none of them tuned:

  coverage   distinct (layer, expert) pairs touched / arena size. The
             coarsest thing that could matter.
  headroom   coverage relative to the cache -- pairs touched / capacity.
             Below 1 the cache can hold everything it will ever be asked for.
  entropy    Shannon entropy of the routing distribution over pairs, in bits,
             normalised by log2(arena). Coverage counts what is touched;
             entropy weights it by how often, which is what a frequency-ranked
             cache actually exploits.

Predictions scored against three outcomes measured elsewhere:

  cache_ratio     device-cache transfers / positional-cache transfers
                  (<1 means the expert-keyed cache is worth having)
  demand_wins     does LRU beat static placement
  gate3_gain      adaptive re-placement vs static

Spearman rank correlation, tie-averaged, with the sign each hypothesis
predicts stated up front so a wrong-signed correlation reads as a refutation
rather than a discovery.
"""
import argparse
import json
import math
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "kernel"))
sys.path.insert(0, HERE)

from replay_dev_cache import positional_transfers, replay      # noqa: E402
from reuse_profile import _spearman                            # noqa: E402
from score_policies import demand_p, load, static_p, adaptive_p  # noqa: E402


def measures(recs, warm, arena, cap):
    ev = recs[warm:]
    c = Counter((int(L), e) for r in ev for L, ex in r["routed"].items()
                for e in ex)
    total = sum(c.values())
    ent = -sum((v / total) * math.log2(v / total) for v in c.values())
    # Rows one decode STEP asks for: every layer contributes its top-k. This
    # is the quantity a cache has to hold before it can retain anything at
    # all across steps, and it is not proportional to the arena -- a model
    # with twice the layers demands twice the rows per step from an arena
    # only a quarter larger.
    per_step = sum(len(v) for v in ev[0]["routed"].values())
    return {"coverage": len(c) / arena,
            "headroom": len(c) / cap,
            "entropy": ent / math.log2(arena),
            "steps_held": cap / per_step,
            "per_step_rows": per_step}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=HERE)
    ap.add_argument("--models", default="olmoe:1024,granite:1280")
    ap.add_argument("--prompts", default="prose,code,math,dialogue")
    ap.add_argument("--fracs", default="0.125,0.375,0.5")
    ap.add_argument("--warm", type=int, default=256)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    models = dict((s.split(":")[0], int(s.split(":")[1]))
                  for s in a.models.split(","))
    prompts = a.prompts.split(",")
    fracs = [float(x) for x in a.fracs.split(",")]

    cells = []
    print("%-8s %-9s %5s | %8s %8s %8s %10s | %10s %7s %9s" % (
        "model", "prompt", "cap", "coverage", "headroom", "entropy",
        "steps_held", "cache/pos", "demand", "gate3"))
    for m, arena in models.items():
        for p in prompts:
            meta, recs = load(os.path.join(a.dir, "%s_%s.jsonl" % (m, p)))
            warm, ev = recs[:a.warm], recs[a.warm:]
            pos = positional_transfers(meta, recs)
            for fr in fracs:
                cap = int(arena * fr)
                mm = measures(recs, a.warm, arena, cap)
                fills, _ = replay(meta, recs, cap)
                st = static_p(warm, ev, cap)[0]
                dm = demand_p(warm, ev, cap)[0]
                ad = adaptive_p(warm, ev, cap, period=32)[0]
                cell = dict(model=m, prompt=p, cap=cap, **mm,
                            cache_ratio=fills / pos,
                            demand_wins=1.0 if dm < st else 0.0,
                            demand_vs_static=dm / st,
                            gate3_gain=(ad - st) / st)
                cells.append(cell)
                print("%-8s %-9s %5d | %8.3f %8.3f %8.3f %10.2f | %10.3f "
                      "%7s %+8.3f%s" % (
                          m, p, cap, mm["coverage"], mm["headroom"],
                          mm["entropy"], mm["steps_held"], cell["cache_ratio"],
                          "yes" if dm < st else "no", cell["gate3_gain"],
                          "  <-- cache LOSES" if fills > pos else ""))

    # Signs are stated BEFORE the numbers: more concentration (lower coverage,
    # lower headroom, lower entropy) should mean a cache that helps more
    # (lower ratio), demand-paging more likely to win, and less for adaptive
    # re-placement to find.
    hyps = [("steps_held", "cache_ratio", -1,
             "cache holds more than one step -> it can retain, so it helps"),
            ("coverage", "cache_ratio", +1,
             "less coverage -> cache helps more"),
            ("headroom", "cache_ratio", +1,
             "cache holds more of the working set -> helps more"),
            ("entropy", "cache_ratio", +1,
             "flatter routing -> cache helps less"),
            ("headroom", "demand_vs_static", +1,
             "cache covers the working set -> demand-paging wins"),
            ("coverage", "gate3_gain", -1,
             "less coverage -> more for re-placement to gain")]
    print("\n%-10s %-18s %6s %8s   %s" % ("measure", "outcome", "sign",
                                          "rho", "verdict"))
    out = {"cells": cells, "hypotheses": []}
    for meas, outc, sign, why in hyps:
        rho = _spearman([c[meas] for c in cells], [c[outc] for c in cells])
        ok = (rho * sign) > 0.5
        out["hypotheses"].append({"measure": meas, "outcome": outc,
                                  "predicted_sign": sign, "rho": rho,
                                  "supported": bool(ok), "why": why})
        print("%-10s %-18s %+6d %8.3f   %s  (%s)" % (
            meas, outc, sign, rho,
            "SUPPORTED" if ok else "not supported", why))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=2)
        print("\nreceipt ->", a.out)


if __name__ == "__main__":
    main()
