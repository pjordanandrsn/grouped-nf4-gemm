"""Score PREREG-third-model.md against the Qwen1.5-MoE-A2.7B traces.

Runs the three registered predictions and emits one receipt. The sweeps are
the ones the preregistration names -- they are not re-chosen here.

P1 is scored against the REAL DevRowCache, not the standalone LRU simulation
in score_crossover.py, because the prediction is about "the expert-keyed
device cache" and the two disagree at exactly the predicted boundary. The
simulation counts a hit at capacity == one step where the cache takes none;
scoring the simulation would have recorded a pass the cache does not earn.
"""
import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "kernel"))
sys.path.insert(0, HERE)

from replay_dev_cache import positional_transfers, replay   # noqa: E402
from score_policies import demand_p, load, static_p         # noqa: E402
from score_demand import counts                             # noqa: E402

PROMPTS = ("prose", "code", "math", "dialogue")
STEPS_HELD = (0.5, 0.75, 0.9, 1.0, 1.25, 1.5)               # P1, as registered
FRACS = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,     # P2, as registered
         0.8, 0.9, 1.0)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def p1(d, model, extra=(), prompts=PROMPTS):
    """Cache vs positional across the registered steps_held sweep."""
    cells, zero_below = [], []
    for p in prompts:
        meta, recs = load(os.path.join(d, "%s_%s.jsonl" % (model, p)))
        per = meta["layers"] * meta["top_k"]
        pos = positional_transfers(meta, recs)
        for sh in STEPS_HELD:
            cap = max(2, int(round(per * sh)))
            f, _ = replay(meta, recs, cap)
            # A cache that retains nothing transfers one row per routed
            # row-slot; that is the zero-hit signature P1b names.
            routed = sum(len(ex) for r in recs for ex in r["routed"].values())
            cell = {"prompt": p, "steps_held": sh, "cap": cap, "per_step": per,
                    "cache": f, "positional": pos, "ratio": f / pos,
                    "routed_rowslots": routed, "hits": routed - f}
            cells.append(cell)
            if sh < 1.0:
                zero_below.append(cell["hits"] == 0)
        for cap in extra:                      # locate the true crossover
            f, _ = replay(meta, recs, cap)
            cells.append({"prompt": p, "steps_held": cap / per, "cap": cap,
                          "per_step": per, "cache": f, "positional": pos,
                          "ratio": f / pos, "probe": True})
    reg = [c for c in cells if not c.get("probe")]
    bad_below = [c for c in reg if c["steps_held"] < 1.0 and c["ratio"] < 1.0]
    bad_above = [c for c in reg if c["steps_held"] >= 1.0 and c["ratio"] > 1.0]
    return {"cells": cells,
            "violations_below": bad_below, "violations_above": bad_above,
            "P1": "CONFIRMED" if not (bad_below or bad_above) else "REFUTED",
            "P1b": "CONFIRMED" if all(zero_below) else "REFUTED",
            "P1b_zero_hit_cells": "%d of %d" % (sum(zero_below),
                                                len(zero_below))}


def p2(d, model, arena, warm=256, prompts=PROMPTS):
    cells = []
    for p in prompts:
        meta, recs = load(os.path.join(d, "%s_%s.jsonl" % (model, p)))
        wm, ev = recs[:warm], recs[warm:]
        ws = len(counts(ev))
        for fr in FRACS:
            cap = max(1, int(round(arena * fr)))
            s = static_p(wm, ev, cap)
            dm = demand_p(wm, ev, cap)
            cells.append({"prompt": p, "frac": fr, "cap": cap, "ws": ws,
                          "headroom": ws / cap, "static": s, "demand": dm,
                          "demand_wins": dm < s})
    fp = [c for c in cells if c["headroom"] <= 1.0 and not c["demand_wins"]]
    tp = [c for c in cells if c["headroom"] <= 1.0 and c["demand_wins"]]
    fn = [c for c in cells if c["headroom"] > 1.0 and c["demand_wins"]]
    return {"cells": cells, "false_positives": fp,
            "tp": len(tp), "fp": len(fp), "fn": len(fn),
            "tn": len(cells) - len(tp) - len(fp) - len(fn),
            "P2": "CONFIRMED" if not fp else "REFUTED"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=HERE)
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--arena", type=int, default=1440)
    ap.add_argument("--probe", default="97,98,99,100,104")
    ap.add_argument("--prompts", default=",".join(PROMPTS),
                    help="subset for robustness checks; the REGISTERED "
                         "result is all four and is what the receipt in "
                         "RESULTS-third-model.json holds")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    extra = [int(x) for x in a.probe.split(",") if x]

    prompts = tuple(x for x in a.prompts.split(",") if x)
    r1 = p1(a.dir, a.model, extra, prompts)
    r2 = p2(a.dir, a.model, a.arena, prompts=prompts)

    print("P1  -- crossover at layers x top-k rows")
    print("%-9s %6s %5s %10s %10s %8s %s"
          % ("prompt", "steps", "cap", "cache", "positional", "ratio", ""))
    for c in r1["cells"]:
        if c.get("probe"):
            continue
        bad = (c["steps_held"] < 1.0 and c["ratio"] < 1.0) or \
              (c["steps_held"] >= 1.0 and c["ratio"] > 1.0)
        print("%-9s %6.2f %5d %10d %10d %8.3f %s"
              % (c["prompt"], c["steps_held"], c["cap"], c["cache"],
                 c["positional"], c["ratio"], "VIOLATION" if bad else ""))
    print("\nP1:  %s   (%d violations below, %d at/above)"
          % (r1["P1"], len(r1["violations_below"]),
             len(r1["violations_above"])))
    print("P1b: %s   zero-hit in %s cells below threshold"
          % (r1["P1b"], r1["P1b_zero_hit_cells"]))

    print("\nP2  -- headroom <= 1 predicts demand beats static")
    print("  TP %d / FP %d / FN %d / TN %d"
          % (r2["tp"], r2["fp"], r2["fn"], r2["tn"]))
    for c in r2["false_positives"]:
        print("  FALSE POSITIVE %s frac=%.2f headroom=%.2f"
              % (c["prompt"], c["frac"], c["headroom"]))
    print("P2:  %s" % r2["P2"])

    if a.out:
        traces = {"%s_%s.jsonl" % (a.model, p):
                  sha(os.path.join(a.dir, "%s_%s.jsonl" % (a.model, p)))
                  for p in prompts}
        with open(a.out, "w") as f:
            json.dump({"prereg": "PREREG-third-model.md",
                       "prereg_sha256": sha(os.path.join(
                           a.dir, "PREREG-third-model.md")),
                       "model": a.model, "arena": a.arena,
                       "prompts": list(prompts),
                       "traces_sha256": traces,
                       "steps_held_sweep": list(STEPS_HELD),
                       "fracs_sweep": list(FRACS),
                       "P1": r1, "P2": r2}, f, indent=1)
        print("\nreceipt -> %s" % a.out)


if __name__ == "__main__":
    main()
