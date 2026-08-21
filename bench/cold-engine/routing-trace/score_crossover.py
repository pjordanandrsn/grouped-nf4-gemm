"""Why the device cache fails below one decode step -- and it is not capacity.

RESULTS-concentration.md found `steps_held = capacity / (layers x top-k)`
separated every configuration where the cache helped from every one where it
lost, and explained it as "a cache smaller than one step is evicted before its
own next request, so it retains nothing". That explanation is testable in the
same way the concentration story was, and it turns out to be half right: the
threshold is real and the reason is not capacity, it is LRU.

Routing per step is a near-cyclic scan of the same layers x top-k rows.
**LRU on a cyclic reference pattern with capacity below the cycle length is
the textbook worst case: zero hits, every time.** So is FIFO. The prediction
that distinguishes "capacity" from "policy" is that a policy WITHOUT the
pathology should get hits in exactly the regime where these get none, and
should not beat them above the threshold.
"""
import argparse
import json
import os
import random
import sys
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "kernel"))
sys.path.insert(0, HERE)

from replay_dev_cache import positional_transfers                 # noqa: E402
from score_policies import load                                   # noqa: E402
from score_policies import steps_capacity  # noqa: E402


def simulate(recs, cap, policy, seed=0):
    """Transfers under one eviction policy. Keys are (layer, expert)."""
    rng = random.Random(seed)
    if policy == "lru":
        c, fills = OrderedDict(), 0
        for r in recs:
            for L, ex in r["routed"].items():
                for e in ex:
                    k = (int(L), e)
                    if k in c:
                        c.move_to_end(k)
                        continue
                    fills += 1
                    c[k] = 1
                    if len(c) > cap:
                        c.popitem(last=False)
        return fills
    c, order, fills = set(), [], 0
    for r in recs:
        for L, ex in r["routed"].items():
            for e in ex:
                k = (int(L), e)
                if k in c:
                    continue
                fills += 1
                if len(c) >= cap:
                    victim = (rng.choice(list(c)) if policy == "random"
                              else order.pop(0))
                    c.discard(victim)
                    if policy == "fifo" and victim in order:
                        order.remove(victim)
                c.add(k)
                if policy == "fifo":
                    order.append(k)
    return fills


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=HERE)
    ap.add_argument("--models", default="olmoe:128,granite:256",
                    help="name:rows_per_step (layers x top-k)")
    ap.add_argument("--prompts", default="prose,code,math,dialogue")
    ap.add_argument("--steps-held", default="0.5,0.75,0.9,1.0,1.25,1.5")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    models = dict((s.split(":")[0], int(s.split(":")[1]))
                  for s in a.models.split(","))
    shs = [float(x) for x in a.steps_held.split(",")]

    out, zero_lru, zero_rand = {"points": []}, 0, 0
    print("%-8s %-9s %6s %5s | %8s %8s %8s | %8s %s" % (
        "model", "prompt", "steps", "cap", "LRU", "FIFO", "RANDOM",
        "positional", "notes"))
    for m, per in models.items():
        for p in a.prompts.split(","):
            meta, recs = load(os.path.join(a.dir, "%s_%s.jsonl" % (m, p)))
            routed = sum(len(v) for r in recs for v in r["routed"].values())
            pos = positional_transfers(meta, recs)
            for sh in shs:
                cap = steps_capacity(per, sh)
                res = {q: simulate(recs, cap, q)
                       for q in ("lru", "fifo", "random")}
                zero_lru += res["lru"] == routed
                zero_rand += res["random"] == routed
                note = ""
                if res["lru"] == routed:
                    note = "LRU: ZERO hits"
                    if res["random"] < routed:
                        note += "; random still hits"
                out["points"].append(dict(model=m, prompt=p, steps_held=sh,
                                          cap=cap, routed=routed,
                                          positional=pos, **res))
                print("%-8s %-9s %6.2f %5d | %8d %8d %8d | %8d %s" % (
                    m, p, sh, cap, res["lru"], res["fifo"], res["random"],
                    pos, note))
        print()
    below = [x for x in out["points"] if x["steps_held"] < 1.0]
    above = [x for x in out["points"] if x["steps_held"] >= 1.0]
    out["summary"] = {
        "below_one_step": len(below),
        "lru_zero_hits_below": sum(1 for x in below if x["lru"] == x["routed"]),
        "fifo_zero_hits_below": sum(1 for x in below if x["fifo"] == x["routed"]),
        "random_zero_hits_below": sum(1 for x in below
                                      if x["random"] == x["routed"]),
        "lru_best_above": sum(1 for x in above
                              if x["lru"] <= min(x["fifo"], x["random"])),
        "above_one_step": len(above)}
    s = out["summary"]
    print("below one step (%d cells): LRU zero-hit in %d, FIFO in %d, "
          "RANDOM in %d" % (s["below_one_step"], s["lru_zero_hits_below"],
                            s["fifo_zero_hits_below"],
                            s["random_zero_hits_below"]))
    print("at or above one step (%d cells): LRU is best in %d" % (
        s["above_one_step"], s["lru_best_above"]))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=2)
        print("\nreceipt ->", a.out)


if __name__ == "__main__":
    main()
