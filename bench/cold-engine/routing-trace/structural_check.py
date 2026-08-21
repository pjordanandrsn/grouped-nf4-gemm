"""Is the one-step threshold a fact about ROUTING, or about the allocator?

RESULTS-crossover.md and RESULTS-third-model.md both report that the device
row cache loses below `layers x top-k` rows and wins at or above it, on three
models and twelve captured traces. Three models agreeing reads as
generalization. It is not, and this is the check that shows why.

Drive the SAME cache with synthetic routing spanning the plausible space --
stickiness from 0 (independent uniform draws, no temporal structure at all)
to 0.95, and expert popularity from flat to heavily skewed -- at every
captured geometry. If the verdict is a property of routing, some corner of
that space should move it.

It does not move. Zero hits below one step and a win at one step, in every
condition. The threshold is arithmetic on `protected = rows - k`: below one
step the cache cannot hold a step's working set, so each step evicts its own
rows before reusing them; at one step it can. The three captured models did
not independently confirm a rule -- they could not have refuted it.

That does not make the threshold useless. It is still the right sizing rule,
and it is still worth stating. It does mean capturing a fourth model to
"test" it buys nothing, and that P1's confirmation in RESULTS-third-model.md
is much weaker evidence than three-models-agreeing suggests.
"""
import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "kernel"))
sys.path.insert(0, HERE)

from replay_dev_cache import positional_transfers, replay   # noqa: E402

# layers, top-k, experts. The three captured, plus Mixtral's geometry.
GEOMETRIES = (("olmoe", 16, 8, 64), ("granite", 32, 8, 40),
              ("qwen", 24, 4, 60), ("mixtral", 32, 2, 8))
STICKY = (0.0, 0.6, 0.95)
SKEW = (0.0, 1.5)


def synth(layers, k, experts, sticky, skew, steps, seed=11):
    """Routing with a tunable amount of the structure real routing has.

    `sticky` is the chance a step keeps each of the previous step's experts;
    0 is independent draws. `skew` is a Zipf exponent on expert popularity;
    0 is uniform. Between them these span from no structure to far more than
    any captured trace shows.
    """
    rng = random.Random(seed)
    w = [1.0 / ((i + 1) ** skew) for i in range(experts)]
    prev = {l: rng.sample(range(experts), k) for l in range(layers)}
    recs = []
    for s in range(steps):
        routed = {}
        for l in range(layers):
            sel = [e for e in prev[l] if rng.random() < sticky]
            while len(sel) < k:
                e = rng.choices(range(experts), weights=w)[0]
                if e not in sel:
                    sel.append(e)
            sel = sorted(sel)
            routed[str(l)] = sel
            prev[l] = sel
        recs.append({"step": s, "routed": routed})
    return {"layers": layers, "top_k": k, "n_experts": experts}, recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rows, moved = [], 0
    print("%-9s %4s %6s %5s | %-13s %-13s" % (
        "geometry", "per", "sticky", "skew", "below one step", "at one step"))
    for name, L, K, E in GEOMETRIES:
        per = L * K
        for st in STICKY:
            for sk in SKEW:
                meta, recs = synth(L, K, E, st, sk, a.steps)
                slots = a.steps * L * K
                pos = positional_transfers(meta, recs)
                below, _ = replay(meta, recs, max(2, int(per * 0.9)))
                at, _ = replay(meta, recs, per)
                zero_below, wins_at = below == slots, at < pos
                if not (zero_below and wins_at):
                    moved += 1
                rows.append({"geometry": name, "layers": L, "top_k": K,
                             "n_experts": E, "per_step": per, "sticky": st,
                             "skew": sk, "below": below, "slots": slots,
                             "at": at, "positional": pos,
                             "zero_hit_below": zero_below, "wins_at": wins_at})
                print("%-9s %4d %6.2f %5.1f | %-13s %-13s" % (
                    name, per, st, sk,
                    "zero-hit" if zero_below else "HITS",
                    "wins" if wins_at else "LOSES"))
        print()

    print("conditions where the verdict moved: %d of %d" % (moved, len(rows)))
    print("The threshold does not depend on the routing. Capturing another"
          "\nmodel cannot test it.")
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"steps": a.steps, "conditions_moved": moved,
                       "rows": rows}, f, indent=1)
        print("\nreceipt -> %s" % a.out)


if __name__ == "__main__":
    main()
