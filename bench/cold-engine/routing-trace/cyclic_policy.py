"""Evicting by position in the layer cycle.

Registered in bench/cold-engine/PREREG-cyclic-policy.md.

A decode step walks the layers in a fixed order, once each, so at layer `cur`
the distance to layer `m`'s next visit is known exactly and without any future
knowledge:

    m >  cur :  m - cur              (later in THIS step)
    m <= cur :  (L - cur) + m        (not until the NEXT step)

LRU evicts the least-recently-used row, which under this access order belongs
to the previous step's highest layers -- the rows needed SOONEST. That is the
mechanism behind the zero-hit region in RESULTS-crossover.md, and this policy
is the direct correction: Belady's rule restricted to the component that is
structurally known.

What it does not know is whether an expert recurs at all; the cyclic distance
is to the layer's next VISIT, not to the row's next USE. It converts an
unknown about experts into a certainty about layers, and is exactly as wrong
as "this expert recurs" is wrong.

Residents are bucketed per layer because every row of a layer shares one
distance: eviction then costs O(L) rather than O(capacity), and L is 16-32
where capacity reaches 512.
"""
import argparse
import json
import os
import sys
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "kernel"))
sys.path.insert(0, HERE)

from oracle_headroom import belady, keystream                    # noqa: E402
from policy_headroom import run_policy                           # noqa: E402
from score_policies import load, steps_capacity                  # noqa: E402


def cyclic_transfers(meta, recs, cap):
    """Transfers under cyclic eviction, LRU as the within-layer tie-break."""
    L = int(meta["layers"])
    buckets = [OrderedDict() for _ in range(L)]   # layer -> {expert: None}, LRU first
    n_res = 0
    fills = clock = 0
    for r in recs:
        for lay, experts in r["routed"].items():
            cur = int(lay)
            for e in experts:
                clock += 1
                b = buckets[cur]
                if e in b:
                    b.move_to_end(e)              # hit
                    continue
                fills += 1
                if n_res >= cap:
                    # Furthest next VISIT wins; every row in a bucket shares
                    # the bucket's distance, so the victim is that bucket's
                    # LRU entry.
                    worst, wd = -1, -1
                    for m in range(L):
                        if not buckets[m]:
                            continue
                        d = (m - cur) if m > cur else (L - cur + m)
                        if d > wd:
                            worst, wd = m, d
                    buckets[worst].popitem(last=False)
                    n_res -= 1
                b[e] = None
                n_res += 1
    return fills


def validate(verbose=True):
    """Preregistered preconditions. A failure scores the implementation."""
    import random
    rng = random.Random(11)
    ok = True

    def say(name, passed, detail=""):
        nonlocal ok
        ok &= passed
        if verbose:
            print("  %-52s %s %s" % (name, "PASS" if passed else "FAIL", detail))

    def mk(L, k, E, steps, seed):
        g = random.Random(seed)
        return ({"layers": L, "top_k": k, "n_experts": E},
                [{"routed": {str(l): g.sample(range(E), k) for l in range(L)}}
                 for _ in range(steps)])

    # 1. capacity >= key space: nothing is ever evicted.
    for L, k, E in ((4, 2, 8), (16, 8, 64)):
        meta, recs = mk(L, k, E, 200, 1)
        cap = L * E
        c = cyclic_transfers(meta, recs, cap)
        l = run_policy(recs, cap, "lru")
        b = belady(keystream(recs), cap)
        say("cap >= key space: cyclic = lru = belady (L=%d)" % L,
            c == l == b, "cyclic=%d lru=%d bel=%d" % (c, l, b))

    # 2. bounded by Belady below and all-miss above.
    bad = None
    for _ in range(30):
        L, k, E = rng.randint(2, 20), rng.randint(1, 6), rng.randint(8, 40)
        meta, recs = mk(L, k, E, rng.randint(20, 120), rng.randrange(10 ** 6))
        cap = rng.randint(2, max(3, L * k))
        c = cyclic_transfers(meta, recs, cap)
        allmiss = sum(len(v) for r in recs for v in r["routed"].values())
        b = belady(keystream(recs), cap)
        if not (b <= c <= allmiss):
            bad = (L, k, cap, c, b, allmiss)
            break
    say("belady <= cyclic <= all-miss (30 random)", bad is None,
        "" if bad is None else str(bad))

    # 3. L = 1 makes every distance equal, so cyclic must reduce EXACTLY to
    #    its tie-break. This is the cheapest way to catch a distance computed
    #    with the wrong sign: a flipped comparison still looks plausible on
    #    real traces and is exact here.
    bad = None
    for _ in range(20):
        E = rng.randint(4, 30)
        meta, recs = mk(1, rng.randint(1, 3), E, rng.randint(20, 200),
                        rng.randrange(10 ** 6))
        cap = rng.randint(2, max(3, E - 1))
        c = cyclic_transfers(meta, recs, cap)
        l = run_policy(recs, cap, "lru")
        if c != l:
            bad = (E, cap, c, l)
            break
    say("L=1 reduces exactly to LRU (20 random)", bad is None,
        "" if bad is None else "E=%d cap=%d cyclic=%d lru=%d" % bad)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=HERE)
    ap.add_argument("--models", default="olmoe,granite,qwen")
    ap.add_argument("--prompts", default="prose,code,math,dialogue")
    ap.add_argument("--steps-held", default="1.0,1.5,2.0")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    print("cyclic implementation validation (preregistered precondition)")
    if not validate():
        sys.exit("VALIDATION FAILED -- implementation is being scored, not "
                 "the policy. Nothing else is reported.")
    if a.validate_only:
        return

    # Imported after validation, not at module scope: validate() must be
    # runnable without a sibling policy present. A validation that cannot run
    # because an unrelated import failed is a validation that gets skipped.
    from adaptive_policy import arc_transfers

    rows = []
    print("\n%-8s %-9s %5s %5s | %9s %9s %9s %9s %9s"
          % ("model", "prompt", "held", "cap", "lru", "lfu", "arc", "cyclic",
             "belady"))
    for m in a.models.split(","):
        for p in a.prompts.split(","):
            f = os.path.join(a.dir, "%s_%s.jsonl" % (m, p))
            if not os.path.exists(f):
                continue
            meta, recs = load(f)
            per = meta["layers"] * meta["top_k"]
            keys = keystream(recs)
            for sh in [float(x) for x in a.steps_held.split(",")]:
                cap = steps_capacity(per, sh)
                l = run_policy(recs, cap, "lru")
                lf = run_policy(recs, cap, "lfu")
                ar, _ = arc_transfers(keys, cap)
                cy = cyclic_transfers(meta, recs, cap)
                b = belady(keys, cap)
                rows.append({"model": m, "prompt": p, "steps_held": sh,
                             "cap": cap, "lru": l, "lfu": lf, "arc": ar,
                             "cyclic": cy, "belady": b})
                print("%-8s %-9s %5.2f %5d | %9d %9d %9d %9d %9d"
                      % (m, p, sh, cap, l, lf, ar, cy, b))
            print()
    if a.out:
        with open(a.out, "w") as fh:
            json.dump({"rows": rows}, fh, indent=1)
        print("receipt -> %s" % a.out)


if __name__ == "__main__":
    main()
