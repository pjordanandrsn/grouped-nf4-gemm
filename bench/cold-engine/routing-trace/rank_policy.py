"""Evicting by the router's rank at each layer's last visit.

Registered in bench/cold-engine/PREREG-router-rank.md as S2. S1 -- the premise
that rank predicts recurrence -- was REFUTED as registered (the bar was 1.5x
on all four models; Qwen came in at 1.37x), so this is EXPLORATORY, not a
registered result. It is run anyway because the prereg's stated reason for
skipping it was "a policy cannot exploit a signal that is not there", and the
signal is there: the rank profile is monotone on all four models and clears
1.5x on three.

The policy is online. At layer m's last visit the engine saw that layer's full
ranking, so "rank at last visit" is information it already has; only the cache
ignores it. Lower rank (higher score) is retained, higher rank evicted, LRU
as the tie-break.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "kernel"))
sys.path.insert(0, HERE)

from oracle_headroom import belady, keystream                    # noqa: E402
from policy_headroom import run_policy                           # noqa: E402
from score_policies import load, steps_capacity                  # noqa: E402


def rank_transfers(meta, recs, cap):
    """Transfers when the victim is the resident with the WORST last-seen rank."""
    resident = {}                      # (layer, expert) -> [rank, clock]
    fills = clock = 0
    for r in recs:
        ranked = r.get("routed_rank") or r["routed"]
        for lay, order in ranked.items():
            m = int(lay)
            for pos, e in enumerate(order):
                clock += 1
                k = (m, e)
                if k in resident:
                    resident[k] = [pos, clock]      # refresh rank AND recency
                    continue
                fills += 1
                if len(resident) >= cap:
                    # worst rank first; oldest breaks the tie
                    v = max(resident, key=lambda kk: (resident[kk][0],
                                                      -resident[kk][1]))
                    del resident[v]
                resident[k] = [pos, clock]
    return fills


def validate(verbose=True):
    import random
    rng = random.Random(17)
    ok = True

    def say(n, p, d=""):
        nonlocal ok
        ok &= p
        if verbose:
            print("  %-50s %s %s" % (n, "PASS" if p else "FAIL", d))

    def mk(L, k, E, steps, seed):
        g = random.Random(seed)
        rs = []
        for _ in range(steps):
            d = {str(l): g.sample(range(E), k) for l in range(L)}
            rs.append({"routed": {l: sorted(v) for l, v in d.items()},
                       "routed_rank": d})
        return {"layers": L, "top_k": k, "n_experts": E}, rs

    for L, k, E in ((4, 2, 8), (16, 8, 64)):
        meta, recs = mk(L, k, E, 200, 1)
        cap = L * E
        a = rank_transfers(meta, recs, cap)
        l = run_policy(recs, cap, "lru")
        b = belady(keystream(recs), cap)
        say("cap >= key space: rank = lru = belady (L=%d)" % L,
            a == l == b, "rank=%d lru=%d bel=%d" % (a, l, b))

    bad = None
    for _ in range(30):
        L, k, E = rng.randint(2, 20), rng.randint(1, 6), rng.randint(8, 40)
        meta, recs = mk(L, k, E, rng.randint(20, 120), rng.randrange(10 ** 6))
        cap = rng.randint(2, max(3, L * k))
        a = rank_transfers(meta, recs, cap)
        allmiss = sum(len(v) for r in recs for v in r["routed"].values())
        b = belady(keystream(recs), cap)
        if not (b <= a <= allmiss):
            bad = (L, k, cap, a, b, allmiss)
            break
    say("belady <= rank <= all-miss (30 random)", bad is None, str(bad or ""))

    # k = 1 makes every resident's rank 0, so the policy must collapse onto
    # its tie-break exactly. The degenerate case that catches a comparison
    # with the wrong sign.
    bad = None
    for _ in range(20):
        L, E = rng.randint(2, 12), rng.randint(6, 40)
        meta, recs = mk(L, 1, E, rng.randint(20, 200), rng.randrange(10 ** 6))
        cap = rng.randint(2, max(3, L - 1))
        a = rank_transfers(meta, recs, cap)
        l = run_policy(recs, cap, "lru")
        if a != l:
            bad = (L, E, cap, a, l)
            break
    say("k=1 (all ranks equal) reduces exactly to LRU", bad is None,
        str(bad or ""))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=HERE)
    ap.add_argument("--models", default="olmoe,granite,qwen,gptoss")
    ap.add_argument("--prompts", default="prose,code,math,dialogue")
    ap.add_argument("--steps-held", default="1.0,1.5,2.0")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    print("rank policy validation")
    if not validate():
        sys.exit("VALIDATION FAILED -- implementation is being scored.")
    if a.validate_only:
        return

    rows = []
    print("\n%-8s %-9s %5s %5s | %9s %9s %9s %9s"
          % ("model", "prompt", "held", "cap", "lru", "lfu", "rank", "belady"))
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
                rk = rank_transfers(meta, recs, cap)
                b = belady(keys, cap)
                rows.append({"model": m, "prompt": p, "steps_held": sh,
                             "cap": cap, "lru": l, "lfu": lf, "rank": rk,
                             "belady": b})
                print("%-8s %-9s %5.2f %5d | %9d %9d %9d %9d"
                      % (m, p, sh, cap, l, lf, rk, b))
            print()
    if a.out:
        with open(a.out, "w") as fh:
            json.dump({"rows": rows}, fh, indent=1)
        print("receipt -> %s" % a.out)


if __name__ == "__main__":
    main()
