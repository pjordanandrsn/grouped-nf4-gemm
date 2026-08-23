"""ARC, and the checks that decide whether its numbers mean anything.

Registered in bench/cold-engine/PREREG-adaptive-policy.md. Five
single-variable explanations for when a frequency-aware rule beats a
recency-aware one are refuted, so the question is whether a policy can avoid
needing to know which applies.

ARC (Megiddo & Modha, FAST'03) keeps T1 (seen once, recency-ordered) and T2
(seen again, frequency-ish), plus ghost lists B1/B2 holding keys recently
evicted from each. A hit in B1 says recency is being under-served and grows
the target `p`; a hit in B2 says the opposite. It arbitrates exactly the axis
these models disagree on.

The implementation is deliberately a literal transcription of the paper's
case analysis rather than a tidied version: ARC is easy to get subtly wrong in
ways that still produce plausible transfer counts, and this file's job is to
be checkable, not elegant. `validate()` runs the invariants the
preregistration requires before any result is reported.
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


def arc_transfers(keys, cap, trace_invariants=False):
    """Transfers under ARC. Returns (fills, max_resident_plus_ghost).

    The second value exists only so `validate` can check ARC's own invariant
    |T1|+|T2|+|B1|+|B2| <= 2c, which a broken port violates silently while
    still returning a believable fill count.
    """
    c = cap
    T1, T2, B1, B2 = OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict()
    p = 0
    fills = 0
    worst = 0

    def replace(x_in_b2):
        # Paper's REPLACE(x, p).
        if T1 and ((x_in_b2 and len(T1) == p) or len(T1) > p):
            k, _ = T1.popitem(last=False)      # LRU of T1 -> ghost B1
            B1[k] = True
        elif T2:
            k, _ = T2.popitem(last=False)      # LRU of T2 -> ghost B2
            B2[k] = True
        elif T1:                                # T2 empty; fall back
            k, _ = T1.popitem(last=False)
            B1[k] = True

    for x in keys:
        if x in T1:                             # Case I (hit)
            del T1[x]
            T2[x] = True
        elif x in T2:                           # Case I (hit)
            T2.move_to_end(x)
        elif x in B1:                           # Case II (ghost hit, recency)
            p = min(p + max(len(B2) // max(len(B1), 1), 1), c)
            replace(False)
            del B1[x]
            T2[x] = True
            fills += 1
        elif x in B2:                           # Case III (ghost hit, freq)
            p = max(p - max(len(B1) // max(len(B2), 1), 1), 0)
            replace(True)
            del B2[x]
            T2[x] = True
            fills += 1
        else:                                   # Case IV (miss, not ghosted)
            if len(T1) + len(B1) == c:
                if len(T1) < c:
                    B1.popitem(last=False)
                    replace(False)
                else:
                    T1.popitem(last=False)      # B1 empty: drop LRU of T1
            elif len(T1) + len(B1) < c and \
                    len(T1) + len(T2) + len(B1) + len(B2) >= c:
                if len(T1) + len(T2) + len(B1) + len(B2) >= 2 * c:
                    if B2:
                        B2.popitem(last=False)
                    elif B1:
                        B1.popitem(last=False)
                replace(False)
            T1[x] = True
            fills += 1
        if trace_invariants:
            worst = max(worst, len(T1) + len(T2) + len(B1) + len(B2))
    return fills, worst


def validate(verbose=True):
    """The checks the preregistration makes preconditions on reporting.

    A failure here means the implementation is being scored, not the policy.
    """
    import random
    ok = True

    def say(name, passed, detail=""):
        nonlocal ok
        ok &= passed
        if verbose:
            print("  %-46s %s %s" % (name, "PASS" if passed else "FAIL", detail))

    rng = random.Random(5)
    # 1. capacity >= whole key space: nothing can be evicted, so every policy
    #    must reach the same fill count -- one per distinct key.
    for U in (8, 40, 137):
        keys = [(0, rng.randrange(U)) for _ in range(4000)]
        distinct = len(set(keys))
        a, _ = arc_transfers(keys, U)
        recs = [{"routed": {"0": [k[1]]}} for k in keys]
        l = run_policy(recs, U, "lru")
        f = run_policy(recs, U, "lfu")
        b = belady(keys, U)
        say("cap >= key space, all policies agree (U=%d)" % U,
            a == l == f == b == distinct, "arc=%d lru=%d lfu=%d bel=%d" % (a, l, f, b))

    # 2. bounded: never worse than all-miss, never better than optimal.
    for trial in range(40):
        U = rng.randint(6, 60)
        n = rng.randint(50, 900)
        cap = rng.randint(2, max(3, U - 1))
        keys = [(0, rng.randrange(U)) for _ in range(n)]
        a, _ = arc_transfers(keys, cap)
        b = belady(keys, cap)
        if not (b <= a <= n):
            say("bounded belady <= arc <= all-miss", False,
                "U=%d n=%d cap=%d arc=%d bel=%d" % (U, n, cap, a, b))
            break
    else:
        say("bounded belady <= arc <= all-miss (40 random)", True)

    # 3. ARC's own invariant: |T1|+|T2|+|B1|+|B2| <= 2c, at all times.
    bad = None
    for trial in range(40):
        U = rng.randint(6, 60)
        cap = rng.randint(2, max(3, U - 1))
        keys = [(0, rng.randrange(U)) for _ in range(rng.randint(50, 900))]
        _, worst = arc_transfers(keys, cap, trace_invariants=True)
        if worst > 2 * cap:
            bad = (cap, worst)
            break
    say("resident+ghost <= 2c (40 random)", bad is None,
        "" if bad is None else "cap=%d saw %d" % bad)
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

    print("ARC implementation validation (preregistered precondition)")
    if not validate():
        sys.exit("VALIDATION FAILED -- the implementation is being scored, "
                 "not the policy. Nothing else is reported.")
    if a.validate_only:
        return

    rows = []
    print("\n%-8s %-9s %5s %5s | %9s %9s %9s %9s"
          % ("model", "prompt", "held", "cap", "lru", "lfu", "arc", "belady"))
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
                b = belady(keys, cap)
                rows.append({"model": m, "prompt": p, "steps_held": sh,
                             "cap": cap, "lru": l, "lfu": lf, "arc": ar,
                             "belady": b})
                print("%-8s %-9s %5.2f %5d | %9d %9d %9d %9d"
                      % (m, p, sh, cap, l, lf, ar, b))
            print()
    if a.out:
        with open(a.out, "w") as fh:
            json.dump({"rows": rows}, fh, indent=1)
        print("receipt -> %s" % a.out)


if __name__ == "__main__":
    main()
