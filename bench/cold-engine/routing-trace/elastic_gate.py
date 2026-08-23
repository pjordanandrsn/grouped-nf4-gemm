"""Gate E, offline half: E1 (the paying set exists) and E2 (it is findable).

Registered in bench/cold-engine/PREREG-elastic-promotion.md. The break-even
n* = 2.62-2.80 TOTAL invocations (including the promoted one) comes from
committed phase-0/phase-2 constants, so the qualifying bar is >= 2 REMAINING
recurrences within the window (total >= 3 = ceil(2.80)); the post-G2-fix
sensitivity is >= 4 remaining (total >= 5 = ceil(4.25)). An earlier draft
registered >= 3 remaining -- one reuse stricter than the economics -- caught
by Bugbot on #201 before anything was scored. Everything here runs on the 16 committed rank traces.

Definitions, fixed here so the receipt is self-describing:

* An INVOCATION is one (step t, layer l, expert e) selection.
* Its REUSE COUNT at window W is the number of steps in t+1 .. t+W where e is
  selected at layer l again. Invocations with t + W >= steps leave the
  DENOMINATOR entirely -- the window would run off the trace end, and counting
  a truncated window as "did not recur" would bias E1 down (registered).
* E1 qualifies an invocation when reuse count >= 2 more (>= 4 more reported
  alongside: the break-even if the named G2 CPU fix lands).
* E2a subset: invocations whose expert is that visit's rank 1
  (routed_rank[l][0]).
* E2b subset: invocations whose expert's trailing selection count at layer l
  over max(0, t-32) .. t-1 is at least the ceil(E/4)-th largest such count at
  that (t, l), and nonzero. Ties at the threshold are included; an empty
  trailing window (t = 0) excludes the invocation from the subset.
"""
import argparse
import bisect
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from score_policies import load                              # noqa: E402

W_SWEEP = (8, 32, 128)
W_GATE = 32
NSTAR_MORE = 2
NSTAR_FIXED_MORE = 4


def occurrences(recs):
    """(layer, expert) -> sorted list of steps where it was selected."""
    occ = defaultdict(list)
    for t, r in enumerate(recs):
        for lay, ex in r["routed"].items():
            l = int(lay)
            for e in ex:
                occ[(l, e)].append(t)
    return occ


def reuse_count(occ_list, idx, W):
    """Occurrences after position idx whose step lies in (t, t+W]."""
    t = occ_list[idx]
    return bisect.bisect_right(occ_list, t + W) - (idx + 1)


def score_trace(meta, recs, W=W_GATE):
    occ = occurrences(recs)
    steps = len(recs)
    E = int(meta["n_experts"])
    L = int(meta["layers"])
    # trailing counts, built incrementally per layer
    trail = [defaultdict(int) for _ in range(L)]
    window = [defaultdict(list) for _ in range(L)]   # expert -> steps in window
    pos = defaultdict(int)                            # (l,e) -> next occ index
    n_all = q3_all = q4_all = 0
    n_r1 = q_r1 = 0
    n_fq = q_fq = 0
    for t, r in enumerate(recs):
        for lay in r["routed"]:
            l = int(lay)
            ranked = (r.get("routed_rank") or r["routed"])[lay]
            # threshold for E2b at this (t, l): ceil(E/4)-th largest count
            counts = sorted(trail[l].values(), reverse=True)
            kth = -(-E // 4)
            thr = counts[kth - 1] if len(counts) >= kth else (counts[-1] if counts else None)
            for e in r["routed"][lay]:
                ol = occ[(l, e)]
                i = pos[(l, e)]
                pos[(l, e)] += 1
                if t + W >= steps:
                    continue                        # leaves the denominator
                c = reuse_count(ol, i, W)
                n_all += 1
                q3 = c >= NSTAR_MORE
                q3_all += q3
                q4_all += c >= NSTAR_FIXED_MORE
                if e == ranked[0]:
                    n_r1 += 1
                    q_r1 += q3
                tc = trail[l].get(e, 0)
                if t > 0 and tc > 0 and thr is not None and tc >= thr and thr > 0:
                    n_fq += 1
                    q_fq += q3
        # slide the 32-step trailing window
        for lay, ex in r["routed"].items():
            l = int(lay)
            for e in ex:
                trail[l][e] += 1
                window[l][e].append(t)
        if t >= 31:
            told = t - 31
            for l in range(L):
                for e in list(window[l]):
                    while window[l][e] and window[l][e][0] <= told - 1:
                        window[l][e].pop(0)
                        trail[l][e] -= 1
                    if not window[l][e]:
                        del window[l][e]
                        del trail[l][e]
    base = q3_all / n_all if n_all else 0.0
    return {"W": W, "invocations": n_all,
            "e1_frac_ge2more": base,
            "e1_frac_ge4more": q4_all / n_all if n_all else 0.0,
            "rank1": {"n": n_r1, "frac": q_r1 / n_r1 if n_r1 else 0.0,
                      "lift": (q_r1 / n_r1 / base) if n_r1 and base else 0.0},
            "freq": {"n": n_fq, "frac": q_fq / n_fq if n_fq else 0.0,
                     "lift": (q_fq / n_fq / base) if n_fq and base else 0.0}}


def validate():
    """Preregistered preconditions. A failure scores the harness, not gate E."""
    ok = True

    def say(name, passed, detail=""):
        nonlocal ok
        ok &= passed
        print("  %-56s %s %s" % (name, "PASS" if passed else "FAIL", detail))

    # 1. constructed trace with known counts, one layer, one expert of note
    steps = 80
    hits = [0, 3, 5, 10, 50]
    recs = [{"routed": {"0": [5] if t in hits else [9000 + t]},
             "routed_rank": {"0": [5] if t in hits else [9000 + t]}}
            for t in range(steps)]
    occ = occurrences(recs)
    got = [reuse_count(occ[(0, 5)], i, 32) for i in range(len(hits))]
    say("known reuse counts reproduced exactly", got == [3, 2, 1, 0, 0],
        str(got))
    # and the corrected qualify bar on those counts: >=2 more qualifies the
    # occurrences with 3 and 2 remaining, not the one with 1
    quals = [c >= NSTAR_MORE for c in got]
    say("qualify bar is >=2 MORE (total >=3)",
        quals == [True, True, False, False, False], str(quals))

    # 2. window-edge handling: an invocation in the last W steps leaves the
    #    denominator rather than counting as non-recurring
    meta = {"n_experts": 16, "layers": 1}
    tail = [{"routed": {"0": [1]}, "routed_rank": {"0": [1]}}
            for _ in range(10)]
    r = score_trace(meta, tail, W=32)
    say("last-W invocations leave the denominator", r["invocations"] == 0,
        "denominator=%d" % r["invocations"])

    # 3. falsifiability, E1: a no-reuse synthetic must score exactly 0%
    norep = [{"routed": {"0": [t * 2, t * 2 + 1]},
              "routed_rank": {"0": [t * 2, t * 2 + 1]}} for t in range(300)]
    r = score_trace({"n_experts": 600, "layers": 1}, norep, W=32)
    say("falsifiability: no-reuse trace scores E1 = 0",
        r["invocations"] > 0 and r["e1_frac_ge2more"] == 0.0,
        "frac=%.3f n=%d" % (r["e1_frac_ge2more"], r["invocations"]))

    # 4. falsifiability, E2a: shuffling rank within each visit kills the lift
    import random
    rng = random.Random(3)
    f = os.path.join(HERE, "..", "rank-2026-08-22", "olmoe_prose.jsonl")
    meta, recs = load(f)
    shuf = []
    for r0 in recs:
        rr = dict(r0)
        rr["routed_rank"] = {l: rng.sample(v, len(v))
                             for l, v in r0["routed_rank"].items()}
        shuf.append(rr)
    real = score_trace(meta, recs)["rank1"]["lift"]
    dead = score_trace(meta, shuf)["rank1"]["lift"]
    say("falsifiability: rank-shuffle pushes E2a lift to ~1.0",
        abs(dead - 1.0) < 0.1 and real > dead,
        "real=%.3f shuffled=%.3f" % (real, dead))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(HERE, "..",
                                                  "rank-2026-08-22"))
    ap.add_argument("--models", default="olmoe,granite,qwen,gptoss")
    ap.add_argument("--prompts", default="prose,code,math,dialogue")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    print("gate E offline harness: preregistered preconditions")
    if not validate():
        sys.exit("VALIDATION FAILED -- the harness is being scored, not "
                 "gate E. Nothing else is reported.")
    if a.validate_only:
        return

    rows = []
    print("\n%-9s %-9s %6s | %8s %8s | %14s %14s"
          % ("model", "prompt", "invoc", "E1>=2m", ">=4m",
             "E2a rank1 lift", "E2b freq lift"))
    for m in a.models.split(","):
        for p in a.prompts.split(","):
            f = os.path.join(a.dir, "%s_%s.jsonl" % (m, p))
            if not os.path.exists(f):
                continue
            meta, recs = load(f)
            row = {"model": m, "prompt": p}
            for W in W_SWEEP:
                row["W%d" % W] = score_trace(meta, recs, W)
            g = row["W%d" % W_GATE]
            rows.append(row)
            print("%-9s %-9s %6d | %7.1f%% %7.1f%% | %13.2fx %13.2fx"
                  % (m, p, g["invocations"], 100 * g["e1_frac_ge2more"],
                     100 * g["e1_frac_ge4more"], g["rank1"]["lift"],
                     g["freq"]["lift"]))
        print()
    if a.out:
        with open(a.out, "w") as fh:
            json.dump({"nstar_more": NSTAR_MORE, "nstar_fixed_more": NSTAR_FIXED_MORE,
                       "W_gate": W_GATE, "rows": rows}, fh, indent=1)
        print("receipt -> %s" % a.out)


if __name__ == "__main__":
    main()
