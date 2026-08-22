"""How much of the 1.9x gap to optimal can an IMPLEMENTABLE policy close?

RESULTS-oracle-headroom.md measured the shipped cache at ~1.9x Belady's
optimum and found it is, to within half a percent, plain LRU. Belady is a
bound, not a proposal. This asks what a policy that runs online can actually
recover.

The candidates use only information available at eviction time:

  lru        the current behaviour, baseline
  lfu        evict least-frequently-routed so far, LRU to break ties
  prevstep   evict rows NOT routed in the PREVIOUS step first, LRU within
             each class -- a one-step lookback prior, legitimate because the
             previous step is complete before this one starts
  prevstep_lfu   the same prior, LFU rather than LRU within each class
  belady     the bound, for scale

`prevstep` is the interesting one and it is not a heuristic pulled from the
air: step-to-step routed-set overlap is 41-50% on OLMoE, Granite and gpt-oss
(reuse_overlap.py), 2.0-4.0x chance, so "was routed last step" carries real
information about "will be routed next step" and LRU does not use it.

Reuse distance here is structural: a given (layer, expert) row is touched at
most once per step, so every repeat is exactly one step away. What separates
policies is only WHICH rows to keep when capacity cannot hold the working
set -- which is a prediction problem, and the previous step is the cheapest
predictor available.
"""
import argparse
import json
import os
import sys
from collections import OrderedDict, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "kernel"))
sys.path.insert(0, HERE)

from oracle_headroom import belady, keystream                  # noqa: E402
from replay_dev_cache import positional_transfers              # noqa: E402
from score_policies import load, steps_capacity                # noqa: E402

PROMPTS = ("prose", "code", "math", "dialogue")


def _steps(recs):
    """(key, step_index) for every routed row-slot, in order."""
    out = []
    for s, r in enumerate(recs):
        for L, ex in r["routed"].items():
            for e in ex:
                out.append(((int(L), e), s))
    return out


def run_policy(recs, cap, policy):
    """Transfers under an online policy. No future information is consulted.

    Recency is an explicit monotone clock rather than a position lookup in the
    resident set: an index-of scan is O(cap) per eviction and, cached, is a
    correctness hazard the moment the set mutates. `used[k]` is the tick of
    k's last touch, so "least recently used" is `min(used)` and every policy
    below breaks ties on the same quantity.
    """
    seq = _steps(recs)
    prev_set, cur_set, cur_step = set(), set(), None
    resident = OrderedDict()
    freq, used = defaultdict(int), {}
    fills = clock = 0
    for k, s in seq:
        if s != cur_step:             # step boundary: last step becomes "prev"
            prev_set, cur_set, cur_step = cur_set, set(), s
        clock += 1
        cur_set.add(k)
        freq[k] += 1
        if k in resident:
            used[k] = clock
            resident.move_to_end(k)
            continue
        fills += 1
        if len(resident) >= cap:
            v = _victim(resident, freq, used, prev_set, policy)
            resident.pop(v)
            used.pop(v, None)
        resident[k] = True
        used[k] = clock
    return fills


def _victim(resident, freq, used, prev_set, policy):
    if policy == "lru":
        return next(iter(resident))                       # OrderedDict head
    if policy == "lfu":
        return min(resident, key=lambda r: (freq[r], used[r]))
    if policy in ("prevstep", "prevstep_lfu"):
        # Rows the previous step did not touch lose first; the suffix decides
        # the ordering WITHIN each class.
        cold = [r for r in resident if r not in prev_set]
        pool = cold if cold else list(resident)
        if policy == "prevstep":
            return min(pool, key=lambda r: used[r])
        return min(pool, key=lambda r: (freq[r], used[r]))
    raise ValueError(policy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=HERE)
    ap.add_argument("--models", default="olmoe,granite,qwen")
    ap.add_argument("--prompts", default=",".join(PROMPTS))
    ap.add_argument("--steps-held", default="1.0,1.5,2.0")
    ap.add_argument("--policies", default="lru,lfu,prevstep,prevstep_lfu")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    held = [float(x) for x in a.steps_held.split(",")]
    pols = [p for p in a.policies.split(",") if p]

    rows = []
    print("%-8s %-9s %5s %5s | %s | %9s %8s"
          % ("model", "prompt", "held", "cap",
             " ".join("%9s" % p for p in pols), "belady", "best/bel"))
    for m in a.models.split(","):
        for p in a.prompts.split(","):
            f = os.path.join(a.dir, "%s_%s.jsonl" % (m, p))
            if not os.path.exists(f):
                continue
            meta, recs = load(f)
            per = meta["layers"] * meta["top_k"]
            keys = keystream(recs)
            for sh in held:
                cap = steps_capacity(per, sh)
                got = {pol: run_policy(recs, cap, pol) for pol in pols}
                b = belady(keys, cap)
                best = min(got.values())
                rows.append({"model": m, "prompt": p, "steps_held": sh,
                             "cap": cap, "belady": b,
                             "positional": positional_transfers(meta, recs),
                             **{("p_" + k): v for k, v in got.items()}})
                print("%-8s %-9s %5.2f %5d | %s | %9d %7.2fx"
                      % (m, p, sh, cap,
                         " ".join("%9d" % got[pol] for pol in pols),
                         b, best / b if b else 0))
            print()

    if a.out:
        with open(a.out, "w") as f:
            json.dump({"rows": rows, "policies": pols}, f, indent=1)
        print("receipt -> %s" % a.out)


if __name__ == "__main__":
    main()
