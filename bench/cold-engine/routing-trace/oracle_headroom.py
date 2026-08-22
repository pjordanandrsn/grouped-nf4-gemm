"""How much transfer reduction is the device cache leaving on the table?

Wall is transfer-bound -- r = +0.9872 against transfers/step on real routing,
replicated on two hosts (RESULTS-wall-real-routing.md). So the only way to
move wall is to move transfers, and the question that decides whether more
policy work pays is: how far is the shipped cache from the best any
replacement policy could do at the same capacity?

That bound is computable exactly. Belady's MIN evicts the resident key whose
NEXT use is furthest in the future; no online policy can beat it, and it needs
the whole trace, which offline replay has. Four columns per cell:

  positional   what the engine already has, no expert-keyed cache at all
  cache        the shipped DevRowCache, driven for real
  lru          a pure-LRU simulation at the same capacity
  belady       the optimum -- the floor no policy can go below

`cache - belady` is the headroom. If it is small the cache line of work is
finished and effort belongs elsewhere; if it is large, policy has room.

Belady is a BOUND, not a proposal: it cannot be implemented online. It is here
to size the opportunity, not to be shipped.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "kernel"))
sys.path.insert(0, HERE)

from replay_dev_cache import positional_transfers, replay      # noqa: E402
from score_crossover import simulate                           # noqa: E402
from score_policies import load, steps_capacity                # noqa: E402

PROMPTS = ("prose", "code", "math", "dialogue")


def keystream(recs):
    """Every routed row-slot in order, as (layer, expert)."""
    out = []
    for r in recs:
        for L, ex in r["routed"].items():
            for e in ex:
                out.append((int(L), e))
    return out


def belady_scan(keys, cap):
    """Reference implementation: linear scan for the furthest next use.

    O(n * cap) and obviously correct. Kept because the heap version below is
    the one that runs and the heap version is where a subtle bug would live;
    a fast optimum that is wrong is worse than no bound at all.
    """
    nxt = defaultdict(list)
    for i, k in enumerate(keys):
        nxt[k].append(i)
    pos = {k: 0 for k in nxt}
    resident, fills = set(), 0
    for k in keys:
        pos[k] += 1
        if k in resident:
            continue
        fills += 1
        if len(resident) >= cap:
            victim, far = None, -1
            for r in resident:
                pr = pos[r]
                nu = nxt[r][pr] if pr < len(nxt[r]) else float("inf")
                if nu > far:
                    victim, far = r, nu
                    if far == float("inf"):
                        break
            resident.discard(victim)
        resident.add(k)
    return fills


def belady(keys, cap):
    """Transfers under optimal replacement, heap-based.

    Evicting by furthest-next-use needs the future, which an offline replay
    has and an online policy does not. Entries are pushed as
    (-next_use, key) and validated on pop against the key's CURRENT next use,
    so stale entries left by a re-access are discarded rather than trusted.
    A key with no next use sorts first: it is never needed again, so it is
    free capacity.
    """
    import heapq
    nxt = defaultdict(list)
    for i, k in enumerate(keys):
        nxt[k].append(i)
    pos = {k: 0 for k in nxt}
    INF = float("inf")

    def next_use(k):
        p = pos[k]
        return nxt[k][p] if p < len(nxt[k]) else INF

    resident, heap, fills = set(), [], 0
    for k in keys:
        pos[k] += 1
        if k in resident:
            heapq.heappush(heap, (-next_use(k), k))   # refresh, stale entry stays
            continue
        fills += 1
        if len(resident) >= cap:
            while True:
                negu, cand = heapq.heappop(heap)
                if cand in resident and -negu == next_use(cand):
                    resident.discard(cand)
                    break
        resident.add(k)
        heapq.heappush(heap, (-next_use(k), k))
    return fills


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=HERE)
    ap.add_argument("--models", default="olmoe,granite,qwen")
    ap.add_argument("--prompts", default=",".join(PROMPTS))
    ap.add_argument("--steps-held", default="1.0,1.25,1.5,2.0")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    held = [float(x) for x in a.steps_held.split(",")]

    rows = []
    print("%-8s %-9s %5s %5s | %9s %9s %9s %9s | %8s %8s"
          % ("model", "prompt", "held", "cap", "positional", "cache", "lru",
             "belady", "cache/bel", "headroom"))
    for m in a.models.split(","):
        for p in a.prompts.split(","):
            f = os.path.join(a.dir, "%s_%s.jsonl" % (m, p))
            if not os.path.exists(f):
                continue
            meta, recs = load(f)
            per = meta["layers"] * meta["top_k"]
            keys = keystream(recs)
            pos = positional_transfers(meta, recs)
            for sh in held:
                cap = steps_capacity(per, sh)
                c, _ = replay(meta, recs, cap)
                l = simulate(recs, cap, "lru")
                b = belady(keys, cap)
                rows.append({"model": m, "prompt": p, "steps_held": sh,
                             "cap": cap, "positional": pos, "cache": c,
                             "lru": l, "belady": b,
                             "cache_over_belady": c / b if b else None,
                             "headroom_rows": c - b})
                print("%-8s %-9s %5.2f %5d | %9d %9d %9d %9d | %8.2fx %8d"
                      % (m, p, sh, cap, pos, c, l, b, c / b if b else 0, c - b))
            print()

    if a.out:
        with open(a.out, "w") as f:
            json.dump({"rows": rows}, f, indent=1)
        print("receipt -> %s" % a.out)


if __name__ == "__main__":
    main()
