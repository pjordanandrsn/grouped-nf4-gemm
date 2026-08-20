"""Replay a REAL decode routing sequence through the device row cache.

The committed routing profile is aggregate mass, which cannot answer the only
question that decides whether this cache is worth its VRAM: when an expert is
routed again, is it still resident? That is a property of the ORDER of the
routing, so this drives the real `DevRowCache` -- not a model of it -- with a
captured autoregressive decode trace.

Two baselines, because "fewer transfers than nothing" is not the bar:

  none        every routed row is fetched. 512 x 16 x 8 transfers.
  positional  what the engine ALREADY has. `slots64` row i holds whatever
              expert routed to position i last step, and the gather's
              address test skips it; so a hit is expert_i(t) == expert_i(t-1)
              within a layer. This is the number the cache has to beat.

The positional baseline here is OPTIMISTIC: the captured trace stores each
step's routed set sorted, and sorting makes position far more stable than the
router's own top-k order, which the engine actually uses. Being generous to
the incumbent is the conservative direction for the claim being made.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "..", "kernel"))
from dev_row_cache import DevRowCache, StepTag        # noqa: E402


def load(path):
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    return rows[0]["meta"], rows[1:]


def positional_transfers(meta, recs):
    """Transfers the engine's existing positional cache would still make."""
    prev, total = {}, 0
    for r in recs:
        for L, ex in r["routed"].items():
            p = prev.get(L)
            total += sum(1 for i, e in enumerate(ex)
                         if p is None or i >= len(p) or p[i] != e)
            prev[L] = ex
    return total


def replay(meta, recs, rows, protected=None):
    c = DevRowCache(rows, 8, device="cpu", protected=protected)
    fills = 0
    for r in recs:
        for L, ex in r["routed"].items():
            t = StepTag("cpu")
            _assign, need = c.want(int(L), ex, t)
            t.record()                 # the gather completes before the next
            fills += len(need)
    return fills, c.stats()


def lru_transfers(meta, recs, rows):
    """An ideal LRU of the same capacity, keyed the same way.

    Separates two very different failures. If LRU also loses to the
    positional cache, expert-keyed residency does not pay on this trace and
    no policy fixes that. If LRU wins where DevRowCache loses, the trace is
    fine and the ALLOCATOR is the problem -- VramSlots increments a _clock
    and never reads it, choosing both victims and slots in slot-index order.
    """
    from collections import OrderedDict
    cache, fills = OrderedDict(), 0
    for r in recs:
        for L, ex in r["routed"].items():
            for e in ex:
                key = (L, e)
                if key in cache:
                    cache.move_to_end(key)
                    continue
                fills += 1
                cache[key] = 1
                if len(cache) > rows:
                    cache.popitem(last=False)
    return fills


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--rows", default="128,192,256,384,512,768,1024")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    meta, recs = load(a.trace)
    routed = sum(len(v) for r in recs for v in r["routed"].values())
    pos = positional_transfers(meta, recs)
    pairs = meta["layers"] * meta["n_experts"]

    print(f"trace: {meta['steps']} decode steps x {meta['layers']} layers x "
          f"top-{meta['top_k']} of {meta['n_experts']}  "
          f"({pairs} distinct (layer,expert) pairs)")
    print(f"  routed row-slots        {routed:>8}")
    print(f"  transfers, no cache     {routed:>8}   (100.0%)")
    print(f"  transfers, POSITIONAL   {pos:>8}   ({pos/routed*100:>5.1f}%)  "
          f"<- what the engine already does")
    print()
    print(f"{'rows':>6} {'%pairs':>7} {'DevRowCache':>12} {'vs pos':>7} | "
          f"{'ideal LRU':>10} {'vs pos':>7} | {'gap':>6}")
    out = {"meta": meta, "routed_row_slots": routed, "positional": pos,
           "distinct_pairs": pairs, "points": []}
    for rows in [int(x) for x in a.rows.split(",")]:
        if rows > pairs:
            continue
        fills, st = replay(meta, recs, rows)
        lru = lru_transfers(meta, recs, rows)
        rec = {"rows": rows, "fills": fills, "lru_fills": lru,
               "frac_of_pairs": rows / pairs,
               "vs_none": fills / routed, "vs_positional": fills / pos,
               "lru_vs_positional": lru / pos,
               "policy_gap": fills / lru,
               "resurrections": st["resurrections"], "stalls": st["stalls"],
               "reuse_before_overwrite": st["reuse_before_overwrite"]}
        out["points"].append(rec)
        print(f"{rows:>6} {rows/pairs*100:>6.1f}% {fills:>12} "
              f"{fills/pos*100:>6.1f}% | {lru:>10} {lru/pos*100:>6.1f}% | "
              f"{fills/lru:>5.2f}x")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=2)
        print("\nreceipt ->", a.out)


if __name__ == "__main__":
    main()
