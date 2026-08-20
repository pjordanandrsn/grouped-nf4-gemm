"""Gate 3's central question, scored offline on a real routing sequence.

Gate 3 is the loop the earlier gates left open:

    placement -> execution -> observed cost/reuse -> new placement

The question it exists to answer is whether CLOSING that loop is worth
anything: does re-placing experts from observed behaviour beat placing them
once from a profile? And -- the comparison the directive does not make --
does either beat simply demand-paging the fast tier, which is what the tier
already does without any placement machinery at all?

Three policies, one capacity, one trace, evaluated on the same window:

  static    profile the first `--warm` steps, pin the top-C experts by
            frequency, never change. This is what solve_placement does.
  adaptive  same start, then every `--period` steps re-pick the top-C from
            everything observed so far. Promotions cost a read. This is
            gate 3.
  demand    no placement at all: LRU over the same C rows. What ColdTier
            and DevRowCache already do.

Frequency, not recency, is the adaptive signal -- R4 was scored on this same
trace and short-window recurrence lost at every capacity that carried signal
(RESULTS-r4.md).

A read is a routed (layer, expert) whose row is not resident. Migration
reads are charged to `adaptive`, because a placement change that pretends
its own promotions are free is not a placement policy.
"""
import argparse
import json
from collections import Counter, OrderedDict


def load(path):
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    return rows[0]["meta"], rows[1:]


def freq_top(counter, cap):
    return {k for k, _ in counter.most_common(cap)}


def static_policy(warm, evalw, cap):
    c = Counter(k for r in warm for L, ex in r["routed"].items()
                for k in ((int(L), e) for e in ex))
    resident = freq_top(c, cap)
    reads = sum(1 for r in evalw for L, ex in r["routed"].items()
                for e in ex if (int(L), e) not in resident)
    return {"reads": reads, "migrations": 0}


def adaptive_policy(warm, evalw, cap, period):
    c = Counter(k for r in warm for L, ex in r["routed"].items()
                for k in ((int(L), e) for e in ex))
    resident = freq_top(c, cap)
    reads = migrations = 0
    for i, r in enumerate(evalw):
        if i and i % period == 0:
            new = freq_top(c, cap)
            promoted = new - resident
            # A promoted row has to be brought in. Charging it is the whole
            # difference between a placement policy and a wish.
            migrations += len(promoted)
            resident = new
        for L, ex in r["routed"].items():
            for e in ex:
                k = (int(L), e)
                c[k] += 1
                if k not in resident:
                    reads += 1
    return {"reads": reads + migrations, "migrations": migrations,
            "demand_reads": reads}


def oracle_policy(evalw, cap):
    """The best FIXED set, chosen with knowledge of the evaluation window.

    Not achievable -- it needs the future -- and adaptive is not expected to
    reach it. It is here to size the HEADROOM: how much of what a perfect
    placement would save is still on the table after the loop has run.
    """
    c = Counter(k for r in evalw for L, ex in r["routed"].items()
                for k in ((int(L), e) for e in ex))
    res = freq_top(c, cap)
    reads = sum(1 for r in evalw for L, ex in r["routed"].items()
                for e in ex if (int(L), e) not in res)
    return {"reads": reads, "migrations": 0}


def demand_policy(warm, evalw, cap):
    lru, reads = OrderedDict(), 0
    for r in warm:                        # warm the cache on the same prefix
        for L, ex in r["routed"].items():
            for e in ex:
                k = (int(L), e)
                if k in lru:
                    lru.move_to_end(k)
                    continue
                lru[k] = 1
                if len(lru) > cap:
                    lru.popitem(last=False)
    for r in evalw:
        for L, ex in r["routed"].items():
            for e in ex:
                k = (int(L), e)
                if k in lru:
                    lru.move_to_end(k)
                    continue
                reads += 1
                lru[k] = 1
                if len(lru) > cap:
                    lru.popitem(last=False)
    return {"reads": reads, "migrations": 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--rows", default="64,128,192,256,384,512,768")
    ap.add_argument("--warm", type=int, default=128)
    ap.add_argument("--period", type=int, default=32)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    meta, recs = load(a.trace)
    warm, evalw = recs[:a.warm], recs[a.warm:]
    routed = sum(len(v) for r in evalw for v in r["routed"].values())
    print(f"trace: {meta['steps']} steps x {meta['layers']} layers x "
          f"top-{meta['top_k']} of {meta['n_experts']}")
    print(f"profile on steps 0-{a.warm-1}, score on {a.warm}-{meta['steps']-1} "
          f"({routed} routed slots); adaptive re-places every {a.period}\n")
    print(f"{'rows':>6} {'static':>9} {'adaptive':>9} {'(migr)':>7} "
          f"{'demand':>9} {'oracle':>8} | {'adapt v static':>15} "
          f"{'adapt v oracle':>15}")
    out = {"meta": meta, "warm": a.warm, "period": a.period,
           "eval_routed": routed, "points": []}
    for cap in [int(x) for x in a.rows.split(",")]:
        s = static_policy(warm, evalw, cap)
        ad = adaptive_policy(warm, evalw, cap, a.period)
        d = demand_policy(warm, evalw, cap)
        o = oracle_policy(evalw, cap)
        av = (ad["reads"] - s["reads"]) / s["reads"] * 100
        dv = (d["reads"] - s["reads"]) / s["reads"] * 100
        ov = (ad["reads"] - o["reads"]) / o["reads"] * 100
        out["points"].append({"rows": cap, "static": s["reads"],
                              "adaptive": ad["reads"],
                              "adaptive_migrations": ad["migrations"],
                              "demand": d["reads"], "oracle": o["reads"],
                              "adaptive_vs_static_pct": av,
                              "demand_vs_static_pct": dv,
                              "adaptive_vs_oracle_pct": ov})
        print(f"{cap:>6} {s['reads']:>9} {ad['reads']:>9} "
              f"{ad['migrations']:>7} {d['reads']:>9} {o['reads']:>8} | "
              f"{av:>+14.1f}% {ov:>+14.1f}%")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=2)
        print("\nreceipt ->", a.out)


if __name__ == "__main__":
    main()
