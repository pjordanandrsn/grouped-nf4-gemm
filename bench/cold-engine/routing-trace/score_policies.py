"""Re-placement policies, against the headroom gate 3 left on the table.

RESULTS-gate3.md scored the loop worth closing -- adaptive re-placement beat
static by 6-41% -- but its third control showed the policy it used sits
**4-30% above the best achievable fixed set**, and the gap WIDENS with
capacity. So the loop is worth closing and top-C by cumulative frequency is
not the way to close it. This sweeps policies against that gap.

  static    top-C by prefix frequency, fixed. What solve_placement does.
  adaptive  top-C by CUMULATIVE frequency, re-picked every `period`.
            Gate 3's policy, and the one with headroom above it.
  ewma      top-C by exponentially-decayed frequency. Cumulative counts are
            dominated by early observations, so a newly-hot expert is slow to
            promote and a cooled one slow to demote; a half-life is the
            smallest change that fixes both without becoming a short window,
            which R4 showed loses.
  hybrid    pin the top-(C-m) by cumulative frequency and DEMAND-PAGE the
            remaining m by LRU. Motivated by gate 3's own numbers: placement
            beats LRU below 512 rows and LRU wins at 768, so neither is right
            everywhere and a split should beat both.
  demand    LRU over all C. No placement at all.
  oracle    best FIXED set for the evaluation window. Needs the future; it is
            the ceiling, not a competitor.

A read is a routed (layer, expert) whose row is not resident. Promotions are
charged as reads -- a policy that pretends its own migrations are free is not
a policy.
"""
import argparse
import json
from collections import Counter, OrderedDict


def load(path):
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    return rows[0]["meta"], rows[1:]


def _keys(r):
    for L, ex in r["routed"].items():
        for e in ex:
            yield (int(L), e)


def _top(counter, cap):
    return {k for k, _ in counter.most_common(cap)}


def static_p(warm, ev, cap, **_):
    res = _top(Counter(k for r in warm for k in _keys(r)), cap)
    return sum(1 for r in ev for k in _keys(r) if k not in res), 0


def oracle_p(warm, ev, cap, **_):
    res = _top(Counter(k for r in ev for k in _keys(r)), cap)
    return sum(1 for r in ev for k in _keys(r) if k not in res), 0


def demand_p(warm, ev, cap, **_):
    lru, reads = OrderedDict(), 0
    for r in warm:
        for k in _keys(r):
            lru.pop(k, None)
            lru[k] = 1
            if len(lru) > cap:
                lru.popitem(last=False)
    for r in ev:
        for k in _keys(r):
            if k in lru:
                lru.move_to_end(k)
                continue
            reads += 1
            lru[k] = 1
            if len(lru) > cap:
                lru.popitem(last=False)
    return reads, 0


def adaptive_p(warm, ev, cap, period=32, decay=None, **_):
    """Cumulative frequency, or exponentially decayed when `decay` is set."""
    c = Counter(k for r in warm for k in _keys(r))
    res, reads, migr = _top(c, cap), 0, 0
    for i, r in enumerate(ev):
        if i and i % period == 0:
            if decay is not None:
                for k in list(c):
                    c[k] *= decay
            new = _top(c, cap)
            migr += len(new - res)
            res = new
        for k in _keys(r):
            c[k] += 1
            if k not in res:
                reads += 1
    return reads + migr, migr


def hybrid_p(warm, ev, cap, period=32, pinned_frac=0.75, **_):
    """Pin the top of the frequency ranking; demand-page the remainder."""
    npin = max(0, min(cap, int(cap * pinned_frac)))
    ncache = cap - npin
    c = Counter(k for r in warm for k in _keys(r))
    pin, reads, migr = _top(c, npin), 0, 0
    lru = OrderedDict()
    for r in warm:
        for k in _keys(r):
            if k in pin:
                continue
            lru.pop(k, None)
            lru[k] = 1
            if len(lru) > ncache:
                lru.popitem(last=False)
    for i, r in enumerate(ev):
        if i and i % period == 0:
            new = _top(c, npin)
            migr += len(new - pin)
            for k in new:                 # a pinned row leaves the LRU half
                lru.pop(k, None)
            pin = new
        for k in _keys(r):
            c[k] += 1
            if k in pin:
                continue
            if k in lru:
                lru.move_to_end(k)
                continue
            reads += 1
            if ncache:
                lru[k] = 1
                if len(lru) > ncache:
                    lru.popitem(last=False)
    return reads + migr, migr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--rows", default="128,256,384,512,768")
    ap.add_argument("--warm", type=int, default=256)
    ap.add_argument("--period", type=int, default=32)
    ap.add_argument("--decay", type=float, default=0.5)
    ap.add_argument("--pinned-frac", type=float, default=0.75)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    meta, recs = load(a.trace)
    warm, ev = recs[:a.warm], recs[a.warm:]
    routed = sum(len(v) for r in ev for v in r["routed"].values())
    print("trace: %d steps; profile 0-%d, score %d-%d (%d routed slots)" % (
        meta["steps"], a.warm - 1, a.warm, meta["steps"] - 1, routed))
    print("period=%d  ewma half-life=%.2f/period  hybrid pin=%.0f%%\n" % (
        a.period, a.decay, a.pinned_frac * 100))

    pol = [("static", static_p, {}),
           ("adaptive", adaptive_p, {"period": a.period}),
           ("ewma", adaptive_p, {"period": a.period, "decay": a.decay}),
           ("hybrid", hybrid_p, {"period": a.period,
                                 "pinned_frac": a.pinned_frac}),
           ("demand", demand_p, {}),
           ("oracle", oracle_p, {})]
    print("%6s | %s | %s" % ("rows",
                             " ".join("%9s" % n for n, _, _ in pol),
                             "best non-oracle"))
    out = {"meta": meta, "warm": a.warm, "period": a.period,
           "decay": a.decay, "pinned_frac": a.pinned_frac, "points": []}
    for cap in [int(x) for x in a.rows.split(",")]:
        row, rec = {}, {"rows": cap}
        for name, fn, kw in pol:
            reads, migr = fn(warm, ev, cap, **kw)
            row[name] = reads
            rec[name] = reads
            rec[name + "_migrations"] = migr
        best = min((v, k) for k, v in row.items() if k != "oracle")
        rec["best"] = best[1]
        rec["best_vs_oracle_pct"] = (best[0] - row["oracle"]) / row["oracle"] * 100
        rec["adaptive_vs_oracle_pct"] = (row["adaptive"] - row["oracle"]) / row["oracle"] * 100
        out["points"].append(rec)
        print("%6d | %s | %s (+%.1f%% over oracle, adaptive was +%.1f%%)" % (
            cap, " ".join("%9d" % row[n] for n, _, _ in pol), best[1],
            rec["best_vs_oracle_pct"], rec["adaptive_vs_oracle_pct"]))
    # Both knobs swept and recorded. A policy that wins only at the setting
    # it was introduced with has been fitted, not found.
    caps = [int(x) for x in a.rows.split(",")]
    decays = [1.0, 0.9, 0.75, 0.5, 0.25, 0.1]
    print("\nEWMA decay per re-placement (1.0 = no decay = plain adaptive):")
    print("%6s %s %8s" % ("rows", " ".join("%8s" % d for d in decays), "oracle"))
    out["decay_sweep"] = []
    for cap in caps:
        vals = [adaptive_p(warm, ev, cap, period=a.period,
                           decay=(None if d == 1.0 else d))[0] for d in decays]
        o = oracle_p(warm, ev, cap)[0]
        out["decay_sweep"].append({"rows": cap, "decays": decays,
                                   "reads": vals, "oracle": o})
        print("%6d %s %8d" % (cap, " ".join("%8d" % v for v in vals), o))

    fracs = [0.25, 0.5, 0.75, 0.9, 1.0]
    print("\nHybrid pinned fraction (1.0 = all pinned = plain adaptive):")
    print("%6s %s" % ("rows", " ".join("%8s" % f for f in fracs)))
    out["pin_sweep"] = []
    for cap in caps:
        vals = [hybrid_p(warm, ev, cap, period=a.period, pinned_frac=f)[0]
                for f in fracs]
        out["pin_sweep"].append({"rows": cap, "fracs": fracs, "reads": vals})
        print("%6d %s" % (cap, " ".join("%8d" % v for v in vals)))

    if a.out:
        json.dump(out, open(a.out, "w"), indent=2)
        print("\nreceipt ->", a.out)


if __name__ == "__main__":
    main()
