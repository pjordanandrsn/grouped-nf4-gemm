"""P2-G2: the reuse-law controller in replay, driving the REAL DevRowCache.

Registered in bench/cold-engine/PREREG-p2-g2.md. Spec: SPEC-elastic-phase2.md
S5 (the law), S11 (the gate). Runs offline on the 16 committed rank traces,
cold start, scored in fill/miss trace units -- no box.

The shipped artifact does the caching: every routed set goes through
DevRowCache.want() exactly as the engine's flow does; the SMOOTH_CAP budget
uses the cache's own failed-fill API (discard()) for un-budgeted misses, so
throttling exercises shipped semantics rather than a model of them.
Externally tracked state (ages, per-period hits, the persistent set) is the
controller's own bookkeeping -- the part G2 exists to test.
"""
import argparse
import json
import math
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "kernel"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "kernel"))

from dev_row_cache import DevRowCache, StepTag          # noqa: E402
from replay_dev_cache import load, lru_transfers        # noqa: E402

PERIOD = 64                # trace-scaled (spec default ~256 is for serving;
                           # registered in the prereg: 512-step traces need
                           # boundaries the persistent law can actually cross)
AGE_MIN = 2 * PERIOD
ETA = 0.25
PLATEAU_LO, PLATEAU_HI = 256, 512
CONV_WIN = 32


def run_controller(meta, recs, rows_total, promo_frac=None, spoiler=None,
                   pers_frac=0.25):
    """One cold-start replay. Returns per-step series + totals.

    spoiler: None | 'margin' (protected = rows-1, the I1 trap: margin 1
    against routed sets of k) | 'noretain' (every fill discarded after
    counting -- residency never forms)."""
    k = int(meta["top_k"])
    layers = int(meta["layers"])
    m = layers * k
    rows_p_cap = int(rows_total * pers_frac)
    rows_t = rows_total - rows_p_cap
    protected = (rows_t - 1) if spoiler == "margin" else None
    c = DevRowCache(rows_t, 8, device="cpu", routed=k, protected=protected)
    cap = math.inf if promo_frac is None else math.ceil(promo_frac * m)

    persistent = set()
    filled_at = {}                       # key -> step of the CURRENT fill
    period_hits = defaultdict(int)       # key -> hits this period
    prev_hits = defaultdict(int)         # key -> hits last period
    pers_zero = defaultdict(int)         # key -> consecutive zero periods
    seen = set()
    fills_series, novelty_series, miss_series = [], [], []
    pers_events = []                     # (step, promoted, demoted)

    for t, r in enumerate(recs):
        budget = cap
        fills = novelty = miss = 0
        for lay, ex in sorted(r["routed"].items(), key=lambda kv: int(kv[0])):
            L = int(lay)
            for e in ex:
                if (L, e) not in seen:
                    seen.add((L, e))
                    novelty += 1
            pers = [e for e in ex if (L, e) in persistent]
            for e in pers:
                period_hits[(L, e)] += 1
            rest = [e for e in ex if (L, e) not in persistent]
            if not rest:
                continue
            tag = StepTag("cpu")
            assign, need = c.want(L, rest, tag)
            tag.record()
            hits = [e for e in rest if e not in need]
            for e in hits:
                period_hits[(L, e)] += 1
            if spoiler == "noretain":
                fills += len(need)
                miss += len(need)
                c.discard(L, need)
                continue
            take = need if budget >= len(need) else need[:int(budget)]
            drop = need[len(take):]
            if drop:
                c.discard(L, drop)
            c.note_filled(len(take))
            budget -= len(take)
            fills += len(take)
            miss += len(need)            # cold whether budgeted or not
            for e in take:
                key = (L, e)
                if key in filled_at:
                    pass                 # re-fill: eviction happened; age
                filled_at[key] = t       # resets to this fill (spec S5)
        fills_series.append(fills)
        novelty_series.append(novelty)
        miss_series.append(miss)

        if (t + 1) % PERIOD == 0 and spoiler is None:
            promoted, demoted = [], []
            room = rows_p_cap - len(persistent)
            if room > 0:
                cand = [(prev_hits[k_] + period_hits[k_], k_)
                        for k_, ft in filled_at.items()
                        if (t - ft) >= AGE_MIN and k_ not in persistent
                        and (prev_hits[k_] + period_hits[k_]) > 0]
                cand.sort(reverse=True)
                for _, k_ in cand[:room]:
                    persistent.add(k_)
                    promoted.append(k_)
            for k_ in list(persistent):
                if period_hits[k_] == 0:
                    pers_zero[k_] += 1
                    if pers_zero[k_] >= 2:
                        persistent.discard(k_)
                        demoted.append(k_)
                else:
                    pers_zero[k_] = 0
            if promoted or demoted:
                pers_events.append((t, len(promoted), len(demoted)))
            prev_hits = period_hits
            period_hits = defaultdict(int)

    return {"fills": fills_series, "novelty": novelty_series,
            "miss": miss_series, "pers_events": pers_events,
            "pers_size": len(persistent), "m": m, "rows_t": rows_t,
            "rows_p_cap": rows_p_cap}


def trailing(series, t, w=CONV_WIN):
    lo = max(0, t - w + 1)
    return sum(series[lo:t + 1]) / (t + 1 - lo)


def score(res):
    fills = res["fills"]
    plateau = sorted(fills[PLATEAU_LO:PLATEAU_HI])[len(fills[PLATEAU_LO:PLATEAU_HI]) // 2]
    conv = None
    for t in range(len(fills)):
        if trailing(fills, t) <= 1.10 * plateau + 1.0:
            conv = t
            break
    ew_f = ew_n = None
    a = 1 / 16
    for t in range(PLATEAU_LO, len(fills)):
        ew_f = fills[t] if ew_f is None else (1 - a) * ew_f + a * fills[t]
        ew_n = (res["novelty"][t] if ew_n is None
                else (1 - a) * ew_n + a * res["novelty"][t])
    churn_ok = ew_f <= (1 + ETA) * ew_n + 1.0     # +1 absolute: tiny EWMAs
    return {"plateau_fills_step": plateau, "converged_at": conv,
            "total_fills_128_512": sum(fills[128:512]),
            "ewma_fills": ew_f, "ewma_novelty": ew_n, "churn_ok": churn_ok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="128,256,512,1024")
    ap.add_argument("--fracs", default="none,0.0625,0.125,0.25")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows_list = [int(x) for x in a.rows.split(",")]
    fracs = [None if x == "none" else float(x) for x in a.fracs.split(",")]

    traces = sorted(f for f in os.listdir(HERE)
                    if f.endswith(".jsonl") and "routing_seq" not in f)
    out = {"period": PERIOD, "age_min": AGE_MIN, "eta": ETA, "traces": {}}
    ok_a = tot_a = 0
    ok_b, ok_c, ok_d = [], [], []
    for tr in traces:
        meta, recs = load(os.path.join(HERE, tr))
        pairs = len({(int(L), e) for r in recs
                     for L, ex in r["routed"].items() for e in ex})
        per_tr = {"pairs": pairs, "caps": {}}
        for rows in rows_list:
            if rows > pairs:
                continue
            lru = lru_transfers(meta, recs, rows)
            lru_eval = None            # LRU fills over the eval window
            # recompute LRU over steps 128..512 for the (b) window
            from collections import OrderedDict
            cache, f128 = OrderedDict(), 0
            for t, r in enumerate(recs):
                for L, ex in r["routed"].items():
                    for e in ex:
                        key = (int(L), e)
                        if key in cache:
                            cache.move_to_end(key)
                            continue
                        if t >= 128:
                            f128 += 1
                        cache[key] = 1
                        if len(cache) > rows:
                            cache.popitem(last=False)
            lru_eval = f128
            entry = {"lru_fills_total": lru, "lru_fills_128_512": lru_eval,
                     "arms": {}}
            for frac in fracs:
                res = run_controller(meta, recs, rows, promo_frac=frac)
                sc = score(res)
                sc["pers_size"] = res["pers_size"]
                entry["arms"][str(frac)] = sc
                tag = "unthrottled" if frac is None else f"frac={frac}"
                if frac is None:
                    tot_a += 1
                    if sc["converged_at"] is not None and sc["converged_at"] <= 64:
                        ok_a += 1
                    ok_b.append(sc["total_fills_128_512"] <= 1.10 * lru_eval)
                    if rows == max(x for x in rows_list if x <= pairs):
                        ok_c.append(sc["churn_ok"])
            un = entry["arms"]["None"]
            for frac in fracs:
                if frac is None:
                    continue
                th = entry["arms"][str(frac)]
                conv_ok = (th["converged_at"] is not None
                           and un["converged_at"] is not None
                           and th["converged_at"] <= 2 * max(un["converged_at"], 16))
                fill_ok = th["total_fills_128_512"] <= 1.05 * un["total_fills_128_512"]
                ok_d.append(conv_ok and fill_ok)
            per_tr["caps"][str(rows)] = entry
        # spoilers at the largest applicable capacity, unthrottled
        rows = max(x for x in rows_list if x <= pairs)
        sp = {}
        for spoiler in ("margin", "noretain"):
            res = run_controller(meta, recs, rows, spoiler=spoiler)
            sc = score(res)
            routed_per_step = res["m"]
            sc["all_miss_frac"] = sc["plateau_fills_step"] / routed_per_step
            sp[spoiler] = sc
        per_tr["spoilers"] = sp
        out["traces"][tr] = per_tr
        print("%-24s pairs=%5d  conv=%s  plateau=%s" % (
            tr, pairs,
            per_tr["caps"][str(rows)]["arms"]["None"]["converged_at"],
            per_tr["caps"][str(rows)]["arms"]["None"]["plateau_fills_step"]))

    verdict = {
        "a_convergence": {"ok": ok_a, "total": tot_a, "bar": "<=64 steps on >=14/16 traces (unthrottled, per capacity)"},
        "b_plateau_vs_lru": {"ok": sum(ok_b), "total": len(ok_b)},
        "c_churn_at_max_cap": {"ok": sum(ok_c), "total": len(ok_c)},
        "d_throttle_graceful": {"ok": sum(ok_d), "total": len(ok_d)},
    }
    out["verdict_detail"] = verdict
    print(json.dumps(verdict, indent=1))
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print("receipt ->", a.out)


if __name__ == "__main__":
    main()
