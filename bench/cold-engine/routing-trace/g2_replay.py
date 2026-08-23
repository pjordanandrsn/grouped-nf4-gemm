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
from collections import OrderedDict, defaultdict

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

    spoiler: None | 'margin' (protected = 1, the I1 thrash trap: margin
    rows_t - 1 >> k) | 'noretain' (every fill discarded after counting --
    residency never forms)."""
    k = int(meta["top_k"])
    layers = int(meta["layers"])
    m = layers * k
    rows_p_cap = int(rows_total * pers_frac)
    rows_t = rows_total - rows_p_cap
    # I1's two failure modes: margin < k is UNSERVABLE (want() stalls), so
    # the runnable thrash spoiler is the other side -- protected = 1, margin
    # = rows_t - 1 >> k, the measured 6,144-fills-for-96-keys regime.
    protected = 1 if spoiler == "margin" else None
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
                # a re-fill means the key was evicted since its last fill;
                # overwriting resets its resident-age to this fill (spec S5)
                filled_at[(L, e)] = t
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

    tdir = os.path.join(HERE, "..", "rank-2026-08-22")
    traces = sorted(f for f in os.listdir(tdir) if f.endswith(".jsonl"))
    out = {"period": PERIOD, "age_min": AGE_MIN, "eta": ETA, "traces": {}}
    per_trace_a = {}                 # trace -> all-caps convergence pass
    ok_b, ok_c, ok_d = [], [], []
    spoiler_margin_fail, spoiler_noretain_fail = [], []
    for tr in traces:
        meta, recs = load(os.path.join(tdir, tr))
        pairs = len({(int(L), e) for r in recs
                     for L, ex in r["routed"].items() for e in ex})
        per_tr = {"pairs": pairs, "caps": {}}
        for rows in rows_list:
            if rows > pairs:
                continue
            lru = lru_transfers(meta, recs, rows)
            lru_eval = None            # LRU fills over the eval window
            # recompute LRU over steps 128..512 for the (b) window
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
                    conv_ok = (sc["converged_at"] is not None
                               and sc["converged_at"] <= 64)
                    per_trace_a[tr] = per_trace_a.get(tr, True) and conv_ok
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
        # registered must-fail conditions (PREREG): margin blows churn or
        # plateau-vs-LRU at its capacity; noretain plateaus at >= 0.90 all-miss
        big = per_tr["caps"][str(rows)]
        spoiler_margin_fail.append(
            (not sp["margin"]["churn_ok"])
            or sp["margin"]["total_fills_128_512"] > 1.10 * big["lru_fills_128_512"])
        spoiler_noretain_fail.append(sp["noretain"]["all_miss_frac"] >= 0.90)
        out["traces"][tr] = per_tr
        print("%-24s pairs=%5d  conv=%s  plateau=%s" % (
            tr, pairs,
            per_tr["caps"][str(rows)]["arms"]["None"]["converged_at"],
            per_tr["caps"][str(rows)]["arms"]["None"]["plateau_fills_step"]))

    a_pass = sum(1 for v in per_trace_a.values() if v)
    spoilers_ok = all(spoiler_margin_fail) and all(spoiler_noretain_fail)
    clauses = {
        "a_convergence": {"traces_pass": a_pass, "traces": len(per_trace_a),
                          "ok": a_pass >= min(14, len(per_trace_a))},
        "b_plateau_vs_lru": {"ok_arms": sum(ok_b), "arms": len(ok_b),
                             "ok": all(ok_b)},
        "c_churn_at_max_cap": {"ok_traces": sum(ok_c), "traces": len(ok_c),
                               "ok": all(ok_c)},
        "d_throttle_graceful": {"ok_arms": sum(ok_d), "arms": len(ok_d),
                                "ok": all(ok_d)},
        "spoilers_must_fail": {"margin": sum(spoiler_margin_fail),
                               "noretain": sum(spoiler_noretain_fail),
                               "ok": spoilers_ok},
    }
    if not spoilers_ok:
        verdict = "UNINFORMATIVE"
    elif all(c["ok"] for k_, c in clauses.items() if k_ != "spoilers_must_fail"):
        verdict = "PASS"
    else:
        verdict = "REFUTED"
    out["clauses"] = clauses
    out["verdict"] = verdict
    print(json.dumps(clauses, indent=1))
    print("G2:", verdict)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print("receipt ->", a.out)


if __name__ == "__main__":
    main()
