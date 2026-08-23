"""P2-G2': the single-pool reuse law in replay -- real DevRowCache.

Registered in bench/cold-engine/PREREG-p2-g2p.md. G2's protocol with the
per-key persistent machinery deleted (spec S5 G2' re-founding) and the two
registered scope corrections: sweep to 2048, clause (c) on capacity-adequate
arms only. Shares G2's loader/baselines; the controller body here IS the
whole law -- one DevRowCache, protected = rows - k, budget, burst.
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
from replay_dev_cache import load                       # noqa: E402
from g2_replay import score, ETA                        # noqa: E402

ADEQ = 0.9


def run_controller(meta, recs, rows, promo_frac=None, spoiler=None):
    k = int(meta["top_k"])
    m = int(meta["layers"]) * k
    protected = 1 if spoiler == "margin" else None
    c = DevRowCache(rows, 8, device="cpu", routed=k, protected=protected)
    cap = math.inf if promo_frac is None else math.ceil(promo_frac * m)
    seen = set()
    fills_series, novelty_series, miss_series = [], [], []
    for t, r in enumerate(recs):
        budget = cap
        fills = novelty = miss = 0
        for lay, ex in sorted(r["routed"].items(), key=lambda kv: int(kv[0])):
            L = int(lay)
            for e in ex:
                if (L, e) not in seen:
                    seen.add((L, e))
                    novelty += 1
            tag = StepTag("cpu")
            _assign, need = c.want(L, ex, tag)
            tag.record()
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
            miss += len(need)
        fills_series.append(fills)
        novelty_series.append(novelty)
        miss_series.append(miss)
    return {"fills": fills_series, "novelty": novelty_series,
            "miss": miss_series, "m": m}


def lru_eval_window(recs, rows, count_from=128):
    cache, f = OrderedDict(), 0
    for t, r in enumerate(recs):
        for L, ex in r["routed"].items():
            for e in ex:
                key = (int(L), e)
                if key in cache:
                    cache.move_to_end(key)
                    continue
                if t >= count_from:
                    f += 1
                cache[key] = 1
                if len(cache) > rows:
                    cache.popitem(last=False)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="128,256,512,1024,2048")
    ap.add_argument("--fracs", default="none,0.0625,0.125,0.25")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows_list = [int(x) for x in a.rows.split(",")]
    fracs = [None if x == "none" else float(x) for x in a.fracs.split(",")]

    tdir = os.path.join(HERE, "..", "rank-2026-08-22")
    traces = sorted(f for f in os.listdir(tdir) if f.endswith(".jsonl"))
    out = {"eta": ETA, "adequate_frac": ADEQ, "traces": {}}
    per_trace_a = {}
    ok_b, ok_c, ok_d = [], [], []
    sp_margin_fail, sp_noretain_fail = [], []
    for tr in traces:
        meta, recs = load(os.path.join(tdir, tr))
        pairs = len({(int(L), e) for r in recs
                     for L, ex in r["routed"].items() for e in ex})
        per_tr = {"pairs": pairs, "caps": {}}
        adequate_any = False
        for rows in rows_list:
            if rows > pairs:
                # one past-pairs capacity is kept so every trace has an
                # adequate arm even when pairs falls between sweep points
                if adequate_any:
                    continue
            adequate = pairs <= ADEQ * rows
            adequate_any = adequate_any or adequate
            lru_f = lru_eval_window(recs, rows)
            entry = {"lru_fills_128_512": lru_f, "adequate": adequate,
                     "arms": {}}
            for frac in fracs:
                res = run_controller(meta, recs, rows, promo_frac=frac)
                sc = score(res)
                entry["arms"][str(frac)] = sc
                if frac is None:
                    conv_ok = (sc["converged_at"] is not None
                               and sc["converged_at"] <= 64)
                    per_trace_a[tr] = per_trace_a.get(tr, True) and conv_ok
                    ok_b.append(sc["total_fills_128_512"] <= 1.10 * lru_f)
                    if adequate:
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
        big = max(int(x) for x in per_tr["caps"])
        bige = per_tr["caps"][str(big)]
        sp = {}
        for spoiler in ("margin", "noretain"):
            res = run_controller(meta, recs, big, spoiler=spoiler)
            sc = score(res)
            sc["all_miss_frac"] = sc["plateau_fills_step"] / res["m"]
            sp[spoiler] = sc
        per_tr["spoilers"] = sp
        sp_margin_fail.append(
            (not sp["margin"]["churn_ok"])
            or sp["margin"]["total_fills_128_512"] > 1.10 * bige["lru_fills_128_512"])
        sp_noretain_fail.append(sp["noretain"]["all_miss_frac"] >= 0.90)
        out["traces"][tr] = per_tr
        print("%-24s pairs=%5d  conv=%s  plateau=%s  adequate_any=%s" % (
            tr, pairs, bige["arms"]["None"]["converged_at"],
            bige["arms"]["None"]["plateau_fills_step"], adequate_any))
        if not adequate_any:
            out["verdict"] = "UNINFORMATIVE"
            print("trace with no adequate arm -- registered UNINFORMATIVE")

    a_pass = sum(1 for v in per_trace_a.values() if v)
    spoilers_ok = all(sp_margin_fail) and all(sp_noretain_fail)
    clauses = {
        "a_convergence": {"traces_pass": a_pass, "traces": len(per_trace_a),
                          "ok": a_pass >= min(14, len(per_trace_a))},
        "b_plateau_vs_lru": {"ok_arms": sum(ok_b), "arms": len(ok_b),
                             "ok": all(ok_b)},
        "c_churn_adequate": {"ok_arms": sum(ok_c), "arms": len(ok_c),
                             "ok": all(ok_c)},
        "d_throttle_graceful": {"ok_arms": sum(ok_d), "arms": len(ok_d),
                                "ok": all(ok_d)},
        "spoilers_must_fail": {"margin": sum(sp_margin_fail),
                               "noretain": sum(sp_noretain_fail),
                               "ok": spoilers_ok},
    }
    if out.get("verdict") == "UNINFORMATIVE" or not spoilers_ok:
        verdict = "UNINFORMATIVE"
    elif all(c["ok"] for k_, c in clauses.items()
             if k_ != "spoilers_must_fail"):
        verdict = "PASS"
    else:
        verdict = "REFUTED"
    out["clauses"] = clauses
    out["verdict"] = verdict
    print(json.dumps(clauses, indent=1))
    print("G2':", verdict)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print("receipt ->", a.out)


if __name__ == "__main__":
    main()
