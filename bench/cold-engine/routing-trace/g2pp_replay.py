"""P2-G2'': the single-pool replay scored by the spec's own equilibrium
definition. Registered in bench/cold-engine/PREREG-p2-g2pp.md.

Reuses g2p_replay's controller and baselines verbatim -- the ONLY change is
scoring: (a) = time-to-sustained-equilibrium (S5's predicate), (b)/(d) fill
bars gain a one-routed-set absolute guard.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from replay_dev_cache import load                             # noqa: E402
from g2_replay import ETA                                     # noqa: E402
from g2p_replay import ADEQ, lru_eval_window, run_controller  # noqa: E402

ALPHA = 1.0 / 16


def sustained_equilibrium_step(fills, novelty):
    """First step where the S5 predicate holds and keeps holding."""
    ew_f = ew_n = None
    ok_from = None
    for t in range(len(fills)):
        ew_f = fills[t] if ew_f is None else (1 - ALPHA) * ew_f + ALPHA * fills[t]
        ew_n = (novelty[t] if ew_n is None
                else (1 - ALPHA) * ew_n + ALPHA * novelty[t])
        if ew_f <= (1 + ETA) * ew_n + 1.0:
            if ok_from is None:
                ok_from = t
        else:
            ok_from = None
    return ok_from


def score2(res):
    fills, nov = res["fills"], res["novelty"]
    conv = sustained_equilibrium_step(fills, nov)
    ew_f = ew_n = None
    for t in range(256, len(fills)):
        ew_f = fills[t] if ew_f is None else (1 - ALPHA) * ew_f + ALPHA * fills[t]
        ew_n = nov[t] if ew_n is None else (1 - ALPHA) * ew_n + ALPHA * nov[t]
    lo = sorted(fills[256:512])
    return {"converged_at": conv,
            "total_fills_128_512": sum(fills[128:512]),
            "plateau_fills_step": lo[len(lo) // 2],
            "ewma_fills": ew_f, "ewma_novelty": ew_n,
            "churn_ok": ew_f <= (1 + ETA) * ew_n + 1.0}


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
    out = {"eta": ETA, "alpha": ALPHA, "traces": {}}
    per_trace_a = {}
    ok_b, ok_c, ok_d = [], [], []
    sp_margin_fail, sp_noretain_fail = [], []
    for tr in traces:
        meta, recs = load(os.path.join(tdir, tr))
        m = int(meta["layers"]) * int(meta["top_k"])
        pairs = len({(int(L), e) for r in recs
                     for L, ex in r["routed"].items() for e in ex})
        per_tr = {"pairs": pairs, "m": m, "caps": {}}
        adequate_any = False
        for rows in rows_list:
            if rows > pairs and adequate_any:
                continue
            adequate = pairs <= ADEQ * rows
            adequate_any = adequate_any or adequate
            lru_f = lru_eval_window(recs, rows)
            entry = {"lru_fills_128_512": lru_f, "adequate": adequate,
                     "arms": {}}
            for frac in fracs:
                res = run_controller(meta, recs, rows, promo_frac=frac)
                sc = score2(res)
                entry["arms"][str(frac)] = sc
                if frac is None:
                    conv_ok = (sc["converged_at"] is not None
                               and sc["converged_at"] <= 64)
                    per_trace_a[tr] = per_trace_a.get(tr, True) and conv_ok
                    ok_b.append(sc["total_fills_128_512"]
                                <= 1.10 * lru_f + m)
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
                fill_ok = (th["total_fills_128_512"]
                           <= 1.05 * un["total_fills_128_512"] + m)
                ok_d.append(conv_ok and fill_ok)
            per_tr["caps"][str(rows)] = entry
        big = max(int(x) for x in per_tr["caps"])
        bige = per_tr["caps"][str(big)]
        sp = {}
        for spoiler in ("margin", "noretain"):
            res = run_controller(meta, recs, big, spoiler=spoiler)
            sc = score2(res)
            sc["all_miss_frac"] = sc["plateau_fills_step"] / m
            sp[spoiler] = sc
        per_tr["spoilers"] = sp
        sp_margin_fail.append(
            (not sp["margin"]["churn_ok"])
            or sp["margin"]["total_fills_128_512"] > 1.10 * bige["lru_fills_128_512"] + m)
        sp_noretain_fail.append(sp["noretain"]["all_miss_frac"] >= 0.90)
        out["traces"][tr] = per_tr
        print("%-24s pairs=%5d  conv=%s" % (
            tr, pairs, bige["arms"]["None"]["converged_at"]))

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
    if not spoilers_ok:
        verdict = "UNINFORMATIVE"
    elif all(c["ok"] for k_, c in clauses.items()
             if k_ != "spoilers_must_fail"):
        verdict = "PASS"
    else:
        verdict = "REFUTED"
    out["clauses"] = clauses
    out["verdict"] = verdict
    print(json.dumps(clauses, indent=1))
    print("G2pp:", verdict)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print("receipt ->", a.out)


if __name__ == "__main__":
    main()
