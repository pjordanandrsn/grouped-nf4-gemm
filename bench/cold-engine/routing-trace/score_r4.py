"""R4, scored on a real captured routing sequence.

Registered (PREREG-tribrid-stage3, R4):

    short-window recurrence predicts resurrection better than long-run
    expert frequency
    -- REFUTED IF global frequency predicts as well or better.

Both predictors already live in `reuse_profile.ReuseProfile`; what was
missing was ground truth. A resurrection is a hit on a row that had lost
capacity ownership but had not yet been overwritten, so it only exists
relative to a cache of some size -- which is why this sweeps capacity rather
than reporting one number. If the answer flips with capacity, that is the
result, not a nuisance.

Ground truth is read from the slot state BEFORE the request: a routed expert
whose slot is RECLAIMABLE at that moment is about to be resurrected. Taking
it from VramSlots' aggregate counter would give a total with no per-expert
attribution, and R4 is a question about ranking experts.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "kernel"))
from dev_row_cache import DevRowCache, StepTag        # noqa: E402
from reuse_profile import ReuseProfile                 # noqa: E402


def score(recs, rows, window):
    cache = DevRowCache(rows, 8, device="cpu")
    prof = ReuseProfile(window=window)
    n_res = 0
    for step, r in enumerate(recs):
        for L, experts in r["routed"].items():
            layer = int(L)
            # BEFORE the request: which of these are reclaimable right now?
            res = set()
            for e in experts:
                s = cache.slots.slot_of((layer, e))
                if s is not None and cache.slots.state(s) == "reclaimable":
                    res.add(e)
            t = StepTag("cpu")
            cache.want(layer, experts, t)
            t.record()
            for e in experts:
                prof.observe((layer, e), step, resurrected=(e in res))
            n_res += len(res)
    return prof.predictor_scores(), n_res, cache.stats()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--rows", default="128,192,256,384,512,768")
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--windows", default="4,8,16,32,64,128",
                    help="recurrence windows to sweep. R4 is a claim about "
                         "SHORT windows, so a result that holds at only one "
                         "of them is tuning, not a finding.")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    with open(a.trace) as f:
        rowsj = [json.loads(line) for line in f]
    meta, recs = rowsj[0]["meta"], rowsj[1:]
    print(f"trace: {meta['steps']} steps x {meta['layers']} layers x "
          f"top-{meta['top_k']} of {meta['n_experts']}; window={a.window} ticks")
    print(f"\n{'rows':>6} {'resurrections':>13} {'recency ρ':>10} "
          f"{'frequency ρ':>12} {'verdict':>22}")
    out = {"meta": meta, "window": a.window, "points": []}
    for rows in [int(x) for x in a.rows.split(",")]:
        sc, n_res, st = score(recs, rows, a.window)
        rec_, frq = sc.get("recency"), sc.get("frequency")
        if rec_ is None or frq is None:
            verdict = "undefined"
        elif rec_ > frq:
            verdict = "R4 holds"
        elif rec_ == frq:
            verdict = "REFUTED (tie)"
        else:
            verdict = "REFUTED (freq wins)"
        out["points"].append({"rows": rows, "resurrections": n_res,
                              "recency": rec_, "frequency": frq,
                              "n_keys": sc.get("n"), "verdict": verdict})
        rs = "None" if rec_ is None else f"{rec_:.4f}"
        fs = "None" if frq is None else f"{frq:.4f}"
        print(f"{rows:>6} {n_res:>13} {rs:>10} {fs:>12} {verdict:>22}")
    # The window sweep. Reported at every capacity that produced
    # resurrections, because a predictor that only wins at one window width
    # has been fitted rather than tested.
    wins = [int(x) for x in a.windows.split(",")]
    grid, tally = [], {"recency": 0, "frequency": 0, "tie": 0, "undefined": 0}
    print(f"\n{'rows':>6} {'window':>7} {'res':>5} {'recency':>9} "
          f"{'frequency':>10} {'winner':>10}")
    for pt in out["points"]:
        if not pt["resurrections"]:
            continue
        for w in wins:
            sc, n, _ = score(recs, pt["rows"], w)
            r, f = sc.get("recency"), sc.get("frequency")
            if r is None or f is None:
                who = "undefined"
                rs = fs = "None"
            else:
                who = ("recency" if r > f else "tie" if r == f else "frequency")
                rs, fs = f"{r:.4f}", f"{f:.4f}"
            tally[who] += 1
            grid.append({"rows": pt["rows"], "window": w, "resurrections": n,
                         "recency": r, "frequency": f, "winner": who})
            print(f"{pt['rows']:>6} {w:>7} {n:>5} {rs:>9} {fs:>10} {who:>10}")
    out["window_sweep"] = grid
    out["tally"] = tally
    print(f"\ncells: {tally}")
    print("R4 REFUTED" if tally["frequency"] >= tally["recency"]
          else "R4 holds")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=2)
        print("receipt ->", a.out)


if __name__ == "__main__":
    main()
