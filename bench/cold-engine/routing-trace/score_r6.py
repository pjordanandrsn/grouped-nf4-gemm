"""R6, scored in REFILLS: is there a pressure band where reclaimable wins?

Registered (PREREG-tribrid-stage3, R6):

    largest gains at active working set ~1.1-2x protected fast-tier capacity
    -- REFUTED IF gains are flat across pressure.

R6 needs a *gain* to locate. `RESULTS-tribrid-reclaimable.md` scored it
CONFIRMED on the grounds that P -- the resurrection rate -- rises
monotonically as ownership tightens. That reading does not survive the
campaign's own conclusion: STAGE3-SYNTHESIS retired the resurrection rate as
a metric that can carry a claim ("a capacity-relative bookkeeping event, not
a saving"), because a cache that demoted better would never have demoted the
row it now gets credit for resurrecting. Scoring R6 on it measures how much
churn the policy creates, not how much work it saves.

This scores it on PHYSICAL REFILLS, the metric that survived, at MATCHED
CAPACITY -- both arms holding the same physical rows, differing only in
whether ownership is capped below that number. That is R10's design, and it
is the only one where "gain" means anything: giving the soft arm fewer
protected rows AND the same total is the comparison a scheduler faces.

x-axis is pressure: the trace's distinct-row working set over the arm's
protected capacity. R6 places its peak at 1.1-2x.
"""
import argparse
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "kernel"))

from nvme_arena import bake, load_index                # noqa: E402
from nvme_residency import ColdTier                    # noqa: E402


def build(tmp, layers, experts):
    import test_nvme_arena as tna
    tna.L, tna.E = layers, experts
    snap = os.path.join(tmp, "snap")
    tna.make_snapshot(snap)
    path = os.path.join(tmp, "toy.arena")
    bake(snap, path, align=4096, log=lambda *a: None)
    return path, load_index(path)


def replay(path, index, recs, rows, protected):
    t = ColdTier(path, hot_rows=rows, pinned=False, index=index,
                 protected_rows=protected, qd=1)     # qd=1: deterministic
    try:
        for r in recs:
            for L, ex in r["routed"].items():
                t.ensure(int(L), ex)
        s = t.stats()
        # P is the tier's OWN reuse_before_overwrite: resurrections over
        # rows that RESOLVED (were reused or overwritten). Recomputing it as
        # resurrections/refills is a different quantity and can exceed 100%.
        return (s.get("misses", 0), s.get("evictions", 0),
                s.get("reuse_before_overwrite"))
    finally:
        t.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--rows", default="128,192,256,384,512,768,1024")
    ap.add_argument("--frac", type=float, default=0.75,
                    help="protected fraction of rows; holds the ownership "
                         "cap CONSTANT while pressure varies. A fixed row "
                         "GAP instead would shrink the reclaimable fraction "
                         "as rows grows (8/128=6%% vs 8/768=1%%), driving "
                         "gain->0 by construction and faking a pressure "
                         "trend. That confound is the whole risk here.")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    with open(a.trace) as f:
        rowsj = [json.loads(line) for line in f]
    meta, recs = rowsj[0]["meta"], rowsj[1:]
    L, E = meta["layers"], meta["n_experts"]
    ws = len({(int(lyr), int(e)) for r in recs
              for lyr, ex in r["routed"].items() for e in ex})
    print(f"trace: {meta['steps']} steps x {L}L x top-{meta['top_k']} of {E}")
    print(f"working set (distinct rows touched): {ws}")
    out = {"meta": meta, "working_set": ws, "points": []}
    with tempfile.TemporaryDirectory() as tmp:
        path, index = build(tmp, L, E)
        print(f"\n  frac={a.frac} (reclaimable fraction held CONSTANT)")
        print(f"{'rows':>5} {'prot':>5} {'ws/prot':>8} {'hard refills':>13} "
              f"{'soft refills':>13} {'gain':>8} {'gain %':>8} """
              f"{'P (retired)':>13}")
        for rows in [int(v) for v in a.rows.split(",")]:
            prot = max(1, int(round(rows * a.frac)))
            if prot >= rows:
                continue
            hm, he, _ = replay(path, index, recs, rows, rows)   # hard
            sm, se, sr = replay(path, index, recs, rows, prot)  # soft
            gain = hm - sm                     # positive => soft reads less
            p = {"rows": rows, "protected": prot, "pressure": ws / prot,
                 "hard_refills": hm, "soft_refills": sm, "gain": gain,
                 "gain_pct": gain / hm * 100 if hm else 0.0,
                 "hard_evictions": he, "soft_evictions": se,
                 "soft_P_reuse_before_overwrite": sr}
            out["points"].append(p)
            print(f"{rows:>5} {prot:>5} {ws/prot:>8.2f} {hm:>13} {sm:>13} "
                  f"{gain:>+8} {p['gain_pct']:>+7.2f}% "
                  f"{(f'{sr * 100:.2f}%' if sr is not None else '-'):>13}")
    best = max(out["points"], key=lambda x: x["gain"])
    print(f"\nbest gain {best['gain']:+d} refills at pressure "
          f"{best['pressure']:.2f}x (R6 predicts a peak in 1.1-2.0x)")
    if best["gain"] <= 0:
        print("NO CONFIGURATION GAINS: there is no peak to locate.")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=2)
        print("receipt ->", a.out)


if __name__ == "__main__":
    main()
