"""R7, scored: does reclaimable residency move the NVMe knee?

Registered (PREREG-tribrid-stage3, R7):

    reclaimable residency moves the NVMe knee outward by 20-50% on workloads
    with temporal locality -- REFUTED IF knee unmoved.

The knee is in CAPACITY space: the point below which shrinking the pool
starts costing reads sharply. "Outward" means reaching the cheap regime with
fewer rows.

Operationalised as **the smallest pool at which reads stay within `--tol` of
the reads at full capacity**, where full capacity holds the entire working
set and so pays only compulsory misses. That is a threshold on the curve
rather than a curvature estimate, because curvature on a 14-point sweep is
mostly a statement about where the points were placed.

Both framings are reported, for the same reason R10 needed both:

  matched    hard = (rows, protected=rows) vs soft = (rows, protected<rows)
             same physical memory; only the ownership cap differs.
  R1-shape   hard = (P, protected=P) vs soft = (1024, protected=P)
             the arms R1 used, where the soft one simply has more rows.
"""
import argparse
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "kernel"))
sys.path.insert(0, HERE)

from score_r10 import run                       # noqa: E402
from score_r3 import build_arena                # noqa: E402


def knee(curve, tol):
    """Smallest pool whose reads are within `tol` of the best on the curve."""
    best = min(r for _, r in curve)
    for rows, reads in sorted(curve):
        if reads <= best * (1.0 + tol):
            return rows, reads
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--rows", default="64,96,128,160,192,256,320,384,448,512,"
                                      "640,768,896,1024")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--tol", type=float, default=0.10)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    with open(a.trace) as f:
        rowsj = [json.loads(line) for line in f]
    meta, recs = rowsj[0]["meta"], rowsj[1:]
    sweep = [int(x) for x in a.rows.split(",")]
    print(f"trace: {meta['steps']} steps x {meta['layers']} layers x "
          f"top-{meta['top_k']} of {meta['n_experts']}; knee tol={a.tol:.0%}")

    out = {"meta": meta, "tol": a.tol, "sweep": sweep}
    with tempfile.TemporaryDirectory() as tmp:
        path, index = build_arena(tmp, meta["layers"], meta["n_experts"])
        hard, soft = [], []
        print(f"\n{'rows':>6} {'hard reads':>11} {'soft reads':>11} {'Δ':>8}")
        for rows in sweep:
            h = run(path, index, recs, rows, rows)["reads"]
            s = run(path, index, recs, rows, max(1, rows - a.k))["reads"]
            hard.append((rows, h))
            soft.append((rows, s))
            print(f"{rows:>6} {h:>11} {s:>11} {(s-h)/h*100:>+7.1f}%")
        out["hard_curve"], out["soft_curve"] = hard, soft

        hk, hr = knee(hard, a.tol)
        sk, sr = knee(soft, a.tol)
        print(f"\nMATCHED capacity (same rows, ownership cap differs):")
        print(f"  hard knee: {hk} rows ({hr} reads)")
        print(f"  soft knee: {sk} rows ({sr} reads)")
        if hk and sk:
            move = (hk - sk) / hk * 100
            print(f"  knee moves {move:+.1f}%  "
                  f"({'outward' if move > 0 else 'inward or unmoved'})")
            out["matched"] = {"hard_knee": hk, "soft_knee": sk,
                              "move_pct": move,
                              "verdict": "R7 holds" if move >= 20
                                         else "REFUTED"}
            print(f"  -> {out['matched']['verdict']} "
                  f"(needs >= +20% outward)")

        # R1's arm shape, for continuity: soft keeps the FULL pool.
        full = max(sweep)
        r1 = [(P, run(path, index, recs, full, P)["reads"]) for P in sweep
              if P <= full]
        out["r1_shape_curve"] = r1
        r1k, r1r = knee(r1, a.tol)
        print(f"\nR1 SHAPE (soft always holds the full {full}-row pool):")
        print(f"  hard knee: {hk} rows")
        print(f"  soft knee: {r1k} protected ({r1r} reads) — but the pool is "
              f"always {full} rows")
        out["r1_shape"] = {"knee_protected": r1k, "pool": full}
    if a.out:
        json.dump(out, open(a.out, "w"), indent=2)
        print("\nreceipt ->", a.out)


if __name__ == "__main__":
    main()
