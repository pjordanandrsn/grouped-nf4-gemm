"""Is ColdTier reproducible? Repeat one replay N times and report the spread.

Counters are compared across arms all over this campaign, and a comparison is
only meaningful against the instrument's own run-to-run spread. `qd=1` was
chosen for the demote probe and for R6 *because* higher queue depths were not
reproducible -- but that reason was never written down, so the spread has
never been quantified or attached to the claims that depend on it.

Mechanism, and why qd is the axis: with qd>1 the tier has several reads in
flight and services them in completion order, which the kernel and device do
not guarantee to be issue order. Completion order decides which row lands
first, which decides what the next demote sees. At qd=1 there is exactly one
outstanding read, so ordering is forced and the replay is a pure function of
the trace.

Reports min/max/spread per counter. Spread 0 => reproducible at that qd.
"""
import argparse
import json
import os
import statistics
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "kernel"))

from nvme_arena import bake, load_index                # noqa: E402
from nvme_residency import ColdTier                    # noqa: E402

COUNTERS = ("misses", "evictions", "resurrections", "spec_resurrections",
            "reclaimable_overwritten", "hits")


def build(tmp, layers, experts):
    import test_nvme_arena as tna
    tna.L, tna.E = layers, experts
    snap = os.path.join(tmp, "snap")
    tna.make_snapshot(snap)
    path = os.path.join(tmp, "toy.arena")
    bake(snap, path, align=4096, log=lambda *a: None)
    return path, load_index(path)


def replay(path, index, recs, rows, protected, qd):
    t = ColdTier(path, hot_rows=rows, pinned=False, index=index,
                 protected_rows=protected, qd=qd)
    try:
        for r in recs:
            for L, ex in r["routed"].items():
                t.ensure(int(L), ex)
        s = t.stats()
        return {c: s.get(c) for c in COUNTERS if s.get(c) is not None}
    finally:
        t.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--rows", type=int, default=512)
    ap.add_argument("--frac", type=float, default=0.90)
    ap.add_argument("--qds", default="1,2,4,8")
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    with open(a.trace) as f:
        rows_j = [json.loads(line) for line in f]
    meta, recs = rows_j[0]["meta"], rows_j[1:]
    prot = max(1, int(round(a.rows * a.frac)))
    print(f"trace {meta['steps']} steps; rows={a.rows} protected={prot} "
          f"repeats={a.repeats}")
    out = {"meta": meta, "rows": a.rows, "protected": prot,
           "repeats": a.repeats, "by_qd": {}}
    with tempfile.TemporaryDirectory() as tmp:
        path, index = build(tmp, meta["layers"], meta["n_experts"])
        for qd in [int(v) for v in a.qds.split(",")]:
            runs = [replay(path, index, recs, a.rows, prot, qd)
                    for _ in range(a.repeats)]
            print(f"\n  qd={qd}")
            rec = {}
            for c in runs[0]:
                vals = [r[c] for r in runs]
                spread = max(vals) - min(vals)
                rec[c] = {"min": min(vals), "max": max(vals),
                          "spread": spread,
                          "stdev": statistics.pstdev(vals) if len(vals) > 1
                          else 0.0, "values": vals}
                flag = "" if spread == 0 else f"   <-- VARIES by {spread}"
                print(f"    {c:26} {min(vals):>8} .. {max(vals):<8}"
                      f" spread {spread:>4}{flag}")
            out["by_qd"][str(qd)] = rec
            tot = sum(v["spread"] for v in rec.values())
            print(f"    {'REPRODUCIBLE' if tot == 0 else 'NOT reproducible'}")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=2)
        print("\nreceipt ->", a.out)


if __name__ == "__main__":
    main()
