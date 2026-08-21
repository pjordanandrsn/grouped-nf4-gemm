"""R10, scored: does reclaimable residency actually reduce refills?

Registered (PREREG-tribrid-stage3, R10):

    reclaimable residency reduces promotion churn, NVMe rereads and H2D
    refills without reducing effective hit rate
    -- REFUTED IF churn unchanged or hit rate drops.

Unlike R1-R3 this is stated in the metric that survives scrutiny: physical
refills, not resurrection rate (see RESULTS-r3.md). So it is answerable as
written, provided the two arms are matched on CAPACITY.

They are matched here, which R1's arms were not. R1 compared hard eviction
at `hot_rows == protected == P` against a 128-row pool with P protected --
the soft arm simply had more memory. Both arms below hold the SAME number of
physical rows; the only difference is whether ownership is capped below that
number, leaving the remainder readable-but-unowned.

  hard   hot_rows = N, protected_rows = N   reclaimable set empty; this is
                                            the pre-Stage-3 tier exactly
  soft   hot_rows = N, protected_rows = P   N - P rows reclaimable
"""
import argparse
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "kernel"))
sys.path.insert(0, HERE)

from nvme_residency import ColdTier            # noqa: E402
from score_r3 import build_arena               # noqa: E402


def run(path, index, recs, rows, protected, qd=1):
    t = ColdTier(path, hot_rows=rows, pinned=False, index=index,
                 protected_rows=protected, qd=qd)
    try:
        for r in recs:
            for L, experts in r["routed"].items():
                t.ensure(int(L), experts)
        st = dict(t.stats())
        st["reads"] = t.reader.traffic()["reads"]
        return st
    finally:
        t.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--rows", default="128,192,256,384,512")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--qd", type=int, default=1,
                    help="reader queue depth. ColdTier defaults to None, which sizes the queue from the host CPU count -- so counters are neither reproducible run-to-run nor comparable across boxes. qd=1 forces completion order and makes this replay a pure function of the trace; see qd_jitter.py.")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    with open(a.trace) as f:
        rowsj = [json.loads(line) for line in f]
    meta, recs = rowsj[0]["meta"], rowsj[1:]
    routed = sum(len(v) for r in recs for v in r["routed"].values())
    # Granite's capture recorded `n_experts: null` -- capture_routing.py read
    # only `num_experts` and GraniteMoE spells it `num_local_experts`, fixed
    # since. Older traces still carry the null and crashed build_arena with a
    # bare TypeError. Fall back to the largest id actually routed, which is a
    # LOWER BOUND (an expert no token ever picked is invisible) and is
    # reported as one rather than silently standing in for the real count.
    n_exp = meta.get("n_experts")
    exact = n_exp is not None
    if not exact:
        n_exp = max(e for r in recs for ex in r["routed"].values() for e in ex) + 1
        meta = dict(meta, n_experts=n_exp, n_experts_inferred=True)
    print(f"trace: {meta['steps']} steps x {meta['layers']} layers x "
          f"top-{meta['top_k']} of {n_exp}{'' if exact else ' (INFERRED lower bound)'};"
          f" {routed} routed slots")
    print("\nBoth arms hold the SAME physical rows. Only the ownership cap "
          "differs.\n")

    out = {"meta": meta, "routed": routed, "points": []}
    with tempfile.TemporaryDirectory() as tmp:
        path, index = build_arena(tmp, meta["layers"], meta["n_experts"])
        print(f"{'rows':>5} {'prot':>5} | {'hard reads':>11} {'soft reads':>11} "
              f"{'Δ reads':>9} | {'hard evict':>11} {'soft evict':>11} "
              f"{'Δ churn':>9} | {'verdict':>10}")
        for rows in [int(x) for x in a.rows.split(",")]:
            hard = run(path, index, recs, rows, rows, a.qd)
            for prot in (max(1, rows // 2), max(1, rows - a.k)):
                soft = run(path, index, recs, rows, prot, a.qd)
                hr, sr = hard["reads"], soft["reads"]
                he, se = hard["evictions"], soft["evictions"]
                d_r = (sr - hr) / hr * 100 if hr else 0.0
                d_c = (se - he) / he * 100 if he else 0.0
                # R10 wants FEWER reads and less churn, with the hit rate not
                # dropping. Fewer reads IS a higher hit rate over a fixed
                # request stream, so the hit-rate clause is checked, not
                # assumed.
                h_hit = hard["hits"] / (hard["hits"] + hard["misses"])
                s_hit = soft["hits"] / (soft["hits"] + soft["misses"])
                ok = sr < hr and se <= he and s_hit >= h_hit
                verdict = "R10 holds" if ok else "REFUTED"
                out["points"].append({
                    "rows": rows, "protected": prot,
                    "hard_reads": hr, "soft_reads": sr, "delta_reads_pct": d_r,
                    "hard_evictions": he, "soft_evictions": se,
                    "delta_churn_pct": d_c,
                    "hard_hit_rate": h_hit, "soft_hit_rate": s_hit,
                    "verdict": verdict})
                print(f"{rows:>5} {prot:>5} | {hr:>11} {sr:>11} {d_r:>+8.1f}% | "
                      f"{he:>11} {se:>11} {d_c:>+8.1f}% | {verdict:>10}")
    tally = {}
    for p in out["points"]:
        tally[p["verdict"]] = tally.get(p["verdict"], 0) + 1
    out["tally"] = tally
    print(f"\ntally: {tally}")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=2)
        print("receipt ->", a.out)


if __name__ == "__main__":
    main()
