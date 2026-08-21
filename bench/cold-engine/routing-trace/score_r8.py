"""R8, scored: does the nominal placement miss rate stop tracking real I/O?

Registered (PREREG-tribrid-stage3, R8):

    nominal placement miss rate becomes a poor I/O metric; physical refill
    rate is the operational one -- REFUTED IF THE TWO STAY CLOSE.

A placement solver reports what fraction of routed invocations land on an
expert it placed in NVMe. That is the number a solver optimizes and a
scoreboard quotes. R8 says it stops meaning what it looks like it means once
a tier sits in front of the arena, because a nominally-cold expert routed
twenty times costs one read, not twenty.

Scored on the captured routing sequence, driven through the real ColdTier at
several cold masses and capacities. Both quantities come out of the SAME
replay:

  nominal miss  -- a routed (layer, expert) whose placement tier is NVMe.
                   Counted from the placement alone; no tier involved.
  physical refill -- a read the tier actually issued, from its own counter.

The ratio between them is the whole prediction. "Stay close" is read as a
refill rate within 2x of nominal -- anything nearer than that and the two
metrics would rank placements the same way, which is what R8 denies.

The cold set is chosen by MASS TAIL, matching force_cold_mass(order="tail")
in the serving path: the lowest-mass experts go cold first. Mass here is
counted from the trace itself, so the placement and the workload are
consistent by construction rather than by a profile that might have been
captured elsewhere.
"""
import argparse
import json
import os
import sys
import tempfile
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
KERNEL = os.path.join(HERE, "..", "..", "..", "kernel")
sys.path.insert(0, KERNEL)

from nvme_arena import bake, load_index               # noqa: E402
from nvme_residency import ColdTier                   # noqa: E402


def build_arena(tmp, layers, experts):
    """A synthetic arena with the trace's geometry (bytes are irrelevant to a
    residency question; geometry and the request stream are what matter)."""
    import test_nvme_arena as tna
    tna.L, tna.E = layers, experts
    snap = os.path.join(tmp, "snap")
    tna.make_snapshot(snap)
    path = os.path.join(tmp, "toy.arena")
    bake(snap, path, align=4096, log=lambda *a: None)
    return path, load_index(path)


def cold_set(recs, frac):
    """The lowest-mass (layer, expert) pairs holding `frac` of routed mass,
    which is what force_cold_mass(order='tail') selects in the serving path."""
    mass = Counter()
    for r in recs:
        for L, experts in r["routed"].items():
            for e in experts:
                mass[(int(L), int(e))] += 1
    total = sum(mass.values())
    cold, acc = set(), 0
    for key, m in sorted(mass.items(), key=lambda kv: kv[1]):
        if acc + m > frac * total:
            break
        cold.add(key)
        acc += m
    return cold, acc / total if total else 0.0


def replay(path, index, recs, cold, rows, qd=1):
    """One pass. Nominal misses from the placement, refills from the tier."""
    nominal = 0
    t = ColdTier(path, hot_rows=rows, pinned=False, index=index, qd=qd)
    try:
        for r in recs:
            for L, experts in r["routed"].items():
                li = int(L)
                want = [e for e in experts if (li, int(e)) in cold]
                nominal += len(want)
                if want:
                    t.ensure(li, want)
        st = dict(t.stats())
    finally:
        t.close()
    return nominal, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--cold", default="0.05,0.10,0.20")
    ap.add_argument("--rows", default="128,256,384")
    ap.add_argument("--qd", type=int, default=1,
                    help="reader queue depth. ColdTier defaults to None, which sizes the queue from the host CPU count -- so counters are neither reproducible run-to-run nor comparable across boxes. qd=1 forces completion order and makes this replay a pure function of the trace; see qd_jitter.py.")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    with open(a.trace) as f:
        rowsj = [json.loads(line) for line in f]
    meta, recs = rowsj[0]["meta"], rowsj[1:]
    L, E = meta["layers"], meta["n_experts"]
    routed_total = sum(len(v) for r in recs for v in r["routed"].values())
    print(f"trace: {meta['steps']} steps x {L} layers x top-{meta['top_k']} "
          f"of {E} -- {routed_total} routed invocations")

    out = {"meta": meta, "routed_invocations": routed_total, "points": []}
    with tempfile.TemporaryDirectory() as tmp:
        path, index = build_arena(tmp, L, E)
        print(f"synthetic arena: {index['n_layers']}L x "
              f"{index['n_experts_per_layer']}E, row {index['row_bytes']} B\n")
        print(f"{'cold%':>6} {'rows':>5} {'nominal':>9} {'refills':>8} "
              f"{'nominal/routed':>15} {'refill/routed':>14} {'ratio':>7}")
        for frac in [float(x) for x in a.cold.split(",")]:
            cold, achieved = cold_set(recs, frac)
            for rows in [int(x) for x in a.rows.split(",")]:
                nominal, st = replay(path, index, recs, cold, rows, a.qd)
                refills = st.get("misses", 0)
                nr = nominal / routed_total
                rr = refills / routed_total
                ratio = (nominal / refills) if refills else None
                out["points"].append({
                    "cold_frac_target": frac, "cold_frac_achieved": achieved,
                    "cold_rows": len(cold), "hot_rows": rows,
                    "nominal_misses": nominal, "physical_refills": refills,
                    "nominal_rate": nr, "refill_rate": rr,
                    "nominal_over_refill": ratio,
                    "disk_reads": st.get("disk_reads"),
                    "hits": st.get("hits"), "evictions": st.get("evictions")})
                print(f"{frac*100:>5.0f}% {rows:>5} {nominal:>9} {refills:>8} "
                      f"{nr:>14.1%} {rr:>13.2%} "
                      f"{(f'{ratio:.1f}x' if ratio else '-'):>7}")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=2)
        print("\nreceipt ->", a.out)


if __name__ == "__main__":
    main()
