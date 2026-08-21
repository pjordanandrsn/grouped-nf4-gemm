"""R3, scored: is the DRAM resurrection rate really above the VRAM one?

Registered (PREREG-tribrid-stage3, R3):

    DRAM resurrection rate (~10-30%) exceeds VRAM (~3-15%)
    -- REFUTED IF VRAM >= DRAM.

The two sides have never been measured against each other. R1's DRAM figures
came from a model run on a rented box; the VRAM figures came from a fixture.
Different traces, different pressure, so the gap could be a property of the
tiers or an artefact of the comparison never having been matched.

This matches it: ONE captured routing sequence, driven through both state
machines at the SAME capacity and the SAME protected budget. `ColdTier` is
the DRAM side (pinned-DRAM rows over an NVMe arena) and `DevRowCache` /
`VramSlots` is the VRAM side. Both publish the same quantity --
resurrections over resolved logical evictions -- so the comparison is
like-for-like by construction.

The DRAM side needs a real arena because ColdTier reads one, so a synthetic
16x64 arena of tiny rows is baked here. Bytes are irrelevant to a residency
question; the geometry and the request stream are what matter.
"""
import argparse
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
KERNEL = os.path.join(HERE, "..", "..", "..", "kernel")
sys.path.insert(0, KERNEL)

from dev_row_cache import DevRowCache, StepTag        # noqa: E402
from nvme_arena import bake, load_index               # noqa: E402
from nvme_residency import ColdTier                   # noqa: E402


def build_arena(tmp, layers, experts):
    """A synthetic arena with the trace's geometry."""
    import test_nvme_arena as tna
    tna.L, tna.E = layers, experts          # module constants drive the shapes
    snap = os.path.join(tmp, "snap")
    tna.make_snapshot(snap)
    path = os.path.join(tmp, "toy.arena")
    bake(snap, path, align=4096, log=lambda *a: None)
    return path, load_index(path)


def dram_side(path, index, recs, rows, protected, qd=1):
    t = ColdTier(path, hot_rows=rows, pinned=False, index=index,
                 protected_rows=protected, qd=qd)
    try:
        for r in recs:
            for L, experts in r["routed"].items():
                t.ensure(int(L), experts)
        st = dict(t.stats())
        # ColdTier's misses ARE its physical refills: a miss is a row that is
        # not resident, which forces a read.
        st["physical_refills"] = st.get("misses", 0)
        return st
    finally:
        t.close()


def vram_side(recs, rows, protected):
    c = DevRowCache(rows, 8, device="cpu", protected=protected)
    fills = 0
    for r in recs:
        for L, experts in r["routed"].items():
            tag = StepTag("cpu")
            _a, need = c.want(int(L), experts, tag)
            tag.record()
            fills += len(need)
    st = dict(c.stats())
    st["physical_refills"] = fills
    return st


def rate(st):
    """resurrections / resolved logical evictions, from each side's OWN keys."""
    res = (st.get("resurrections", 0) or 0) + (st.get("spec_resurrections", 0) or 0)
    over = st.get("reclaimable_overwritten", st.get("overwritten", 0)) or 0
    return (res / (res + over)) if (res + over) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--rows", default="128,192,256,384,512")
    ap.add_argument("--budgets", default="half,rows-k",
                    help="protected budgets to sweep. R3 does not pin one, "
                         "and the verdict turns out to depend on it.")
    ap.add_argument("--k", type=int, default=8, help="routed set size")
    ap.add_argument("--qd", type=int, default=1,
                    help="reader queue depth. ColdTier defaults to None, which sizes the queue from the host CPU count -- so counters are neither reproducible run-to-run nor comparable across boxes. qd=1 forces completion order and makes this replay a pure function of the trace; see qd_jitter.py.")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    with open(a.trace) as f:
        rowsj = [json.loads(line) for line in f]
    meta, recs = rowsj[0]["meta"], rowsj[1:]
    L, E = meta["layers"], meta["n_experts"]
    print(f"trace: {meta['steps']} steps x {L} layers x top-{meta['top_k']} "
          f"of {E}  ({L*E} distinct rows)")

    out = {"meta": meta, "budgets": a.budgets, "points": []}
    with tempfile.TemporaryDirectory() as tmp:
        path, index = build_arena(tmp, L, E)
        print(f"synthetic arena: {index['n_layers']}L x "
              f"{index['n_experts_per_layer']}E, row {index['row_bytes']} B\n")
        # Physical refills are reported beside the rates ON PURPOSE. A rate
        # is only interesting if it tracks the cost it is supposed to proxy,
        # and these two do not move together.
        print(f"{'rows':>6} {'prot':>5} {'budget':>7} | {'DRAM rate':>10} "
              f"{'DRAM refills':>13} | {'VRAM rate':>10} "
              f"{'VRAM refills':>13} | {'verdict':>10}")
        combos = [(rows, b) for b in a.budgets.split(",")
                  for rows in [int(x) for x in a.rows.split(",")]]
        tally = {"R3 holds": 0, "REFUTED": 0, "undefined": 0}
        for rows, budget in combos:
            prot = (max(1, rows // 2) if budget == "half"
                    else max(1, rows - a.k))
            d, v = (dram_side(path, index, recs, rows, prot, a.qd),
                    vram_side(recs, rows, prot))
            dr, vr = rate(d), rate(v)
            if dr is None or vr is None:
                verdict = "undefined"
            elif vr >= dr:
                verdict = "REFUTED"
            else:
                verdict = "R3 holds"
            d_res = (d.get("resurrections", 0) or 0) + (d.get("spec_resurrections", 0) or 0)
            tally[verdict] = tally.get(verdict, 0) + 1
            out["points"].append({
                "rows": rows, "protected": prot, "budget": budget,
                "dram_resurrections": d_res,
                "dram_overwritten": d.get("reclaimable_overwritten"),
                "dram_rate": dr,
                "vram_resurrections": v.get("resurrections"),
                "vram_overwritten": v.get("overwritten"),
                "vram_rate": vr, "verdict": verdict,
                "dram_physical_refills": d.get("physical_refills"),
                "vram_physical_refills": v.get("physical_refills")})
            ds = "None" if dr is None else f"{dr*100:.1f}%"
            vs = "None" if vr is None else f"{vr*100:.1f}%"
            print(f"{rows:>6} {prot:>5} {budget:>7} | {ds:>10} "
                  f"{d.get('physical_refills', 0):>13} | {vs:>10} "
                  f"{v.get('physical_refills', 0):>13} | {verdict:>10}")
        out["tally"] = tally
        print(f"\ntally: {tally}")
        byb = {}
        for pt in out["points"]:
            byb.setdefault(pt["budget"], []).append(pt["verdict"])
        for b, vs_ in byb.items():
            print(f"  budget={b}: " + ", ".join(
                f"{k}={vs_.count(k)}" for k in ("R3 holds", "REFUTED")))
        print("\nR3 is UNDETERMINED as registered: the verdict is decided by "
              "a\nprotected budget the prediction does not pin.")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=2)
        print("\nreceipt ->", a.out)


if __name__ == "__main__":
    main()
