"""R2's PREMISE, scored: do VRAM resurrections reach 2-5% of routed work?

Registered (PREREG-tribrid-stage3, R2):

    VRAM resurrection is disproportionately valuable -- even 2-5% of routed
    invocations moves wall time -- REFUTED BY no measurable wall effect.

The wall half needs an MXFP4 model this program does not have (the device
arena lives in Mxfp4NvmeResidency, and no gpt-oss/K3-lineage arena has been
baked here). But the clause carries a quantity that can be checked without
one: R2 asserts resurrections reach **2-5% of ROUTED INVOCATIONS**, and if
they never do, the wall claim has no mass behind it whatever a chooser does.

Note the denominator. RESULTS-r3.md reports resurrections over RESOLVED
LOGICAL EVICTIONS -- a rate that ran 0.0-33.9% and was shown there to depend
on a budget R3 never pinned. That is a different quantity. A tier can
resurrect nearly every evicted row while resurrections stay a rounding error
against the work the model actually does, which is the number R2 names.

Both are reported side by side here so the two cannot be confused again.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
KERNEL = os.path.join(HERE, "..", "..", "..", "kernel")
sys.path.insert(0, KERNEL)

from dev_row_cache import DevRowCache, StepTag        # noqa: E402


def replay(recs, rows, protected):
    c = DevRowCache(rows, 8, device="cpu", protected=protected)
    routed = fills = 0
    for r in recs:
        for L, experts in r["routed"].items():
            routed += len(experts)
            tag = StepTag("cpu")
            _assign, need = c.want(int(L), experts, tag)
            fills += len(need)
            tag.record()
    st = dict(c.stats())
    res = (st.get("resurrections", 0) or 0) + (st.get("spec_resurrections", 0) or 0)
    over = st.get("reclaimable_overwritten", st.get("overwritten", 0)) or 0
    return {
        "rows": rows, "protected": protected, "routed": routed,
        # reported so a high resurrection rate cannot be read as a healthy
        # cache: at rows=128/prot=120 two thirds of requests still MISS, and
        # resurrections are what a thrashing cache does, not what it avoids
        "fills": fills, "fill_rate": fills / routed if routed else 0.0,
        "resurrections": res, "overwritten": over,
        # R2's quantity: resurrections as a share of the work the model does
        "per_routed": res / routed if routed else 0.0,
        # R3's quantity, beside it so the two are never conflated again
        "per_resolved_eviction": (res / (res + over)) if (res + over) else None,
        "stats": st,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--rows", default="128,256,384,512,1024")
    ap.add_argument("--protected", default="quarter,half,three-quarter,rows-k",
                    help="budgets. R3 showed the resurrection rate is decided "
                         "by this, so a single budget would say nothing -- and "
                         "`rows-k` must be among them: it is DevRowCache's own "
                         "default and the budget RESULTS-r3.md measured. "
                         "Omitting it is how the first version of this scorer "
                         "concluded the 2-5%% antecedent never arrives when at "
                         "that budget it reaches 33.9%% (Bugbot, gnf4#151).")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    with open(a.trace) as f:
        rowsj = [json.loads(line) for line in f]
    meta, recs = rowsj[0]["meta"], rowsj[1:]
    frac = {"quarter": 0.25, "half": 0.5, "three-quarter": 0.75}
    print(f"trace: {meta['steps']} steps x {meta['layers']} layers x "
          f"top-{meta['top_k']} of {meta['n_experts']}")
    print(f"\n{'rows':>5} {'prot':>6} {'budget':>14} {'resurr':>8} "
          f"{'overwr':>8} {'/routed (R2)':>13} {'/resolved (R3)':>15}")
    out = {"meta": meta, "points": []}
    for rows in [int(x) for x in a.rows.split(",")]:
        for name in a.protected.split(","):
            prot = (rows - meta["top_k"]) if name == "rows-k" else max(
                1, int(frac[name] * rows))
            if prot < 1 or prot > rows:
                continue
            try:
                p = replay(recs, rows, prot)
            except (ValueError, RuntimeError) as exc:
                # RuntimeError too: VramSlots refuses a budget that starves the
                # next request ("no slot available"), which a near-full
                # protected budget on a small cache does. Catching only
                # ValueError turned that into a crash.
                print(f"{rows:>5} {prot:>6} {name:>14}  refused: "
                      f"{str(exc).split(chr(10))[0][:38]}")
                continue
            p["budget"] = name
            out["points"].append(p)
            pr = p["per_resolved_eviction"]
            print(f"{rows:>5} {prot:>6} {name:>14} {p['resurrections']:>8} "
                  f"{p['overwritten']:>8} {p['per_routed']:>12.2%} "
                  f"{(f'{pr:.1%}' if pr is not None else '-'):>15}")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=2, default=str)
        print("\nreceipt ->", a.out)


if __name__ == "__main__":
    main()
