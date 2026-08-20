"""R9, scored: can declining a VRAM copy for the DRAM one ever win?

Registered (PREREG-tribrid-stage3, R9):

    choosing between simultaneously-valid DRAM and VRAM copies by slack
    beats always taking the highest tier -- REFUTED IF highest-tier-always
    ties or wins.

gnf4#150 scored the precondition: two valid copies exist for 0-57.3% of
invocations depending on device-cache capacity. This scores the CHOICE.

It needed no deadline estimator in the end, for two reasons.

FIRST, the currency is transfers, not seconds. gnf4#152 measured wall on
this path against row transfers at r=+0.975, slope 546 us per 13.22 MB row
(24.2 GB/s against a ~28 GB/s PCIe ceiling). So a policy that moves fewer
rows is faster, with a measured conversion factor -- a transfer count IS a
wall comparison here.

SECOND, and more decisively, the state machine makes the comparison a
DOMINANCE argument rather than an empirical one. In `VramSlots._want_locked`,
a resurrection promotes RECLAIMABLE -> ACTIVE on the slot that ALREADY HOLDS
that expert: no transfer, and nothing else evicted. Declining it to take the
DRAM copy still has to `_claim` a slot -- consuming the same capacity, quite
possibly that very slot -- and pays a transfer on top. Same slot pressure,
strictly more traffic.

So highest-tier-always cannot lose, and no slack signal changes that: the
quantity a slack policy would trade against does not exist here, because
taking the higher tier costs no capacity the alternative would have saved.

This scorer exists to check that argument against the real state machine
rather than trust it. `discard()` before `want()` is the "decline the VRAM
copy" policy: it unpublishes the row, so the request must fetch it.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "kernel"))

from dev_row_cache import DevRowCache, StepTag        # noqa: E402

US_PER_ROW = 545.9          # gnf4#152, measured on gpt-oss-20b rows


def replay(recs, rows, protected, decline):
    """`decline=True` refuses any copy the cache still holds reclaimable,
    forcing the DRAM path for it -- R9's alternative policy."""
    c = DevRowCache(rows, 8, device="cpu", protected=protected)
    routed = xfer = declined = 0
    for r in recs:
        for L, experts in r["routed"].items():
            li = int(L)
            routed += len(experts)
            if decline:
                # Take the DRAM copy for every VRAM copy this request could
                # otherwise reuse for free: unpublish it so `want` must claim
                # and fill instead.
                #
                # BOTH non-active states count, and getting that wrong made
                # this policy far weaker than advertised (Bugbot, gnf4#156).
                # State is inspected BEFORE `want`, and `want` runs its own
                # settle pass, so a row demoted on the previous step is still
                # RETIRING here -- it flips to RECLAIMABLE inside `want` and
                # resurrects for free. Filtering on "reclaimable" alone
                # therefore skipped exactly the copies _want_locked reuses via
                # its RETIRING self-hit path.
                give_up = [e for e in experts
                           if (s := c.slots.slot_of((li, int(e)))) is not None
                           and c.slots.state(s) in ("reclaimable", "retiring")]
                if give_up:
                    declined += c.discard(li, give_up)
            tag = StepTag("cpu")
            _a, need = c.want(li, experts, tag)
            xfer += len(need)
            tag.record()
    st = c.stats()
    return {"rows": rows, "protected": protected, "decline": decline,
            "routed": routed, "transfers": xfer, "declined": declined,
            "resurrections": st.get("resurrections", 0),
            "est_ms": xfer * US_PER_ROW / 1000.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--rows", default="256,384,512,1024")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    with open(a.trace) as f:
        rowsj = [json.loads(line) for line in f]
    meta, recs = rowsj[0]["meta"], rowsj[1:]
    k = meta["top_k"]
    print(f"trace: {meta['steps']} steps x {meta['layers']} layers x top-{k} "
          f"of {meta['n_experts']}")
    print(f"\n{'rows':>5} {'prot':>5} | {'keep xfer':>10} {'decline xfer':>13} "
          f"{'delta':>8} | {'declined':>9} {'est cost ms':>12}")
    out = {"meta": meta, "us_per_row": US_PER_ROW, "points": []}
    worse = ties = better = 0
    for rows in [int(v) for v in a.rows.split(",")]:
        for prot in (rows // 2, rows - k):
            if prot < 1 or prot >= rows:
                continue
            keep = replay(recs, rows, prot, decline=False)
            dec = replay(recs, rows, prot, decline=True)
            d = dec["transfers"] - keep["transfers"]
            worse += d > 0
            ties += d == 0
            better += d < 0
            out["points"].append({"keep": keep, "decline": dec, "delta": d})
            print(f"{rows:>5} {prot:>5} | {keep['transfers']:>10} "
                  f"{dec['transfers']:>13} {d:>+8} | {dec['declined']:>9} "
                  f"{d * US_PER_ROW / 1000.0:>+11.1f}")
    print(f"\ndecline is WORSE at {worse} points, ties at {ties}, better at {better}")
    out["summary"] = {"worse": worse, "ties": ties, "better": better}
    if a.out:
        json.dump(out, open(a.out, "w"), indent=2)
        print("receipt ->", a.out)


if __name__ == "__main__":
    main()
