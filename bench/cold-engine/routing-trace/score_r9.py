"""R9's PRECONDITION, scored: how often are two valid copies available at all?

Registered (PREREG-tribrid-stage3, R9):

    choosing between simultaneously-valid DRAM and VRAM copies by slack
    beats always taking the highest tier -- REFUTED IF highest-tier-always
    ties or wins.

R9 cannot be scored as registered. Choosing "by slack" needs a time-to-
deadline estimate, and gate 2 established there is no deadline estimator to
supply one (PREREG amendment 1). Building a chooser to score a prediction
about choosers would be scoring the chooser.

What CAN be settled without one is whether the situation R9 describes ever
arises. The prediction is about invocations where BOTH copies are valid; if
that set is empty, or negligible, then no chooser -- slack-based or
otherwise -- has anything to decide, and R9 is bounded before any policy is
written. That bound is worth having on its own: it is the same shape as R1's
uncontended finding, where P collapsed to 1.000 because the event whose
probability it asked about never occurred.

Configuration is #133's: a DevRowCache in FRONT of a ColdTier, which is the
arrangement that creates two copies of one expert -- the device row and the
DRAM row it was filled from. A placement-tiered VRAM/DRAM split cannot
produce them, because placement makes the tiers disjoint by construction.

"Valid" is taken from each side's own state, non-mutatingly and BEFORE the
step's requests are issued:
  DRAM -- ColdTier.slot_of(layer, expert) is not None
  VRAM -- VramSlots.slot_of((layer, expert)) is not None, which is true in
          ACTIVE, RETIRING and RECLAIMABLE alike. A reclaimable row is
          exactly a valid copy nobody owns, so excluding it would define
          R9's opportunity away.
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
    import test_nvme_arena as tna
    tna.L, tna.E = layers, experts
    snap = os.path.join(tmp, "snap")
    tna.make_snapshot(snap)
    path = os.path.join(tmp, "toy.arena")
    bake(snap, path, align=4096, log=lambda *a: None)
    return path, load_index(path)


def replay(path, index, recs, dram_rows, vram_rows, protected, qd=1):
    tier = ColdTier(path, hot_rows=dram_rows, pinned=False, index=index, qd=qd)
    cache = DevRowCache(vram_rows, 8, device="cpu", protected=protected)
    n = both = dram_only = vram_only = neither = 0
    try:
        for r in recs:
            for L, experts in r["routed"].items():
                li = int(L)
                for e in experts:
                    d = tier.slot_of(li, int(e)) is not None
                    v = cache.slots.slot_of((li, int(e))) is not None
                    n += 1
                    both += d and v
                    dram_only += d and not v
                    vram_only += v and not d
                    neither += not d and not v
                # issue the step AFTER observing, so the counts describe what
                # a chooser would have seen, not what this request created
                tier.ensure(li, experts)
                tag = StepTag("cpu")
                cache.want(li, experts, tag)
                tag.record()
        return {"routed": n, "both_valid": both, "dram_only": dram_only,
                "vram_only": vram_only, "neither": neither,
                "both_rate": both / n if n else 0.0,
                "tier": dict(tier.stats()), "cache": dict(cache.stats())}
    finally:
        tier.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--dram-rows", default="256,384")
    ap.add_argument("--vram-rows", default="64,128,256")
    ap.add_argument("--protected", default="half")
    ap.add_argument("--qd", type=int, default=1,
                    help="reader queue depth. ColdTier defaults to None, which sizes the queue from the host CPU count -- so counters are neither reproducible run-to-run nor comparable across boxes. qd=1 forces completion order and makes this replay a pure function of the trace; see qd_jitter.py.")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    with open(a.trace) as f:
        rowsj = [json.loads(line) for line in f]
    meta, recs = rowsj[0]["meta"], rowsj[1:]
    L, E = meta["layers"], meta["n_experts"]
    print(f"trace: {meta['steps']} steps x {L} layers x top-{meta['top_k']} of {E}")

    out = {"meta": meta, "points": []}
    with tempfile.TemporaryDirectory() as tmp:
        path, index = build_arena(tmp, L, E)
        print(f"\n{'dram':>5} {'vram':>5} {'prot':>5} {'both':>9} {'dram-only':>10} "
              f"{'vram-only':>10} {'neither':>9} {'both rate':>10}")
        for dr in [int(x) for x in a.dram_rows.split(",")]:
            for vr in [int(x) for x in a.vram_rows.split(",")]:
                prot = vr // 2 if a.protected == "half" else int(a.protected)
                st = replay(path, index, recs, dr, vr, prot, a.qd)
                st.update({"dram_rows": dr, "vram_rows": vr, "protected": prot})
                out["points"].append(st)
                print(f"{dr:>5} {vr:>5} {prot:>5} {st['both_valid']:>9} "
                      f"{st['dram_only']:>10} {st['vram_only']:>10} "
                      f"{st['neither']:>9} {st['both_rate']:>9.1%}")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=2, default=str)
        print("\nreceipt ->", a.out)


if __name__ == "__main__":
    main()
