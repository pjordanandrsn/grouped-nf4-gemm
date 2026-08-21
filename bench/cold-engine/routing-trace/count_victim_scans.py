"""How much work does one victim selection cost, now that it is heap-driven?

`_victim` was 40% of the tier's CPU as an O(hot_rows) sweep. The LFU heap
replaced that with a few pops, so the quantity that matters is now POPS PER
SELECTION -- how many stale or excluded entries stand between the heap top and
a usable victim.

Counts, not times: timing could not resolve changes to this function on a
laptop (a wall A/B once reported the soft arm 64% SLOWER from a patch that
strictly removes work, and CPU-time end-to-end gave 6-25% spreads against a
~10% effect). Pop counts are deterministic and have no spread.

`stale` are entries whose slot or rank moved on; `held` were live but excluded
and get pushed back. Their sum is the real cost of a selection, against
hot_rows for the sweep it replaced.
"""
import argparse
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "kernel"))

from nvme_arena import bake, load_index          # noqa: E402
import nvme_residency                            # noqa: E402


def build(tmp, layers, experts):
    import test_nvme_arena as tna
    tna.L, tna.E = layers, experts
    snap = os.path.join(tmp, "snap")
    tna.make_snapshot(snap)
    path = os.path.join(tmp, "toy.arena")
    bake(snap, path, align=4096, log=lambda *a: None)
    return path, load_index(path)


def run(path, index, recs, rows, prot):
    """Count selections and the heap pops they consumed."""
    counts = {"calls": 0, "pops": 0, "heap": 0}
    ColdTier = nvme_residency.ColdTier
    original = ColdTier._victim

    def counting(self, excluded):
        counts["calls"] += 1
        before = len(self._vheap)
        got = original(self, excluded)
        # held entries are pushed back, so the net drop is the stale discards;
        # add the one selected entry, which stays on the heap.
        counts["pops"] += max(0, before - len(self._vheap))
        counts["heap"] = len(self._vheap)
        return got

    ColdTier._victim = counting
    try:
        t = ColdTier(path, hot_rows=rows, pinned=False, index=index,
                     protected_rows=prot, qd=1)
        for r in recs:
            for L, ex in r["routed"].items():
                t.ensure(int(L), ex)
        t.close()
    finally:
        ColdTier._victim = original
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    a = ap.parse_args()
    with open(a.trace) as f:
        rows_j = [json.loads(line) for line in f]
    meta, recs = rows_j[0]["meta"], rows_j[1:]
    print(f"requests in trace: {sum(len(r['routed']) for r in recs)}")
    print(f"\n{'config':<24} {'selections':>11} {'stale pops':>11} "
          f"{'pops/call':>10} {'heap left':>10} {'vs sweep':>9}")
    with tempfile.TemporaryDirectory() as tmp:
        path, index = build(tmp, meta["layers"], meta["n_experts"])
        for rows, prot in ((128, 120), (256, None), (256, 248), (512, 504)):
            c = run(path, index, recs, rows, prot)
            r = c["pops"] / c["calls"] if c["calls"] else 0
            print(f"rows={rows:<5} prot={str(prot):<10} {c['calls']:>11} "
                  f"{c['pops']:>11} {r:>10.2f} {c['heap']:>10} "
                  f"{rows / r if r else float('inf'):>8.1f}x")


if __name__ == "__main__":
    main()
