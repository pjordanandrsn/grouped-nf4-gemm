"""How many full O(hot_rows) victim scans does a replay actually do?

`_victim` was 40% of the tier's CPU. Timing could not resolve changes to it on
a laptop -- a wall-clock A/B reported the soft arm 64% SLOWER from a patch that
strictly removes work, and CPU-time end-to-end still gave 6-25% spreads against
a ~10% effect, with configurations disagreeing on the sign.

Scan count has no spread. It is deterministic, and it is the quantity the
batching in `_victim` is about: one ranking per plan loop instead of one scan
per claimed slot. Reported alongside `_victim` calls so the ratio is visible.
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
    """Count calls, and how many of them fell through to a full scan."""
    counts = {"calls": 0, "scans": 0}
    ColdTier = nvme_residency.ColdTier
    original = ColdTier._victim

    def counting(self, excluded):
        counts["calls"] += 1
        # A call is served from the batch iff a non-empty batch existed on
        # entry and the returned slot came off its front. Anything else means
        # the enumerate loop ran.
        had = list(self._victim_batch or ())
        got = original(self, excluded)
        if not (had and got in had):
            counts["scans"] += 1
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
    print(f"\n{'config':<24} {'_victim calls':>14} {'full scans':>11} "
          f"{'scans/call':>11}")
    with tempfile.TemporaryDirectory() as tmp:
        path, index = build(tmp, meta["layers"], meta["n_experts"])
        for rows, prot in ((128, 120), (256, None), (256, 248), (512, 504)):
            c = run(path, index, recs, rows, prot)
            r = c["scans"] / c["calls"] if c["calls"] else 0
            print(f"rows={rows:<5} prot={str(prot):<10} {c['calls']:>14} "
                  f"{c['scans']:>11} {r:>11.2f}")


if __name__ == "__main__":
    main()
