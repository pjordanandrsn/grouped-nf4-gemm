"""Which part of ensure() carries the soft arm's extra non-read work?

RESULTS-read-timing.md put 86.5% of #153's residual OUTSIDE the read: at qd=1
the soft arm's reads cost what their count says (+2.2% on +1.4% more reads)
while its work around them costs +26.6%. That named a region, not a mechanism.

This profiles the region. `non_read_ns` is Python bookkeeping -- planning,
slot reservation, pending events, as_completed dispatch, demote -- and the
control flow through it is decided by POLICY, not by row size. That is the
same property that let score_locality.py measure read ORDER offline: the toy
arena reproduces which slots are touched and which demotes fire, so the
function-level profile is representative even though the bytes are not.

EXPLORATORY. A profile ranks suspects; it does not confirm one. Anything this
surfaces needs its own registered test before it counts as the mechanism.

Caveat stated up front: cProfile's per-call overhead inflates functions with
many small calls relative to few large ones. Both arms pay it identically per
call, so the ARM DIFFERENCE is meaningful while the absolute shares are not.
"""
import argparse
import cProfile
import json
import os
import pstats
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "kernel"))

from nvme_arena import bake, load_index                # noqa: E402
from nvme_residency import ColdTier                    # noqa: E402


def build(tmp, layers, experts):
    import test_nvme_arena as tna
    tna.L, tna.E = layers, experts
    snap = os.path.join(tmp, "snap")
    tna.make_snapshot(snap)
    path = os.path.join(tmp, "toy.arena")
    bake(snap, path, align=4096, log=lambda *a: None)
    return path, load_index(path)


def profile_arm(path, index, recs, rows, protected, qd):
    t = ColdTier(path, hot_rows=rows, pinned=False, index=index,
                 protected_rows=protected, qd=qd)
    pr = cProfile.Profile()
    try:
        pr.enable()
        for r in recs:
            for L, ex in r["routed"].items():
                t.ensure(int(L), ex)
        pr.disable()
        st = pstats.Stats(pr)
        rows_out = {}
        for func, (cc, nc, tt, ct, _cal) in st.stats.items():
            fname = f"{os.path.basename(func[0])}:{func[1]}({func[2]})"
            rows_out[fname] = {"ncalls": nc, "tottime": tt, "cumtime": ct}
        return rows_out, dict(t.stats())
    finally:
        t.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--rows", type=int, default=256)
    ap.add_argument("--protected", type=int, default=248)
    ap.add_argument("--qd", type=int, default=1)
    ap.add_argument("--top", type=int, default=18)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    with open(a.trace) as f:
        rows_j = [json.loads(line) for line in f]
    meta, recs = rows_j[0]["meta"], rows_j[1:]
    print(f"rows={a.rows}  hard(prot={a.rows}) vs soft(prot={a.protected})  "
          f"qd={a.qd}   [the read-timing configuration]")
    with tempfile.TemporaryDirectory() as tmp:
        path, index = build(tmp, meta["layers"], meta["n_experts"])
        hard, hst = profile_arm(path, index, recs, a.rows, a.rows, a.qd)
        soft, sst = profile_arm(path, index, recs, a.rows, a.protected, a.qd)
    hm, sm = hst.get("misses", 0), sst.get("misses", 0)
    print(f"reads: hard {hm}  soft {sm}  ({(sm - hm) / hm * 100:+.2f}%)")
    print("\nfunctions by EXTRA total time in the soft arm "
          "(tottime, self only):\n")
    print(f"{'Δ tottime':>10} {'Δ per-read':>11} {'Δ ncalls':>10}  function")
    deltas = []
    for fn in set(hard) | set(soft):
        h = hard.get(fn, {"ncalls": 0, "tottime": 0.0})
        s = soft.get(fn, {"ncalls": 0, "tottime": 0.0})
        d = s["tottime"] - h["tottime"]
        # per-read normalisation: the soft arm legitimately does 1.4% more
        # work because it issues 1.4% more reads. Scale hard up to soft's read
        # count so what is left is EXCESS, not volume.
        scaled_h = h["tottime"] * (sm / hm) if hm else 0.0
        deltas.append((s["tottime"] - scaled_h, d, s["ncalls"] - h["ncalls"],
                       fn, h["tottime"], s["tottime"]))
    deltas.sort(reverse=True)
    for excess, d, dn, fn, ht, st_ in deltas[:a.top]:
        print(f"{d:+10.3f} {excess:+11.3f} {dn:+10d}  {fn}")
        print(f"{'':>10} {'':>11} {'':>10}    hard {ht:7.3f}s  soft {st_:7.3f}s")
    if a.out:
        json.dump({"meta": meta, "rows": a.rows, "protected": a.protected,
                   "qd": a.qd, "hard_reads": hm, "soft_reads": sm,
                   "top": [{"fn": f, "delta_tottime": d,
                            "excess_over_read_scaling": e,
                            "delta_ncalls": dn, "hard": ht, "soft": s_}
                           for e, d, dn, f, ht, s_ in deltas[:40]]},
                  open(a.out, "w"), indent=2)
        print("\nreceipt ->", a.out)


if __name__ == "__main__":
    main()
