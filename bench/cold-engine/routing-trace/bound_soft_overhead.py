"""Account for the soft arm's extra non-read work, piece by piece.

Seven candidates have been tested one at a time and none dominates: dispatch
~10%, demote selection >=6.3%, the victim sweep <=16%, locality nil, per-read
storage cost nil. The shape suggests the asymmetry is SPREAD, so the useful
question stops being "which one" and becomes "do the known pieces add up".

Everything the soft arm does that the hard arm does not is the reclaimable
machinery: with protected_rows=None, _demote_locked early-returns, nothing is
ever reclaimable, and no row is ever resurrected. So the extra work is
_demote_locked + _resurrect_locked + the reclaimable term in _victim's rank +
the extra reads the policy causes.

Timed with targeted counters rather than cProfile. These functions are called
thousands of times, not millions, so per-call timer overhead is negligible --
unlike the 1.97M-call key lambda, where cProfile's ~1us/call inflated demote's
apparent share to 85% against an unprofiled 6.3%.
"""
import argparse
import json
import os
import sys
import tempfile
import time

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


def run(path, index, recs, prot, rows):
    """One arm, with per-function accumulators on the reclaimable machinery."""
    C = nvme_residency.ColdTier
    acc = {"demote_ns": 0, "resurrect_ns": 0, "victim_ns": 0,
           "demote_calls": 0, "resurrect_calls": 0, "victim_calls": 0}
    orig = {n: getattr(C, n) for n in
            ("_demote_locked", "_resurrect_locked", "_victim")}

    def wrap(name, key):
        f = orig[name]

        def inner(self, *a, **k):
            t0 = time.perf_counter_ns()
            try:
                return f(self, *a, **k)
            finally:
                acc[key + "_ns"] += time.perf_counter_ns() - t0
                acc[key + "_calls"] += 1
        return inner

    for n, k in (("_demote_locked", "demote"), ("_resurrect_locked", "resurrect"),
                 ("_victim", "victim")):
        setattr(C, n, wrap(n, k))
    try:
        t = C(path, hot_rows=rows, pinned=False, index=index,
              protected_rows=prot, qd=1)
        c0 = time.process_time()
        for r in recs:
            for L, ex in r["routed"].items():
                t.ensure(int(L), ex)
        cpu = time.process_time() - c0
        st = t.stats()
        t.close()
    finally:
        for n in orig:
            setattr(C, n, orig[n])
    return cpu, acc, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--rows", type=int, default=256)
    ap.add_argument("--protected", type=int, default=248)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    with open(a.trace) as f:
        rows_j = [json.loads(line) for line in f]
    meta, recs = rows_j[0]["meta"], rows_j[1:]
    with tempfile.TemporaryDirectory() as tmp:
        path, index = build(tmp, meta["layers"], meta["n_experts"])
        res = {}
        for _ in range(a.reps):                       # interleave the arms
            for arm, prot in (("hard", a.rows), ("soft", a.protected)):
                cpu, acc, st = run(path, index, recs, prot, a.rows)
                res.setdefault(arm, []).append((cpu, acc, st))
    out = {"rows": a.rows, "protected": a.protected, "reps": a.reps}
    pick = {}
    for arm in ("hard", "soft"):
        i = min(range(a.reps), key=lambda j: res[arm][j][0])   # min-CPU rep
        cpu, acc, st = res[arm][i]
        pick[arm] = (cpu, acc, st)
        print(f"{arm}: cpu {cpu:6.3f}s  reads {st['misses']}  "
              f"demote {acc['demote_ns']/1e9:.3f}s/{acc['demote_calls']}  "
              f"resurrect {acc['resurrect_ns']/1e9:.3f}s/{acc['resurrect_calls']}  "
              f"victim {acc['victim_ns']/1e9:.3f}s/{acc['victim_calls']}")
    (hc, ha, hs), (sc, sa, ss) = pick["hard"], pick["soft"]
    gap = sc - hc
    extra_reads = ss["misses"] - hs["misses"]
    per_read_hard = (hc - ha["victim_ns"] / 1e9) / hs["misses"]
    terms = [
        ("_demote_locked", (sa["demote_ns"] - ha["demote_ns"]) / 1e9),
        ("_resurrect_locked", (sa["resurrect_ns"] - ha["resurrect_ns"]) / 1e9),
        ("_victim (reclaimable rank term)",
         (sa["victim_ns"] - ha["victim_ns"]) / 1e9),
        (f"{extra_reads} extra reads x rest-of-tier",
         extra_reads * per_read_hard),
    ]
    named = sum(v for _n, v in terms)
    print(f"\nsoft-hard CPU gap: {gap:.3f}s")
    for n, v in terms:
        print(f"  {n:<34} {v:7.3f}s  {v/gap*100:5.1f}%")
    print(f"  {'ACCOUNTED':<34} {named:7.3f}s  {named/gap*100:5.1f}%")
    print(f"  {'unaccounted':<34} {gap-named:7.3f}s  {(gap-named)/gap*100:5.1f}%")
    out["gap_s"] = gap
    out["terms"] = {n: v for n, v in terms}
    out["accounted_s"] = named
    out["unaccounted_s"] = gap - named
    if a.out:
        json.dump(out, open(a.out, "w"), indent=2)
        print("receipt ->", a.out)


if __name__ == "__main__":
    main()
