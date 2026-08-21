"""Do the hard and soft arms read the arena in a different ORDER?

#153's residual is a per-read cost: soft costs ~6.4% more wall while issuing
~1.35% more reads, and the gap does not move with queue depth. Four candidates
are eliminated. Read locality is the shape nothing in the campaign records.

The offset sequence is decided by POLICY, not by row size -- which row is read
when is a function of the trace and the eviction rules -- so it is recoverable
offline from the toy arena and interpretable against the real geometry.
Distances are reported in rows (the policy-level quantity) and in bytes at the
real 3.3 MB stride.

See PREREG-read-locality.md, registered before this was run.
"""
import argparse
import json
import os
import statistics
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "kernel"))

from nvme_arena import bake, load_index, row_offset      # noqa: E402
from nvme_residency import ColdTier                      # noqa: E402

REAL_STRIDE = 3342336          # bytes/row in the baked OLMoE arena (qd1.json)


def build(tmp, layers, experts):
    import test_nvme_arena as tna
    tna.L, tna.E = layers, experts
    snap = os.path.join(tmp, "snap")
    tna.make_snapshot(snap)
    path = os.path.join(tmp, "toy.arena")
    bake(snap, path, align=4096, log=lambda *a: None)
    return path, load_index(path)


def read_sequence(path, index, recs, rows, protected, qd=1):
    """Replay, returning the ordered list of rows that caused a disk read.

    Reads are captured by wrapping the READER, so this observes exactly the
    physical reads the tier issued, in issue order, without altering what it
    does. Deriving the sequence by calling ensure() one expert at a time
    instead would change the tier's batching -- and the batching is part of
    what decides the read order this is trying to measure.
    """
    t = ColdTier(path, hot_rows=rows, pinned=False, index=index,
                 protected_rows=protected, qd=qd)
    seq = []
    real_read_row = t.reader.read_row

    def spy(layer, expert, *a, **k):
        seq.append((int(layer), int(expert)))
        return real_read_row(layer, expert, *a, **k)

    t.reader.read_row = spy
    try:
        for r in recs:
            for L, ex in r["routed"].items():
                t.ensure(int(L), ex)
        return seq, t.stats()
    finally:
        t.reader.read_row = real_read_row
        t.close()


def locality(seq, index):
    offs = [row_offset(index, L, e) for L, e in seq]
    stride = min(b - a for a, b in zip(sorted(set(offs)), sorted(set(offs))[1:]))
    rows_apart = [abs(b - a) // stride for a, b in zip(offs, offs[1:])]
    if not rows_apart:
        return None
    q = statistics.quantiles(rows_apart, n=4)
    return {
        "reads": len(seq), "transitions": len(rows_apart),
        "mean_rows": statistics.fmean(rows_apart),
        "median_rows": statistics.median(rows_apart),
        "q1_rows": q[0], "q3_rows": q[2],
        "adjacent_pct": 100.0 * sum(d <= 1 for d in rows_apart) / len(rows_apart),
        "within_8_pct": 100.0 * sum(d <= 8 for d in rows_apart) / len(rows_apart),
        "mean_bytes": statistics.fmean(rows_apart) * REAL_STRIDE,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--rows", type=int, default=256)
    ap.add_argument("--protected", type=int, default=248)
    ap.add_argument("--qd", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    with open(a.trace) as f:
        rows_j = [json.loads(line) for line in f]
    meta, recs = rows_j[0]["meta"], rows_j[1:]
    print(f"config: rows={a.rows} hard(prot={a.rows}) vs soft(prot={a.protected})"
          f"  qd={a.qd}   [matches the qd probe]")
    out = {"meta": meta, "rows": a.rows, "protected": a.protected,
           "qd": a.qd, "real_stride_bytes": REAL_STRIDE, "arms": {}}
    with tempfile.TemporaryDirectory() as tmp:
        path, index = build(tmp, meta["layers"], meta["n_experts"])
        for name, prot in (("hard", a.rows), ("soft", a.protected)):
            seq, st = read_sequence(path, index, recs, a.rows, prot, a.qd)
            loc = locality(seq, index)
            out["arms"][name] = {**loc, "tier_misses": st.get("misses")}
            print(f"\n  {name}: {loc['reads']} reads "
                  f"(tier misses {st.get('misses')})")
            print(f"    mean {loc['mean_rows']:.2f} rows   "
                  f"median {loc['median_rows']:.1f}   "
                  f"IQR {loc['q1_rows']:.1f}-{loc['q3_rows']:.1f}")
            print(f"    adjacent (<=1 row) {loc['adjacent_pct']:.1f}%   "
                  f"within 8 rows {loc['within_8_pct']:.1f}%")
            print(f"    mean gap at real stride: "
                  f"{loc['mean_bytes'] / 1e6:.1f} MB")
    h, s = out["arms"]["hard"], out["arms"]["soft"]
    for k in ("mean_rows", "median_rows", "adjacent_pct", "within_8_pct"):
        d = s[k] - h[k]
        rel = (d / h[k] * 100) if h[k] else float("nan")
        out.setdefault("delta", {})[k] = {"abs": d, "pct": rel}
        print(f"\n  Δ {k:14} {d:+.2f}  ({rel:+.1f}%)")
    print("\nregistered: CONFIRMED needs a clear separation (soft materially "
          "less local);\n            REFUTED if medians are within a few "
          "percent and quantiles overlap.")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=2)
        print("receipt ->", a.out)


if __name__ == "__main__":
    main()
