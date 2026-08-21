"""Hard vs soft eviction in WALL TIME, at matched capacity, on real NVMe.

RESULTS-r10.md and RESULTS-r7.md measured this comparison in READS, offline,
and found soft eviction costs ~1% more everywhere. Both carried the same
caveat: reads are not wall, and R5 reports soft eviction *faster* than hard
when the tier is contended. This closes that gap.

It also scores R2:

    R2 -- VRAM resurrection is disproportionately valuable; even 2-5% of
    routed invocations moves wall time. REFUTED IF no measurable wall effect.

Same trace, same two arms, but on a box with real NVMe and a pinned tier, so
a resurrection is a genuinely skipped read rather than a counter that did not
increment.

The arena is synthetic with OLMoE's geometry (16x64, ~3.3 MB rows). Bytes do
not matter; row SIZE does, because it sets the bytes each miss actually
moves. Written one layer at a time so a 3.4 GB arena does not need 10 GB of
RAM to build.
"""
import argparse
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "kernel"))

from nvme_arena import bake, load_index          # noqa: E402
from nvme_residency import ColdTier              # noqa: E402

SUF = ("mlp.experts.gate_up_proj_blocks", "mlp.experts.gate_up_proj_scales",
       "mlp.experts.down_proj_blocks", "mlp.experts.down_proj_scales")


def _st_bytes(tensors):
    hdr, blobs, off = {}, [], 0
    for name, (shape, data) in tensors.items():
        hdr[name] = {"dtype": "U8", "shape": list(shape),
                     "data_offsets": [off, off + len(data)]}
        blobs.append(data)
        off += len(data)
    raw = json.dumps(hdr).encode()
    pad = (-len(raw)) % 8
    raw += b" " * pad
    return len(raw).to_bytes(8, "little") + raw + b"".join(blobs)


def make_snapshot(root, layers, experts, shapes, seed=7):
    """One shard per LAYER, so peak memory is one layer, not the arena."""
    os.makedirs(root, exist_ok=True)
    weight_map = {}
    for lay in range(layers):
        shard = "model-%03d.safetensors" % lay
        tensors = {}
        for suf, es in zip(SUF, shapes):
            name = "model.layers.%d.%s" % (lay, suf)
            n = experts * es[0] * es[1]
            tensors[name] = ((experts,) + es, os.urandom(n))
            weight_map[name] = shard
        with open(os.path.join(root, shard), "wb") as f:
            f.write(_st_bytes(tensors))
        del tensors
    with open(os.path.join(root, "model.safetensors.index.json"), "w") as f:
        json.dump({"weight_map": weight_map}, f)


def drop_caches():
    """A cold page cache, or the second arm just reads the first arm's RAM."""
    try:
        os.sync()
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3")
        return True
    except Exception:
        return False


def run(path, index, recs, rows, protected, pinned, qd=4, time_reads=False):
    """One arm. With time_reads, also partitions the wall into time spent
    INSIDE read_row and time spent everywhere else.

    #153's residual is a per-read cost that survives five eliminations. The
    one question that splits it cleanly is whether the extra time is inside
    the read or outside it, and nothing so far has measured that. The reader
    is wrapped rather than edited, the same non-invasive capture
    score_locality.py uses; the wrapper costs both arms the same per read, so
    it biases `read_ns` upward equally and cannot manufacture a difference
    beyond the read-count difference already being accounted for.
    """
    t = ColdTier(path, hot_rows=rows, pinned=pinned, index=index,
                 protected_rows=protected, qd=qd)
    read_ns = 0
    real_read_row = t.reader.read_row
    if time_reads:
        def timed(*a, **k):
            nonlocal read_ns
            s0 = time.perf_counter_ns()
            try:
                return real_read_row(*a, **k)
            finally:
                read_ns += time.perf_counter_ns() - s0
        t.reader.read_row = timed
    try:
        t0 = time.perf_counter_ns()
        for r in recs:
            for L, experts in r["routed"].items():
                t.ensure(int(L), experts)
        wall = time.perf_counter_ns() - t0
        st = dict(t.stats())
        st["wall_ns"] = wall
        tr = t.reader.traffic()
        st["reads"] = tr["reads"]
        # O_DIRECT or bust. The arena is ~3.4 GB and a run issues ~32k reads,
        # so every row is fetched ~31 times: buffered, the page cache serves
        # almost all of it and the run measures RAM, not storage. The reader
        # falls back to buffered silently enough that this has to be recorded.
        st["reader_mode"] = tr.get("mode")
        if time_reads:
            st["read_ns"] = read_ns
            st["non_read_ns"] = wall - read_ns
        return st
    finally:
        if time_reads:
            t.reader.read_row = real_read_row
        t.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--arena", required=True)
    ap.add_argument("--build", metavar="SNAPDIR", default=None,
                    help="build the synthetic arena here first, then run")
    ap.add_argument("--row-shapes", default="2048x1024,2048x64,1024x1024,1024x64",
                    help="per-expert [n,k] for the four segments; the default "
                         "gives a ~3.3 MB row, matching OLMoE's 3.54 MB")
    ap.add_argument("--rows", default="128,256,384")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--protected-frac", type=float, default=None,
                    help="protected = rows * FRAC instead of rows - k. "
                         "R2 is a claim about wall time when 2-5%% of routed "
                         "invocations are resurrections; at rows-k the rate "
                         "is ~0.5%%, too low to test it, and a smaller "
                         "protected budget is what raises it.")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--pinned", action="store_true")
    ap.add_argument("--qd", type=int, default=4,
                    help="reader queue depth. The residual is a per-read cost "
                         "(soft achieves lower effective bandwidth at the same "
                         "row size), and CPU work between reads can only cost "
                         "bandwidth when there is a queue to drain -- so qd=1 "
                         "vs qd=4 separates direct CPU cost from overlap loss.")
    ap.add_argument("--time-reads", action="store_true",
                    help="partition the wall into time inside read_row vs "
                         "everywhere else. This is the targeted instrument "
                         "for #153's residual: five candidates are eliminated "
                         "and the surviving shape is a per-read cost, so the "
                         "question is whether it lives in the read or around "
                         "it. Run WITHOUT this flag too -- the residual must "
                         "be unchanged by measuring it.")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    with open(a.trace) as f:
        rowsj = [json.loads(line) for line in f]
    meta, recs = rowsj[0]["meta"], rowsj[1:]

    if a.build:
        shapes = [tuple(int(v) for v in seg.split("x"))
                  for seg in a.row_shapes.split(",")]
        row = sum(n * k for n, k in shapes)
        print("building %dL x %dE snapshot, %.2f MB/row (%.1f GB arena)..." % (
            meta["layers"], meta["n_experts"], row / 1e6,
            meta["layers"] * meta["n_experts"] * row / 1e9))
        t0 = time.perf_counter()
        make_snapshot(a.build, meta["layers"], meta["n_experts"], shapes)
        print("  snapshot in %.0fs; baking..." % (time.perf_counter() - t0))
        t0 = time.perf_counter()
        bake(a.build, a.arena, align=4096, log=lambda *x: None)
        print("  baked in %.0fs" % (time.perf_counter() - t0))

    index = load_index(a.arena)
    print("arena: %dL x %dE, row %d B, pinned=%s" % (
        index["n_layers"], index["n_experts_per_layer"], index["row_bytes"],
        a.pinned))
    print("trace: %d steps, %d layers, top-%d of %d\n" % (
        meta["steps"], meta["layers"], meta["top_k"], meta["n_experts"]))

    routed = sum(len(v) for r in recs for v in r["routed"].values())
    out = {"meta": meta, "arena_row_bytes": index["row_bytes"],
           "routed_slots": routed,
           "pinned": a.pinned, "repeats": a.repeats, "qd": a.qd,
           "points": []}
    print("%6s %6s | %11s %11s %8s | %10s %10s %8s | %s" % (
        "rows", "prot", "hard ms", "soft ms", "d wall", "hard reads",
        "soft reads", "d reads", "resurrections"))
    for rows in [int(x) for x in a.rows.split(",")]:
        prot = (max(1, int(rows * a.protected_frac)) if a.protected_frac
                else max(1, rows - a.k))
        hw, sw, hr, sr, res = [], [], None, None, None
        hrd, srd = [], []          # per-repeat ns spent INSIDE read_row
        dropped = []               # did the page cache ACTUALLY get dropped?
        # A/B/A: alternate the arms so a drift in the box shows up as
        # disagreement between the two A runs rather than as a result.
        for i in range(a.repeats):
            dropped.append(drop_caches())
            h = run(a.arena, index, recs, rows, rows, a.pinned, a.qd,
                    a.time_reads)
            dropped.append(drop_caches())
            s = run(a.arena, index, recs, rows, prot, a.pinned, a.qd,
                    a.time_reads)
            hw.append(h["wall_ns"])
            sw.append(s["wall_ns"])
            if a.time_reads:
                hrd.append(h["read_ns"])
                srd.append(s["read_ns"])
            hr, sr = h["reads"], s["reads"]
            res = (s.get("resurrections", 0) or 0) + \
                  (s.get("spec_resurrections", 0) or 0)
        hm, sm = statistics.median(hw), statistics.median(sw)
        part = {}
        if a.time_reads:
            # Medians are taken per-quantity, so read and non-read medians need
            # not sum to the wall median. Reported anyway because the COMPARISON
            # is hard-vs-soft within each quantity, not the decomposition.
            hrm, srm = statistics.median(hrd), statistics.median(srd)
            hnm, snm = hm - hrm, sm - srm
            part = {
                "hard_read_ns": hrd, "soft_read_ns": srd,
                "hard_read_median_ns": hrm, "soft_read_median_ns": srm,
                "delta_read_ns_pct": (srm - hrm) / hrm * 100 if hrm else None,
                "hard_non_read_median_ns": hnm, "soft_non_read_median_ns": snm,
                "delta_non_read_pct": (snm - hnm) / hnm * 100 if hnm else None,
                "hard_read_frac_of_wall": hrm / hm if hm else None,
                "soft_read_frac_of_wall": srm / sm if sm else None,
            }
        out["points"].append({**part,
            "rows": rows, "protected": prot,
            "hard_wall_ns": hw, "soft_wall_ns": sw,
            "hard_wall_median_ns": hm, "soft_wall_median_ns": sm,
            "delta_wall_pct": (sm - hm) / hm * 100,
            "hard_reads": hr, "soft_reads": sr,
            "delta_reads_pct": (sr - hr) / hr * 100,
            "soft_resurrections": res,
            "reader_mode": h.get("reader_mode"),
            "page_cache_dropped": all(dropped),
            "drop_caches_attempts": len(dropped),
            "resurrection_frac_of_routed": res / routed})
        print("%6d %6d | %11.1f %11.1f %+7.1f%% | %10d %10d %+7.1f%% | "
              "%7d (%.2f%% of routed)" % (
                  rows, prot, hm / 1e6, sm / 1e6, (sm - hm) / hm * 100,
                  hr, sr, (sr - hr) / hr * 100, res, res / routed * 100))
        if a.time_reads:
            print("        read_ns  %11.1f %11.1f %+7.1f%%   "
                  "(read is %.1f%% / %.1f%% of wall)" % (
                      part["hard_read_median_ns"] / 1e6,
                      part["soft_read_median_ns"] / 1e6,
                      part["delta_read_ns_pct"],
                      part["hard_read_frac_of_wall"] * 100,
                      part["soft_read_frac_of_wall"] * 100))
            print("    non_read_ns  %11.1f %11.1f %+7.1f%%   <-- "
                  "the residual lives in whichever of these tracks wall" % (
                      part["hard_non_read_median_ns"] / 1e6,
                      part["soft_non_read_median_ns"] / 1e6,
                      part["delta_non_read_pct"]))
    modes = {p.get("reader_mode") for p in out["points"]}
    if any(m is None or "buffered" in str(m) for m in modes):
        print("\n*** READS WERE BUFFERED, NOT O_DIRECT ***\n"
              f"    reader mode: {sorted(str(m) for m in modes)}\n"
              "    The arena is ~3.4 GB and a run issues ~32k reads, so each\n"
              "    row is fetched ~31 times. Buffered, the page cache serves\n"
              "    nearly all of that and these numbers measure RAM. Re-run on\n"
              "    a filesystem that supports O_DIRECT.")
        out["INVALID"] = "reads were buffered, not O_DIRECT"
    if not all(p.get("page_cache_dropped") for p in out["points"]):
        print("\n*** PAGE CACHE WAS NOT DROPPED ***\n"
              "    /proc/sys/vm/drop_caches was not writable, so the second\n"
              "    arm read the first arm's RAM. These wall numbers do NOT\n"
              "    measure storage and must not be compared. Re-run on a host\n"
              "    where the container can drop caches (privileged).")
        out["INVALID"] = "page cache not dropped"
    if a.out:
        json.dump(out, open(a.out, "w"), indent=2)
        print("\nreceipt ->", a.out)


if __name__ == "__main__":
    main()
