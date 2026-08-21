"""Does the R2 wall null survive REAL routing?

Registered in bench/cold-engine/PREREG-wall-real-routing.md, sha256
3d9039038ba3f561... , committed and pushed before this box was rented.

RESULTS-r2-wall.md refuted R2 -- resurrections reached 5.37-15.72% of routed
work and moved wall by nothing. It drives the engine with `routes()`, which
draws FRESH `torch.randn` logits every step, so routing is independent across
steps and step-to-step reuse sits at chance, k/E = 4/32 = 12.5%. Every
captured real trace runs 2.0-3.3x chance. The null was measured on the
routing LEAST favourable to the mechanism it was testing.

This runs the identical measurement against gpt-oss-20b's OWN captured
routing -- same model the arena was baked from, so E=32 and k=4 with no id
remapping -- paired against the synthetic sequence in the same process.

`run_arm` is IMPORTED from run_r2_wall rather than copied. Its measurement
boundary and counter whitelist carry three separate Bugbot fixes (lifetime
vs windowed counters, gauge-differencing, host_to_cache_rows vs overwritten);
a copy would silently fork them.
"""
import argparse
import json
import os
import statistics
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "r2-wall-2026-08-20"))
from run_r2_wall import (flatten_block_shapes, load_index,   # noqa: E402
                         routes, run_arm)


def routes_from_trace(path, layer, k, E, device="cuda"):
    """The captured sequence for ONE layer, shaped exactly like `routes()`.

    Scores are uniform 1/k. The trace records routed IDS, not gate weights,
    and weights do not change which rows are fetched or when -- they are a
    combine coefficient. Wall here is dominated by row transfer and the
    gather; using uniform weights keeps the ROUTING identical to what the
    model actually did, which is the variable under test, and is recorded in
    the receipt so it is not mistaken for the model's own gating.
    """
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    meta, recs = rows[0]["meta"], rows[1:]
    if int(meta["top_k"]) != k:
        raise SystemExit("trace top_k=%s != engine k=%s" % (meta["top_k"], k))
    if meta.get("n_experts") is not None and int(meta["n_experts"]) != E:
        raise SystemExit("trace E=%s != arena E=%s" % (meta["n_experts"], E))
    key = str(layer)
    seq = []
    for r in recs:
        ex = r["routed"].get(key)
        if ex is None:
            continue
        if max(ex) >= E:
            raise SystemExit("expert id %d >= arena E=%d" % (max(ex), E))
        idx = torch.tensor([ex], dtype=torch.int64, device=device)
        sc = torch.full((1, k), 1.0 / k, dtype=torch.bfloat16, device=device)
        seq.append((idx, sc))
    if not seq:
        raise SystemExit("layer %s absent from every step of %s" % (layer, path))
    return seq, meta


def overlap(seq):
    """Mean step-to-step routed-set overlap, the quantity this is all about."""
    sets = [set(idx.reshape(-1).tolist()) for idx, _ in seq]
    v = [len(sets[i] & sets[i - 1]) / len(sets[i]) for i in range(1, len(sets))]
    return sum(v) / len(v) if v else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arena", required=True)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=16)
    ap.add_argument("--hot-rows", type=int, default=32)
    ap.add_argument("--rows", default="12,16,24,32,48")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    index = flatten_block_shapes(load_index(a.arena))
    E = index["n_experts_per_layer"]
    seg = {s["suffix"].rsplit(".", 1)[-1]: s for s in index["segments"]}
    hidden = seg["down_proj_blocks"]["shape_per_expert"][0]
    x = torch.randn(1, hidden, dtype=torch.bfloat16, device="cuda")

    syn = routes(a.steps, E, a.k, a.seed)
    real, tmeta = routes_from_trace(a.trace, a.layer, a.k, E)
    real = real[:a.steps]
    seqs = {"synthetic": syn, "captured": real}
    ov = {n: overlap(s) for n, s in seqs.items()}
    print("arena: %dL x %dE row=%dB stride=%dB hidden=%d"
          % (index["n_layers"], E, index["row_bytes"], index["row_stride"], hidden))
    print("steps: synthetic=%d captured=%d (layer %d of %s)"
          % (len(syn), len(real), a.layer, os.path.basename(a.trace)))
    print("step-to-step overlap: synthetic %.1f%%  captured %.1f%%  (chance %.1f%%)"
          % (100 * ov["synthetic"], 100 * ov["captured"], 100 * a.k / E))

    out = {"schema": "wall-real-routing/1", "gpu": torch.cuda.get_device_name(0),
           "prereg": "bench/cold-engine/PREREG-wall-real-routing.md",
           "config": vars(a), "trace_meta": tmeta, "overlap": ov, "arms": []}
    print("\n%-10s %5s %6s %5s %11s %9s %11s %9s"
          % ("seq", "rows", "prot", "rep", "median ms", "resurr", "per routed", "fills"))
    for rows in [int(v) for v in a.rows.split(",")]:
        for label, prot in (("quarter", rows // 4), ("rows-k", rows - a.k)):
            if prot < 1 or prot >= rows:
                continue
            for rep in range(a.repeats):
                # ALTERNATING inside the repeat loop, not one sequence then
                # the other: thermal drift and any host-side warmup then land
                # on both arms equally instead of on whichever ran second.
                for name in ("synthetic", "captured"):
                    try:
                        arm = run_arm(a.arena, index, a.layer, a.k, a.hot_rows,
                                      rows, prot, seqs[name], x, a.warmup)
                    except (RuntimeError, ValueError) as exc:
                        print("%-10s %5d %6d %5d  refused: %s"
                              % (name, rows, prot, rep + 1,
                                 str(exc).split(chr(10))[0][:44]))
                        continue
                    arm["sequence"] = name
                    arm["budget"] = label
                    arm["rep"] = rep + 1
                    out["arms"].append(arm)
                    print("%-10s %5d %6d %5d %11.3f %9d %10.2f%% %9s"
                          % (name, rows, prot, rep + 1,
                             arm["median_ns"] / 1e6, arm["resurrections"],
                             100 * arm["per_routed"], arm["fills"]))
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)
    print("\nreceipt -> %s (%d arms)" % (a.out, len(out["arms"])))


if __name__ == "__main__":
    main()
