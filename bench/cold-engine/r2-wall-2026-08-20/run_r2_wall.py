"""R2's wall half: does a resurrection rate of 2-5%+ move wall time?

Registered (PREREG-tribrid-stage3, R2):

    VRAM resurrection is disproportionately valuable -- even 2-5% of routed
    invocations moves wall time -- REFUTED BY no measurable wall effect.

gnf4#151 scored the antecedent offline: the resurrection rate is a KNOB,
spanning 0.00% to 33.87% of routed invocations across cache configurations.
That is what makes this measurable -- the rate can be swept as an
independent variable while the routing stays fixed.

THE CONTROL IS THE HARD PART, and one obvious design does not work.
`protected = rows` would make the reclaimable set empty and resurrections
impossible -- but it also leaves nothing demotable, so `_claim` raises "no
slot available" and the arm cannot run at all. There is no configuration
that disables resurrections while leaving the cache otherwise identical.

So capacity is held FIXED and `protected` is swept, which is the knob that
moves the resurrection rate (gnf4#151 measured it spanning 0.00-33.87%).

What that cannot do is isolate resurrections from fills, and this harness
does not pretend otherwise: a resurrection IS an avoided fill, so the two
move together by construction. FILLS ARE THEREFORE REPORTED BESIDE WALL AND
RESURRECTIONS at every point. If wall tracks fills, the honest reading is
that capacity retention moved it; only a wall change that fills do not
explain can be laid at resurrection's door. R2's claim is really about that
joint quantity, and stating it as one variable would be the easy lie here.

Driven through the REAL Mxfp4NvmeResidency on a REAL gpt-oss arena, one
layer, decode-shaped (T=1 per step), which is the path R2 is about. Wall is
the median over the timed steps after warmup; the tier's own counters give
the resurrection rate beside it, so the two are never inferred from one
another.
"""
import argparse
import json
import statistics
import sys
import time

import torch

sys.path.insert(0, "/root/src/gnf4/kernel")
from mxfp4_residency import Mxfp4NvmeResidency          # noqa: E402
from nvme_arena import load_index                       # noqa: E402
from dev_row_cache import DevRowCache                   # noqa: E402


def flatten_block_shapes(index):
    """gpt-oss bakes blocks as [n, nblocks, 16]; the engine wants [n, k].

    `nvme_arena.bake` records the checkpoint's own shape, which is what makes
    sha256(arena) == sha256(source) provenance hold. The engine's segment map
    requires 2 dims, and for good reason: it separates blocks from scales by
    the packing invariant that a projection's blocks width is exactly 16x its
    scales width, and on the 3D shape both read 90 so the discriminator
    collapses. Flattened, blocks are 1440 against scales' 90 and it works.

    The BYTES do not move -- 5760x90x16 and 5760x1440 are the same 8,294,400
    per expert -- so this is metadata only, and it is done on a COPY here
    rather than in the bake or the kernel. Making it the arena's or the
    engine's job is a real decision with provenance consequences either way,
    and a measurement harness is the wrong place to make it. Recorded as a
    gap: serving a gpt-oss arena through Mxfp4NvmeResidency needs this step
    and nothing in the shipped path performs it.
    """
    import copy
    out = copy.deepcopy(index)
    for g in out["segments"]:
        shp = list(g["shape_per_expert"])
        if len(shp) > 2:
            flat = 1
            for d in shp[1:]:
                flat *= d
            g["shape_per_expert"] = [shp[0], flat]
    return out


def routes(steps, E, k, seed, device="cuda"):
    """A fixed routing sequence, identical for every arm."""
    g = torch.Generator(device=device).manual_seed(seed)
    out = []
    for _ in range(steps):
        logits = torch.randn(1, E, device=device, generator=g)
        sc, idx = torch.topk(torch.softmax(logits, -1), k=k, dim=-1)
        out.append((idx, sc.to(torch.bfloat16)))
    return out


def run_arm(path, index, layer, k, hot_rows, rows, protected, seq, x, warmup):
    cache = None
    if rows:
        cache = DevRowCache(rows, index["row_stride"], device="cuda",
                            protected=protected)
    eng = Mxfp4NvmeResidency(path, layer, hot_ids=(), k_slots=k,
                             hot_rows=hot_rows, device="cuda", index=index,
                             dev_cache=cache)
    try:
        with torch.no_grad():
            for i in range(warmup):
                idx, sc = seq[i % len(seq)]
                eng.forward(x, idx, sc)
            torch.cuda.synchronize()
            per = []
            for idx, sc in seq:
                t0 = time.perf_counter_ns()
                eng.forward(x, idx, sc)
                torch.cuda.synchronize()
                per.append(time.perf_counter_ns() - t0)
        t = eng.traffic() if hasattr(eng, "traffic") else {}
        dc = t.get("dev_cache", {}) if isinstance(t, dict) else {}
        res = (dc.get("resurrections", 0) or 0) + (dc.get("spec_resurrections", 0) or 0)
        routed = len(seq) * k
        # Reported beside wall on purpose: a resurrection is an avoided fill,
        # so a wall change that the fill count already explains must not be
        # credited to resurrection (see the module docstring).
        fills = dc.get("fills", dc.get("misses", dc.get("overwritten")))
        return {"median_ns": statistics.median(per), "steps": len(per),
                "rows": rows, "protected": protected, "fills": fills,
                "resurrections": res, "per_routed": res / routed if routed else 0.0,
                "dev_cache": dc, "traffic": {kk: vv for kk, vv in t.items()
                                             if not isinstance(vv, dict)}}
    finally:
        del eng
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arena", default="/root/models/gptoss.arena")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=16)
    ap.add_argument("--hot-rows", type=int, default=64)
    ap.add_argument("--rows", default="64,96,128")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    index = flatten_block_shapes(load_index(a.arena))
    E = index["n_experts_per_layer"]
    # hidden is the DOWN projection's output dim, not gate_up's -- gate_up's
    # first dim is 2*intermediate (the fused gate and up halves), which for
    # gpt-oss-20b is 5760 against a hidden of 2880. Feeding the engine a 5760
    # wide activation would fail far from its cause.
    seg = {s["suffix"].rsplit(".", 1)[-1]: s for s in index["segments"]}
    hidden = seg["down_proj_blocks"]["shape_per_expert"][0]
    print(f"arena: {index['n_layers']}L x {E}E row={index['row_bytes']}B "
          f"stride={index['row_stride']}B hidden={hidden}")
    seq = routes(a.steps, E, a.k, a.seed)
    x = torch.randn(1, hidden, dtype=torch.bfloat16, device="cuda")
    out = {"schema": "r2-wall/1", "gpu": torch.cuda.get_device_name(0),
           "config": vars(a), "arms": []}
    print(f"\n{'rows':>5} {'prot':>6} {'budget':>16} {'median ms':>10} "
          f"{'resurr':>8} {'per routed':>11} {'fills':>8}")
    for rows in [int(v) for v in a.rows.split(",")]:
        # PAIRED: same rows, resurrections off vs on. Capacity is held.
        budgets = [("quarter", rows // 4), ("half", rows // 2),
                   ("three-quarter", 3 * rows // 4), ("rows-k", rows - a.k)]
        for label, prot in budgets:
            if prot < 1 or prot >= rows:
                continue
            for rep in range(a.repeats):
                try:
                    arm = run_arm(a.arena, index, a.layer, a.k, a.hot_rows,
                                  rows, prot, seq, x, a.warmup)
                except (RuntimeError, ValueError) as exc:
                    print(f"{rows:>5} {prot:>6} {label:>18}  refused: "
                          f"{str(exc).split(chr(10))[0][:40]}")
                    break
                arm["label"] = f"{label}#{rep+1}"
                out["arms"].append(arm)
                print(f"{rows:>5} {prot:>6} {arm['label']:>16} "
                      f"{arm['median_ns']/1e6:>10.3f} {arm['resurrections']:>8} "
                      f"{arm['per_routed']:>10.2%} {str(arm['fills']):>8}")
        json.dump(out, open(a.out, "w"), indent=2, default=str)
    json.dump(out, open(a.out, "w"), indent=2, default=str)
    print("\nreceipt ->", a.out)


if __name__ == "__main__":
    main()
