"""Direct-scatter vs copy: what does removing one memcpy per segment buy?

Isolates the ONE variable. Both arms read the same bytes off the same disk
with the same syscall count -- one preadv per expert row. The copy arm lands
that row in a ColdTier slot and then memcpys each segment into the
kernel-shaped stacks; the direct arm scatters the segments into those stacks
in the read itself. Nothing else differs: same tier, same hot_rows, same
routing trace, same miss sequence.

No GPU and no model: this measures host copy + storage, which is what the
change touches. An end-to-end serving number is a different (and noisier)
question.

A/B/A ordering per instrument law 6, and a self-pair, because the claim is a
difference of medians.
"""
import argparse
import json
import os
import random
import statistics
import struct
import sys
import time

sys.path.insert(0, "/root/gnf4/kernel")

import numpy as np  # noqa: E402

from cold_cpu_view import ColdCpuView  # noqa: E402
from nvme_arena import bake_expert_tensors, load_index  # noqa: E402
from nvme_residency import ColdTier  # noqa: E402

# OLMoE-1B-7B expert geometry, which is what the gate-1 runs used.
N, K = 2048, 2048
KINDS = ("nf4.gate_up_blocks", "nf4.gate_up_absmax")
SHAPES = {"nf4.gate_up_blocks": (N, K // 2),        # 2 MiB, 4096-aligned
          "nf4.gate_up_absmax": (N, K // 64 * 4)}   # 512 KiB, 4096-aligned
TEMPLATE = "model.layers.{layer}.mlp.experts.{expert}.{kind}"


def _st(tensors):
    hdr, blobs, off = {}, [], 0
    for name, (data, shape) in tensors.items():
        hdr[name] = {"dtype": "U8", "shape": list(shape),
                     "data_offsets": [off, off + len(data)]}
        blobs.append(data)
        off += len(data)
    hj = json.dumps(hdr).encode()
    return struct.pack("<Q", len(hj)) + hj + b"".join(blobs)


def build(root, layers, experts, seed=5):
    rng = np.random.default_rng(seed)
    shard, wm = {}, {}
    for lay in range(layers):
        for e in range(experts):
            for kind in KINDS:
                a = rng.integers(0, 256, SHAPES[kind], dtype=np.uint8)
                nm = TEMPLATE.format(layer=lay, expert=e, kind=kind)
                shard[nm] = (a.tobytes(), SHAPES[kind])
                wm[nm] = "a.safetensors"
    os.makedirs(root, exist_ok=True)
    open(os.path.join(root, "a.safetensors"), "wb").write(_st(shard))
    json.dump({"weight_map": wm}, open(
        os.path.join(root, "model.safetensors.index.json"), "w"))


def trace(layers, experts, steps, k, seed=20260819):
    """Skewed routing, like a real MoE: a hot head plus a long tail."""
    rng = random.Random(seed)
    w = [1.0 / (i + 1) ** 1.1 for i in range(experts)]
    out = []
    for _ in range(steps):
        for lay in range(layers):
            out.append((lay, rng.choices(range(experts), weights=w, k=k)))
    return out


def run(arena, index, sufs, hot_rows, tr, direct):
    if direct:
        holder = {}

        def landing(layer, expert, slot):
            return holder["v"].landing(layer, expert, slot)

        tier = ColdTier(arena, hot_rows=hot_rows, pinned=False, index=index,
                        landing=landing)
        holder["v"] = view = ColdCpuView(tier, index, sufs, direct=True)
    else:
        tier = ColdTier(arena, hot_rows=hot_rows, pinned=False, index=index)
        view = ColdCpuView(tier, index, sufs)
    per = []
    for lay, experts in tr:
        t0 = time.perf_counter_ns()
        view.ensure(lay, experts)
        per.append(time.perf_counter_ns() - t0)
    st = tier.stats()
    out = {"median_ns": statistics.median(per), "total_ns": sum(per),
           "calls": len(per), "disk_reads": st["disk_reads"],
           "misses": st["misses"], "hits": st["hits"],
           "materializations": view.stats()["materializations"]}
    tier.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/root/bench_arena")
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--experts", type=int, default=64)
    ap.add_argument("--hot-rows", type=int, default=96)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--out", default="/root/bench_direct.json")
    a = ap.parse_args()

    snap = os.path.join(a.root, "snap")
    arena = os.path.join(a.root, "bench.arena")
    if not os.path.exists(arena):
        build(snap, a.layers, a.experts)
        bake_expert_tensors(snap, arena, name_template=TEMPLATE, kinds=KINDS,
                            align=4096, log=lambda *x: None)
    index = load_index(arena)
    sufs = list(KINDS)
    tr = trace(a.layers, a.experts, a.steps, a.topk)
    print("arena: L=%d E=%d row=%.2f MB | hot_rows=%d | calls=%d"
          % (index["n_layers"], index["n_experts_per_layer"],
             index["row_bytes"] / 1e6, a.hot_rows, len(tr)))

    res = {"config": vars(a), "arms": {}}
    for tag, direct in (("copy-A", False), ("direct", True), ("copy-B", False)):
        r = run(arena, index, sufs, a.hot_rows, tr, direct)
        res["arms"][tag] = r
        print("%-8s median %8.1f us | total %7.1f ms | reads %5d | miss %5d | mat %5d"
              % (tag, r["median_ns"] / 1e3, r["total_ns"] / 1e6,
                 r["disk_reads"], r["misses"], r["materializations"]))

    a1, a2 = res["arms"]["copy-A"], res["arms"]["copy-B"]
    d = res["arms"]["direct"]
    sp = abs(a1["total_ns"] - a2["total_ns"]) / min(a1["total_ns"], a2["total_ns"])
    base = min(a1["total_ns"], a2["total_ns"])
    res["self_pair_spread"] = sp
    res["direct_vs_copy"] = (d["total_ns"] - base) / base
    res["reads_match"] = a1["disk_reads"] == d["disk_reads"] == a2["disk_reads"]
    print("\nself-pair spread: %.2f%%   direct vs copy: %+.2f%%   reads matched: %s"
          % (sp * 100, res["direct_vs_copy"] * 100, res["reads_match"]))
    if abs(res["direct_vs_copy"]) <= sp:
        print("VERDICT: inside the instrument's own spread -- no measurable difference")
    json.dump(res, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
