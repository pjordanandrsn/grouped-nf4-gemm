"""Does the direct landing help a MIXED destination, where it was barred?

gnf4#143 measured the direct landing at -10.1% on a pure-CPU destination and
showed it is what reordered the destinations. A deadline scheduler never got
it: any destination that can route a row to the GPU was forced onto the copy
path, so its CPU-routed rows paid the slow fill while a pure-CPU run did not.
That biases the comparison the scheduler exists to make.

With the GPU stack reading the cold view, neither half of a mixed step calls
tier.row(), so the bar lifts. This measures whether lifting it pays.
"""
import argparse
import json
import statistics
import sys

import torch
from experts4bit_qlora.engines import hybrid as hy
from experts4bit_qlora.engines.placement import (force_cold_mass,
                                                 load_routing_mass,
                                                 solve_placement)
from experts4bit_qlora.loader import load_moe_4bit_streaming
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from run_gate1 import run_steps  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--calib", required=True)
ap.add_argument("--model", default="/root/models/olmoe")
ap.add_argument("--arena", default="/root/models/olmoe.arena")
ap.add_argument("--profile", default="/root/olmoe_profile_ORIGINAL.jsonl")
ap.add_argument("--dest", default="4.0", help="threshold => mixed destination")
ap.add_argument("--sweep", default="0.05,0.20")
ap.add_argument("--hot-rows", type=int, default=384)
ap.add_argument("--steps", type=int, default=128)
ap.add_argument("--warmup", type=int, default=6)
ap.add_argument("--repeats", type=int, default=2)
ap.add_argument("--out", required=True)
a = ap.parse_args()

calib = json.load(open(a.calib))
box = {"gpu": torch.cuda.get_device_name(0),
       "b_dram_gbs": calib["cpu_bench"]["triad_best"]["gbs"],
       "g0_verdict": calib["gate_g0"]["verdict"]}
print("box:", box)
from nvme_arena import load_index  # noqa: E402
idx = load_index(a.arena)
L, E, rb = idx["n_layers"], idx["n_experts_per_layer"], idx["row_bytes"]
mass, _ = load_routing_mass(a.profile, L, E)
base = solve_placement(n_layers=L, n_experts=E, bytes_per_expert=rb,
                       vram_budget_bytes=int(0.25 * L * E) * rb,
                       dram_budget_bytes=(L * E) * rb, calibration=a.calib,
                       profile_path=a.profile, top_k=8, batch=1)
from transformers import AutoTokenizer  # noqa: E402
PROSE = ("The question of how memory works has occupied philosophers and "
         "scientists for centuries. When we recall an event, we do not "
         "replay a recording; we reconstruct it, and the reconstruction "
         "is shaped by everything we have learned since. This is why "
         "eyewitness testimony is less reliable than juries assume. ")
ids = AutoTokenizer.from_pretrained(a.model)(
    PROSE * 4, return_tensors="pt").input_ids[:, :64].to("cuda")

out = {"schema": "e4b-direct-mixed/1", "box": box, "config": vars(a), "points": []}
for frac in [float(x) for x in a.sweep.split(",")]:
    man = force_cold_mass(base, mass, frac, order="tail", source="dram")
    pt = {"cold_frac": man["forced_cold"]["achieved_frac"], "arms": {}}
    print("\n=== cold %.0f%%, dest=%s ===" % (frac * 100, a.dest))
    ref = None
    for direct in (False, True):
        for r in range(a.repeats):
            name = f"{'direct' if direct else 'copy'}#{r+1}"
            model, _ = load_moe_4bit_streaming(
                a.model, device="cuda", dtype=torch.bfloat16, r=8, alpha=16,
                quant_type="nf4", arena=a.arena)
            n = hy.enable_hybrid_tier(model, a.arena, man, hot_rows=a.hot_rows,
                                      cold_dest=a.dest, cold_direct=direct,
                                      gpu_stacks_via_view=True, verbose=False)
            assert n > 0
            pre = {}
            steps, _, toks = run_steps(
                model, ids, a.steps, a.warmup,
                on_measure_start=lambda m=model: pre.update(hy.cold_stats(m)))
            cs = hy.cold_stats(model)
            tier = next(getattr(m, "_e4b_cold_tier", None)
                        for _, m in model.named_modules()
                        if getattr(m, "_e4b_cold_tier", None) is not None)
            v = getattr(tier, "_e4b_cold_view", None)
            vs = v.stats() if v is not None else {}
            arm = {"median_ns": statistics.median(steps),
                   "reads_in_window": cs.get("disk_reads", 0) - pre.get("disk_reads", 0),
                   "cold_rows_cpu": cs.get("cold_rows_cpu"),
                   "cold_rows_gpu": cs.get("cold_rows_gpu"),
                   "view_hits": vs.get("view_hits"),
                   "materializations": vs.get("materializations"),
                   "e4b_path": getattr(v, "e4b_path", None),
                   "cold_direct": direct}
            if ref is None:
                ref = toks
            arm["tokens_match"] = (toks == ref)
            pt["arms"][name] = arm
            print("  %-9s median=%8.2f ms cpu/gpu=%6d/%-6d mat=%-6s path=%-14s toks_ok=%s" % (
                name, arm["median_ns"]/1e6, arm["cold_rows_cpu"] or 0,
                arm["cold_rows_gpu"] or 0, arm["materializations"],
                arm["e4b_path"], arm["tokens_match"]))
            hy.disable_hybrid_tier(model)
            del model
            torch.cuda.empty_cache()
    out["points"].append(pt)
    json.dump(out, open(a.out, "w"), indent=2, default=str)
print("\nreceipt ->", a.out)
