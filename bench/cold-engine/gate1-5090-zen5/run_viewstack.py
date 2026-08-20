"""A/B: does a GPU-destined cold stack reading the view beat rebuilding it?

Gate 1 attributes ~98% of cold cost to staging rather than disk
(RESULTS-tribrid-gate1.md, corrected 2026-08-20). `_TieredStack.index_select`
was rebuilding every routed row with `segment_tensor` on every call, even for
experts the cold view had materialized on an earlier step. This measures the
change end to end.

The control is not a degraded mode: `gpu_stacks_via_view=False` is the engine
that shipped before the change, so both arms run the same code path except
for that switch (the rule gnf4#133 set for DevRowCache).

Self-paired: each arm runs twice, so an effect smaller than the instrument's
own disagreement is reported as noise rather than a win.
"""
import argparse
import json
import statistics

import torch

from experts4bit_qlora.engines import hybrid as hy
from experts4bit_qlora.engines.placement import (force_cold_mass,
                                                 load_routing_mass,
                                                 solve_placement)
from experts4bit_qlora.loader import load_moe_4bit_streaming

import sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from run_gate1 import run_steps  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/models/olmoe")
    ap.add_argument("--arena", default="/root/models/olmoe.arena")
    ap.add_argument("--profile", default="/root/olmoe_profile_ORIGINAL.jsonl")
    ap.add_argument("--calib", required=True)
    ap.add_argument("--sweep", default="0.05,0.20")
    ap.add_argument("--hot-rows", type=int, default=384)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=6)
    ap.add_argument("--vram-frac", type=float, default=0.25)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--direct", default="both", choices=("both","true","false"),
                    help="cold_direct for the GPU destination. 'both' A/Bs "
                         "the preadv landing, which was illegal here until "
                         "the stack learned to read the view.")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    calib = json.load(open(a.calib))
    box = {"gpu": torch.cuda.get_device_name(0),
           "b_dram_gbs": calib["cpu_bench"]["triad_best"]["gbs"],
           "g0_verdict": calib["gate_g0"]["verdict"],
           "b_nvme_seq_gbs": max(
               (p["gbs"] for p in ((calib["cpu_bench"].get("nvme") or {})
                                   .get("points") or [])
                if p.get("mode") == "seq"), default=None)}
    print("box:", box)

    from nvme_arena import load_index
    idx = load_index(a.arena)
    L, E, rb = idx["n_layers"], idx["n_experts_per_layer"], idx["row_bytes"]
    mass, _ = load_routing_mass(a.profile, L, E)
    base = solve_placement(
        n_layers=L, n_experts=E, bytes_per_expert=rb,
        vram_budget_bytes=int(a.vram_frac * L * E) * rb,
        dram_budget_bytes=(L * E) * rb,
        calibration=a.calib, profile_path=a.profile, top_k=8, batch=1)
    assert not base["tiers"]["nvme"]

    from transformers import AutoTokenizer
    PROSE = ("The question of how memory works has occupied philosophers and "
             "scientists for centuries. When we recall an event, we do not "
             "replay a recording; we reconstruct it, and the reconstruction "
             "is shaped by everything we have learned since. This is why "
             "eyewitness testimony is less reliable than juries assume. ")
    ids = AutoTokenizer.from_pretrained(a.model)(
        PROSE * 4, return_tensors="pt").input_ids[:, :a.seq].to("cuda")

    out = {"schema": "e4b-gpu-stacks-via-view/1", "box": box,
           "config": vars(a), "points": []}
    for frac in [float(x) for x in a.sweep.split(",")]:
        man = force_cold_mass(base, mass, frac, order="tail", source="dram")
        pt = {"cold_frac": man["forced_cold"]["achieved_frac"],
              "experts_moved": man["forced_cold"]["experts_moved"], "arms": {}}
        print("\n=== cold %.0f%% (%d experts) ===" % (
            frac * 100, pt["experts_moved"]))
        ref_toks = None
        arms = ([(True, True), (True, False)] if a.direct == "both"
                else [(True, a.direct == "true")])
        for via, direct in arms:
            for rep in range(a.repeats):
                name = f"{'direct' if direct else 'copy'}#{rep + 1}"
                model, _ = load_moe_4bit_streaming(
                    a.model, device="cuda", dtype=torch.bfloat16, r=8,
                    alpha=16, quant_type="nf4", arena=a.arena)
                n = hy.enable_hybrid_tier(
                    model, a.arena, man, hot_rows=a.hot_rows, cold_dest="gpu",
                    gpu_stacks_via_view=via, cold_direct=direct,
                    verbose=False)
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
                       "view_hits": vs.get("view_hits"),
                       "materializations": vs.get("materializations"),
                       "rows_requested": vs.get("rows_requested"),
                       "cold_rows_gpu": cs.get("cold_rows_gpu"),
                       "serve_gpu_stacks": via, "cold_direct": direct,
                       "e4b_path": getattr(v, "e4b_path", None)}
                if ref_toks is None:
                    ref_toks = toks
                arm["tokens_match"] = (toks == ref_toks)
                pt["arms"][name] = arm
                print("  %-10s median=%8.2f ms reads=%5d view_hits=%-8s "
                      "materializations=%-8s toks_ok=%s" % (
                          name, arm["median_ns"] / 1e6, arm["reads_in_window"],
                          arm["view_hits"], arm["materializations"],
                          arm["tokens_match"]), "path=", arm["e4b_path"])
                hy.disable_hybrid_tier(model)
                del model
                torch.cuda.empty_cache()
        out["points"].append(pt)
        json.dump(out, open(a.out, "w"), indent=2, default=str)
    json.dump(out, open(a.out, "w"), indent=2, default=str)
    print("\nreceipt ->", a.out)


if __name__ == "__main__":
    main()
