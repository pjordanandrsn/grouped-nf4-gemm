"""Stage 3, gate 2: does choosing a destination by DEADLINE beat choosing by
threshold?

Registered in PREREG-tribrid-stage3.md as amended by amendment 1 (stamped
0d5f9dbe...). The amendment matters to how this is scored:

  * the arm gate 1 shipped as "dynamic" is a rows-per-unique-expert
    THRESHOLD; it is this gate's BASELINE, not its treatment. A deadline
    model has to beat the cheap rule, not merely beat fixed-GPU;
  * the primary axis is COMPUTE-side load asymmetry, not disk pressure --
    gate 1 measured storage at 5-11% of cold-path cost;
  * that destinations flip at all is already shown and is not re-litigated.

Arms, same manifest / routing / box, differing only in cold_dest:

  gpu        fixed, the pre-Stage-3 path
  cpu        fixed
  threshold  rows-per-unique rule (the BASELINE)
  deadline   predicted time-to-contribution incl. both engines' backlog

Load asymmetry is created by moving the VRAM/DRAM split, not by adding
synthetic work: a placement with more VRAM mass commits the GPU, one with
more DRAM mass commits the CPU. That keeps every arm running the same model
on the same routing.
"""
import argparse
import json
import statistics
import time

import torch
from transformers import AutoTokenizer

from experts4bit_qlora.engines import hybrid as hy
from experts4bit_qlora.engines.placement import (force_cold_mass,
                                                 load_routing_mass,
                                                 solve_placement)
from experts4bit_qlora.loader import load_moe_4bit_streaming

ARMS = ("gpu", "cpu", "threshold", "deadline")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/models/olmoe")
    ap.add_argument("--arena", default="/root/models/olmoe.arena")
    ap.add_argument("--profile", default="/root/olmoe_profile.jsonl")
    ap.add_argument("--calib", required=True)
    ap.add_argument("--cold", type=float, default=0.20)
    ap.add_argument("--hot-rows", type=int, default=128)
    ap.add_argument("--threshold", type=float, default=4.0)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--cpu-us-fixed", type=float, default=55.0)
    ap.add_argument("--cpu-us-per-row", type=float, default=2.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import cold_deadline
    from nvme_arena import load_index
    idx = load_index(a.arena)
    L, E, rb = idx["n_layers"], idx["n_experts_per_layer"], idx["row_bytes"]
    calib = json.load(open(a.calib))
    costs = cold_deadline.Costs.from_blob(
        calib, cpu_us_fixed=a.cpu_us_fixed,
        cpu_us_per_row=a.cpu_us_per_row, bytes_per_expert=rb)
    print("costs: cpu %.0f+%.1f/row us | b_dram %.1f b_vram %.1f b_link %.2f"
          % (costs.cpu_us_fixed, costs.cpu_us_per_row, costs.b_dram_gbs,
             costs.b_vram_gbs, costs.b_link_gbs))

    mass, _ = load_routing_mass(a.profile, L, E)
    PROSE = ("The question of how memory works has occupied philosophers and "
             "scientists for centuries. When we recall an event, we do not "
             "replay a recording; we reconstruct it, and the reconstruction "
             "is shaped by everything we have learned since. ")
    tk = AutoTokenizer.from_pretrained(a.model)
    ids = tk(PROSE * 4, return_tensors="pt").input_ids[:, :64].to("cuda")

    # Load asymmetry by MOVING THE SPLIT, not by adding synthetic work:
    # a fat VRAM budget commits the GPU, a thin one commits the CPU.
    regimes = {"gpu-loaded": 0.60, "cpu-loaded": 0.10}
    out = {"schema": "e4b-tribrid-gate2/1",
           "prereg": "PREREG-tribrid-stage3.md + amendment1",
           "config": vars(a), "regimes": {}}

    for rname, vfrac in regimes.items():
        base = solve_placement(
            n_layers=L, n_experts=E, bytes_per_expert=rb,
            vram_budget_bytes=int(vfrac * L * E) * rb,
            dram_budget_bytes=L * E * rb, calibration=a.calib,
            profile_path=a.profile, top_k=8, batch=1)
        man = force_cold_mass(base, mass, a.cold, order="tail", source="dram")
        print("\n=== %s: vram %d dram %d nvme %d ===" % (
            rname, len(man["tiers"]["vram"]), len(man["tiers"]["dram"]),
            len(man["tiers"]["nvme"])))
        out["regimes"][rname] = {"vram_frac": vfrac, "arms": {}}

        for arm in ARMS:
            dest = a.threshold if arm == "threshold" else arm
            model, _ = load_moe_4bit_streaming(
                a.model, device="cuda", dtype=torch.bfloat16, r=8, alpha=16,
                quant_type="nf4", arena=a.arena)
            n = hy.enable_hybrid_tier(
                model, a.arena, man, hot_rows=a.hot_rows, cold_dest=dest,
                costs=costs if arm == "deadline" else None)
            assert n == 16, n
            toks, per = [], []
            with torch.no_grad():
                for _ in range(3):
                    o = model(ids, use_cache=True)
                    c, p = o.logits[:, -1:].argmax(-1), o.past_key_values
                    for _ in range(4):
                        o = model(c, past_key_values=p, use_cache=True)
                        c, p = o.logits[:, -1:].argmax(-1), o.past_key_values
                torch.cuda.synchronize()
                o = model(ids, use_cache=True)
                p, c = o.past_key_values, o.logits[:, -1:].argmax(-1)
                for _ in range(a.steps):
                    t0 = time.perf_counter_ns()
                    o = model(c, past_key_values=p, use_cache=True)
                    torch.cuda.synchronize()
                    per.append(time.perf_counter_ns() - t0)
                    p, c = o.past_key_values, o.logits[:, -1:].argmax(-1)
                    toks.append(int(c.item()))
            cs = hy.cold_stats(model)
            out["regimes"][rname]["arms"][arm] = {
                "median_ns": statistics.median(per),
                "cold_rows_cpu": cs.get("cold_rows_cpu"),
                "cold_rows_gpu": cs.get("cold_rows_gpu"),
                "deadline_decisions": cs.get("deadline_decisions"),
                "deadline_flips": cs.get("deadline_flips"),
                "tokens": toks[:32]}
            print("  %-10s median %7.2f ms | cold cpu/gpu %6d/%-6d | "
                  "decisions %5s flips %5s" % (
                      arm, statistics.median(per) / 1e6,
                      cs.get("cold_rows_cpu", 0), cs.get("cold_rows_gpu", 0),
                      cs.get("deadline_decisions"), cs.get("deadline_flips")))
            hy.disable_hybrid_tier(model)
            del model
            torch.cuda.empty_cache()
        json.dump(out, open(a.out, "w"), indent=2)

    # score: deadline vs its BASELINE (the threshold), per amendment 1
    print()
    for rname, r in out["regimes"].items():
        A = r["arms"]
        if "threshold" in A and "deadline" in A:
            t, d = A["threshold"]["median_ns"], A["deadline"]["median_ns"]
            r["deadline_vs_threshold"] = (d - t) / t
            print("%-11s deadline vs threshold: %+.2f%%   (vs best fixed "
                  "%+.2f%%)" % (
                      rname, 100 * (d - t) / t,
                      100 * (d - min(A["gpu"]["median_ns"],
                                     A["cpu"]["median_ns"]))
                      / min(A["gpu"]["median_ns"], A["cpu"]["median_ns"])))
    json.dump(out, open(a.out, "w"), indent=2)
    print("\nreceipt ->", a.out)


if __name__ == "__main__":
    main()
