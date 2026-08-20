"""R1 re-measurement on the corrected view path.

The published P(reuse before overwrite) of 11-60%
(RESULTS-tribrid-reclaimable.md) was WITHDRAWN in gnf4#130: every arm ran
through ColdCpuView.ensure, which materialized a batch by calling
segment_into per expert per segment, and segment_into issued its OWN demand
ColdTier.ensure. Each of those replaced the demand window and ran the
demotion pass, so a batch logically evicted its own members and the next
segment's hit "resurrected" them. Both terms of the ratio were inflated by
self-inflicted cycles.

The bias has a direction. A self-inflicted demotion is nearly always
followed by a hit on that same batch member, so contaminated pairs enter the
ratio at close to 1.0 and pull P UP. This harness therefore expects the
corrected number to be LOWER, and the registered band (5-20%) to be a live
possibility rather than a foregone pass.

Identical to the withdrawn runs in every other respect -- same model, same
prompt, same routing trace, same placement, same two configurations -- so
the only thing that changed is the code under test.
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
from run_gate1 import run_steps  # noqa: E402  same decode-shaped loop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/models/olmoe")
    ap.add_argument("--arena", default="/root/models/olmoe.arena")
    ap.add_argument("--profile", default="/root/olmoe_profile.jsonl")
    ap.add_argument("--calib", required=True)
    ap.add_argument("--cold-frac", type=float, required=True)
    ap.add_argument("--hot-rows", type=int, required=True)
    ap.add_argument("--protected", required=True,
                    help="comma-separated protected_rows sweep, descending; "
                         "the first should equal --hot-rows (the arm where "
                         "the reclaimable set is empty by construction)")
    ap.add_argument("--order", default="tail")
    ap.add_argument("--source", default="dram", choices=("dram", "vram"))
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--warmup", type=int, default=6)
    ap.add_argument("--vram-frac", type=float, default=0.25)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--cold-direct", default="true",
                    choices=("true", "false"),
                    help="preadv scatter straight into kernel-shaped stacks. "
                         "DEFAULT TRUE in e4b, and it bypasses segment_into "
                         "entirely -- so the #112 self-ensure bug is "
                         "unreachable under it, and a run at the default "
                         "cannot reproduce OR refute the withdrawn numbers, "
                         "which predate the direct path being wired. Use "
                         "false to measure the configuration those numbers "
                         "were taken in.")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    calib = json.load(open(a.calib))
    # Recorded for provenance, not used in the solve (which takes the path).
    # A receipt without its box's measured ceilings cannot be compared to
    # another box's, and the whole point of this run is comparability.
    box = {"gpu": torch.cuda.get_device_name(0),
           "b_vram_gbs": max(d["b_vram_triad_gbs"]
                             for d in calib["gpu_bench"]["devices"]),
           "b_dram_gbs": calib["cpu_bench"]["triad_best"]["gbs"],
           # SEQUENTIAL ceiling only. There is no "seq_best_gbs" key -- the
           # blob is {"o_direct", "points"} with per-point "mode" -- so the
           # older spelling silently recorded None, and a max() over ALL
           # points would quote the random-QD peak (6.63 here vs 5.54 seq),
           # overstating the device. Bugbot caught that in the training
           # harness (gnf4#129); fixed here before it could repeat.
           "b_nvme_seq_gbs": max(
               (p["gbs"] for p in ((calib["cpu_bench"].get("nvme") or {})
                                   .get("points") or [])
                if p.get("mode") == "seq"), default=None),
           "g0_verdict": calib["gate_g0"]["verdict"]}
    print("calibration: %s" % box)

    from nvme_arena import load_index
    idx = load_index(a.arena)
    n_layers, n_experts = idx["n_layers"], idx["n_experts_per_layer"]
    row_bytes = idx["row_bytes"]

    mass, _ = load_routing_mass(a.profile, n_layers, n_experts)
    vram_slots = int(a.vram_frac * n_layers * n_experts)
    base = solve_placement(
        n_layers=n_layers, n_experts=n_experts, bytes_per_expert=row_bytes,
        vram_budget_bytes=vram_slots * row_bytes,
        dram_budget_bytes=(n_layers * n_experts) * row_bytes,
        calibration=a.calib, profile_path=a.profile, top_k=8, batch=1)
    assert not base["tiers"]["nvme"], "control must have NO cold experts"
    forced = force_cold_mass(base, mass, a.cold_frac, order=a.order,
                             source=a.source)
    print("cold %.0f%% -> achieved %.4f, %d experts moved" % (
        a.cold_frac * 100, forced["forced_cold"]["achieved_frac"],
        forced["forced_cold"]["experts_moved"]))

    from transformers import AutoTokenizer
    PROSE = ("The question of how memory works has occupied philosophers and "
             "scientists for centuries. When we recall an event, we do not "
             "replay a recording; we reconstruct it, and the reconstruction "
             "is shaped by everything we have learned since. This is why "
             "eyewitness testimony is less reliable than juries assume. ")
    _tk = AutoTokenizer.from_pretrained(a.model)
    ids = _tk(PROSE * 4, return_tensors="pt").input_ids[:, :a.seq].to("cuda")

    receipt = {"schema": "e4b-tribrid-reclaimable/2",
               "supersedes": "e4b-tribrid-reclaimable/1 (WITHDRAWN, gnf4#130)",
               "hot_rows": a.hot_rows, "dest": "cpu",
               "cold_direct": a.cold_direct == "true",
               "cold_frac_achieved": forced["forced_cold"]["achieved_frac"],
               "box": box,
               "config": vars(a), "arms": []}

    ref_toks = None
    for prot in [int(x) for x in a.protected.split(",")]:
        model, _ = load_moe_4bit_streaming(
            a.model, device="cuda", dtype=torch.bfloat16, r=8, alpha=16,
            quant_type="nf4", arena=a.arena)
        n = hy.enable_hybrid_tier(
            model, a.arena, forced, hot_rows=a.hot_rows, cold_dest="cpu",
            protected_rows=prot, cold_direct=(a.cold_direct == "true"),
            verbose=False)
        assert n > 0, "tier not engaged"
        pre = hy.cold_stats(model)
        steps, _, toks = run_steps(model, ids, a.steps, a.warmup)
        cs = hy.cold_stats(model)
        arm = {
            "protected_rows": prot,
            "median_ns": statistics.median(steps),
            "reads_in_window": cs.get("disk_reads", 0) - pre.get("disk_reads", 0),
            "logical_evictions": cs.get("logical_evictions", 0),
            "resurrections": cs.get("resurrections", 0),
            "spec_resurrections": cs.get("spec_resurrections", 0),
            "reclaimable_overwritten": cs.get("reclaimable_overwritten", 0),
            "reuse_before_overwrite": cs.get("reuse_before_overwrite"),
            "resurrection_bytes_saved": cs.get("resurrection_bytes_saved", 0),
            "physical_evictions": cs.get("evictions", 0),
            "mean_ticks_to_overwrite": cs.get("mean_ticks_to_overwrite"),
            "mean_ticks_to_resurrection": cs.get("mean_ticks_to_resurrection"),
        }
        # Reclaimable residency is bookkeeping: it must not change a single
        # emitted token at ANY protected_rows. If it does, the mechanism is
        # not what it claims to be and no latency number from it is usable.
        if ref_toks is None:
            ref_toks = toks
        arm["tokens_match"] = (toks == ref_toks)
        receipt["arms"].append(arm)
        p = arm["reuse_before_overwrite"]
        print("  prot=%4d median=%8.2f ms reads=%5d logEv=%5d resurr=%5d "
              "overwr=%5d P=%s toks_ok=%s" % (
                  prot, arm["median_ns"] / 1e6, arm["reads_in_window"],
                  arm["logical_evictions"], arm["resurrections"],
                  arm["reclaimable_overwritten"],
                  ("%.3f" % p) if p is not None else "-", arm["tokens_match"]))
        hy.disable_hybrid_tier(model)
        del model
        torch.cuda.empty_cache()
        json.dump(receipt, open(a.out, "w"), indent=2, default=str)

    receipt["all_tokens_identical"] = all(x["tokens_match"] for x in receipt["arms"])
    json.dump(receipt, open(a.out, "w"), indent=2, default=str)
    print("\nall_tokens_identical =", receipt["all_tokens_identical"])
    print("receipt ->", a.out)


if __name__ == "__main__":
    main()
