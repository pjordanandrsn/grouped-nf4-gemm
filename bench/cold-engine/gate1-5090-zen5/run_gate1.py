"""Stage-3 gate 1, executed. Registered in
grouped-nf4-gemm/bench/cold-engine/PREREG-tribrid-stage3.md (stamped
7bf5b2be87aef56dc514f67cf90ea219ba1289004aaf0901e5c3230503a52ef5).

Four arms per cold-mass point, one model, one prompt, one routing trace,
one placement, one box. The only thing that differs between arms is where
cold experts execute.
"""
import argparse
import json
import statistics
import time

import torch

from experts4bit_qlora.engines import hybrid as hy
from experts4bit_qlora.engines.placement import (force_cold_mass,
                                                 load_routing_mass,
                                                 solve_placement)
from experts4bit_qlora.loader import load_moe_4bit_streaming

ARMS = ("control", "cold-gpu", "cold-cpu", "dynamic")


def arm_dest(arm, thr):
    return {"control": "gpu", "cold-gpu": "gpu",
            "cold-cpu": "cpu", "dynamic": thr}[arm]


def run_steps(model, ids, n_steps, warmup, on_measure_start=None):
    """DECODE-shaped measurement: prefill once, then one token per step
    against a KV cache.

    The earlier version re-ran the whole prompt every step. That is a
    PREFILL, and with top-8-of-64 routing a 64-token prefill touches 62-64
    of 64 experts in every layer — essentially the whole arena, every step.
    Under that workload no tier size avoids thrash and "cold" stops being a
    controlled fraction of routed work, which is the premise gate 1 rests
    on. Decode touches at most top_k experts per layer per token, so the
    routed working set is small enough for cold mass to mean something.
    """
    with torch.no_grad():
        for _ in range(warmup):
            out = model(ids, use_cache=True)
            cur = out.logits[:, -1:].argmax(-1)
            past = out.past_key_values
            for _ in range(4):
                out = model(cur, past_key_values=past, use_cache=True)
                cur = out.logits[:, -1:].argmax(-1)
                past = out.past_key_values
        torch.cuda.synchronize()

        out = model(ids, use_cache=True)
        past = out.past_key_values
        cur = out.logits[:, -1:].argmax(-1)
        torch.cuda.synchronize()

        # THE measurement boundary. Everything above is warmup prefills,
        # warmup decodes, and the prefill that builds the KV cache -- a
        # prefill touches 62-64 of 64 experts per layer, which is the exact
        # access shape this function exists to keep out of the measurement.
        # A tier-counter snapshot taken before run_steps() therefore charges
        # all of it into the "window", scoring reads and eviction
        # bookkeeping on prefill+decode while wall is scored on decode
        # alone. Callers that diff counters MUST snapshot here.
        # (Bugbot, gnf4#132.)
        if on_measure_start is not None:
            on_measure_start()

        per_step, toks = [], []
        for _ in range(n_steps):
            t0 = time.perf_counter_ns()
            out = model(cur, past_key_values=past, use_cache=True)
            torch.cuda.synchronize()
            per_step.append(time.perf_counter_ns() - t0)
            past = out.past_key_values
            cur = out.logits[:, -1:].argmax(-1)
            toks.append(int(cur.item()))
    return per_step, out.logits[:, -1, :].detach().float().cpu(), toks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/models/olmoe")
    ap.add_argument("--arena", default="/root/models/olmoe.arena")
    ap.add_argument("--profile", default="/root/olmoe_profile.jsonl")
    ap.add_argument("--calib", required=True)
    ap.add_argument("--sweep", default="0.01,0.05,0.10,0.20")
    ap.add_argument("--order", default="tail")
    ap.add_argument("--source", default="dram",
                    choices=("dram", "vram", "both"),
                    help="which tier the forced-cold experts come OUT of. "
                         "This decides the control arm's ARITHMETIC: a DRAM "
                         "expert executes on the CPU, a VRAM expert on the "
                         "GPU. source='dram' therefore gives cold_dest='cpu' "
                         "a matched reference and 'gpu' a mismatched one; "
                         "source='vram' swaps that. Getting this wrong is "
                         "what produced the retracted e4b#171.")
    ap.add_argument("--threshold", type=float, default=4.0)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--warmup", type=int, default=6)
    ap.add_argument("--vram-frac", type=float, default=0.25,
                    help="fraction of experts the VRAM budget admits")
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--prefetch", action="store_true",
                    help="speculative prefetch (route L+1 from L, warm its "
                         "NVMe rows off the critical path). Gate 1 asks "
                         "whether cold latency can be HIDDEN, and hiding is "
                         "this mechanism -- a no-prefetch sweep measures how "
                         "exposed cold work is, which is the control for the "
                         "hypothesis rather than a test of it.")
    ap.add_argument("--hot-rows", type=int, default=384,
                    help="cold-tier slots. Decode touches <= top_k rows per "
                         "layer per step (8*16=128 here), so this holds the "
                         "working set with 3x headroom while staying far "
                         "below the 1024-row arena -- cold experts really "
                         "miss instead of the tier becoming a full mirror.")
    ap.add_argument("--tol", type=float, default=5e-2,
                    help="cross-placement logit tolerance (NOT bitwise: a "
                         "moved expert changes its rounding path)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    calib = json.load(open(a.calib))
    # Read the blob's OWN field names rather than assuming: the ceilings are
    # the whole point of calibrating, and silently defaulting one would put a
    # spec-sheet number into the placement solve.
    b_vram = max(d["b_vram_triad_gbs"]
                 for d in calib["gpu_bench"]["devices"])
    b_dram = calib["cpu_bench"]["triad_best"]["gbs"]
    # Derived from the SEQUENTIAL points, not read from a key. `seq_best_gbs`
    # is not a field any calibration this repo has ever produced -- the blobs
    # carry `nvme.points[{mode,qd,gbs}]` -- so this silently resolved to None
    # and every gate-1 receipt recorded `b_nvme_gbs: null` while the printed
    # line said `B_nvme=None`. The attribution constant then had to be picked
    # by hand from the blob, and was picked wrong (rand qd16 6.26 GB/s in
    # place of the sequential 5.51). Compute it, and refuse rather than
    # default: a ceiling that was never measured cannot be charged against.
    _nv = (calib["cpu_bench"].get("nvme") or {})
    _seq = [q["gbs"] for q in (_nv.get("points") or [])
            if str(q.get("mode", "")).startswith("seq") and q.get("ok")
            and isinstance(q.get("gbs"), (int, float))]
    if not _seq:
        raise SystemExit(
            "calibration has no usable SEQUENTIAL NVMe point; disk time is "
            "charged against the sequential ceiling and there is nothing to "
            "charge it against. Re-run bench/calibrate.py with --nvme-dir on "
            f"the drive under test. nvme block: {_nv!r}")
    b_nvme = max(_seq)
    print("calibration: B_vram=%.1f B_dram=%.1f B_nvme=%s (G0 %s)" % (
        b_vram, b_dram, b_nvme, calib["gate_g0"]["verdict"]))

    from nvme_arena import load_index
    idx = load_index(a.arena)
    n_layers = idx["n_layers"]
    n_experts = idx["n_experts_per_layer"]
    row_bytes = idx["row_bytes"]
    print("arena: L=%d E=%d row=%d B" % (n_layers, n_experts, row_bytes))

    mass, prof_sha = load_routing_mass(a.profile, n_layers, n_experts)
    total_mass = sum(mass.values())

    vram_slots = int(a.vram_frac * n_layers * n_experts)
    base = solve_placement(
        n_layers=n_layers, n_experts=n_experts, bytes_per_expert=row_bytes,
        vram_budget_bytes=vram_slots * row_bytes,
        dram_budget_bytes=(n_layers * n_experts) * row_bytes,
        calibration=a.calib, profile_path=a.profile, top_k=8, batch=1)
    print("control placement: vram=%d dram=%d nvme=%d (nvme_frac=%.4f)" % (
        len(base["tiers"]["vram"]), len(base["tiers"]["dram"]),
        len(base["tiers"]["nvme"]), base["masses"]["nvme_frac"]))
    assert not base["tiers"]["nvme"], "control must have NO cold experts"

    from transformers import AutoTokenizer
    PROSE = ("The question of how memory works has occupied philosophers and "
             "scientists for centuries. When we recall an event, we do not "
             "replay a recording; we reconstruct it, and the reconstruction "
             "is shaped by everything we have learned since. This is why "
             "eyewitness testimony is less reliable than juries assume. ")
    _tk = AutoTokenizer.from_pretrained(a.model)
    tok_ids = _tk(PROSE * 4, return_tensors="pt").input_ids[:, :a.seq].to("cuda")
    print("prompt: real prose, %d tokens" % tok_ids.shape[1])

    receipt = {"schema": "e4b-tribrid-gate1/1", "prefetch": a.prefetch,
               "prereg_sha256": "7bf5b2be87aef56dc514f67cf90ea219ba1289004a"
                                "af0901e5c3230503a52ef5",
               "box": {"gpu": torch.cuda.get_device_name(0),
                       "b_vram_gbs": b_vram, "b_dram_gbs": b_dram,
                       "b_nvme_gbs": b_nvme,
                       "g0_pct_of_triad": calib["gate_g0"]["scatter_pct_of_triad"]},
               "config": vars(a), "points": []}

    ref_logits = None
    for frac in [float(x) for x in a.sweep.split(",")]:
        point = {"cold_frac_target": frac, "arms": {}}
        forced = force_cold_mass(base, mass, frac, order=a.order,
                                 source=a.source)
        point["cold_frac_achieved"] = forced["forced_cold"]["achieved_frac"]
        point["experts_moved"] = forced["forced_cold"]["experts_moved"]
        print("\n=== cold %.0f%% (achieved %.4f, %d experts) ===" % (
            frac * 100, point["cold_frac_achieved"], point["experts_moved"]))

        # control runs TWICE: the second is the self-pair. `exposed` is a
        # difference of medians, so an arm whose exposure is inside the
        # instrument's own disagreement is noise, not a measurement.
        for arm in ("control", "control#2") + ARMS[1:]:
            man = base if arm.startswith("control") else forced
            model, _ = load_moe_4bit_streaming(
                a.model, device="cuda", dtype=torch.bfloat16, r=8, alpha=16,
                quant_type="nf4", arena=a.arena)
            n = hy.enable_hybrid_tier(
                model, a.arena, man, hot_rows=a.hot_rows,
                cold_dest=arm_dest(arm.replace("#2", ""), a.threshold),
                prefetch=a.prefetch, verbose=False)
            if n == 0:
                print("  %-9s NOT-ENGAGED (0 modules patched)" % arm)
                point["arms"][arm] = {"engaged": 0}
                del model
                torch.cuda.empty_cache()
                continue
            # Snapshot at LOAD as well as at the measurement boundary, so a
            # single run reports both windows and the size of the difference
            # is measured on one trace rather than inferred across runs. The
            # load snapshot is what the pre-#132 harness was unknowingly
            # differencing against.
            at_load = hy.cold_stats(model)
            pre = {}
            steps, logits, toks = run_steps(
                model, tok_ids, a.steps, a.warmup,
                # bind by value: `model` is deleted at the end of each arm
                on_measure_start=lambda m=model: pre.update(hy.cold_stats(m)))
            assert pre, "on_measure_start never fired — window is unmeasured"
            cs = hy.cold_stats(model)
            # Reads charged to the MEASURED window, not the whole process:
            # warmup cold-start traffic is not what gate 1 asks about, and
            # counting it would make every arm look disk-bound.
            #
            # This comment described the INTENT and the code did not
            # implement it: `pre` was snapshotted before run_steps(), which
            # runs the warmup prefills INSIDE itself, so every warmup
            # prefill's near-full-arena sweep landed in the "window"
            # (Bugbot, gnf4#132). Now snapshotted at the measurement
            # boundary via on_measure_start. THE PUBLISHED GATE-1 READ
            # COUNTS IN RESULTS-tribrid-gate1.md PREDATE THIS FIX and would
            # need a re-run to correct; gate 1's MISS verdict rests on
            # prefetch coverage rather than on those counts.
            cs["reads_in_window"] = (cs.get("disk_reads", 0)
                                     - pre.get("disk_reads", 0))
            cs["cold_rows_in_window"] = (
                (cs.get("cold_rows", 0)) - (pre.get("cold_rows", 0)))
            # The warmup-inclusive figure, named for what it is. Reported so
            # the published numbers can be located against this run instead
            # of merely declared wrong: reads_since_load is the quantity the
            # old code called "in window".
            cs["reads_since_load"] = (cs.get("disk_reads", 0)
                                      - at_load.get("disk_reads", 0))
            cs["warmup_reads"] = (cs["reads_since_load"]
                                  - cs["reads_in_window"])
            point["arms"][arm] = {
                "engaged": n, "median_ns": statistics.median(steps),
                "steps": steps, "cold_stats": cs,
            }
            if arm == "control":
                ref_logits, ref_toks = logits, toks
            else:
                # Cross-placement tolerance, NOT bitwise: the CPU tier
                # dequantizes in fp32 while the GPU rounds through the
                # module's compute dtype, so a moved expert legitimately
                # differs. Report the number, don't assert equality.
                d = (logits - ref_logits).abs().max().item()
                point["arms"][arm]["max_abs_logit_diff"] = d
                point["arms"][arm]["equivalent"] = bool(d <= a.tol)
                # The metric that actually matters for a served model: did
                # the greedy token sequence change? Registered tolerance
                # stays as filed; this is reported ALONGSIDE it, not
                # instead of it.
                point["arms"][arm]["tokens_match"] = (toks == ref_toks)
                point["arms"][arm]["first_divergence"] = next(
                    (i for i, (x, y) in enumerate(zip(toks, ref_toks))
                     if x != y), None)
            print("  %-10s patched=%2d median=%8.2f ms  cold cpu/gpu=%6d/%-6d "
                  "win_reads=%5s dmax=%s" % (
                      arm, n, statistics.median(steps) / 1e6,
                      cs.get("cold_rows_cpu", 0), cs.get("cold_rows_gpu", 0),
                      cs.get("reads_in_window"),
                      ("%.4f" % point["arms"][arm]["max_abs_logit_diff"])
                      if "max_abs_logit_diff" in point["arms"][arm] else "ref"))
            hy.disable_hybrid_tier(model)
            del model
            torch.cuda.empty_cache()
        receipt["points"].append(point)
        json.dump(receipt, open(a.out, "w"), indent=2, default=str)

    json.dump(receipt, open(a.out, "w"), indent=2, default=str)
    print("\nreceipt ->", a.out)


if __name__ == "__main__":
    main()
