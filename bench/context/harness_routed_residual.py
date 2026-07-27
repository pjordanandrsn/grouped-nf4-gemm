"""Harness for PREREG-routed-residual (`bench/context/PREREG-routed-residual.md`,
OTS-stamped `ba7a29ce…`). Decomposes the 5.2 ms/layer non-byte residual finding
#22 left, and prices the sync fix against it.

RUN
    E4B_OFFLOAD_STATS=1 python harness_routed_residual.py --out receipt.json

WHY THE ARMS ARE ENV SWITCHES AND NOT TWO CHECKOUTS
    The prereg registers the arms as commits (`e62a7a0` control, `94e4004`
    treatment) and *also* requires them interleaved within one process. Those
    two clauses cannot both hold across a git boundary, and the second one is
    the load-bearing half: cross-provision anchor drift on the same serial has
    measured ~7%, which is larger than the effect R6 registers. So the control
    is reproduced in-process by `E4B_ABLATE_ROUTED_IDS=device` +
    `E4B_ABLATE_ROW_PLAN=dict`, which restore the pre-`94e4004` code paths
    exactly, and the harness flips the module globals between reps.

    That substitution is a real assumption and it gets checked, not asserted:
    `--verify-switch` prints the command to run the true `e62a7a0` build in a
    separate process, whose s/token must land inside the self-pair spread of the
    switched control. If it does not, the switch is not a faithful control and
    every number here is void. Run it BEFORE trusting the ladder.

REPS = 6, REGISTERED (not a deviation)
    Amendment 1 (`PREREG-routed-residual-amendment-1.md`, OTS-stamped
    `2397612e…`, written PRE-DATA) moves the fixture from the original's "median
    of 2" to **median of 6**. The original prereg is NOT edited — its stamp
    `ba7a29ce…` still binds its original bytes, so the two can be diffed by
    anyone.

    Amendment 1 also registers, ahead of data: ABBA ordering (A2); the rule that
    R6 yields NO verdict when the self-pair spread meets or exceeds its band
    width (A3); and the status of `logit_identity` / `arm_fidelity` /
    `position_balance` as gates rather than predictions (A4).

    `reps` must be EVEN, or `interleave` cannot balance positions.

REGISTERED vs EXPLORATORY
    Registered (R1–R6): arms `C` and `T1` only. `T1s` and `T1c` split T1 into
    its sync half and its copy-plan half — free once the switches exist, but
    **not registered**, so they are reported under `exploratory` and may not be
    quoted as confirmatory results. Promoting them needs its own prereg.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import torch

from experts4bit_qlora import (
    enable_routed_staging,
    load_moe_4bit_streaming,
    offload_stats_report,
    report_offload_environment,
    reset_offload_stats,
    routed_ids_counts,
    row_plan_counts,
)
from experts4bit_qlora import offload as offload_mod

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e4b_ladder import PROMPT, _collect_handles, decode  # noqa: E402
from routed_residual_verdicts import REGISTERED, evaluate, interleave  # noqa: E402,F401

#: (routed_ids, row_plan) per arm. `device`/`dict` are the pre-94e4004 paths.
ARMS = {
    "C":   ("device", "dict"),   # control == e62a7a0 behaviour
    "T1":  ("auto",   "flat"),   # treatment == 94e4004
    "T1s": ("auto",   "dict"),   # EXPLORATORY: sync half only
    "T1c": ("device", "flat"),   # EXPLORATORY: copy-plan half only
}


def _set_arm(arm: str) -> None:
    """Flip the ablation switches. They are module globals read once at import,
    so assigning them here is the whole mechanism — no re-import, one process."""
    ids_mode, plan_mode = ARMS[arm]
    offload_mod._ABLATE_ROUTED_IDS = ids_mode
    offload_mod._ABLATE_ROW_PLAN = plan_mode


def _assert_confounds_off(handles) -> None:
    """The prereg requires speculative staging and the expert cache OFF in every arm.

    Both landed on `private` after #22 and both change which bytes cross the link,
    so either one left on silently invalidates the comparison against 0.936 s.
    This fails the run rather than footnoting it — a confound discovered in the
    write-up is a wasted rental.
    """
    for i, h in enumerate(handles):
        if getattr(h, "_pool", None) is not None:
            raise SystemExit(f"ABORT: expert cache is enabled on handle {i}; prereg requires it off")
        if getattr(h, "_spec_dev", None) is not None or getattr(h, "_prefetch_next", None) is not None:
            raise SystemExit(f"ABORT: speculative/prefetch staging active on handle {i}; prereg requires it off")


@torch.no_grad()
def run(args) -> dict:
    # A STRING, not torch.device: the loader hands this straight to
    # safetensors.safe_open, which rejects a torch.device
    # ("device device(type='cuda', index=0) is invalid"). e4b_ladder uses "cuda".
    dev = "cuda"
    # offload=True is not optional decoration: without it no _offload handles are
    # built, _collect_handles returns [], enable_routed_staging is a no-op, and the
    # run silently measures a resident model. r/alpha are the ladder's values --
    # LoRA is untouched by this prereg but the loader requires them.
    model, _cfg = load_moe_4bit_streaming(args.model, dev, torch.bfloat16, r=8, alpha=16,
                                          offload=True, pin=True)
    model.eval()
    handles = _collect_handles(model)
    if not handles:
        raise SystemExit("ABORT: no offload handles found -- offload did not engage, nothing to measure")
    enable_routed_staging(handles)
    _assert_confounds_off(handles)

    env = report_offload_environment(dev, log=print) or {}
    ceiling = env.get("ceiling_pinned_gbps") or env.get("ceiling_pageable_gbps") or 0.0

    from transformers import AutoTokenizer
    # args.model, not model.config._name_or_path: after a meta-init streaming load the
    # config's path field is not reliably the hub id.
    ids = AutoTokenizer.from_pretrained(args.model)(PROMPT, return_tensors="pt").input_ids.to(dev)

    # Position-balanced (ABBA), not plain repetition: plain repetition pins each
    # arm to one position and turns run drift into a treatment effect. See
    # interleave()'s docstring for the smoke that demonstrated it.
    order = interleave(args.arms, args.reps)
    if args.reps % 2:
        print(f"# WARNING: reps={args.reps} is odd — positions cannot balance; prefer an even reps")
    records = []
    for rep_i, arm in enumerate(order):
        _set_arm(arm)
        reset_offload_stats()                                     # also zeroes the branch counters
        s_per_token, first_logits, greedy = _timed_decode(model, ids, args.tokens, args.warmup)
        rep = offload_stats_report() or {}
        routed = (rep.get("by_policy") or {}).get("routed") or {}
        records.append({
            "arm": arm, "rep": rep_i // len(args.arms), "position": rep_i,
            "s_per_token": s_per_token,
            "greedy_ids": greedy,
            "max_abs_logit": float(first_logits.abs().max()),
            "_logits": first_logits,
            "counts": routed_ids_counts(),
            "row_plan": row_plan_counts(),
            "routed_gbps": routed.get("gbps"),
            "routed_copies": routed.get("copies"),
        })
        print(f"  {arm:<4} rep{rep_i // len(args.arms)}  {s_per_token:.4f} s/tok  "
              f"routed {routed.get('gbps', 0):.2f} GB/s  counts {records[-1]['counts']}")

    # Finding #24 established that gates run on LOGITS on a natural prompt, because
    # greedy ids can agree while logits differ. The stamped prereg's R1 says greedy
    # ids, and the stamp cannot be edited -- so satisfy R1 as written and report the
    # stronger check next to it rather than silently substituting one for the other.
    base = next(r["_logits"] for r in records if r["arm"] == "C")
    for r in records:
        r["max_delta_logit"] = float((r.pop("_logits") - base).abs().max())
    worst = max(r["max_delta_logit"] for r in records)
    verdicts = evaluate(records, ceiling)
    verdicts["logit_identity"] = {
        "pass": worst == 0.0, "max_delta_logit": worst,
        "detail": "exact logit agreement across all arms (stronger than R1's greedy-id gate; "
                  "#24's standard). NOT a registered prediction -- R1 as stamped is the "
                  "registered gate and is evaluated separately.",
    }
    if worst != 0.0:
        verdicts["gates_passed"] = False
    return {"ceiling_gbps": ceiling, "records": records, "verdicts": verdicts}


@torch.no_grad()
def _timed_decode(model, ids, n_tokens, warmup):
    """Timing, greedy ids, and first-token logits from ONE pass.

    An earlier draft generated a second time just to collect ids, doubling the
    decode work per arm. It also risked the two passes disagreeing, which would
    have read as a correctness failure.

    Both gates come out of here: `greedy` for the stamped R1, and `logits` for the
    stronger logit check the harness adds on top -- see `logit_identity` in main().
    """
    out = model(ids, use_cache=True)
    past = out.past_key_values
    first_logits = out.logits[:, -1, :].float().clone()
    nxt = out.logits[:, -1:, :].argmax(-1)
    greedy, times = [int(nxt)], []
    for step in range(n_tokens + warmup):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        o = model(nxt, past_key_values=past, use_cache=True)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        past = o.past_key_values
        nxt = o.logits[:, -1:, :].argmax(-1)
        greedy.append(int(nxt))
        if step >= warmup:
            times.append(dt)
    return statistics.median(times), first_logits, greedy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-235B-A22B")
    ap.add_argument("--tokens", type=int, default=12)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--reps", type=int, default=6,
                    help="median of N per arm. Registered by amendment 1 (2397612e). "
                         "Must be even (ABBA balance).")
    ap.add_argument("--arms", default="C,T1,T1s,T1c",
                    help="comma list; C and T1 are registered, the rest exploratory")
    ap.add_argument("--out", default="")
    ap.add_argument("--verify-switch", action="store_true",
                    help="print the cross-check that the switched control matches the real e62a7a0")
    args = ap.parse_args()
    args.arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    if args.verify_switch:
        print(
            "Switch-fidelity cross-check — run BEFORE trusting the ladder:\n"
            "  git -C ~/code/experts4bit-qlora worktree add /tmp/ctl e62a7a0\n"
            "  (cd /tmp/ctl && E4B_OFFLOAD_STATS=1 python .../harness_routed_residual.py --arms C --reps 2)\n"
            "The true-e62a7a0 s/token must land INSIDE the switched control's self-pair spread.\n"
            "If it does not, the switch is not a faithful control and the ladder is void."
        )
        return

    if os.environ.get("E4B_OFFLOAD_STATS") != "1":
        raise SystemExit("ABORT: set E4B_OFFLOAD_STATS=1 — R4 is the point of this run and needs it")

    t0 = time.time()
    result = run(args)
    result["wall_s"] = time.time() - t0
    result["prereg"] = ("bench/context/PREREG-routed-residual.md ba7a29ce "
                        "+ amendment-1 2397612e")

    v = result["verdicts"]
    print("\n=== GATES ===")
    for k in ("R1_bit_identity", "R2_engagement"):
        print(f"  {k}: {'PASS' if v['registered'][k]['pass'] else 'FAIL'} — {v['registered'][k].get('detail','')}")
    if not v["gates_passed"]:
        print("\nGATES FAILED — timings are not interpretable. Do not report them.")
    print("\n=== REGISTERED ===")
    print(json.dumps(v["registered"], indent=2, default=str))
    if v["exploratory"]:
        print("\n=== EXPLORATORY (not registered, not quotable) ===")
        print(json.dumps(v["exploratory"], indent=2, default=str))

    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nreceipt -> {args.out}")


if __name__ == "__main__":
    main()
