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
from routed_residual_verdicts import REGISTERED, evaluate  # noqa: E402,F401

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
    model, _cfg = load_moe_4bit_streaming(args.model)
    handles = _collect_handles(model)
    enable_routed_staging(handles)
    _assert_confounds_off(handles)

    env = report_offload_environment(torch.device("cuda:0"), log=print) or {}
    ceiling = env.get("ceiling_pinned_gbps") or env.get("ceiling_pageable_gbps") or 0.0

    tok = model.config._name_or_path if hasattr(model.config, "_name_or_path") else args.model
    from transformers import AutoTokenizer
    ids = AutoTokenizer.from_pretrained(tok).encode(PROMPT, return_tensors="pt").cuda()

    order = [a for _ in range(args.reps) for a in args.arms]      # interleaved, NOT blocked
    records = []
    for rep_i, arm in enumerate(order):
        _set_arm(arm)
        reset_offload_stats()                                     # also zeroes the branch counters
        s_per_token, _logits = decode(model, ids, args.tokens, args.warmup)
        rep = offload_stats_report() or {}
        routed = (rep.get("by_policy") or {}).get("routed") or {}
        records.append({
            "arm": arm, "rep": rep_i // len(args.arms),
            "s_per_token": s_per_token,
            "greedy_ids": _greedy_ids(model, ids, args.tokens),
            "counts": routed_ids_counts(),
            "row_plan": row_plan_counts(),
            "routed_gbps": routed.get("gbps"),
            "routed_copies": routed.get("copies"),
        })
        print(f"  {arm:<4} rep{rep_i // len(args.arms)}  {s_per_token:.4f} s/tok  "
              f"routed {routed.get('gbps', 0):.2f} GB/s  counts {records[-1]['counts']}")

    return {"ceiling_gbps": ceiling, "records": records,
            "verdicts": evaluate(records, ceiling)}


@torch.no_grad()
def _greedy_ids(model, ids, n_tokens) -> list:
    """R1's gate: the actual token ids, generated separately from the timed loop
    so timing jitter can never be mistaken for a correctness difference."""
    out = model(ids, use_cache=True)
    past, nxt = out.past_key_values, out.logits[:, -1:, :].argmax(-1)
    got = [int(nxt)]
    for _ in range(n_tokens - 1):
        o = model(nxt, past_key_values=past, use_cache=True)
        past, nxt = o.past_key_values, o.logits[:, -1:, :].argmax(-1)
        got.append(int(nxt))
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-235B-A22B")
    ap.add_argument("--tokens", type=int, default=12)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--reps", type=int, default=2, help="median of N per arm (prereg: 2)")
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
    result["prereg"] = "bench/context/PREREG-routed-residual.md ba7a29ce"

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
