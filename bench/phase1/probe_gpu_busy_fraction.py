#!/usr/bin/env python3
"""Settle it: is the training step actually host-bound, and by how much?

Leg 3 found per-call spans 3.76x longer on the H100 than leg 2 measured for the
same arm (0.44 -> 1.68 ms), and 0.60x on the 4090. I attributed that to the step
being CPU-bound. **That attribution was stated more confidently than the
evidence supports** -- the story predicts leg 2's sync-per-call timing should
have captured the starvation gap too, since events record on the stream and a
GPU waiting on the host sits inside the span either way. It read 0.44 ms. So
either the model of those two timing paths is wrong or something else differs,
and the honest move is to measure the quantity directly instead of inferring it
from a discrepancy.

WHAT THIS MEASURES, three numbers per arm per cell, from one process:

  wall_per_step   N steps back to back, NO syncs between them, wall/N. This is
                  the per-step cost a training loop actually pays, because that
                  is how a training loop runs.
  gpu_busy_per_step  summed CUDA kernel self-time from the profiler, over M
                  steps, divided by M. This is how long the GPU is doing work.
  busy_fraction   the second over the first. If it is near 1 the GPU is the
                  bottleneck; if it is small the host is, and kernel choice
                  cannot move the wall clock much at this size.

It also re-measures the SAME cell with leg 2's instrument (sync per iteration)
and leg 3's (interleaved, no sync), so all four numbers sit in one receipt on
one box at one moment and the 3.76x either explains itself or does not.

PREDICTION, written before the run so it can fail: if the host-bound story is
right, gpu_busy_per_step should land near leg 2's figure, wall_per_step near
leg 3's, and busy_fraction well below 1 on the H100 and near 1 on the 4090.

Report-only. No prereg grades it, no arm changes.
"""
from __future__ import annotations

import argparse
import importlib.util as _iu
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

import torch

_ROOT = Path(os.environ.get("DQF_REPO", Path.cwd())).resolve()
if not (_ROOT / "bench" / "phase1" / "harness.py").exists():
    raise SystemExit(f"DQF_REPO/cwd={_ROOT} is not the repo root. Set DQF_REPO.")
sys.path.insert(0, str(_ROOT / "bench" / "phase1"))
sys.path.insert(0, str(_ROOT / "kernel"))
import harness as H  # noqa: E402


def _load(name, path):
    s = _iu.spec_from_file_location(name, path)
    m = _iu.module_from_spec(s)
    s.loader.exec_module(m)
    return m


dqf1 = _load("dqf1", _ROOT / "bench" / "phase1" / "train_dequant_forward.py")
ff = _load("ff", _ROOT / "bench" / "phase1" / "train_dequant_forward_floorfree.py")
il = _load("il", _ROOT / "bench" / "phase1" / "interleave.py")
RANK = 16


def wall_per_step(step, n):
    """The training-loop number: n steps back to back, one sync at the end."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        step()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / n


def gpu_busy_per_step(step, m):
    """Summed CUDA kernel self-time / m, from the profiler."""
    from torch.profiler import ProfilerActivity, profile
    for _ in range(3):
        step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as pr:
        for _ in range(m):
            step()
        torch.cuda.synchronize()
    total_us = 0.0
    for e in pr.key_averages():
        v = getattr(e, "self_device_time_total", None)
        if v is None:
            v = getattr(e, "self_cuda_time_total", 0.0)
        total_us += float(v or 0.0)
    return total_us / 1000.0 / m


def cell(spec, regime, device, args, stack):
    groups = H.make_activations(spec, regime, device)
    sizes = [a.shape[0] for _, a in groups]
    eids = [int(e) for e, _ in groups]
    a_cat = torch.cat([a for _, a in groups]).detach().requires_grad_(True)
    B_pack, A_scale = stack.fusedpack()
    row = {"model": spec.model, "proj": spec.proj, "regime": regime,
           "groups": len(groups), "rows": a_cat.shape[0], "arms": {}}

    gb = ff.g_base_arm(a_cat, B_pack, A_scale, sizes, eids)
    db = ff.d_base_arm(stack, a_cat, sizes, eids)

    def mk(f):
        def run():
            a_cat.grad = None
            f().float().pow(2).mean().backward()
        return run

    steps = {"G_base": mk(gb), "D_base": mk(db)}
    try:
        for name, step in steps.items():
            dqf1._warm(step, args.warm_s)
            w = st.median([wall_per_step(step, args.steps) for _ in range(3)])
            busy = gpu_busy_per_step(step, args.prof_steps)
            # the two prior instruments, same cell, same moment
            leg2 = dqf1._timed(step, min(200, args.steps))
            ta, tb, _o = il.interleaved_pairs(step, step, 60, torch_mod=torch)
            leg3 = st.median(ta)
            row["arms"][name] = {
                "wall_per_step_ms": w,
                "gpu_busy_per_step_ms": busy,
                "busy_fraction": (busy / w) if w else None,
                "leg2_sync_per_iter_ms": leg2,
                "leg3_interleaved_percall_ms": leg3,
                "leg3_over_leg2": (leg3 / leg2) if leg2 else None,
                "wall_over_busy": (w / busy) if busy else None}
        row["status"] = "ok"
    except Exception as e:  # pragma: no cover
        row.update({"status": "skipped",
                    "reason": f"{type(e).__name__}: {str(e)[:200]}"})
    torch.cuda.empty_cache()
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["OLMoE", "gpt-oss"])
    ap.add_argument("--regimes", nargs="*", default=["decode_m8", "tokbudget_2048"])
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--prof-steps", type=int, default=50)
    ap.add_argument("--warm-s", type=float, default=1.5)
    ap.add_argument("--out", default=os.environ.get("DQF_OUT", "/root/dqf-out"))
    ap.add_argument("--tag", default="busy1")
    args = ap.parse_args()

    out = {"probe": "GPU busy fraction of the training step",
           "tier": "EXPLORATORY / report-only",
           "prediction": "if the host-bound reading is right: gpu_busy ~ leg2's "
                         "number, wall ~ leg3's, busy_fraction << 1 on H100 and "
                         "~1 on the 4090",
           "gpu": torch.cuda.get_device_name(0),
           "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
           "torch": torch.__version__, "steps": args.steps,
           "prof_steps": args.prof_steps, "rows": []}
    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    art = dest / f"gpu_busy_{args.tag}.json"

    for spec in H.census_specs(H.REPO / "census" / "shape_census.json", args.models):
        stack = H.QuantStack(spec, "cuda")
        for regime in args.regimes:
            r = cell(spec, regime, "cuda", args, stack)
            out["rows"].append(r)
            art.write_text(json.dumps(out, indent=1, default=str))
            if r.get("status") == "ok":
                for n, a in r["arms"].items():
                    print("%-16s %-8s %-14s %-7s wall %7.3f  gpu %7.3f  busy %5.1f%%"
                          "  | leg2 %7.3f leg3 %7.3f (%.2fx)" % (
                              r["model"].split("/")[-1][:16], r["proj"], regime, n,
                              a["wall_per_step_ms"], a["gpu_busy_per_step_ms"],
                              100 * a["busy_fraction"], a["leg2_sync_per_iter_ms"],
                              a["leg3_interleaved_percall_ms"], a["leg3_over_leg2"]),
                          flush=True)
            else:
                print("  skipped", r.get("reason"), flush=True)
        del stack
        torch.cuda.empty_cache()

    ok = [r for r in out["rows"] if r.get("status") == "ok"]
    if ok:
        bf = [a["busy_fraction"] for r in ok for a in r["arms"].values()]
        print("\nmedian GPU busy fraction: %.1f%%  (min %.1f%%, max %.1f%%)" % (
            100 * st.median(bf), 100 * min(bf), 100 * max(bf)))
    print("GPU_BUSY_DONE")


if __name__ == "__main__":
    main()
