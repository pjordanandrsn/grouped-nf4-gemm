#!/usr/bin/env python3
"""LEG 4 — the same comparison on a routing-faithful fixture, reporting the
kernel ratio and the step ratio separately because they are not the same thing.

Two corrections at once, both forced by earlier legs:

1. THE FIXTURE. `decode_m*` builds exactly `top_k` groups; `tokbudget_*` builds
   all E with equal counts. Real routing at 32 tokens hits ~58 of 64 experts
   with 1-17 rows each, and at 2048 tokens hits all 64 but with counts running
   31-795 (cv 0.506). Both fictions move BOTH arms -- the baseline's dequant
   call count and the fused arm's group count move together -- so the net effect
   has to be measured. `routing_fixture.py` draws from this repo's own measured
   histograms and refuses to invent routing for the two census models that have
   none.

2. THE QUANTITY. Leg 3's follow-up measured that at decode band the GPU is busy
   only 4-33% of the step: those cells were comparing one kernel launch against
   a per-expert python loop, ~90% host. So every cell here reports BOTH
   `wall` (what a training loop pays) and `gpu` (summed CUDA kernel self-time,
   what the kernels do), plus the busy fraction that says which regime it is in.
   Reporting only one of those is how the earlier legs got mislabelled.

MATCHED PAIRS, MEASURED IN ONE PROCESS. Each token count is run on BOTH the
fictional fixture and the faithful one, on the same box back to back, at
identical total rows. Comparing against the earlier legs' numbers instead would
be a cross-run comparison -- the exact error that produced, and then retracted,
a 3.76x that did not exist.

  T=32   fiction `decode_m32`     (top_k groups x 32 rows)  vs routed_32
  T=2048 fiction `tokbudget_2048` (all E, uniform)          vs routed_2048
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
rf = _load("rf", _ROOT / "bench" / "phase1" / "routing_fixture.py")
RESULTS = _ROOT / "bench" / "phase1" / "results"


def wall_ms(step, n):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        step()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / n


def gpu_ms(step, m):
    from torch.profiler import ProfilerActivity, profile
    for _ in range(3):
        step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as pr:
        for _ in range(m):
            step()
        torch.cuda.synchronize()
    tot = 0.0
    for e in pr.key_averages():
        v = getattr(e, "self_device_time_total", None)
        if v is None:
            v = getattr(e, "self_cuda_time_total", 0.0)
        tot += float(v or 0.0)
    return tot / 1000.0 / m


def measure(stack, spec, groups, args):
    """wall and gpu per step for both arms on one fixture."""
    sizes = [a.shape[0] for _, a in groups]
    eids = [int(e) for e, _ in groups]
    a_cat = torch.cat([a for _, a in groups]).detach().requires_grad_(True)
    B_pack, A_scale = stack.fusedpack()

    def mk(f, act=a_cat):
        def run():
            act.grad = None
            f().float().pow(2).mean().backward()
        return run

    arms = {"G": mk(ff.g_base_arm(a_cat, B_pack, A_scale, sizes, eids)),
            "D": mk(ff.d_base_arm(stack, a_cat, sizes, eids))}
    out = {}
    for name, step in arms.items():
        dqf1._warm(step, args.warm_s)
        w = st.median([wall_ms(step, args.steps) for _ in range(3)])
        g = gpu_ms(step, args.prof_steps)
        out[name] = {"wall_ms": w, "gpu_ms": g,
                     "busy_fraction": (g / w) if w else None,
                     # wall minus summed kernel time = launch gaps + host
                     # stalls. Summed kernel time does NOT count the gaps
                     # between launches, so a single-launch design can show an
                     # equal `gpu_ms` and still be much faster in wall terms.
                     # That distinction is the whole point of splitting these.
                     "gap_ms": w - g}
    out["d_over_g_wall"] = out["D"]["wall_ms"] / out["G"]["wall_ms"]
    out["d_over_g_gpu"] = (out["D"]["gpu_ms"] / out["G"]["gpu_ms"]
                           if out["G"]["gpu_ms"] else None)
    out["d_over_g_gap"] = (out["D"]["gap_ms"] / out["G"]["gap_ms"]
                           if out["G"]["gap_ms"] else None)
    # REGISTERED LABEL (C4): either arm below the bar makes this fixture's
    # ratios step ratios, not kernel measurements. This leg already measured
    # busy_fraction; what was missing was applying the rule to it and printing
    # the verdict beside the numbers.
    out["measurement_class"], out["min_busy_fraction"], _ = H.measurement_class(
        {k: out[k] for k in ("G", "D")})
    out["groups"] = len(groups)
    out["total_rows"] = int(sum(sizes))
    del a_cat
    torch.cuda.empty_cache()
    return out


def cell(spec, tokens, device, args, stack):
    row = {"model": spec.model, "proj": spec.proj, "tokens": tokens,
           "E": spec.E, "top_k": spec.top_k}
    try:
        faithful = rf.routed_groups(spec, tokens, RESULTS, device,
                                    seed=args.seed)
    except LookupError as e:
        row.update({"status": "not_run", "reason": str(e)})
        return row
    row["routing"] = rf.summarise(faithful, spec.E)

    # the matched fiction at IDENTICAL total rows
    fiction_regime = (f"decode_m{tokens}" if tokens * spec.top_k
                      <= spec.E * spec.top_k and tokens < 512
                      else f"tokbudget_{tokens}")
    fiction = H.make_activations(spec, fiction_regime, device)
    row["fiction_regime"] = fiction_regime
    row["fiction_groups"] = len(fiction)
    row["fiction_rows"] = int(sum(a.shape[0] for _, a in fiction))

    try:
        row["fiction"] = measure(stack, spec, fiction, args)
        row["faithful"] = measure(stack, spec, faithful, args)
        for arm in ("G", "D"):
            row[f"fixture_effect_{arm}_wall"] = (
                row["faithful"][arm]["wall_ms"] / row["fiction"][arm]["wall_ms"])
            row[f"fixture_effect_{arm}_gpu"] = (
                row["faithful"][arm]["gpu_ms"] / row["fiction"][arm]["gpu_ms"]
                if row["fiction"][arm]["gpu_ms"] else None)
        row["ratio_shift_wall"] = (row["faithful"]["d_over_g_wall"]
                                   / row["fiction"]["d_over_g_wall"])
        row["ratio_shift_gpu"] = (row["faithful"]["d_over_g_gpu"]
                                  / row["fiction"]["d_over_g_gpu"]
                                  if row["fiction"]["d_over_g_gpu"] else None)
        row["status"] = "ok"
    except Exception as e:  # pragma: no cover
        row.update({"status": "skipped",
                    "reason": f"{type(e).__name__}: {str(e)[:200]}"})
    torch.cuda.empty_cache()
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--tokens", type=int, nargs="*", default=[32, 2048])
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--prof-steps", type=int, default=40)
    ap.add_argument("--warm-s", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.environ.get("DQF_OUT", "/root/dqf-out"))
    ap.add_argument("--tag", default="rt1")
    args = ap.parse_args()

    out = {"leg": "4 - routing-faithful fixture, kernel and step ratios split",
           "prereg": "kernel/prereg_dequant_forward_routed.json",
           "gpu": torch.cuda.get_device_name(0),
           "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
           "torch": torch.__version__, "tokens": args.tokens,
           "seed": args.seed, "rows": []}
    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    art = dest / f"routed_{args.tag}.json"

    for spec in H.census_specs(H.REPO / "census" / "shape_census.json", args.models):
        stack = H.QuantStack(spec, "cuda")
        for tokens in args.tokens:
            r = cell(spec, tokens, "cuda", args, stack)
            out["rows"].append(r)
            art.write_text(json.dumps(out, indent=1, default=str))
            if r.get("status") == "ok":
                rt = r["routing"]
                print("%-16s %-8s T=%-5d fiction %3d grp -> faithful %3d grp "
                      "(occ %.2f cv %.2f) | d/g wall %.3f->%.3f  gpu %.3f->%.3f "
                      "| busy G %.0f%% D %.0f%%" % (
                          r["model"].split("/")[-1][:16], r["proj"], tokens,
                          r["fiction_groups"], rt["hit_experts"],
                          rt["occupancy"], rt["cv"],
                          r["fiction"]["d_over_g_wall"],
                          r["faithful"]["d_over_g_wall"],
                          r["fiction"]["d_over_g_gpu"],
                          r["faithful"]["d_over_g_gpu"],
                          100 * r["faithful"]["G"]["busy_fraction"],
                          100 * r["faithful"]["D"]["busy_fraction"]), flush=True)
            else:
                print("%-16s %-8s T=%-5d %s: %s" % (
                    r["model"].split("/")[-1][:16], r["proj"], tokens,
                    r["status"].upper(), str(r.get("reason"))[:70]), flush=True)
        del stack
        torch.cuda.empty_cache()

    ok = [r for r in out["rows"] if r.get("status") == "ok"]
    nr = [r for r in out["rows"] if r.get("status") == "not_run"]
    if ok:
        print("\nmedian ratio shift (faithful/fiction): wall %.3f  gpu %.3f" % (
            st.median([r["ratio_shift_wall"] for r in ok]),
            st.median([r["ratio_shift_gpu"] for r in ok])))
    print("NOT-RUN (no measured routing): %d cells" % len(nr))
    print("ROUTED_DONE")


if __name__ == "__main__":
    main()
