#!/usr/bin/env python3
"""EXPLORATORY probe: what KIND of noise does this box have?

Three legs of this program have each guessed at a pairing granularity and paid
for the guess. Leg 1 lost a device to a cold GPU. Leg 2's amendment 2 raised
the iteration ceiling and the two devices went OPPOSITE ways -- the 4090's
self-pair improved 5.6x while the H100's got 2.3x worse. Leg 3 pairs at the
iteration instead. Each was a reasonable guess and none of them measured the
thing that decides the answer.

**And the amendment-2 comparison is confounded**: runs 1 and 2 were different
pods, so block length is entangled with instance luck. This probe removes that
confound by sweeping block length WITHIN ONE PROCESS ON ONE BOX.

WHAT IT MEASURES. For a fixed cell, time the same arm against itself at a
ladder of block lengths N and record |1 - self-pair|. The way that falls with N
identifies the noise:

  ~ 1/sqrt(N)   white, within-block  -> averaging works; use long blocks
  flat or rising  low-frequency drift -> averaging cannot help; pair finer

Both are actionable and they point opposite ways, which is exactly why guessing
has cost three legs. The fitted slope on a log-log plot is the number: -0.5 is
white, 0 is drift-dominated.

It also runs the INTERLEAVED self-pair at the same cell, so the two instruments
are compared on one box at one moment rather than across pods.

Report-only. Touches no registered leg, changes no arm, and produces no number
that any prereg grades.
"""
from __future__ import annotations

import argparse
import importlib.util as _iu
import json
import math
import os
import statistics as st
import sys
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


def block_selfpair(step, iters):
    """|1 - self-pair| the way legs 1 and 2 measured it: two ADJACENT blocks of
    `iters`, median each, divided."""
    a = dqf1._timed(step, iters)
    b = dqf1._timed(step, iters)
    return b / a, a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["OLMoE", "gpt-oss"])
    ap.add_argument("--regimes", nargs="*", default=["decode_m8", "tokbudget_2048"])
    ap.add_argument("--ladder", type=int, nargs="*",
                    default=[25, 50, 100, 200, 400, 800, 1600])
    ap.add_argument("--reps", type=int, default=9,
                    help="repeats per rung. RAISED from 5 after a smoke: at 2-3 "
                         "reps the fitted slope flipped its own conclusion on "
                         "identical hardware (-0.524 'white' vs -0.091 'drift', "
                         "same box, same cell). The slope is the fragile "
                         "statistic here; monotone_frac and the interleaved "
                         "comparison are the robust ones.")
    ap.add_argument("--warm-s", type=float, default=1.5)
    ap.add_argument("--out", default=os.environ.get("DQF_OUT", "/root/dqf-out"))
    ap.add_argument("--tag", default="ns1")
    args = ap.parse_args()

    out = {"probe": "noise spectrum vs block length, WITHIN one process",
           "tier": "EXPLORATORY / report-only",
           "scope_note": "This characterises THIS BOX's measurement noise. It "
                         "is not a kernel-speed measurement and licenses no "
                         "statement about how fast any arm is. On the "
                         "correctness-only A2000 that distinction is what makes "
                         "running it there legitimate at all.",
           "gpu": torch.cuda.get_device_name(0),
           "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
           "torch": torch.__version__, "ladder": args.ladder,
           "reps": args.reps, "rows": []}
    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    art = dest / f"noise_spectrum_{args.tag}.json"

    for spec in H.census_specs(H.REPO / "census" / "shape_census.json", args.models):
        stack = H.QuantStack(spec, "cuda")
        for regime in args.regimes:
            groups = H.make_activations(spec, regime, "cuda")
            sizes = [a.shape[0] for _, a in groups]
            eids = [int(e) for e, _ in groups]
            a_cat = torch.cat([a for _, a in groups]).detach().requires_grad_(True)
            B_pack, A_scale = stack.fusedpack()
            gb = ff.g_base_arm(a_cat, B_pack, A_scale, sizes, eids)

            def make_step(gb=gb, act=a_cat):
                def step():
                    act.grad = None
                    gb().float().pow(2).mean().backward()
                return step
            step = make_step()

            row = {"model": spec.model, "proj": spec.proj, "regime": regime,
                   "rows": a_cat.shape[0], "groups": len(groups), "rungs": []}
            try:
                dqf1._warm(step, args.warm_s)
                for n in args.ladder:
                    devs, cell_ms = [], []
                    for _ in range(args.reps):
                        sp, ms = block_selfpair(step, n)
                        devs.append(abs(1.0 - sp))
                        cell_ms.append(ms)
                    row["rungs"].append({
                        "iters": n, "block_ms": st.median(cell_ms) * n,
                        "cell_ms": st.median(cell_ms),
                        "dev_median": st.median(devs), "dev_max": max(devs),
                        "devs": devs})
                # interleaved self-pair on the SAME cell, same moment
                pairs = max(30, min(600, int(args.ladder[-1])))
                ta, tb, orders = il.interleaved_pairs(step, step, pairs,
                                                      torch_mod=torch)
                s = il.pair_stats(ta, tb, orders)
                row["interleaved"] = {"pairs": pairs,
                                      "dev": abs(1.0 - s["ratio_median"]),
                                      "ratio_median": s["ratio_median"],
                                      "halves_ratio": s["halves_ratio"],
                                      "iqr": s["ratio_iqr"]}
                # log-log slope of dev vs iters: -0.5 white, ~0 drift-dominated
                xs = [math.log(r["iters"]) for r in row["rungs"]]
                ys = [math.log(max(r["dev_median"], 1e-9)) for r in row["rungs"]]
                mx, my = st.mean(xs), st.mean(ys)
                den = sum((x - mx) ** 2 for x in xs)
                row["loglog_slope"] = (sum((x - mx) * (y - my)
                                           for x, y in zip(xs, ys)) / den
                                       if den else None)
                # ROBUSTNESS, learned from a smoke where the slope flipped its
                # own verdict at low reps: report assumption-light companions.
                d = [r["dev_median"] for r in row["rungs"]]
                pairs_ = list(zip(d, d[1:]))
                row["monotone_frac"] = (sum(1 for a, b in pairs_ if b < a)
                                        / len(pairs_)) if pairs_ else None
                row["dev_first_over_last"] = (d[0] / d[-1]) if d and d[-1] else None
                # per-rep slopes, so the slope's OWN spread is visible rather
                # than a single point estimate being taken on trust
                slopes = []
                for k in range(min(len(r_["devs"]) for r_ in row["rungs"])):
                    yy = [math.log(max(r_["devs"][k], 1e-9)) for r_ in row["rungs"]]
                    myk = st.mean(yy)
                    slopes.append(sum((x - mx) * (y - myk)
                                      for x, y in zip(xs, yy)) / den)
                row["slope_per_rep"] = slopes
                row["slope_spread"] = (max(slopes) - min(slopes)) if slopes else None
                row["reading"] = ("white / within-block — average it"
                                  if row["loglog_slope"] is not None
                                  and row["loglog_slope"] < -0.35 else
                                  "drift-dominated — averaging will not help")
                row["reading_trustworthy"] = bool(
                    row["slope_spread"] is not None and row["slope_spread"] < 0.5)
                row["status"] = "ok"
            except Exception as e:  # pragma: no cover
                row.update({"status": "skipped",
                            "reason": f"{type(e).__name__}: {str(e)[:200]}"})
            out["rows"].append(row)
            print(json.dumps(row), flush=True)
            art.write_text(json.dumps(out, indent=1, default=str))
            del a_cat, groups
            torch.cuda.empty_cache()
        del stack
        torch.cuda.empty_cache()

    ok = [r for r in out["rows"] if r.get("status") == "ok"]
    print()
    for r in ok:
        print("%-16s %-8s %-14s slope %6.3f  %s" % (
            r["model"].split("/")[-1][:16], r["proj"], r["regime"],
            r["loglog_slope"], r["reading"]))
        print("    block dev: " + "  ".join(
            "N=%d:%.4f" % (x["iters"], x["dev_median"]) for x in r["rungs"]))
        print("    interleaved dev: %.4f (pairs %d)   best block rung: %.4f"
              % (r["interleaved"]["dev"], r["interleaved"]["pairs"],
                 min(x["dev_median"] for x in r["rungs"])))
        print("    monotone %.2f  slope spread %.3f  trustworthy=%s" % (
            r["monotone_frac"], r["slope_spread"], r["reading_trustworthy"]))
    print("NOISE_SPECTRUM_DONE")


if __name__ == "__main__":
    main()
