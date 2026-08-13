#!/usr/bin/env python3
"""Does the baseline's CUDA-graph advantage survive DYNAMIC routing?

The adversarial race showed the baseline gains 8.6-12.5x from a CUDA graph
while the fused path cannot be captured at all -- which removed this leg's
small-batch speed headline. But that race used a STATIC fixture, and CUDA
graphs require static shapes. **MoE routing does not provide them**: per-expert
group sizes change every step, so a graph captured at one routing draw is
invalid at the next.

A real trainer can only use graphs by padding every expert to a fixed capacity
C, re-capturing per step, or bucketing (which is padding with extra steps).
Each has a price, and this probe measures it:

  CAPACITY TAX   C is set so no token is dropped across `--draws` independent
                 routing draws; the padded problem is then E groups x C rows,
                 against faithful routing's much smaller and skewed total.
                 rows_padded / rows_faithful is the compute a graphed arm pays
                 on every step for the privilege of having static shapes.
  CAPTURE COST   wall time to capture once, which is what a re-capture-per-step
                 strategy would pay.
  THE REAL RACE  D_base graphed AT PADDED SHAPES vs G_base at FAITHFUL shapes.
                 That is what a user actually gets on each side.

Token-dropping (choosing a smaller C and discarding overflow) is the other
standard option and is NOT measured here, because it changes what the model
computes and so is not a like-for-like comparison.
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
gr = _load("gr", _ROOT / "bench" / "phase1" / "probe_graphed_race.py")
RESULTS = _ROOT / "bench" / "phase1" / "results"


def capacity_over_draws(spec, tokens, draws):
    """C that drops no token across `draws` independent routing draws, and the
    per-draw faithful row totals for contrast."""
    per_layer, E, k = rf.routing_for(spec.model, RESULTS)
    counts = per_layer[len(per_layer) // 2]
    worst, totals, hits = 0, [], []
    for d in range(draws):
        s = rf.sample_group_sizes(counts, tokens, k, seed=1000 + d)
        worst = max(worst, max(s.values()))
        totals.append(sum(s.values()))
        hits.append(len(s))
    return worst, st.median(totals), st.median(hits), E


def padded_groups(spec, C, device, seed=7):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return [(e, (torch.randn(C, spec.K, generator=g, dtype=torch.float32) * 0.5)
             .to(device=device, dtype=torch.bfloat16)) for e in range(spec.E)]


def cell(spec, stack, tokens, device, args):
    row = {"model": spec.model, "proj": spec.proj, "tokens": tokens, "E": spec.E}
    if rf.routing_for(spec.model, RESULTS) is None:
        row.update({"status": "not_run", "reason": "no measured routing"})
        return row
    try:
        C, rows_faithful, hits, E = capacity_over_draws(spec, tokens, args.draws)
        rows_padded = C * E
        row.update({"capacity_C": C, "rows_faithful": rows_faithful,
                    "rows_padded": rows_padded, "median_hit_experts": hits,
                    "capacity_tax": rows_padded / rows_faithful,
                    "draws": args.draws})

        faithful = rf.routed_groups(spec, tokens, RESULTS, device, seed=args.seed)
        padded = padded_groups(spec, C, device)

        sf = gr.build(stack, faithful, device)
        dqf1._warm(sf["G"], args.warm_s)
        g_faith = st.median([gr.wall_ms(sf["G"], args.steps) for _ in range(3)])
        d_faith = st.median([gr.wall_ms(sf["D"], args.steps) for _ in range(3)])
        del sf
        torch.cuda.empty_cache()

        sp = gr.build(stack, padded, device)
        dqf1._warm(sp["D"], args.warm_s)
        t0 = time.perf_counter()
        graph, note = gr.try_capture(sp["D"])
        row["capture_ms"] = (time.perf_counter() - t0) * 1000.0
        row["capture_note"] = note
        d_pad_graph = (st.median([gr.wall_ms(graph.replay, args.steps)
                                  for _ in range(3)]) if graph else None)
        d_pad_plain = st.median([gr.wall_ms(sp["D"], args.steps) for _ in range(3)])

        row.update({
            "g_faithful_ms": g_faith, "d_faithful_ms": d_faith,
            "d_padded_plain_ms": d_pad_plain, "d_padded_graphed_ms": d_pad_graph,
            "d_over_g_faithful_ungraphed": d_faith / g_faith,
            # THE REAL RACE: what each side actually gets
            "d_graphed_padded_over_g_faithful": (
                d_pad_graph / g_faith) if d_pad_graph else None,
            # what the static-shape race claimed, for contrast
            "padding_cost_on_baseline": d_pad_plain / d_faith,
            "status": "ok"})
        del sp, graph
    except Exception as e:  # pragma: no cover
        row.update({"status": "skipped", "reason": f"{type(e).__name__}: {str(e)[:200]}"})
    torch.cuda.empty_cache()
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--tokens", type=int, nargs="*", default=[32, 2048])
    ap.add_argument("--draws", type=int, default=64)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--warm-s", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.environ.get("DQF_OUT", "/root/dqf-out"))
    ap.add_argument("--tag", default="dyn1")
    args = ap.parse_args()

    out = {"probe": "does the baseline's graph advantage survive dynamic routing",
           "prereg": "kernel/prereg_dequant_forward_dynamic.json",
           "gpu": torch.cuda.get_device_name(0),
           "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
           "torch": torch.__version__, "rows": []}
    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    art = dest / f"dynamic_{args.tag}.json"
    for spec in H.census_specs(H.REPO / "census" / "shape_census.json", args.models):
        stack = H.QuantStack(spec, "cuda")
        for tokens in args.tokens:
            r = cell(spec, stack, tokens, "cuda", args)
            out["rows"].append(r)
            art.write_text(json.dumps(out, indent=1, default=str))
            if r.get("status") == "ok":
                print("%-14s %-8s T=%-5d C=%-4d tax %5.2fx | d/g faithful-ungraphed "
                      "%7.3f | REAL RACE d-graphed-padded/g-faithful %7.3f | capture %6.0f ms"
                      % (r["model"].split("/")[-1][:14], r["proj"], tokens,
                         r["capacity_C"], r["capacity_tax"],
                         r["d_over_g_faithful_ungraphed"],
                         r["d_graphed_padded_over_g_faithful"] or float("nan"),
                         r["capture_ms"]), flush=True)
            else:
                print("%-14s %-8s T=%-5d %s" % (r["model"].split("/")[-1][:14],
                                                r["proj"], tokens, r["status"].upper()),
                      flush=True)
        del stack
        torch.cuda.empty_cache()

    ok = [r for r in out["rows"] if r.get("status") == "ok"]
    if ok:
        print("\ncapacity tax median %.2fx | REAL RACE median %.3f | "
              "static-shape race said %.3f"
              % (st.median([r["capacity_tax"] for r in ok]),
                 st.median([r["d_graphed_padded_over_g_faithful"] for r in ok
                            if r["d_graphed_padded_over_g_faithful"]]),
                 st.median([r["d_over_g_faithful_ungraphed"] for r in ok])))
    print("DYNAMIC_DONE")


if __name__ == "__main__":
    main()
