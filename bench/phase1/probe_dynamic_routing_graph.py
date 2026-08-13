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
standard option. It was excluded from the registered run because it changes what
the model computes; `--capacity-factors` adds it as a SEPARATE, report-only arm
(amendment 1), for the reason below.

WHY THE cf ARM EXISTS, AND WHAT IT CANNOT SAY. The no-drop capacity above is the
honest bound for "same computation, static shapes", but it is not what trainers
actually run: real MoE training commonly sets capacity_factor 1.0-2.0 and drops
the overflow. Leaving that unpriced would leave the baseline's cheapest
configuration unmeasured, so it is measured here -- but at cf the row tax is
~cf BY CONSTRUCTION (E*ceil(cf*T*k/E) / (T*k)), which makes rows a useless
currency. The quantity that actually varies is how much routed computation gets
DISCARDED, so every cf row reports `drop_rate` from the SAME draws used for the
no-drop capacity, and no timing number from this arm may be quoted without it.

The comparison is deliberately UNFAIR TO US, in two stacked ways: the cf arm
gets a CUDA graph the fused path cannot have, AND it is excused from computing
`drop_rate` of the work the fused path still does in full. It is therefore not
like-for-like and cannot be reported as a speed result. What it can establish is
a bound: if the fused path still wins while the baseline is both graphed and
excused part of the problem, the margin is not an artifact of denying the
baseline its standard optimisation. What it cannot establish is the quality cost
of those drops -- that needs a training run and an eval, not a timer.

THE DROP RATE IS AN UPPER BOUND, AND THE REASON IS THE FIXTURE. Both routing
histograms are ONE 2048-token sequence per model (`seq`/`tokens` = 2048 in
routing_olmoe.json / routing_qwen.json), captured from a trained model at
inference. Two things make that more concentrated than a real training step:

  * one sequence is topically narrow, while a training batch mixes documents and
    spreads load across more experts;
  * these models train with a load-balancing auxiliary loss actively flattening
    the router, which is NOT acting on an inference capture.

So the measured drop rates here are a CEILING on what a real trainer at the same
capacity_factor would discard, and they are quoted that way. The direction of
that bias favours the baseline being reported as worse than it is, which is why
it is stated rather than buried: a reader who wants the number for their own
router should re-measure on their own batch.
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


def draw_counts(spec, tokens, draws):
    """The `draws` independent routing draws themselves, so the no-drop capacity
    and every capacity-factor drop rate are derived from the SAME sample rather
    than from two independently drawn ones."""
    per_layer, E, k = rf.routing_for(spec.model, RESULTS)
    counts = per_layer[len(per_layer) // 2]
    return [rf.sample_group_sizes(counts, tokens, k, seed=1000 + d)
            for d in range(draws)], E, k


def capacity_over_draws(spec, tokens, draws):
    """C that drops no token across `draws` independent routing draws, and the
    per-draw faithful row totals for contrast."""
    samples, E, _k = draw_counts(spec, tokens, draws)
    worst = max(max(s.values()) for s in samples)
    totals = [sum(s.values()) for s in samples]
    hits = [len(s) for s in samples]
    return worst, st.median(totals), st.median(hits), E


def capacity_at_factor(spec, tokens, cf, draws):
    """Price a capacity_factor: C = ceil(cf*T*k/E), overflow DISCARDED.

    The row tax here is ~cf by construction, so it is reported but is not the
    finding. `drop_rate` is: the fraction of routed (token, expert) assignments
    that never get computed, median and worst across draws. `overflow_experts`
    says how concentrated that loss is -- a few hot experts eating the whole
    budget is a different failure from every expert shedding a little.
    """
    samples, E, k = draw_counts(spec, tokens, draws)
    C = -(-int(round(cf * tokens * k)) // E)  # ceil
    assigns = tokens * k
    drops, over = [], []
    for s in samples:
        d = sum(max(0, c - C) for c in s.values())
        drops.append(d / assigns)
        over.append(sum(1 for c in s.values() if c > C) / E)

    # PRIMARY: the drop rate implied by the MEASURED histogram itself, with the
    # sampling model removed entirely -- each layer's counts rescaled to this
    # token budget and clipped at capacity. Reported across ALL layers, not just
    # the representative one, because the spread between layers is large (OLMoE
    # cv 0.397-0.995, Qwen 0.956-2.193) and a single layer would understate it.
    per_layer, _E, _k = rf.routing_for(spec.model, RESULTS)
    meas = []
    for c in per_layer:
        tot = sum(c)
        cap = cf * tot / E
        meas.append(sum(max(0.0, x - cap) for x in c) / tot)
    return {"capacity_factor": cf, "capacity_C": C, "rows_padded": C * E,
            "row_tax": C * E / assigns,
            "drop_rate_measured": st.median(meas),
            "drop_rate_measured_worst_layer": max(meas),
            "drop_rate_measured_best_layer": min(meas),
            "drop_rate_sampled": st.median(drops),
            "drop_rate_sampled_max": max(drops),
            "overflow_expert_frac": st.median(over), "draws": draws}


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
            # the cf a no-drop policy is implicitly paying, so the cf arm below
            # is read on the same axis rather than as a separate universe
            "implied_no_drop_capacity_factor": rows_padded / (tokens * spec.top_k),
            "status": "ok"})
        del sp, graph
        torch.cuda.empty_cache()

        # ---- amendment 1: the capacity_factor arm (report-only) -------------
        # Same draws as the no-drop capacity, so drop rates and C are
        # commensurable. Timing is against the SAME `g_faith` measured above --
        # the fused arm is not re-timed and is not excused any work.
        cf_rows = []
        for cf in (args.capacity_factors or []):
            info = capacity_at_factor(spec, tokens, cf, args.draws)
            try:
                scf = gr.build(stack, padded_groups(spec, info["capacity_C"],
                                                    device), device)
                dqf1._warm(scf["D"], args.warm_s)
                gcf, note_cf = gr.try_capture(scf["D"])
                info["capture_note"] = note_cf
                info["d_cf_graphed_ms"] = (
                    st.median([gr.wall_ms(gcf.replay, args.steps)
                               for _ in range(3)]) if gcf else None)
                info["d_cf_plain_ms"] = st.median(
                    [gr.wall_ms(scf["D"], args.steps) for _ in range(3)])
                info["d_cf_graphed_over_g_faithful"] = (
                    info["d_cf_graphed_ms"] / g_faith
                    if info["d_cf_graphed_ms"] else None)
                del scf, gcf
            except Exception as e:  # pragma: no cover
                info["error"] = f"{type(e).__name__}: {str(e)[:160]}"
            torch.cuda.empty_cache()
            cf_rows.append(info)
        if cf_rows:
            row["capacity_factor_arm"] = cf_rows
    except Exception as e:  # pragma: no cover
        row.update({"status": "skipped", "reason": f"{type(e).__name__}: {str(e)[:200]}"})
    torch.cuda.empty_cache()
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--tokens", type=int, nargs="*", default=[32, 2048])
    ap.add_argument("--draws", type=int, default=64)
    ap.add_argument("--capacity-factors", type=float, nargs="*", default=None,
                    help="amendment 1: also price token-DROPPING capacities "
                         "(e.g. 1.0 1.25 2.0). Report-only; every timing from "
                         "this arm carries its drop_rate.")
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
                for c in r.get("capacity_factor_arm", []):
                    print("      cf=%-5.2f C=%-5d row tax %5.2fx | DROPS %5.1f%% "
                          "of routed work (worst layer %5.1f%%, %4.1f%% of experts "
                          "overflow) | graphed vs fused %s"
                          % (c["capacity_factor"], c["capacity_C"], c["row_tax"],
                             100 * c["drop_rate_measured"],
                             100 * c["drop_rate_measured_worst_layer"],
                             100 * c["overflow_expert_frac"],
                             ("%.3f" % c["d_cf_graphed_over_g_faithful"])
                             if c.get("d_cf_graphed_over_g_faithful")
                             else str(c.get("capture_note") or c.get("error"))[:40]),
                          flush=True)
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
        by_cf = {}
        for r in ok:
            for c in r.get("capacity_factor_arm", []):
                by_cf.setdefault(c["capacity_factor"], []).append(c)
        for cf in sorted(by_cf):
            cs = by_cf[cf]
            races = [c["d_cf_graphed_over_g_faithful"] for c in cs
                     if c.get("d_cf_graphed_over_g_faithful")]
            print("cf=%.2f: drops median %.1f%% (worst layer %.1f%%) of routed work | "
                  "graphed-vs-fused median %s  <-- NOT like-for-like: the "
                  "baseline is graphed AND skips that work"
                  % (cf, 100 * st.median([c["drop_rate_measured"] for c in cs]),
                     100 * max(c["drop_rate_measured_worst_layer"] for c in cs),
                     ("%.3f" % st.median(races)) if races else "n/a"))
    print("DYNAMIC_DONE")


if __name__ == "__main__":
    main()
