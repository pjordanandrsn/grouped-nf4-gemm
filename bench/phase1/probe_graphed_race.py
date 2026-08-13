#!/usr/bin/env python3
"""ADVERSARIAL: race the baseline WITH the optimisation it can have against the
fused path WITHOUT the one it cannot.

Leg 4 says gnf4's wall advantage is mostly eliminated launches: at training
shape the baseline makes ~59 python-mediated launches to the fused path's one,
and on an H100 it is 6% GPU-busy while doing it. But a user can erase the
baseline's launch overhead TODAY -- it CUDA-graphs cleanly (4/4 attempts, both
devices) -- and cannot do the same for the fused path, which fails capture 8/8.

So the honest race, the one actually available to someone choosing between
these two implementations right now, is:

    D_base CAPTURED IN A CUDA GRAPH   vs   G_base as it ships

If that closes leg 4's 10-11x at training shape, then our headline is largely an
artifact of comparing an optimisable path against an already-optimal one, and it
has to be said that way. This probe exists to find that out before any of leg 4
is published anywhere.

ALSO MEASURED, because leg 4 corrected the fixture and never revisited it:
PEAK TRANSIENT MEMORY under faithful routing. `F.linear` saves its weight for
backward, so the baseline holds every HIT expert's materialised bf16 weight
across the forward-to-backward window. The old fiction gave it 8 hit experts;
real routing gives ~59. That is a ~7x increase in precisely the quantity that
makes the baseline OOM, on the axis where this kernel's claim has always been
strongest -- and leg 1 measured it on the fixture we now know understates it.

TWO HONESTY NOTES, both favouring the BASELINE, i.e. making this test harder on
us, which is the direction an adversarial test should lean:
  * a replayed graph reuses its input buffers; a real loop would copy fresh
    activations in each step and the graph does not pay for that here;
  * the fused arm gets no equivalent help, because none exists for it today.

Report-only against `kernel/prereg_dequant_forward_graphed.json`.
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


def wall_ms(fn, n):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / n


def transient_bytes(step):
    step()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    step()
    torch.cuda.synchronize()
    return int(torch.cuda.max_memory_allocated() - base)


def try_capture(step):
    """Warm on a side stream, capture, replay once. (graph|None, note)."""
    try:
        step()
        torch.cuda.synchronize()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(5):
                step()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            step()
        torch.cuda.synchronize()
        g.replay()
        torch.cuda.synchronize()
        return g, "captured"
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)[:160]}"


def build(spec_stack, groups, device):
    sizes = [a.shape[0] for _, a in groups]
    eids = [int(e) for e, _ in groups]
    a_cat = torch.cat([a for _, a in groups]).detach().requires_grad_(True)
    B_pack, A_scale = spec_stack.fusedpack()
    gb = ff.g_base_arm(a_cat, B_pack, A_scale, sizes, eids)
    db = ff.d_base_arm(spec_stack, a_cat, sizes, eids)

    def zero_inplace():
        if a_cat.grad is not None:
            a_cat.grad.zero_()

    def mk(f):
        def run():
            zero_inplace()
            f().float().pow(2).mean().backward()
        return run

    return {"G": mk(gb), "D": mk(db), "act": a_cat}


def cell(spec, spec_stack, tokens, device, args):
    row = {"model": spec.model, "proj": spec.proj, "tokens": tokens,
           "E": spec.E, "top_k": spec.top_k}
    try:
        faithful = rf.routed_groups(spec, tokens, RESULTS, device, seed=args.seed)
    except LookupError as e:
        row.update({"status": "not_run", "reason": str(e)})
        return row
    row["routing"] = rf.summarise(faithful, spec.E)

    fiction_regime = (f"decode_m{tokens}" if tokens < 512
                      else f"tokbudget_{tokens}")
    fiction = H.make_activations(spec, fiction_regime, device)
    row["fiction_regime"] = fiction_regime

    try:
        # ---- memory, both fixtures, so the fixture's effect on the axis this
        # kernel's claim rests on is visible rather than assumed -------------
        mem = {}
        for label, grp in (("fiction", fiction), ("faithful", faithful)):
            s = build(spec_stack, grp, device)
            mem[label] = {"G": transient_bytes(s["G"]),
                          "D": transient_bytes(s["D"]),
                          "groups": len(grp)}
            mem[label]["D_over_G"] = mem[label]["D"] / max(mem[label]["G"], 1)
            del s
            torch.cuda.empty_cache()
        row["memory"] = mem
        row["memory_fixture_effect_D"] = (mem["faithful"]["D"]
                                          / max(mem["fiction"]["D"], 1))

        # ---- the race, faithful routing ------------------------------------
        s = build(spec_stack, faithful, device)
        dqf1._warm(s["G"], args.warm_s)
        g_wall = st.median([wall_ms(s["G"], args.steps) for _ in range(3)])
        d_wall = st.median([wall_ms(s["D"], args.steps) for _ in range(3)])

        graph_d, note_d = try_capture(s["D"])
        d_graph_wall = (st.median([wall_ms(graph_d.replay, args.steps)
                                   for _ in range(3)]) if graph_d else None)
        # NOT attempted here: capturing the fused arm. It fails 8/8 in the
        # process-isolated feasibility probe, and a failed capture POISONS the
        # CUDA context -- attempting it mid-loop would corrupt every cell after
        # it in this process. That lesson cost the first feasibility probe its
        # detail. The fused arm's incapturability is taken from that probe,
        # where each attempt had its own process, not re-tested here.

        row["g_wall_ms"] = g_wall
        row["d_wall_ms"] = d_wall
        row["d_graphed_wall_ms"] = d_graph_wall
        row["d_capture"] = note_d
        row["d_over_g_ungraphed"] = d_wall / g_wall
        row["d_over_g_GRAPHED"] = (d_graph_wall / g_wall) if d_graph_wall else None
        row["graph_speedup_for_baseline"] = (d_wall / d_graph_wall
                                             if d_graph_wall else None)
        row["status"] = "ok"
        del s, graph_d
    except Exception as e:  # pragma: no cover
        row.update({"status": "skipped",
                    "reason": f"{type(e).__name__}: {str(e)[:200]}"})
    torch.cuda.empty_cache()
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--tokens", type=int, nargs="*", default=[32, 2048])
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--warm-s", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.environ.get("DQF_OUT", "/root/dqf-out"))
    ap.add_argument("--tag", default="gr1")
    args = ap.parse_args()

    out = {"probe": "graphed baseline vs shipped fused, plus memory under "
                    "faithful routing",
           "prereg": "kernel/prereg_dequant_forward_graphed.json",
           "gpu": torch.cuda.get_device_name(0),
           "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
           "torch": torch.__version__, "tokens": args.tokens, "rows": []}
    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    art = dest / f"graphed_race_{args.tag}.json"

    for spec in H.census_specs(H.REPO / "census" / "shape_census.json", args.models):
        spec_stack = H.QuantStack(spec, "cuda")
        for tokens in args.tokens:
            r = cell(spec, spec_stack, tokens, "cuda", args)
            out["rows"].append(r)
            art.write_text(json.dumps(out, indent=1, default=str))
            if r.get("status") == "ok":
                print("%-14s %-8s T=%-5d | d/g ungraphed %7.3f -> GRAPHED %7.3f "
                      "(graph gave baseline %.2fx) | mem D/G %5.2f  fixture x%.2f"
                      % (r["model"].split("/")[-1][:14], r["proj"], tokens,
                         r["d_over_g_ungraphed"],
                         r["d_over_g_GRAPHED"] or float("nan"),
                         r["graph_speedup_for_baseline"] or float("nan"),
                         r["memory"]["faithful"]["D_over_G"],
                         r["memory_fixture_effect_D"]), flush=True)
            else:
                print("%-14s %-8s T=%-5d %s" % (
                    r["model"].split("/")[-1][:14], r["proj"], tokens,
                    r["status"].upper()), flush=True)
        del spec_stack
        torch.cuda.empty_cache()

    ok = [r for r in out["rows"] if r.get("status") == "ok"]
    if ok:
        un = st.median([r["d_over_g_ungraphed"] for r in ok])
        gr = [r["d_over_g_GRAPHED"] for r in ok if r["d_over_g_GRAPHED"]]
        print("\nd/g ungraphed median %.3f  ->  GRAPHED median %.3f  (%d cells)"
              % (un, st.median(gr) if gr else float("nan"), len(gr)))
        print("baseline gained %.2fx median from graphing"
              % st.median([r["graph_speedup_for_baseline"] for r in ok
                           if r["graph_speedup_for_baseline"]]))
        print("memory D/G under faithful routing: median %.2f"
              % st.median([r["memory"]["faithful"]["D_over_G"] for r in ok]))
    print("GRAPHED_RACE_DONE")


if __name__ == "__main__":
    main()
