#!/usr/bin/env python3
"""Uniform-full R-scan: mechanism attribution for the above-band HBM3 cell.

Grades `kernel/prereg_graphed_rscan.json` (OTS-stamped pre-data). PR #93 left
one cell above the HBM3 parity band — down/T=32 at D/G = 1.695 — and the
absolute excesses already in the merged receipts FLIP order between memory
classes (on GDDR6 gate_up's excess > down's, ~byte-proportional; on HBM3
down's > gate_up's despite half the work), so on H100 that excess is not
traffic. Candidates: a per-kernel execution floor times the D-graph's ~3E
kernels (H_floor) vs shape-dependent cuBLAS selection punishing down's
[R,1024]<->[1024,2048] GEMMs on sm_90 (H_select).

Design, verbatim from the registration: proj x R in {1..128}, E=64, lora=False
(matching the anomalous cells), every expert EXACTLY R real rows — zero
padding, so the D arm's kernel shapes reproduce the race's padded slices while
the padding confound is gone. T_real = E*R varies with R by construction;
these cells license no throughput claim. Both arms are the merged
DenseGroupsStep machinery, unmodified, captured and replayed; instrument =
replay self-pairs, void outside [0.967, 1.033].

Graded: P1 Spearman(D/G, R) <= -0.7 per projection; P2 down/gate_up excess
ratio at R=32 >= 1.1; P4 D-arm elasticity R=1->8 <= 0.35 per projection.
P3 (which kernels carry the excess) is decided by pre-named RULES over the
profile, not a numeric band; the profiler carries its own positive control
(3 eager fused steps must show kernels, else every profile field is VOID).
"""
from __future__ import annotations

import argparse
import json
import math
import os
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
import graphed_buckets as gb  # noqa: E402

R_SET = (1, 2, 4, 8, 16, 32, 64, 128)
BANDS = {
    "spearman_max": float(os.environ.get("DQF_RS_SPEAR_MAX", "-0.7")),
    "flip_min": float(os.environ.get("DQF_RS_FLIP_MIN", "1.1")),
    "elast_max": float(os.environ.get("DQF_RS_ELAST_MAX", "0.35")),
    "selfpair_lo": 0.967, "selfpair_hi": 1.033,
}


def uniform_groups(spec, R, device, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return [(e, torch.randn(R, spec.K, generator=g).to(device, torch.bfloat16))
            for e in range(spec.E)]


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = float(pos)
        return r
    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx)
                    * sum((b - my) ** 2 for b in ry))
    return num / den if den else 0.0


def kernel_table(prof, replays):
    """Per-replay kernel count + name-family aggregation from a profiler pass.
    Returns {} if the profiler recorded no device kernels (caller treats that
    as instrument-blind — never as 'zero kernels ran')."""
    from torch.autograd import DeviceType
    rows = []
    for ev in prof.key_averages():
        if getattr(ev, "device_type", None) != DeviceType.CUDA:
            continue
        t = float(getattr(ev, "self_device_time_total",
                          getattr(ev, "self_cuda_time_total", 0.0)))
        if ev.count <= 0 and t <= 0:
            continue
        rows.append({"name": ev.key, "count": ev.count, "self_us": t})
    if not rows:
        return {}

    def family(name):
        n = name.lower()
        if any(s in n for s in ("dequant", "kdequantize")):
            return "dequant"
        if any(s in n for s in ("nf4", "grouped", "lora_delta")):
            return "fused_triton"
        if any(s in n for s in ("gemm", "nvjet", "cutlass", "xmma", "wgmma",
                                "matmul", "gemv", "splitk", "split_k")):
            return "gemm"
        return "elementwise_other"
    fam = {}
    total_n = 0
    for r in rows:
        f = family(r["name"])
        d = fam.setdefault(f, {"count": 0, "self_us": 0.0})
        d["count"] += r["count"]
        d["self_us"] += r["self_us"]
        total_n += r["count"]
    top = sorted(rows, key=lambda r: -r["self_us"])[:12]
    return {
        "kernels_per_replay": total_n / replays,
        "families_per_replay": {
            k: {"count": v["count"] / replays, "self_us": v["self_us"] / replays}
            for k, v in fam.items()},
        "splitk_names": sorted({r["name"] for r in rows
                                if "split" in r["name"].lower()
                                or "reduce" in r["name"].lower()}),
        "top12": top,
    }


def profile_replays(step, draws, replays=3):
    from torch.profiler import ProfilerActivity, profile
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for i in range(replays):
            step.load(draws[i % len(draws)])
            step.replay()
        torch.cuda.synchronize()
    return kernel_table(prof, replays)


def profiler_positive_control(stack, spec, device="cuda"):
    """The profiler must SEE kernels on a path known to launch them (3 eager
    fused steps). Empty -> the instrument is blind on this pod and every
    profile field in the receipt is VOID (registered rule d)."""
    from torch.profiler import ProfilerActivity, profile
    step = gb.DenseGroupsStep("fused", stack, spec, spec.E * 4, 4, False, device)
    step.load(uniform_groups(spec, 4, device, seed=999))
    step.eager()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(3):
            step.eager()
        torch.cuda.synchronize()
    t = kernel_table(prof, 3)
    return {"pass": bool(t), "kernels_per_step": t.get("kernels_per_replay", 0.0)}


def floor_probe(device="cuda", n_kernels=256, reps=50):
    """Captured chain of data-dependent scalar adds: per-kernel replay floor on
    this pod. A yardstick, reported — never subtracted from anything."""
    x = torch.zeros(1, device=device)

    def chain():
        y = x
        for _ in range(n_kernels):
            y = y + 1.0
        return y
    chain(); torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            chain()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        chain()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        g.replay()
    torch.cuda.synchronize()
    per_kernel_us = (time.perf_counter() - t0) * 1e6 / (reps * n_kernels)
    return {"n_kernels": n_kernels, "reps": reps,
            "per_kernel_floor_us": per_kernel_us}


def scan_cell(stack, spec, R, args, device="cuda"):
    draws = [uniform_groups(spec, R, device, seed=s)
             for s in range(args.steps_pool)]
    T_real = spec.E * R
    arms = {}
    for name in ("fused", "base"):
        s_ = gb.DenseGroupsStep(name, stack, spec, T_real, R, False, device)
        s_.load(draws[0])
        s_.capture()
        arms[name] = s_

    def timed_replay_pure(s_, n):
        # GRADED instrument: load() once OUTSIDE the timed region. race_cell's
        # load-inclusive loop would add an arm-shared host cost that dilutes
        # ratios hardest at small R — an instrument failure mode, not a
        # mechanism (registered divergence; see the prereg's instrument note).
        s_.load(draws[0])
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            s_.replay()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000.0 / n

    def timed_replay_with_load(s_, n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(n):
            s_.load(draws[i % len(draws)])
            s_.replay()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000.0 / n

    warm = getattr(__import__("importlib").import_module("e2e_train_arms"),
                   "warm_gpu", None)
    if warm:
        warm(1.0)
    # AMENDMENT 1: attempt 1 voided 12/16 cells, every void the g_a block —
    # the first timed block of a cell reads a settling ramp that lasts about a
    # full block (the e2e ten-drift law at block scale; short burn-ins were
    # falsified there and are not attempted here). Remedy = the first-mode
    # discard translated: one full UNTIMED block per arm before its timed
    # pair, plus a 0.25 s wall floor on block length from a 5-replay pilot.
    pilot_ms = timed_replay_pure(arms["base"], 5)
    n = max(200, min(2000, math.ceil(250.0 / max(pilot_ms, 1e-3))))
    primer = {}
    for name in ("fused", "base"):
        t0 = time.perf_counter()
        timed_replay_pure(arms[name], n)
        primer[name] = {"n": n, "wall_s": time.perf_counter() - t0}
    t = {"g_a": timed_replay_pure(arms["fused"], n),
         "g_b": timed_replay_pure(arms["fused"], n),
         "d_a": timed_replay_pure(arms["base"], n),
         "d_b": timed_replay_pure(arms["base"], n),
         "g_with_load": timed_replay_with_load(arms["fused"], min(n, 100)),
         "d_with_load": timed_replay_with_load(arms["base"], min(n, 100))}
    row = {
        "proj": spec.proj, "R": R, "T_real": T_real, "ms": t,
        "n": n, "pilot_ms": pilot_ms, "primer": primer,
        "g_selfpair": t["g_b"] / t["g_a"], "d_selfpair": t["d_b"] / t["d_a"],
        "d_over_g": (t["d_a"] + t["d_b"]) / (t["g_a"] + t["g_b"]),
        "excess_ms": (t["d_a"] + t["d_b"]) / 2 - (t["g_a"] + t["g_b"]) / 2,
        "d_over_g_with_load": t["d_with_load"] / t["g_with_load"],
    }
    row["live"] = all(BANDS["selfpair_lo"] <= row[k] <= BANDS["selfpair_hi"]
                      for k in ("g_selfpair", "d_selfpair"))
    if R in args.profile_R:
        for name in ("fused", "base"):
            row[f"profile_{name}"] = profile_replays(arms[name], draws)
    return row


def grade(out):
    """P1/P2/P4 against the registered bands; refusals are written as such."""
    rows = out["scan"]
    verdicts = {}
    for proj in ("gate_up", "down"):
        live = [r for r in rows if r["proj"] == proj and r["live"]]
        if len(live) >= 6:
            rho = spearman([float(r["R"]) for r in live],
                           [r["d_over_g"] for r in live])
            verdicts[f"P1_spearman_{proj}"] = {
                "rho": rho, "n_live": len(live),
                "pass": rho <= BANDS["spearman_max"]}
        else:
            verdicts[f"P1_spearman_{proj}"] = {
                "refusal": f"only {len(live)} live points (< 6)"}
        r1 = next((r for r in live if r["R"] == 1), None)
        r8 = next((r for r in live if r["R"] == 8), None)
        if r1 and r8:
            d1 = (r1["ms"]["d_a"] + r1["ms"]["d_b"]) / 2
            d8 = (r8["ms"]["d_a"] + r8["ms"]["d_b"]) / 2
            e = math.log(d8 / d1) / math.log(8)
            verdicts[f"P4_elasticity_{proj}"] = {
                "elasticity_1_to_8": e, "pass": e <= BANDS["elast_max"]}
        else:
            verdicts[f"P4_elasticity_{proj}"] = {
                "refusal": "R=1 or R=8 cell not live"}
    c32 = {r["proj"]: r for r in rows if r["R"] == 32 and r["live"]}
    if "down" in c32 and "gate_up" in c32:
        flip = c32["down"]["excess_ms"] / c32["gate_up"]["excess_ms"]
        verdicts["P2_flip_at_R32"] = {
            "excess_down_over_gate_up": flip,
            "excess_ms": {p: c32[p]["excess_ms"] for p in c32},
            "pass": flip >= BANDS["flip_min"]}
    else:
        verdicts["P2_flip_at_R32"] = {"refusal": "an R=32 cell is not live"}
    graded = [v for v in verdicts.values() if "pass" in v]
    verdicts["ALL_GRADED_PASS"] = bool(graded) and all(v["pass"] for v in graded)
    if not graded:
        verdicts["ALL_GRADED_PASS"] = False
        verdicts["refusal"] = "no graded rows at all — absence is not a pass"
    return verdicts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["OLMoE"])
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--steps-pool", type=int, default=8)
    ap.add_argument("--profile-R", type=int, nargs="*", default=[8, 32])
    ap.add_argument("--fidelity-only", action="store_true")
    ap.add_argument("--out", default=os.environ.get("DQF_OUT", "/root/dqf-out"))
    ap.add_argument("--tag", default="rs1")
    args = ap.parse_args()

    out = {"probe": "uniform-full R-scan: mechanism attribution for the "
                    "above-band HBM3 cell",
           "prereg": "kernel/prereg_graphed_rscan.json",
           "gpu": torch.cuda.get_device_name(0),
           "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
           "torch": torch.__version__, "bands": BANDS,
           "fidelity": [], "scan": []}
    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    art = dest / f"graphed_rscan_{args.tag}.json"

    specs = list(H.census_specs(H.REPO / "census" / "shape_census.json",
                                args.models))
    for spec in specs:
        stack = H.QuantStack(spec, "cuda")
        # F2 spot: the merged bitwise gate on uniform-full draws (base path).
        for R in (1, 32, 128):
            fc = {"proj": spec.proj, "R_requested": R}
            fc.update(gb.fidelity_cell(stack, spec,
                                       uniform_groups(spec, R, "cuda", seed=0),
                                       lora=False))
            fc["F2_pass"] = all(v for k, v in fc.items()
                                if k.endswith("_bitwise"))
            out["fidelity"].append(fc)
            print("F2 %-8s R=%-3d %s" % (
                spec.proj, R, "PASS" if fc["F2_pass"] else "FAIL"), flush=True)
            art.write_text(json.dumps(out, indent=1, default=str))
        if args.fidelity_only:
            del stack
            torch.cuda.empty_cache()
            continue
        if not all(f["F2_pass"] for f in out["fidelity"]
                   if f["proj"] == spec.proj):
            print("F2 failed — the scan does not run (stop rule)")
            del stack
            torch.cuda.empty_cache()
            continue
        if "profiler_control" not in out:
            out["profiler_control"] = profiler_positive_control(stack, spec)
            out["floor_probe"] = floor_probe()
            print("profiler control:", out["profiler_control"],
                  " floor:", out["floor_probe"], flush=True)
        for R in R_SET:
            row = scan_cell(stack, spec, R, args)
            out["scan"].append(row)
            print("RS %-8s R=%-3d d/g %.3f excess %.3fms "
                  "(self %.3f/%.3f)%s" % (
                      row["proj"], R, row["d_over_g"], row["excess_ms"],
                      row["g_selfpair"], row["d_selfpair"],
                      "" if row["live"] else "  VOID"), flush=True)
            art.write_text(json.dumps(out, indent=1, default=str))
        del stack
        torch.cuda.empty_cache()

    if not args.fidelity_only:
        if not out.get("profiler_control", {}).get("pass"):
            for r in out["scan"]:
                for k in list(r):
                    if k.startswith("profile_"):
                        r[k] = {"VOID": "profiler positive control failed"}
        out["verdicts"] = grade(out)
        print("VERDICTS:", json.dumps(out["verdicts"], indent=1,
                                      default=str), flush=True)
        art.write_text(json.dumps(out, indent=1, default=str))
    print("GRAPHED_RSCAN_DONE ->", art)


if __name__ == "__main__":
    main()
