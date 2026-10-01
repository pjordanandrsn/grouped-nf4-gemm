"""Lane K20 bench (``PREREG-k20-k19-plan-sweep-5090.md``): K19's plan space against the served GEMV, on recorded routing.

    python k20_bench.py eids_b16.int16.bin eids_b16.int16.json k20_rows.json [--quick]

Recorded B=16 routing (experts4bit-qlora P60's ids: [steps, 48 layers, 16 rows, top-8], raw little-endian int16, shape
and sha256 in the json). Synthetic int4-b32 stores for 128 experts at Qwen3-30B-A3B's expert shapes (gate_up N1536
K2048, down N2048 K768). The bytes and the L2 behaviour depend on WHICH experts a call touches, not on the values.

Each arm is one CUDA graph per decode step: 48 layers, gate_up then down, as the model runs them:
  served    quant_x_rows + gemv_int4_b32 (its own reduce) per call: the route P86/P87 censused (GEMV + reduce + quantise)
  k19@plan  per layer one build_group_tiles_fused(ids, 128, 16), then K19 gate_up (gather) and K19 down
  tiles     the per-layer tile build alone
  floor     no kernel: the step's distinct-expert bytes / this box's measured device-to-device bandwidth

Selection is separate from evaluation (no winner's curse): the SELECT steps (the first half) choose the best plan by
median step time, and the EVAL steps (the second half) report every arm. The rule reads EVAL only. A plan that fails to
compile or launch is recorded and skipped.

Correctness, per plan, on the first step's first layer: max |out - default plan| / max |default|. A plan changes the
fp32 accumulation order across K chunks, so the check is closeness, not bit-equality. Also max relative error against
an fp32 dequant oracle on 8 rows.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import statistics
import sys
import time

import numpy as np
import torch

from int4_b32 import build_group_tiles_fused, gemv_int4_b32, quant_x_rows
from int4_pack_ref import dequant_int4_ref
from int4_smallm import gemm_int4_b32_grouped_smallm

dev = "cuda"
SHAPES = {"gate_up": (1536, 2048), "down": (2048, 768)}
E = 128
GRID = {"block_n": (32, 64, 128, 256), "kc": (64, 128, 256), "warps": (4, 8), "stages": (2, 3, 4)}
QUICK = {"block_n": (64, 128), "kc": (128,), "warps": (4,), "stages": (2, 3)}
DEFAULT = (64, 128, 4, 2)
ITERS = 20
# The registered rule (PREREG-k20), read on the EVAL medians.
P87_SERVED_MS = 7.729       # P87's B=16 census: GEMV 7.000 + reduce 0.384 + quantise 0.345 ms/step
P87_K19_MS = 6.993          # P87's B=16 census: K19 6.495 + tile build 0.498 ms/step (the default plan)
INSTRUMENT_BAND = 0.15
PLAN_TOL = 0.02             # a plan whose output is further than this from the oracle or the default is not selectable
PROMISING, MARGINAL = 0.77, 0.90


def verdict(rec):
    med = rec["eval_median"]
    for name, ref in (("served", P87_SERVED_MS), ("default", P87_K19_MS)):
        if abs(med[name] / ref - 1) > INSTRUMENT_BAND:
            return "VOID", f"{name} {med[name]:.3f} ms/step is not within {INSTRUMENT_BAND:.0%} of P87's census {ref}"
    r = rec["ratio_best_over_served"]
    if r <= PROMISING:
        return "PROMISING", f"best/served {r:.3f} <= {PROMISING}"
    if r <= MARGINAL:
        return "MARGINAL", f"{PROMISING} < best/served {r:.3f} <= {MARGINAL}"
    return "NO", f"best/served {r:.3f} > {MARGINAL}"


def self_test():
    base = {"eval_median": {"served": 7.7, "default": 7.0, "best": 5.5}, "ratio_best_over_served": 5.5 / 7.7}
    assert verdict(base)[0] == "PROMISING"
    assert verdict(dict(base, ratio_best_over_served=0.85))[0] == "MARGINAL"
    assert verdict(dict(base, ratio_best_over_served=0.95))[0] == "NO"
    assert verdict(dict(base, eval_median={"served": 5.0, "default": 7.0, "best": 4.0}))[0] == "VOID"
    assert verdict(dict(base, eval_median={"served": 7.7, "default": 9.0, "best": 5.0}))[0] == "VOID"
    print("k20 verdict self-test OK (5 cases)")


def bytes_per_expert(N, K):
    return N * (K // 2) + N * (K // 32) * 2                      # packed int4 + fp16 scales


def graph_ms(fn, iters=ITERS, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b))
    del g
    torch.cuda.empty_cache()
    return statistics.median(ts)


def copy_gbps():
    src = torch.empty(512 << 20, dtype=torch.uint8, device=dev)
    dst = torch.empty_like(src)
    ms = graph_ms(lambda: dst.copy_(src))
    return 2 * src.numel() / (ms / 1e3) / 1e9                    # read + write


def main(bin_path, meta_path, out_path, quick=False):
    meta = json.load(open(meta_path))
    raw = open(bin_path, "rb").read()
    assert hashlib.sha256(raw).hexdigest() == meta["sha256"], "eids file digest mismatch"
    eids = torch.from_numpy(np.frombuffer(raw, dtype="<i2").reshape(meta["shape"]).astype(np.int32))    # [S, L, 16, k]
    S, L = eids.shape[:2]
    steps = list(range(4)) if quick else list(range(min(S, 16)))
    half = len(steps) // 2
    select_steps, eval_steps = steps[:half], steps[half:]
    g = torch.Generator().manual_seed(7)
    W = {}
    for i, (name, (N, K)) in enumerate(SHAPES.items()):
        W[name] = dict(N=N, K=K,
                       packed=torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, generator=g).to(dev),
                       scales=(torch.rand(E, N, K // 32, generator=g) * 0.01 + 1e-3).to(torch.float16).to(dev),
                       x=(torch.randn(128, K, generator=g) / 4).to(torch.bfloat16).to(dev))
    layers = {s: [eids[s, li].reshape(-1).to(dev) for li in range(L)] for s in steps}

    def served(el):
        def f():
            for ids in el:
                for name in ("gate_up", "down"):
                    w = W[name]
                    xq, xs = quant_x_rows(w["x"])
                    gemv_int4_b32(xq, xs, w["packed"], w["scales"], ids, w["N"], w["K"])
        return f

    def k19(el, plan):
        bn, kc, wp, st = plan
        wg, wd = W["gate_up"], W["down"]

        def f():
            for ids in el:
                row0, rows, grp, order, _ = build_group_tiles_fused(ids, E, 16)
                gemm_int4_b32_grouped_smallm(wg["x"], wg["packed"], wg["scales"], row0, rows, grp, order,
                                             block_n=bn, kc=kc, warps=wp, stages=st)
                gemm_int4_b32_grouped_smallm(wd["x"], wd["packed"], wd["scales"], row0, rows, grp, None,
                                             block_n=bn, kc=kc, warps=wp, stages=st)
        return f

    def tiles(el):
        def f():
            for ids in el:
                build_group_tiles_fused(ids, E, 16)
        return f

    rec = {"device": torch.cuda.get_device_name(), "sm": torch.cuda.get_device_properties(dev).multi_processor_count,
           "torch": torch.__version__, "steps": steps, "select_steps": select_steps, "eval_steps": eval_steps,
           "grid": QUICK if quick else GRID, "default_plan": DEFAULT}
    t0 = time.time()
    gbps = copy_gbps()
    bpe = sum(bytes_per_expert(N, K) for N, K in SHAPES.values())
    floor = {s: sum(torch.unique(ids).numel() for ids in layers[s]) * bpe / (gbps * 1e9) * 1e3 for s in steps}
    rec["copy_gbps"] = gbps
    rec["floor_ms"] = {str(s): floor[s] for s in steps}
    rec["served_ms"] = {str(s): graph_ms(served(layers[s])) for s in steps}
    rec["tiles_ms"] = {str(s): graph_ms(tiles(layers[s])) for s in steps}
    print(f"copy {gbps:.0f} GB/s; served median {statistics.median(rec['served_ms'].values()):.3f} ms/step; "
          f"tiles {statistics.median(rec['tiles_ms'].values()):.3f}; floor {statistics.median(floor.values()):.3f}", flush=True)

    # correctness reference: the default plan's outputs on step 0, layer 0, and an fp32 oracle on 8 rows
    ids0 = layers[steps[0]][0]
    row0, rows, grp, order, _ = build_group_tiles_fused(ids0, E, 16)
    wg = W["gate_up"]

    def k19_gu(plan):
        bn, kc, wp, st = plan
        return gemm_int4_b32_grouped_smallm(wg["x"], wg["packed"], wg["scales"], row0, rows, grp, order,
                                            block_n=bn, kc=kc, warps=wp, stages=st).float()
    ref_default = k19_gu(DEFAULT)
    sorted_ids = ids0.index_select(0, order)
    oracle = torch.stack([wg["x"][int(order[i])].float().cpu() @ dequant_int4_ref(
        wg["packed"][int(sorted_ids[i])].cpu(), wg["scales"][int(sorted_ids[i])].cpu(), wg["N"], wg["K"]).float().t()
        for i in range(8)])

    grid = QUICK if quick else GRID
    plans = list(itertools.product(grid["block_n"], grid["kc"], grid["warps"], grid["stages"]))
    rec["plans"] = []
    for plan in plans:
        row = {"plan": dict(zip(("block_n", "kc", "warps", "stages"), plan))}
        try:
            out = k19_gu(plan)
            row["rel_vs_default"] = float((out - ref_default).abs().max() / ref_default.abs().max())
            row["rel_vs_oracle"] = float((out[:8].cpu() - oracle).abs().max() / oracle.abs().max())
            row["select_ms"] = {str(s): graph_ms(k19(layers[s], plan)) for s in select_steps}
            row["select_median"] = statistics.median(row["select_ms"].values())
        except Exception as e:                                   # a plan that does not compile/launch is recorded
            row["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        rec["plans"].append(row)
        print(json.dumps(row), flush=True)
    ok = [r for r in rec["plans"] if "select_median" in r
          and r["rel_vs_oracle"] <= PLAN_TOL and r["rel_vs_default"] <= PLAN_TOL]
    assert ok, "no plan ran within tolerance"
    best = min(ok, key=lambda r: r["select_median"])
    bplan = tuple(best["plan"][k] for k in ("block_n", "kc", "warps", "stages"))
    rec["best_plan"] = best["plan"]
    ev = {}
    for name, plan in (("best", bplan), ("default", DEFAULT)):
        ev[name] = {str(s): graph_ms(k19(layers[s], plan)) for s in eval_steps}
    ev["served"] = {str(s): graph_ms(served(layers[s])) for s in eval_steps}
    ev["tiles"] = {str(s): rec["tiles_ms"][str(s)] for s in eval_steps}
    ev["floor"] = {str(s): floor[s] for s in eval_steps}
    med = {k: statistics.median(v.values()) for k, v in ev.items()}
    rec["eval"] = ev
    rec["eval_median"] = med
    rec["ratio_best_over_served"] = med["best"] / med["served"]
    rec["ratio_default_over_served"] = med["default"] / med["served"]
    rec["efficiency"] = {k: med["floor"] / med[k] for k in ("best", "default", "served")}
    rec["wall_s"] = round(time.time() - t0, 1)
    rec["verdict"], rec["verdict_reason"] = verdict(rec)
    json.dump(rec, open(out_path, "w"), indent=1)
    print(f"K20_VERDICT {rec['verdict']}: {rec['verdict_reason']}", flush=True)
    print(f"K20 EVAL best {bplan} {med['best']:.3f} ms/step, default {med['default']:.3f}, served {med['served']:.3f}, "
          f"tiles {med['tiles']:.3f}, floor {med['floor']:.3f}; best/served {rec['ratio_best_over_served']:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
        sys.exit(0)
    args = [a for a in sys.argv[1:] if a != "--quick"]
    sys.exit(main(*args, quick="--quick" in sys.argv))
