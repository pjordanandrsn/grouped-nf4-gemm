"""Lane K24 bench (``PREREG-k24-gptoss-per-layer.md``): K22's bench, re-read with each layer's OWN synthetic stores and
K21's masked-tail plans, on gpt-oss-20b's routing recorded by K22.

    python k24_bench.py eids_b16.pt k24_rows.json --census-nf4-ms <ms/step> [--quick]
    python k24_bench.py --self-test

Changes from ``k22_bench.py``, both from K22's VOID read:
- **Per-layer stores.** K22 shared one weight set across all 24 layers. That let L2 serve experts that consecutive
  layers both touch, and its served NF4 kernel read 17 % under the in-model census. Here every layer has its own NF4
  and MXFP4 stores, generated on the GPU: about 21 GB on a 32 GB card.
- **The masked K tail.** K21 now takes KC 128 / 256 on K = 2880 (#425), so the grid is BLOCK_N {32, 64, 128} × KC
  {64, 128, 256} × warps {4, 8} × stages {2, 3, 4} = 54 plans.

The arms, the select / eval split, the instrument (served ``_gemm_nf4_grouped`` within 15 % of a same-box census) and
the rule are K22's.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import statistics
import sys
import time

import torch

E, H, I_ = 32, 2880, 2880
SHAPES = {"gate_up": (2 * I_, H), "down": (H, I_)}
GRID = {"block_n": (32, 64, 128), "kc": (64, 128, 256), "warps": (4, 8), "stages": (2, 3, 4)}
QUICK = {"block_n": (32,), "kc": (64, 256), "warps": (4,), "stages": (2,)}
DEFAULT = (32, 256, 4, 2)                # K21's default plan; KC 256 masks K 2880's 64-wide tail (#425)
ITERS = 20
INSTRUMENT_BAND = 0.15
PLAN_TOL = 0.02
PROMISING, MARGINAL = 0.77, 0.90
NF4_KERNEL = "_gemm_nf4_grouped"


def mxfp4_bytes(N, K):
    return N * (K // 2) + N * (K // 32)                 # e2m1 nibbles + e8m0 bytes


def verdict(rec):
    census = rec.get("census_nf4_ms")
    bench = rec.get("served_nf4_kernel_ms")
    if not census or not bench:
        return "VOID", f"no instrument: census {census!r}, bench {bench!r}"
    if abs(bench / census - 1) > INSTRUMENT_BAND:
        return "VOID", f"served _gemm_nf4_grouped {bench:.3f} ms/step is not within {INSTRUMENT_BAND:.0%} of the census {census:.3f}"
    r = rec["ratio_best_over_served"]
    if r <= PROMISING:
        return "PROMISING", f"best/served {r:.3f} <= {PROMISING}"
    if r <= MARGINAL:
        return "MARGINAL", f"{PROMISING} < best/served {r:.3f} <= {MARGINAL}"
    return "NO", f"best/served {r:.3f} > {MARGINAL}"


def self_test():
    base = {"census_nf4_ms": 6.0, "served_nf4_kernel_ms": 6.3, "ratio_best_over_served": 0.70}
    assert verdict(base)[0] == "PROMISING"
    assert verdict(dict(base, ratio_best_over_served=0.85))[0] == "MARGINAL"
    assert verdict(dict(base, ratio_best_over_served=0.95))[0] == "NO"
    assert verdict(dict(base, served_nf4_kernel_ms=7.0))[0] == "VOID"            # 16.7 % off the census
    assert verdict(dict(base, census_nf4_ms=None))[0] == "VOID"
    print("k24 verdict self-test OK (5 cases)")


def main(eids_path, out_path, census_nf4_ms, quick=False):
    from int4_b32 import build_group_tiles_fused, quant_x_rows
    from mxfp4_grouped import gemm_mxfp4_grouped_smallm, gemv_mxfp4_b32
    from mxfp4_pack_ref import dequant_mxfp4
    from nf4_grouped import gemm_4bit_grouped_captured
    dev = "cuda"
    eids = torch.load(eids_path, map_location="cpu")["eids"].to(torch.int32)          # [S, L, B, k]
    S, L, B, k = eids.shape
    assert (L, k) == (24, 4), f"not gpt-oss-20b routing: {tuple(eids.shape)}"
    if quick:                                 # rehearsal only (a 12 GB card cannot hold 24 layers' stores): 4 layers
        eids, L = eids[:, :4], 4
    steps = list(range(min(S, 4 if quick else 16)))
    half = len(steps) // 2
    sel, evs = steps[:half], steps[half:]
    g = torch.Generator(device=dev).manual_seed(7)
    WL = []                                   # one store set PER LAYER, as the model has (K22's VOID)
    for _li in range(L):
        Wl = {}
        for name, (N, K) in SHAPES.items():
            Wl[name] = dict(N=N, K=K,
                            mx_b=torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, generator=g, device=dev),
                            mx_s=torch.randint(118, 126, (E, N, K // 32), dtype=torch.uint8, generator=g, device=dev),
                            nf_b=torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, generator=g, device=dev),
                            nf_a=torch.rand(E, N, K // 64, generator=g, device=dev) * 0.05 + 0.01)
        WL.append(Wl)
    X = {name: (torch.randn(B * k, K, generator=g, device=dev) / 4).to(torch.bfloat16) for name, (N, K) in SHAPES.items()}
    W = WL[0]                                 # layer 0's stores: the numerics check below
    for name in SHAPES:
        W[name]["x"] = X[name]
    layers = {s: [eids[s, li].reshape(-1).to(dev) for li in range(L)] for s in steps}

    def served(el):
        def f():
            for li, ids in enumerate(el):
                row0, rows, grp, order, _ = build_group_tiles_fused(ids, E, 16)
                for name in ("gate_up", "down"):
                    w = WL[li][name]
                    xs = X[name].index_select(0, order)
                    gemm_4bit_grouped_captured(xs, w["nf_b"], w["nf_a"], row0, rows, grp, 16)
        return f

    def gemv(el):
        def f():
            for li, ids in enumerate(el):
                for name in ("gate_up", "down"):
                    w = WL[li][name]
                    xq, xsc = quant_x_rows(X[name])
                    gemv_mxfp4_b32(xq, xsc, w["mx_b"], w["mx_s"], ids, w["N"], w["K"])
        return f

    def k21(el, plan):
        bn, kc, wp, st = plan

        def f():
            for li, ids in enumerate(el):
                row0, rows, grp, order, _ = build_group_tiles_fused(ids, E, 16)
                wg, wd = WL[li]["gate_up"], WL[li]["down"]
                gemm_mxfp4_grouped_smallm(X["gate_up"], wg["mx_b"], wg["mx_s"], row0, rows, grp, order,
                                          block_n=bn, kc=kc, warps=wp, stages=st)
                gemm_mxfp4_grouped_smallm(X["down"], wd["mx_b"], wd["mx_s"], row0, rows, grp, None,
                                          block_n=bn, kc=kc, warps=wp, stages=st)
        return f

    def capture(fn, warm=3):
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            fn()
        torch.cuda.synchronize()
        return gr

    def graph_ms(fn):
        gr = capture(fn)
        ts = []
        for _ in range(ITERS):
            a, b = torch.cuda.Event(True), torch.cuda.Event(True)
            a.record()
            gr.replay()
            b.record()
            b.synchronize()
            ts.append(a.elapsed_time(b))
        del gr
        torch.cuda.empty_cache()
        return statistics.median(ts)

    def kernel_ms(fn, name, reps=4):
        gr = capture(fn)
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(reps):
                gr.replay()
            torch.cuda.synchronize()
        tot = sum(ev.self_device_time_total for ev in prof.key_averages() if ev.key == name) / 1e3
        del gr
        torch.cuda.empty_cache()
        return tot / reps

    t0 = time.time()
    def copy_gbps():
        src = torch.empty(512 << 20, dtype=torch.uint8, device=dev)
        dst = torch.empty_like(src)
        return 2 * src.numel() / (graph_ms(lambda: dst.copy_(src)) / 1e3) / 1e9
    gbps = copy_gbps()
    bpe = sum(mxfp4_bytes(N, K) for N, K in SHAPES.values())
    floor = {s: sum(torch.unique(ids).numel() for ids in layers[s]) * bpe / (gbps * 1e9) * 1e3 for s in steps}
    rec = {"device": torch.cuda.get_device_name(), "sm": torch.cuda.get_device_properties(dev).multi_processor_count,
           "torch": torch.__version__, "eids_shape": list(eids.shape),
           "eids_sha256": hashlib.sha256(eids.numpy().tobytes()).hexdigest(), "steps": steps, "select_steps": sel,
           "eval_steps": evs, "grid": QUICK if quick else GRID, "default_plan": DEFAULT, "copy_gbps": gbps,
           "census_nf4_ms": census_nf4_ms, "layers": L, "quick": quick}
    rec["served_nf4_kernel_ms"] = statistics.median(kernel_ms(served(layers[s]), NF4_KERNEL) for s in evs)
    print(f"copy {gbps:.0f} GB/s; served {NF4_KERNEL} {rec['served_nf4_kernel_ms']:.3f} ms/step "
          f"(census {census_nf4_ms})", flush=True)

    # numerics: K21 at each plan against an fp32 MXFP4 dequant oracle on 8 rows of step 0, layer 0, gate_up
    ids0 = layers[steps[0]][0]
    row0, rows, grp, order, _ = build_group_tiles_fused(ids0, E, 16)
    wg = W["gate_up"]
    sids = ids0.index_select(0, order)
    oracle = torch.stack([wg["x"][int(order[i])].float().cpu() @ dequant_mxfp4(
        wg["mx_b"][int(sids[i])].cpu().reshape(wg["N"], wg["K"] // 32, 16), wg["mx_s"][int(sids[i])].cpu()).float().t()
        for i in range(8)])

    def k21_gu(plan):
        bn, kc, wp, st = plan
        return gemm_mxfp4_grouped_smallm(wg["x"], wg["mx_b"], wg["mx_s"], row0, rows, grp, order,
                                         block_n=bn, kc=kc, warps=wp, stages=st).float()
    ref_default = k21_gu(DEFAULT)
    grid = QUICK if quick else GRID
    rec["plans"] = []
    for plan in itertools.product(grid["block_n"], grid["kc"], grid["warps"], grid["stages"]):
        row = {"plan": dict(zip(("block_n", "kc", "warps", "stages"), plan))}
        try:
            out = k21_gu(plan)
            row["rel_vs_default"] = float((out - ref_default).abs().max() / ref_default.abs().max())
            row["rel_vs_oracle"] = float((out[:8].cpu() - oracle).abs().max() / oracle.abs().max())
            row["select_ms"] = {str(s): graph_ms(k21(layers[s], plan)) for s in sel}
            row["select_median"] = statistics.median(row["select_ms"].values())
        except Exception as e:                                       # recorded, never dropped
            row["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        rec["plans"].append(row)
        print(json.dumps(row), flush=True)
    ok = [r for r in rec["plans"] if "select_median" in r and r["rel_vs_oracle"] <= PLAN_TOL and r["rel_vs_default"] <= PLAN_TOL]
    assert ok, "no plan ran within tolerance"
    best = min(ok, key=lambda r: r["select_median"])
    bplan = tuple(best["plan"][x] for x in ("block_n", "kc", "warps", "stages"))
    rec["best_plan"] = best["plan"]
    ev = {"best": {str(s): graph_ms(k21(layers[s], bplan)) for s in evs},
          "default": {str(s): graph_ms(k21(layers[s], DEFAULT)) for s in evs},
          "served": {str(s): graph_ms(served(layers[s])) for s in evs},
          "gemv": {str(s): graph_ms(gemv(layers[s])) for s in evs},
          "floor": {str(s): floor[s] for s in evs}}
    med = {kk: statistics.median(v.values()) for kk, v in ev.items()}
    rec["eval"], rec["eval_median"] = ev, med
    rec["ratio_best_over_served"] = med["best"] / med["served"]
    rec["ratio_gemv_over_served"] = med["gemv"] / med["served"]
    rec["efficiency"] = {kk: med["floor"] / med[kk] for kk in ("best", "default", "served", "gemv")}
    rec["wall_s"] = round(time.time() - t0, 1)
    rec["verdict"], rec["verdict_reason"] = verdict(rec)
    json.dump(rec, open(out_path, "w"), indent=1)
    print(f"K24 EVAL best {bplan} {med['best']:.3f} ms/step, default {med['default']:.3f}, served {med['served']:.3f}, "
          f"gemv {med['gemv']:.3f}, floor {med['floor']:.3f}; best/served {rec['ratio_best_over_served']:.3f}", flush=True)
    print(f"K24_VERDICT {rec['verdict']}: {rec['verdict_reason']}", flush=True)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("eids", nargs="?")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--census-nf4-ms", type=float)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        sys.exit(0)
    if not a.eids or not a.out:
        ap.error("eids and out are required")
    sys.exit(main(a.eids, a.out, a.census_nf4_ms, quick=a.quick))
