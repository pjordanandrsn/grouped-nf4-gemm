"""Lane K34 bench (``PREREG-k34-k16-wide-plan-census.md``): an EXPLORATORY plan census of K16's 32- and 64-row tiles at
Qwen3-30B-A3B's attention shapes on an RTX 5090. Its best plan licenses nothing; it can only name a candidate.

experts4bit-qlora P124 read ``E4B_ATTN_INT4_WIDE`` DEFAULT_ON: the attention projections of a 32- or 64-row decode step
on ``gemm_int4_b32_smallm`` (#522's ``block_m=``) instead of cuBLAS on a cached bf16 copy. At the shipped plan (BLOCK_N
64, KC 128, split-K 4, 4 warps, 2 stages) the 64-row tile still cost 0.74 of the cuBLAS time it replaced, the 32-row tile
0.59. The int4 bytes are a quarter of the bf16 copy's. This census asks whether another plan does markedly better.

**Stores.** Per layer its own synthetic int4-b32 store (random packed bytes, fp16 scales), seeded, at the two projections
P124 served after the q/k/v fusion: ``qkv`` (N 5120, K 2048) and ``o`` (N 2048, K 4096), 48 layers each, so a graph
streams every layer's bytes and nothing stays in L2 (K24's lesson).

**Graphs.** One per (projection, tile, arm or plan): 48 launches, one per layer, of ``gemm_int4_b32_smallm`` on the tile's
rows (32 or 64) with a preallocated workspace sized for 64 rows (shared by the layers, as experts4bit-qlora shares it).
- ``shipped``, ``shipped2``: the shipped plan; the repeat is the instrument;
- ``best``: the plan this run selected (``PLANS``: BLOCK_N {32, 64, 128} x KC {128, 256} x split-K {1, 2, 4, 8} x warps
  {4, 8}, 2 stages: 48 plans);
- ``bf16``: cuBLAS on each layer's dequantised bf16 weight, the path the route replaced (reported).

**Selection** (K20's discipline): every plan is captured and timed for ``SELECT_ROUNDS``; a plan that fails to compile or
launch, or whose layer-0 output is further than one bf16 ulp (``2^-7`` of the output's scale) from the fp32 dequant
reference, is not selectable. The fastest median wins; the arms are then timed afresh for ``ROUNDS`` (order reversed
every round, CUDA events). Within one ulp is a sanity check, not a quality licence: a different split-K or KC reorders
the fp32 sums. **Floor:** the graph's int4 bytes / the box's own copy bandwidth (a 512 MiB copy).

    python k34_bench.py out.json
    python k34_bench.py --self-test
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

SHAPES = {"qkv": (5120, 2048), "o": (2048, 4096)}
PROJ = ("qkv", "o")
LAYERS = 48
TILES = (32, 64)
SHIPPED = (64, 128, 4, 4, 2)                       # block_n, kc, sk, warps, stages: plan_smallm's default + the wrapper's
PLANS = [(bn, kc, sk, w, 2) for bn in (32, 64, 128) for kc in (128, 256) for sk in (1, 2, 4, 8) for w in (4, 8)]
ARMS = ("shipped", "shipped2", "best", "bf16")
WARM, ROUNDS, SELECT_ROUNDS = 20, 200, 40
SELF_LO, SELF_HI = 0.98, 1.02
CANDIDATE_MAX = 0.90                               # best / shipped at or below this on both projections names a candidate
TOL = 2.0 ** -7
CARD = "5090"


def int4_bytes(n, k):
    """Bytes one launch streams: packed nibbles + fp16 scales."""
    return n * (k // 2 + (k // 32) * 2)


def verdict(r):
    """The registered rule (exploratory). The first that applies:
    VOID       the card is not the registered RTX 5090; the copy floor or a timing is missing; no plan was selectable
               for a (projection, tile).
    NOISY      shipped2 / shipped outside [0.98, 1.02] for any (projection, tile).
    CANDIDATE  for a tile, best / shipped <= 0.90 on both projections: that tile's plan is named for a confirmatory read.
    NONE       otherwise.
    A CANDIDATE licenses nothing. Any plan that would move a default gets its own registered confirmatory read in
    experts4bit-qlora (P124 Amendment 1's interleaved blocks on SC2e's served step, P110's teacher-forced bar and a mutant)."""
    if CARD not in str(r.get("device", "")):
        return "VOID", f"card {r.get('device')!r} is not the registered RTX {CARD}"
    if not r.get("copy_gbps"):
        return "VOID", "the copy floor was not measured"
    ms = r.get("median_ms") or {}
    keys = [f"{p}/{t}" for t in TILES for p in PROJ]
    missing = [f"{k}/{a}" for k in keys for a in ARMS if not (ms.get(k) or {}).get(a)]
    if missing:
        return "VOID", f"timings missing: {missing}"
    if not all((r.get("plan") or {}).get(k) for k in keys):
        return "VOID", "no plan was selectable for a (projection, tile)"
    noisy = {k: ms[k]["shipped2"] / ms[k]["shipped"] for k in keys}
    if any(not SELF_LO <= v <= SELF_HI for v in noisy.values()):
        return "NOISY", f"shipped2 / shipped = {noisy}, outside [{SELF_LO}, {SELF_HI}]"
    ratio = {k: ms[k]["best"] / ms[k]["shipped"] for k in keys}
    cand = [t for t in TILES if all(ratio[f"{p}/{t}"] <= CANDIDATE_MAX for p in PROJ)]
    txt = "best / shipped " + ", ".join(f"{k} x{ratio[k]:.3f} ({r['plan'][k]})" for k in keys)
    if cand:
        return "CANDIDATE", txt + f" -- candidate tile(s) {cand}: a confirmatory read in experts4bit-qlora, nothing moves"
    return "NONE", txt + " -- the shipped plan stays"


def self_test():
    def rec(ship=(1.40, 0.70), ship2=None, best=(1.00, 0.55), card="NVIDIA GeForce RTX 5090", gbps=1570.0, plan=True,
            drop=None, tile_best=None):
        ship2 = ship2 or ship
        ms, pl = {}, {}
        for t in TILES:
            for i, p in enumerate(PROJ):
                b = (tile_best or {}).get(t, best)[i] if tile_best else best[i]
                ms[f"{p}/{t}"] = {"shipped": ship[i], "shipped2": ship2[i], "best": b, "bf16": 1.9}
                pl[f"{p}/{t}"] = [32, 256, 2, 8, 2] if plan else None
        if drop:
            ms["qkv/64"].pop(drop)
        return {"device": card, "copy_gbps": gbps, "median_ms": ms, "plan": pl}
    cases = [
        ("candidate at both tiles", verdict(rec())[0] == "CANDIDATE" and "[32, 64]" in verdict(rec())[1]),
        ("candidate at one tile", "[64]" in verdict(rec(tile_best={32: (1.35, 0.68), 64: (1.0, 0.55)}))[1]),
        ("one projection misses", verdict(rec(best=(1.0, 0.68)))[0] == "NONE"),
        ("none", verdict(rec(best=(1.35, 0.68)))[0] == "NONE"),
        ("the bar is inclusive", verdict(rec(best=(1.26, 0.63)))[0] == "CANDIDATE"),
        ("wrong card", verdict(rec(card="NVIDIA RTX A2000 12GB"))[0] == "VOID"),
        ("no floor", verdict(rec(gbps=None))[0] == "VOID"),
        ("missing arm", verdict(rec(drop="bf16"))[0] == "VOID"),
        ("no plan", verdict(rec(plan=False))[0] == "VOID"),
        ("noisy", verdict(rec(ship2=(1.45, 0.70)))[0] == "NOISY"),
        ("plan grid", len(PLANS) == 48 and SHIPPED in PLANS and len(set(PLANS)) == 48),
        ("bytes", int4_bytes(5120, 2048) == 5120 * (1024 + 128)),
    ]
    bad = [n for n, ok in cases if not ok]
    print(f"k34_bench self-test {'OK' if not bad else 'FAILED ' + str(bad)} ({len(cases)} cases)")
    return 0 if not bad else 1


def main(out_path, quick=False):
    import torch
    import triton

    from int4_pack_ref import dequant_int4_ref
    from int4_smallm import gemm_int4_b32_smallm, smallm_workspace
    dev = "cuda"
    t_start = time.time()
    rounds, select_rounds, plans = (20, 5, [SHIPPED, PLANS[0]]) if quick else (ROUNDS, SELECT_ROUNDS, PLANS)
    rec = {"device": torch.cuda.get_device_name(), "cc": list(torch.cuda.get_device_capability()),
           "sm_count": torch.cuda.get_device_properties(0).multi_processor_count, "torch": torch.__version__,
           "triton": triton.__version__, "shapes": SHAPES, "layers": LAYERS, "tiles": TILES, "shipped": SHIPPED,
           "rounds": rounds, "select_rounds": select_rounds, "plans": plans, "quick": quick,
           "median_ms": {}, "quartiles_ms": {}, "floor_ms": {}, "plan": {}, "plan_ms": {}, "plan_refused": {},
           "numerics": {}}
    g = torch.Generator(device=dev).manual_seed(34)

    src = torch.empty(512 << 20, dtype=torch.uint8, device=dev)            # the copy floor
    dst = torch.empty_like(src)
    for _ in range(3):
        dst.copy_(src)
    torch.cuda.synchronize()
    ts = []
    for _ in range(20):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        dst.copy_(src)
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b))
    rec["copy_gbps"] = 2 * src.numel() / (statistics.median(ts) * 1e6)
    del src, dst

    def time_graphs(graphs, n_rounds):
        times = {a: [] for a in graphs}
        order = list(graphs)
        for rnd in range(WARM + n_rounds):
            for a_ in (order if rnd % 2 == 0 else order[::-1]):
                s, e = torch.cuda.Event(True), torch.cuda.Event(True)
                s.record()
                graphs[a_].replay()
                e.record()
                e.synchronize()
                if rnd >= WARM:
                    times[a_].append(s.elapsed_time(e))
        return times

    def capture(fn):
        fn()                                                                # warm (compile) outside the capture
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            fn()
        torch.cuda.synchronize()
        return gr

    for proj in PROJ:
        N, K = SHAPES[proj]
        stores = [(torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, generator=g, device=dev),
                   (torch.rand(N, K // 32, generator=g, device=dev) * 0.004 + 0.002).to(torch.float16))
                  for _ in range(LAYERS)]
        x_all = (torch.randn(max(TILES), K, generator=g, device=dev) * 0.5).to(torch.bfloat16)
        w0 = dequant_int4_ref(stores[0][0].cpu(), stores[0][1].cpu(), N, K).to(torch.bfloat16)
        for tile in TILES:
            key = f"{proj}/{tile}"
            x = x_all[:tile].contiguous()
            rec["floor_ms"][key] = LAYERS * int4_bytes(N, K) / (rec["copy_gbps"] * 1e6)
            ref = (x.float().cpu() @ w0.float().t())
            scale = ref.abs().max().clamp_min(1e-6)

            def chain(plan, ws):
                bn, kc, sk, w, st = plan
                return [gemm_int4_b32_smallm(x, P, S, block_n=bn, kc=kc, sk=sk, warps=w, stages=st, workspace=ws,
                                             block_m=tile) for P, S in stores]

            # ---- plan selection: numerics on layer 0 first (one bf16 ulp of the reference), then SELECT rounds
            cand, num = {}, {}
            for plan in plans:
                bn, kc, sk, w, st = plan
                try:
                    ws = smallm_workspace(N, block_m=max(TILES), block_n=bn, sk=sk, device=dev)
                    y0 = gemm_int4_b32_smallm(x, stores[0][0], stores[0][1], block_n=bn, kc=kc, sk=sk, warps=w,
                                              stages=st, workspace=ws, block_m=tile)
                    torch.cuda.synchronize()
                    err = ((y0.float().cpu() - ref).abs().max() / scale).item()
                    num[",".join(map(str, plan))] = err
                    if err > TOL:
                        rec["plan_refused"].setdefault(key, []).append([list(plan), f"numerics {err:.2e} > {TOL:.2e}"])
                        continue
                    cand[plan] = capture(lambda p=plan, s=ws: chain(p, s))
                except Exception as exc:                                     # noqa: BLE001 -- unrunnable: not selectable
                    rec["plan_refused"].setdefault(key, []).append([list(plan), f"{type(exc).__name__}: {str(exc)[:120]}"])
            rec["numerics"][key] = num
            if not cand:
                continue
            sel = time_graphs(cand, select_rounds)
            rec["plan_ms"][key] = {",".join(map(str, p)): statistics.median(v) for p, v in sel.items()}
            best = min(sel, key=lambda p: statistics.median(sel[p]))
            rec["plan"][key] = list(best)
            del cand

            # ---- the arms, timed afresh
            ws_ship = smallm_workspace(N, block_m=max(TILES), block_n=SHIPPED[0], sk=SHIPPED[2], device=dev)
            ws_best = smallm_workspace(N, block_m=max(TILES), block_n=best[0], sk=best[2], device=dev)
            wbf = [dequant_int4_ref(P.cpu(), S.cpu(), N, K).to(torch.bfloat16).to(dev) for P, S in stores]
            graphs = {"shipped": capture(lambda: chain(SHIPPED, ws_ship)),
                      "shipped2": capture(lambda: chain(SHIPPED, ws_ship)),
                      "best": capture(lambda: chain(best, ws_best)),
                      "bf16": capture(lambda: [x @ w_.t() for w_ in wbf])}
            t = time_graphs(graphs, rounds)
            rec["median_ms"][key] = {a: statistics.median(v) for a, v in t.items()}
            rec["quartiles_ms"][key] = {a: statistics.quantiles(v, n=4) for a, v in t.items()}
            print(f"K34 {key} best {list(best)} median ms " + json.dumps({a: round(v, 4) for a, v in rec["median_ms"][key].items()})
                  + f" floor {rec['floor_ms'][key]:.4f}", flush=True)
            graphs = wbf = None                                              # release this tile's graphs and copies
            torch.cuda.empty_cache()
        stores = None
        torch.cuda.empty_cache()

    rec["wall_s"] = round(time.time() - t_start, 1)
    rec["verdict"], rec["verdict_reason"] = verdict(rec)
    json.dump(rec, open(out_path, "w"), indent=1)
    print(f"K34_VERDICT {rec['verdict']}: {rec['verdict_reason']}", flush=True)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--quick", action="store_true", help="a rehearsal: two plans, 20 rounds (never a reading)")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(self_test())
    if not a.out:
        ap.error("out is required")
    sys.exit(main(a.out, quick=a.quick))
