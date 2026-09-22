"""Lane K18 bench (``PREREG-k18-grouped-expert-gemv.md``): the grouped split-K GEMV against the served one.

    python k18_bench.py eids_b16.int16.bin eids_b16.int16.json k18_rows.json

P2 -- REPLAY of recorded routing (experts4bit-qlora P60's ids: [steps, 48 layers, 16 rows, 8], raw little-endian int16,
shape and sha256 in the json): synthetic int4-b32 stores for 128 experts at Qwen3-30B-A3B's expert shapes (gate_up N1536
K2048, down N2048 K768); for each recorded step its 96 calls (48 layers x gate_up + down, R = 128) captured into one CUDA
graph, 20 replays, median; arms served (`gemv_int4_b32`), grouped (`gemv_int4_b32_grouped`, mt=4), dedup (one row per
distinct expert: P60's ceiling). Every step's grouped output is checked bitwise against served on its first layer.

P3 -- small R: the six K17 shapes at R in {8, 16} (E=8, ids cycling), served vs grouped, K14's graph timer.
"""
from __future__ import annotations

import hashlib
import json
import statistics
import sys

import numpy as np
import torch

from int4_b32 import _plan, _sm_count, gemv_int4_b32, gemv_int4_b32_grouped, quant_x_rows

dev = "cuda"
SHAPES = {"gate_up": (1536, 2048), "down": (2048, 768)}
K17_SHAPES = [("expert_gate_up", 1536, 2048), ("expert_down", 2048, 768), ("attn_q", 4096, 2048), ("attn_kv", 512, 2048),
              ("attn_o", 2048, 4096), ("attn_qkv_fused", 5120, 2048)]


def store(E, N, K, seed):
    g = torch.Generator().manual_seed(seed)
    packed = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, generator=g).to(dev)
    scales = (torch.rand(E, N, K // 32, generator=g) * 0.01 + 1e-3).to(torch.float16).to(dev)
    return packed, scales


def graph_ms(fn, iters=20, warm=3):
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
    return statistics.median(ts)


def main(bin_path, meta_path, out_path):
    meta = json.load(open(meta_path))
    raw = open(bin_path, "rb").read()
    assert hashlib.sha256(raw).hexdigest() == meta["sha256"], "eids file digest mismatch"
    eids = torch.from_numpy(np.frombuffer(raw, dtype="<i2").reshape(meta["shape"]).astype(np.int32))    # [S, L, 16, k]
    S, L = eids.shape[:2]
    sm = _sm_count(dev)
    W = {}
    for i, (name, (N, K)) in enumerate(SHAPES.items()):
        packed, scales = store(128, N, K, seed=10 + i)
        x = (torch.randn(128, K, generator=torch.Generator().manual_seed(20 + i)) / 4).to(torch.bfloat16).to(dev)
        xq, xs = quant_x_rows(x)
        sk = max(_plan(N, K, r, sm)[2] for r in (1, 2, 4, 8, 16, 32, 64, 128))
        W[name] = dict(N=N, K=K, packed=packed, scales=scales, xq=xq, xs=xs,
                       part=torch.empty(sk * 128, N, dtype=torch.float32, device=dev),
                       out=torch.empty(128, N, dtype=torch.bfloat16, device=dev), sk=sk)

    def step(el, fn_kind):
        def f():
            for e in el:
                R = e.numel()
                for name in ("gate_up", "down"):
                    w = W[name]
                    kw = dict(part=w["part"][: w["sk"] * R], out=w["out"][:R])
                    if fn_kind == "grouped":
                        gemv_int4_b32_grouped(w["xq"][:R], w["xs"][:R], w["packed"], w["scales"], e, w["N"], w["K"], **kw)
                    else:
                        gemv_int4_b32(w["xq"][:R], w["xs"][:R], w["packed"], w["scales"], e, w["N"], w["K"], fused_reduce=False, **kw)
        return f

    rows = {"device": torch.cuda.get_device_name(), "sm_count": sm, "torch": torch.__version__, "steps": S, "layers": L,
            "eids_sha256": meta["sha256"], "plans": {n: W[n]["sk"] for n in W}, "p1_replay_mismatches": 0, "replay": {}}
    per = {"served": [], "grouped": [], "dedup": []}
    for s in range(S):
        el = [eids[s, li].reshape(-1).to(dev) for li in range(L)]
        # P1 on real routing: grouped == served on this step's first layer, both projections
        for name in ("gate_up", "down"):
            w = W[name]
            a = gemv_int4_b32(w["xq"], w["xs"], w["packed"], w["scales"], el[0], w["N"], w["K"], fused_reduce=False)
            b = gemv_int4_b32_grouped(w["xq"], w["xs"], w["packed"], w["scales"], el[0], w["N"], w["K"])
            if not torch.equal(a, b):
                rows["p1_replay_mismatches"] += 1
        per["served"].append(graph_ms(step(el, "served")))
        per["grouped"].append(graph_ms(step(el, "grouped")))
        per["dedup"].append(graph_ms(step([torch.unique(e) for e in el], "served")))
    for k, v in per.items():
        rows["replay"][k] = {"step_ms_median": statistics.median(v), "step_ms_mean": statistics.mean(v), "min": min(v), "max": max(v)}
    rv = rows["replay"]
    print(f"P2 replay ({S} steps): served {rv['served']['step_ms_median']:.3f}  grouped {rv['grouped']['step_ms_median']:.3f}  "
          f"dedup {rv['dedup']['step_ms_median']:.3f} ms/step  | saving {rv['served']['step_ms_median'] - rv['grouped']['step_ms_median']:+.3f}  "
          f"| P1 mismatches on replay: {rows['p1_replay_mismatches']}", flush=True)
    small = []
    for name, N, K in K17_SHAPES:
        packed, scales = store(8, N, K, seed=3)
        for R in (8, 16):
            x = (torch.randn(R, K, generator=torch.Generator().manual_seed(5)) / 4).to(torch.bfloat16).to(dev)
            xq, xs = quant_x_rows(x)
            e = (torch.arange(R) % 8).to(torch.int32).to(dev)
            sk = _plan(N, K, R, sm)[2]
            part = torch.empty(sk * R, N, dtype=torch.float32, device=dev)
            out = torch.empty(R, N, dtype=torch.bfloat16, device=dev)
            eq = torch.equal(gemv_int4_b32(xq, xs, packed, scales, e, N, K, fused_reduce=False),
                             gemv_int4_b32_grouped(xq, xs, packed, scales, e, N, K))
            t0 = graph_ms(lambda: gemv_int4_b32(xq, xs, packed, scales, e, N, K, part=part, out=out, fused_reduce=False), iters=200)
            t1 = graph_ms(lambda: gemv_int4_b32_grouped(xq, xs, packed, scales, e, N, K, part=part, out=out), iters=200)
            small.append({"shape": name, "N": N, "K": K, "R": R, "p1_equal": eq, "served_us": t0 * 1e3, "grouped_us": t1 * 1e3, "ratio": t1 / t0})
            print(f"   P3 {name:16s} R={R:2d} served {t0*1e3:7.2f} us  grouped {t1*1e3:7.2f} us  ratio {t1/t0:.3f}  equal {eq}", flush=True)
    rows["small_r"] = small
    json.dump(rows, open(out_path, "w"), indent=1)
    bad = rows["p1_replay_mismatches"] + sum(1 for r in small if not r["p1_equal"])
    print(f"rows written; P1 failures: {bad}", flush=True)
    return 0 if bad == 0 else 3


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
