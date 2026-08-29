"""Blackwell re-tune of the M-tile grouped GEMM at the B=16 census cells.

The v6 config rule (BLOCK_N=128, warps=4, stages=3, variant 1) was tuned
on an A5000 (sm_86). Two rejected knobs were rejected for sm_86 reasons
that do not hold on sm_120: GROUPS=2 (BLOCK_K=128, "SMEM blowout" at
~100 KB; Blackwell has 228 KB) and variant 3 bf16 MMA ("slower
everywhere" on A5000's weak HMMA). This sweeps the matrix on a 5090.

Correctness gates every cell: variant-1 GROUPS=1 cells must be BITWISE
equal to the incumbent's output (same math order); GROUPS=2 and
variant-3 cells (different accumulation grouping / documented looser
P-fid) gate on max|d| <= max|ref| * 2^-7 (the K6 relative frame).
A cell that fails its gate is recorded and EXCLUDED from the podium.

Real routing draws at B=16 top-8 over E=128 (per-shape, 6 draws, cells
timed round-robin across draws).
"""
import json, statistics, time, torch
from nf4_grouped import gemm_4bit_grouped
from nf4_pack_ref import quantize_pack_nf4

dev = "cuda"
torch.manual_seed(0)
E, TOPK, B_ = 128, 8, 16
R = B_ * TOPK
SHAPES = [("gate_up", 1536, 2048), ("down", 2048, 768)]
DRAWS, INNER = 6, 15

def pack(n, k):
    w = torch.randn(E * n, k) * 0.02
    p, a = quantize_pack_nf4(w)
    return (p.view(E, n, k // 2).to(dev).contiguous(),
            a.view(E, n, -1).to(dev).float().contiguous())

CELLS = []
for variant in (1, 3):
    for groups in (1, 2):
        for bn in (64, 128, 256):
            for warps in (4, 8):
                for stages in (2, 3, 4):
                    CELLS.append((variant, groups, bn, warps, stages))

out = {"sm": torch.cuda.get_device_capability(), "cells": {}, "fails": []}
for tag, N, K in SHAPES:
    if K % 128 and any(c[1] == 2 for c in CELLS):
        pass  # groups=2 needs K%128==0; guarded per cell below
    Bt, At = pack(N, K)
    draws = []
    for d in range(DRAWS):
        g = torch.Generator().manual_seed(500 + d)
        ids = torch.stack([torch.randperm(E, generator=g)[:TOPK]
                           for _ in range(B_)]).reshape(-1)
        sids, order = torch.sort(ids)
        uniq, counts = torch.unique_consecutive(sids, return_counts=True)
        x = (torch.randn(R, K, generator=g) * 0.1).to(dev, torch.bfloat16)
        xs = x.index_select(0, order.to(dev)).contiguous()
        draws.append((xs, counts.tolist(), uniq.to(dev).to(torch.int32)))
    def run(cell, dr):
        v, gp, bn, w, st = cell
        xs, sizes, eids = dr
        return gemm_4bit_grouped(xs, Bt, At, sizes, eids,
                                 prefill_variant=v, prefill_groups=gp,
                                 prefill_config=(bn, w, st))
    refs = [gemm_4bit_grouped(d[0], Bt, At, d[1], d[2]) for d in draws]
    res = {}
    for cell in CELLS:
        v, gp, bn, w, st = cell
        if gp == 2 and K % 128:
            continue
        key = f"v{v}g{gp}bn{bn}w{w}s{st}"
        try:
            outs = [run(cell, d) for d in draws]
            ok = True
            for o, r in zip(outs, refs):
                if v == 1 and gp == 1:
                    if not torch.equal(o, r): ok = False; break
                else:
                    dmax = (o.float() - r.float()).abs().max()
                    if dmax > r.float().abs().max() * 2 ** -7: ok = False; break
            if not ok:
                out["fails"].append({"shape": tag, "cell": key, "why": "numeric"})
                continue
            ts = []
            for d in draws:
                for _ in range(3): run(cell, d)
                torch.cuda.synchronize(); a = time.perf_counter()
                for _ in range(INNER): run(cell, d)
                torch.cuda.synchronize()
                ts.append((time.perf_counter() - a) * 1000 / INNER)
            res[key] = statistics.median(ts)
        except Exception as e:
            out["fails"].append({"shape": tag, "cell": key,
                                 "why": repr(e)[:90]})
    # incumbent timing (rule config resolved internally)
    ts = []
    for d in draws:
        for _ in range(3): gemm_4bit_grouped(d[0], Bt, At, d[1], d[2])
        torch.cuda.synchronize(); a = time.perf_counter()
        for _ in range(INNER): gemm_4bit_grouped(d[0], Bt, At, d[1], d[2])
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - a) * 1000 / INNER)
    inc = statistics.median(ts)
    top = sorted(res.items(), key=lambda kv: kv[1])[:6]
    out["cells"][tag] = {"incumbent_ms": inc, "swept": len(res),
                         "top": top}
    print(f"=== {tag} (N={N} K={K}) incumbent {inc:.4f} ms, "
          f"{len(res)} cells clean, {len([f for f in out['fails'] if f['shape']==tag])} failed ===")
    for k, ms in top:
        print(f"  {k:20s} {ms:.4f} ms  {inc/ms:6.3f}x vs incumbent")
json.dump(out, open("./blackwell_tune.json", "w"), indent=1)
