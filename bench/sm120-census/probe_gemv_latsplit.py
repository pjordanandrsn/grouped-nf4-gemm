"""Split the decode GEMV's ~65 us/call into host-issue vs device latency.

Same cell (T=1 top-8 singleton, both expert shapes). Three timings:
  eager     streamed back-to-back calls (the prior harness)
  graph     the SAME 40-call loop captured once and replayed -- replay
            has no host issue, so its per-call time IS device latency
  floor     warmed DRAM read probe (the retraction lesson applied)

If graph ~= eager: the kernel itself is latency-bound; fusion (fewer,
deeper kernels) is the only lever. If graph << eager: the harness was
host-bound and the certified b1d graph lane already pays only the
device part.
"""
import json, statistics, time, torch
from nf4_grouped import gemm_4bit_grouped
from nf4_pack_ref import quantize_pack_nf4

dev = "cuda"
torch.manual_seed(0)
E, TOPK = 128, 8
SHAPES = [("gate_up", 1536, 2048), ("down", 2048, 768)]
CALLS, REPS = 40, 15

big = torch.empty(int(2e9 // 4), dtype=torch.float32, device=dev)
big.sum(); torch.cuda.synchronize()          # WARM before timing
a = time.perf_counter()
for _ in range(20): big.sum()
torch.cuda.synchronize()
ROOF = (big.numel() * 4 * 20) / (time.perf_counter() - a) / 1e9
del big; torch.cuda.empty_cache()
out = {"roof_gbps": ROOF, "cells": {}}
print(f"warmed roofline {ROOF:.0f} GB/s")

for tag, N, K in SHAPES:
    W = torch.randn(E, N, K) * 0.02
    packs = [quantize_pack_nf4(W[e]) for e in range(E)]
    Bt = torch.stack([p.reshape(N, K // 2) for p, _ in packs]).to(dev).contiguous()
    At = torch.stack([a2.reshape(N, K // 64) for _, a2 in packs]).float().to(dev).contiguous()
    g = torch.Generator().manual_seed(77)
    ids = torch.randperm(E, generator=g)[:TOPK].sort().values.to(dev).to(torch.int32)
    x = (torch.randn(TOPK, K, generator=g) * 0.1).to(dev, torch.bfloat16).contiguous()
    ones = [1] * TOPK
    call = lambda: gemm_4bit_grouped(x, Bt, At, ones, ids)

    for _ in range(10): call()
    torch.cuda.synchronize()
    ts = []
    for _ in range(REPS):
        a = time.perf_counter()
        for _ in range(CALLS): call()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - a) * 1e6 / CALLS)
    eager = statistics.median(ts)

    gobj = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): call()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    with torch.cuda.graph(gobj):
        for _ in range(CALLS): call()
    for _ in range(3): gobj.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(REPS):
        a = time.perf_counter()
        gobj.replay()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - a) * 1e6 / CALLS)
    graph = statistics.median(ts)

    bytes_call = TOPK * (N * K // 2 + N * (K // 64) * 4)
    floor_us = bytes_call / (ROOF * 1e3)
    out["cells"][tag] = {"eager_us": eager, "graph_us": graph,
                         "host_share_us": eager - graph,
                         "floor_us": floor_us,
                         "graph_over_floor": graph / floor_us}
    print(f"=== {tag} ===")
    print(f"  eager {eager:7.2f} us/call | graph-replay {graph:7.2f} | "
          f"host share {eager-graph:6.2f}")
    print(f"  byte floor {floor_us:6.2f} us -> graph is {graph/floor_us:.2f}x over floor")
json.dump(out, open("gemv_latsplit.json", "w"), indent=1)
