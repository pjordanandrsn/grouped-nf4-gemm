"""Blackwell sweep of the DECODE GEMV path — the single-stream expert kernel.

BWTUNE swept only the M-tile prefill path. The decode path (max(sizes)==1,
the b1d single-stream cell: top-8 singleton groups) has its own knobs —
decode_config (BLOCK_N, warps), split_k, GNF4_GEMV_DOTPAD, WIDE/VEC loads —
tuned in the k6b/k7 prereg era. The ceiling memory records the cell at
3.8x its byte floor and says nobody profiled where that goes.

Cells: dotpad {0,1} x bn {64,128,256,512} x warps {2,4,8} x sk {1,2,4,8},
wide/vec defaults plus a wide=0 arm for the winner. Gates: bitwise within
kernel family (scalar vs scalar), K6 relative across families (dotpad=MMA
TF32 vs scalar fp32). Both timings: streamed (graph-like back-to-back) and
the read-floor arm for GB/s context. Shapes: gate_up + down at T=1 top-8.
"""
import itertools, json, os, statistics, time, torch
from nf4_grouped import gemm_4bit_grouped
from nf4_pack_ref import quantize_pack_nf4

dev = "cuda"
torch.manual_seed(0)
E, TOPK = 128, 8
SHAPES = [("gate_up", 1536, 2048), ("down", 2048, 768)]
DRAWS, INNER = 5, 40
out = {"cells": {}, "env": {k: os.environ.get(k) for k in
       ("GNF4_GEMV_DOTPAD", "GNF4_GEMV_WIDE_LOADS", "GNF4_GEMV_VEC_LOADS")}}

def floor_gbps():
    big = torch.empty(int(2e9 // 4), dtype=torch.float32, device=dev)
    torch.cuda.synchronize(); a = time.perf_counter()
    for _ in range(10): big.sum()
    torch.cuda.synchronize()
    return (big.numel() * 4 * 10) / (time.perf_counter() - a) / 1e9
ROOF = floor_gbps()

for tag, N, K in SHAPES:
    W = torch.randn(E, N, K) * 0.02
    packs = [quantize_pack_nf4(W[e]) for e in range(E)]
    Bt = torch.stack([p.reshape(N, K // 2) for p, _ in packs]).to(dev).contiguous()
    At = torch.stack([a.reshape(N, K // 64) for _, a in packs]).float().to(dev).contiguous()
    draws = []
    for d in range(DRAWS):
        g = torch.Generator().manual_seed(1300 + d)
        ids = torch.randperm(E, generator=g)[:TOPK].sort().values
        x = (torch.randn(TOPK, K, generator=g) * 0.1).to(dev, torch.bfloat16)
        draws.append((x.contiguous(), ids.to(dev).to(torch.int32)))
    ones = [1] * TOPK
    bytes_call = TOPK * (N * K // 2 + N * (K // 64) * 4)

    def t_of(fn):
        ts = []
        for d in draws:
            for _ in range(5): fn(d)
            torch.cuda.synchronize(); a = time.perf_counter()
            for _ in range(INNER): fn(d)
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - a) * 1e6 / INNER)
        return statistics.median(ts)

    def run(d, dot, cfg, sk):
        os.environ["GNF4_GEMV_DOTPAD"] = str(dot)
        return gemm_4bit_grouped(d[0], Bt, At, ones, d[1],
                                 decode_config=cfg, split_k=sk)
    os.environ["GNF4_GEMV_DOTPAD"] = "1"
    inc = lambda d: gemm_4bit_grouped(d[0], Bt, At, ones, d[1])
    us_inc = t_of(inc)
    ref = inc(draws[0])

    res = {}
    for dot, bn, w, sk in itertools.product((1, 0), (64, 128, 256, 512),
                                            (2, 4, 8), (1, 2, 4, 8)):
        key = f"d{dot}bn{bn}w{w}sk{sk}"
        try:
            o = run(draws[0], dot, (bn, w), sk)
            rel = (o.float() - ref.float()).abs().max() / ref.float().abs().max().clamp_min(1e-5)
            if rel > 2 ** -7: continue
            res[key] = t_of(lambda d, dot=dot, bn=bn, w=w, sk=sk: run(d, dot, (bn, w), sk))
        except Exception:
            pass
    os.environ["GNF4_GEMV_DOTPAD"] = "1"
    top = sorted(res.items(), key=lambda kv: kv[1])[:6]
    gb = bytes_call / 1e9
    out["cells"][tag] = {"incumbent_us": us_inc,
                         "incumbent_gbps": gb / (us_inc / 1e6),
                         "roofline_gbps": ROOF, "swept": len(res), "top": top}
    print(f"=== {tag} bytes/call {gb*1000:.1f} MB, roofline {ROOF:.0f} GB/s ===")
    print(f"  incumbent  {us_inc:8.2f} us  {gb/(us_inc/1e6):7.1f} GB/s "
          f"({100*gb/(us_inc/1e6)/ROOF:.1f}% of roof)")
    for k, us in top:
        print(f"  {k:18s} {us:8.2f} us  {us_inc/us:6.3f}x vs incumbent")
json.dump(out, open("gemv_tune.json", "w"), indent=1)
