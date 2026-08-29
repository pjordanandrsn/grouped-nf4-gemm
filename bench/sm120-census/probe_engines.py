"""GEMM P0b: is the M-tile path the WRONG PATH at B=16 routing?

Mean group size at the census cell is 128/~81 = 1.6 rows/expert -- GEMV
territory, where the Marlin law says M-tile machinery is strictly
negative. gnf4 already ships a certified GEMV branch (max(sizes)==1);
the router just never takes it when any group has 2+ rows.

Arms at the census draws (both shapes):
  mtile      incumbent path (grouped sizes, register-LUT M-tile)
  singleton  SAME kernel entry, sizes=[1]*R, per-row eids -- the decode
             GEMV branch; pays 128/81 = 1.58x weight bytes for a
             bandwidth-shaped kernel
  gmm_bf16   torch._grouped_mm on bf16 dequant-truth weights (2x bytes,
             real HMMA grouped engine) -- the honest ceiling referee,
             skipped with a note if this torch lacks it

Outputs are compared against the m-tile arm (singleton is bitwise per
BV3's own note: per-row arithmetic identical; gmm on the K6 frame).
"""
import json, statistics, time, torch
from nf4_grouped import gemm_4bit_grouped, dequant_ref
from nf4_pack_ref import quantize_pack_nf4

dev = "cuda"
torch.manual_seed(0)
E, TOPK, B_ = 128, 8, 16
R = B_ * TOPK
SHAPES = [("gate_up", 1536, 2048), ("down", 2048, 768)]
DRAWS, INNER = 5, 15

out = {"arms": {}}
for tag, N, K in SHAPES:
    W = torch.randn(E, N, K) * 0.02
    packs = [quantize_pack_nf4(W[e]) for e in range(E)]
    Bt = torch.stack([p.reshape(N, K // 2) for p, _ in packs]).to(dev).contiguous()
    At = torch.stack([a.reshape(N, K // 64) for _, a in packs]).float().to(dev).contiguous()
    Wq = torch.stack([dequant_ref(p.reshape(N, K // 2), a, N, K)
                      for p, a in packs]).to(dev, torch.bfloat16).contiguous()
    draws = []
    for d in range(DRAWS):
        g = torch.Generator().manual_seed(900 + d)
        ids = torch.stack([torch.randperm(E, generator=g)[:TOPK]
                           for _ in range(B_)]).reshape(-1)
        sids, order = torch.sort(ids)
        uniq, counts = torch.unique_consecutive(sids, return_counts=True)
        x = (torch.randn(R, K, generator=g) * 0.1).to(dev, torch.bfloat16)
        xs = x.index_select(0, order.to(dev)).contiguous()
        # _grouped_mm wants ONE offset per batch of b (all E experts,
        # empty groups as repeated offsets), not per touched group
        counts_full = torch.bincount(sids, minlength=E)
        offs = torch.cumsum(counts_full, 0).to(dev, torch.int32)
        draws.append({"xs": xs, "sizes": counts.tolist(),
                      "eids": uniq.to(dev).to(torch.int32),
                      "row_ids": sids.to(dev).to(torch.int32),
                      "offs": offs})
    def t_of(fn):
        ts = []
        for d in draws:
            for _ in range(3): fn(d)
            torch.cuda.synchronize(); a = time.perf_counter()
            for _ in range(INNER): fn(d)
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - a) * 1000 / INNER)
        return statistics.median(ts)

    mt = lambda d: gemm_4bit_grouped(d["xs"], Bt, At, d["sizes"], d["eids"])
    ones = [1] * R
    sg = lambda d: gemm_4bit_grouped(d["xs"], Bt, At, ones, d["row_ids"])
    ms_mt, ms_sg = t_of(mt), t_of(sg)
    bit = all(torch.equal(mt(d), sg(d)) for d in draws)

    gmm = None
    if hasattr(torch, "_grouped_mm"):
        try:
            def gm(d):
                return torch._grouped_mm(d["xs"], Wq.transpose(1, 2),
                                         offs=d["offs"])
            o, r = gm(draws[0]), mt(draws[0])
            rel = float((o.float() - r.float()).abs().max()
                        / r.float().abs().max().clamp_min(1e-5))
            gmm = {"ms": t_of(gm), "rel_err": rel}
        except Exception as e:
            gmm = {"error": repr(e)[:140]}
    nf4_gb = (len(draws[0]["sizes"]) * (N * K // 2 + N * (K // 64) * 4)) / 1e9
    out["arms"][tag] = {"mtile_ms": ms_mt, "singleton_ms": ms_sg,
                        "singleton_bitwise": bool(bit),
                        "speedup_singleton": ms_mt / ms_sg,
                        "gmm_bf16": gmm, "nf4_gb": nf4_gb}
    print(f"=== {tag} ===")
    print(f"  mtile     {ms_mt:8.4f} ms   {nf4_gb/(ms_mt/1e3):7.1f} GB/s")
    print(f"  singleton {ms_sg:8.4f} ms   speedup {ms_mt/ms_sg:6.3f}x  bitwise={bit}")
    print(f"  gmm_bf16  {gmm}")
json.dump(out, open("./gemm_p0b.json", "w"), indent=1)
