"""Per-bm time and tile count for gemm_4bit_grouped's prefill path across routed batches -- data for a tile-height cost model."""
import sys, time, torch, json, math
sys.path.insert(0, sys.argv[1])
import nf4_grouped as ng
from nf4_pack_ref import quantize_pack_nf4
torch.manual_seed(0); dev = "cuda"; E, topk = 128, 8
def stack(N, K):
    p, a = quantize_pack_nf4((torch.randn(E * N, K) * 0.02))
    return p.reshape(E, N, K // 2).to(dev), a.reshape(E, N, K // 64).float().to(dev)
W = {n: stack(N, K) for n, N, K in (("gate_up", 1536, 2048), ("down", 2048, 768))}
for alpha in (2.0, 0.5, 0.15):
    for tokens in (64, 160, 380, 520, 1024, 2048, 4096):
        pop = torch.distributions.Dirichlet(torch.full((E,), alpha)).sample()
        idx = torch.multinomial(pop.expand(tokens, E), topk, replacement=False).flatten()
        counts = torch.bincount(idx, minlength=E).tolist()
        eids = [e for e in range(E) if counts[e]]; sizes = [counts[e] for e in eids]; T = sum(sizes)
        for name, (B, am) in W.items():
            K = B.shape[2] * 2
            a = torch.randn(T, K, device=dev, dtype=torch.bfloat16)
            out = {}
            for bm in (16, 32, 64, 128):
                for _ in range(2): ng.gemm_4bit_grouped(a, B, am, sizes, eids, block_m=bm)
                torch.cuda.synchronize(); t0 = time.perf_counter()
                for _ in range(6): ng.gemm_4bit_grouped(a, B, am, sizes, eids, block_m=bm)
                torch.cuda.synchronize()
                out[bm] = dict(ms=round((time.perf_counter() - t0) * 1e3 / 6, 4), tiles=sum(-(-r // bm) for r in sizes))
            print(json.dumps(dict(alpha=alpha, tokens=tokens, proj=name, sizes=sizes, by_bm=out)), flush=True)
