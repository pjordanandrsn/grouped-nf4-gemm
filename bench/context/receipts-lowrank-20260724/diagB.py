import sys, statistics, torch
sys.path.insert(0, "/root/g/kernel")
from nf4_kv import quantize_kv, attend_nf4_kv
T, D, BT = 32768, 128, 128
def bench(fn, reps=15):
    for _ in range(5): fn()
    torch.cuda.synchronize(); ts=[]
    for _ in range(reps):
        s,e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return statistics.median(ts)
print("Is the cost per QUERY head (redundant dequant) or per KV head (bytes)?")
print("H_kv fixed at 4; if time scales with H_q, each query head is")
print("re-dequantizing the same kv bytes -- 16x wasted ALU at GQA 16:1.\n")
g = torch.Generator(device="cpu").manual_seed(0)
k = (torch.randn(T, 4, D, generator=g)*.5).cuda().bfloat16()
v = (torch.randn(T, 4, D, generator=g)*.5).cuda().bfloat16()
kp, ka = quantize_kv(k); vp, va = quantize_kv(v)
print(f"{'H_q':>5} {'GQA':>5} {'ms':>8} {'ms per q-head':>14}")
base = None
for H_q in (4, 8, 16, 32, 64):
    q = torch.randn(H_q, D, device="cuda", dtype=torch.float32)
    t = bench(lambda: attend_nf4_kv(q, kp, ka, vp, va, block_t=BT))
    if base is None: base = t
    print(f"{H_q:>5} {H_q//4:>4}:1 {t:>7.3f}m {t/H_q:>13.4f}m")
print(f"\n64-head time / 4-head time = {bench(lambda: attend_nf4_kv(torch.randn(64,D,device='cuda'), kp,ka,vp,va,block_t=BT))/base:.2f}x")
print("(bytes are IDENTICAL across all rows -- same 4 kv heads)")
