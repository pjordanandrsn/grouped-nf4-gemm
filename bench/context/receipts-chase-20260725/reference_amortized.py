import statistics, sys, time, torch
sys.path.insert(0, "/root/g/kernel")
from nf4_kv import quantize_kv, dequant_kv_ref, dequant_kv_fused
T,H,D = 4096,16,128
g = torch.Generator(device="cpu").manual_seed(0)
x = (torch.randn(T,H,D,generator=g)*0.5).cuda(); p,a = quantize_kv(x)
moved = (p.numel()+a.numel()*4+T*H*D*2)/1e9
def amort(fn, n):
    for _ in range(10): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/n
def sync1(fn, reps=20):
    ts=[]
    for i in range(reps+5):
        torch.cuda.synchronize(); t0=time.perf_counter(); fn(); torch.cuda.synchronize()
        if i>=5: ts.append(time.perf_counter()-t0)
    return statistics.median(ts)
r_s = sync1(lambda: dequant_kv_ref(p,a,D,dtype=torch.bfloat16))
r_a = amort(lambda: dequant_kv_ref(p,a,D,dtype=torch.bfloat16), 60)
f_s = sync1(lambda: dequant_kv_fused(p,a,D,dtype=torch.bfloat16))
f_a = amort(lambda: dequant_kv_fused(p,a,D,dtype=torch.bfloat16), 300)
print(f"[{T},{H},{D}] -> bf16 on {torch.cuda.get_device_name(0)}")
print(f"  reference  synced {r_s*1e3:7.3f} ms   amortized {r_a*1e3:7.3f} ms  ({moved/r_a:6.1f} GB/s)")
print(f"  fused      synced {f_s*1e3:7.3f} ms   amortized {f_a*1e3:7.3f} ms  ({moved/f_a:6.1f} GB/s)")
print(f"  speedup    synced {r_s/f_s:6.2f}x     amortized {r_a/f_a:6.2f}x")
print(f"  -> the synced ratio UNDERSTATES by {(r_a/f_a)/(r_s/f_s):.2f}x, because the")
print(f"     ~{(1-f_a/f_s)*100:.0f}% sync overhead on the fused arm is only ~{(1-r_a/r_s)*100:.0f}% on the reference")
