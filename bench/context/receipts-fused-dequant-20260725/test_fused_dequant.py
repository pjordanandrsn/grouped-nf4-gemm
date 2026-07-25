"""H1b gate + H1a timing, before anything is wired into the cache."""
import statistics, sys, time, torch
sys.path.insert(0, "/root/g/kernel")
from nf4_kv import quantize_kv, dequant_kv_ref, dequant_kv_fused

ok = True
for (T, H, D) in ((4096, 16, 128), (1024, 4, 128), (777, 3, 64), (256, 2, 256), (1, 1, 128)):
    g = torch.Generator(device="cpu").manual_seed(T)
    x = (torch.randn(T, H, D, generator=g) * 0.5).cuda()
    p, a = quantize_kv(x)
    for dt in (torch.bfloat16, torch.float32):
        r = dequant_kv_ref(p, a, D, dtype=dt)
        f = dequant_kv_fused(p, a, D, dtype=dt)
        same = torch.equal(r, f)
        ok &= same
        if not same:
            d = (r.float() - f.float()).abs()
            print(f"  MISMATCH T={T} H={H} D={D} {dt}: max {d.max():.3e}, "
                  f"{(d>0).sum().item()} of {d.numel()} differ")
        else:
            print(f"  ok  T={T:>5} H={H:>2} D={D:>3} {str(dt).split('.')[-1]:<8} bit-identical")
print("H1b GATE:", "CONFIRMED" if ok else "FALSIFIED")
if not ok: raise SystemExit(1)

T, H, D = 4096, 16, 128
g = torch.Generator(device="cpu").manual_seed(0)
x = (torch.randn(T, H, D, generator=g) * 0.5).cuda()
p, a = quantize_kv(x)
def timed(fn, reps=30):
    ts=[]
    for i in range(reps+5):
        torch.cuda.synchronize(); t0=time.perf_counter(); fn(); torch.cuda.synchronize()
        if i>=5: ts.append(time.perf_counter()-t0)
    return statistics.median(ts)*1e3
tr = timed(lambda: dequant_kv_ref(p, a, D, dtype=torch.bfloat16))
tf = timed(lambda: dequant_kv_fused(p, a, D, dtype=torch.bfloat16))
moved = (p.numel() + a.numel()*4 + T*H*D*2) / 1e9
print(f"\n[{T},{H},{D}] -> bf16")
print(f"  reference {tr:8.3f} ms   ({moved/tr*1e3:6.1f} GB/s)")
print(f"  fused     {tf:8.3f} ms   ({moved/tf*1e3:6.1f} GB/s)")
print(f"  H1a speedup = {tr/tf:.2f}x   (>=5 confirms, <2 falsifies)")
