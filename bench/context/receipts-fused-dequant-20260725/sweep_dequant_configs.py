import statistics, sys, time, torch
sys.path.insert(0, "/root/g/kernel")
import nf4_kv
from nf4_kv import quantize_kv, dequant_kv_ref, dequant_kv_fused
T, H, D = 4096, 16, 128
g = torch.Generator(device="cpu").manual_seed(0)
x = (torch.randn(T, H, D, generator=g) * 0.5).cuda()
p, a = quantize_kv(x)
moved = (p.numel() + a.numel()*4 + T*H*D*2) / 1e9
def timed(fn, reps=30):
    ts=[]
    for i in range(reps+5):
        torch.cuda.synchronize(); t0=time.perf_counter(); fn(); torch.cuda.synchronize()
        if i>=5: ts.append(time.perf_counter()-t0)
    return statistics.median(ts)*1e3
print("H2b gate first:")
ok = True
for (t_,h_,d_) in ((4096,16,128),(777,3,64),(256,2,256),(1,1,128)):
    gg = torch.Generator(device="cpu").manual_seed(t_)
    xx = (torch.randn(t_,h_,d_, generator=gg)*0.5).cuda()
    pp, aa = quantize_kv(xx)
    for dt in (torch.bfloat16, torch.float32):
        same = torch.equal(dequant_kv_ref(pp,aa,d_,dtype=dt), dequant_kv_fused(pp,aa,d_,dtype=dt))
        ok &= same
        if not same: print(f"  MISMATCH {t_}x{h_}x{d_} {dt}")
print("  H2b:", "CONFIRMED" if ok else "FALSIFIED")
if not ok: raise SystemExit(1)
print(f"\nreference: {timed(lambda: dequant_kv_ref(p,a,D,dtype=torch.bfloat16)):.3f} ms")
print(f"\n{'rows':>5} {'warps':>6} {'ms':>8} {'GB/s':>8}")
best=None
for rows in (1,2,4,8,16,32,64):
    for w in (1,2,4,8):
        try:
            ms = timed(lambda: dequant_kv_fused(p,a,D,dtype=torch.bfloat16,
                                                rows_per_prog=rows, num_warps=w), reps=20)
        except Exception as e:
            print(f"{rows:>5} {w:>6}   skip ({type(e).__name__})"); continue
        gbs = moved/ms*1e3
        print(f"{rows:>5} {w:>6} {ms:>8.3f} {gbs:>8.1f}")
        if best is None or gbs > best[0]: best = (gbs, rows, w, ms)
print(f"\nBEST: {best[0]:.1f} GB/s at rows={best[1]} warps={best[2]} ({best[3]:.3f} ms)")
print(f"H2a >=150 GB/s: {'CONFIRMED' if best[0] >= 150 else ('FALSIFIED' if best[0] < 120 else 'outside interval')}")
