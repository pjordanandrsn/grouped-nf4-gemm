"""K1 of PREREG-chase-unresolved.md — is the ceiling the kernel or the stopwatch?"""
import statistics, sys, time, torch
sys.path.insert(0, "/root/g/kernel")
from nf4_kv import quantize_kv, dequant_kv_ref, dequant_kv_fused
H, D = 16, 128

def synced(fn, reps=20):
    ts = []
    for i in range(reps + 5):
        torch.cuda.synchronize(); t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
        if i >= 5: ts.append(time.perf_counter() - t0)
    return statistics.median(ts)

def amortized(fn, n=200):
    for _ in range(10): fn()                      # warm
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n

print(f"{'T':>7} {'bytes MB':>9} {'synced us':>10} {'amort us':>9} {'sync GB/s':>10} "
      f"{'amort GB/s':>11} {'sync/amort':>11}")
res = {}
for T in (1024, 4096, 8192, 16384, 32768, 65536):
    g = torch.Generator(device="cpu").manual_seed(T)
    x = (torch.randn(T, H, D, generator=g) * 0.5).cuda()
    p, a = quantize_kv(x)
    if T == 4096:
        assert torch.equal(dequant_kv_ref(p, a, D, dtype=torch.bfloat16),
                           dequant_kv_fused(p, a, D, dtype=torch.bfloat16)), "K1c GATE FAILED"
    moved = (p.numel() + a.numel() * 4 + T * H * D * 2) / 1e9
    f = lambda: dequant_kv_fused(p, a, D, dtype=torch.bfloat16)
    s, am = synced(f), amortized(f)
    res[T] = (moved / s, moved / am, s, am)          # seconds -> GB/s, no 1e3
    print(f"{T:>7} {moved*1e3:>9.1f} {s*1e6:>10.1f} {am*1e6:>9.1f} "
          f"{moved/s:>10.1f} {moved/am:>11.1f} {s/am:>10.3f}x")
    del x, p, a; torch.cuda.empty_cache()

print("\n=== scoring ===")
k1a = res[32768][1] / res[4096][1]
print(f"K1a GB/s(32768)/GB/s(4096), amortized = {k1a:.3f}   >=1.30 confirms, <1.05 falsifies  "
      f"{'CONFIRMED' if k1a >= 1.30 else ('FALSIFIED' if k1a < 1.05 else 'outside interval')}")
k1b = 1 - res[4096][3] / res[4096][2]
print(f"K1b amortized faster than synced @4096 = {k1b*100:.1f}%   >=5% confirms, <0% falsifies  "
      f"{'CONFIRMED' if k1b >= 0.05 else ('FALSIFIED' if k1b < 0 else 'outside interval')}")
print("K1c GATE: CONFIRMED (bit-identical)")
print(f"\nfor the record: peak amortized {max(v[1] for v in res.values()):.1f} GB/s "
      f"at T={max(res, key=lambda k: res[k][1])}; A2000 HBM is ~288 GB/s")
