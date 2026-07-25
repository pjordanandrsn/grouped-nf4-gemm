import sys, json, statistics, torch
sys.path.insert(0, "/root/g/kernel")
from nf4_kv import quantize_kv, attend_nf4_kv, attend_nf4_kv_fused, dequant_kv_ref
free, total = torch.cuda.mem_get_info()
print(f"free {free/2**30:.2f} GB of {total/2**30:.2f} GB   {torch.cuda.get_device_name(0)}")
H_Q, H_KV, D, BT, REPS = 64, 4, 128, 128, 25

def bench(fn, reps=REPS):
    for _ in range(5): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        st, en = torch.cuda.Event(True), torch.cuda.Event(True)
        st.record(); fn(); en.record(); torch.cuda.synchronize()
        ts.append(st.elapsed_time(en))
    return statistics.median(ts)

rows = []
for T in (4096, 32768):
    g = torch.Generator(device="cpu").manual_seed(T)
    k = (torch.randn(T, H_KV, D, generator=g) * .5).cuda().bfloat16()
    v = (torch.randn(T, H_KV, D, generator=g) * .5).cuda().bfloat16()
    kp, ka = quantize_kv(k); vp, va = quantize_kv(v)
    q = torch.randn(H_Q, D, device="cuda", dtype=torch.float32)
    kf = dequant_kv_ref(kp, ka, D).to(torch.float16).transpose(0, 1).unsqueeze(0).contiguous()
    vf = dequant_kv_ref(vp, va, D).to(torch.float16).transpose(0, 1).unsqueeze(0).contiguous()
    qf = q.to(torch.float16).unsqueeze(0).unsqueeze(2)
    rep = H_Q // H_KV
    kf2, vf2 = kf.repeat_interleave(rep, 1), vf.repeat_interleave(rep, 1)
    t_two = bench(lambda: attend_nf4_kv(q, kp, ka, vp, va, block_t=BT))
    t_one = bench(lambda: attend_nf4_kv_fused(q, kp, ka, vp, va, block_t=BT))
    t_f16 = bench(lambda: torch.nn.functional.scaled_dot_product_attention(qf, kf2, vf2))
    rows.append((T, t_two, t_one, t_f16))

print(f"\n{'ctx':>7} {'two-pass':>9} {'fused':>8} {'fp16 sdpa':>10} "
      f"{'fused vs two':>13} {'fused vs fp16':>14}")
for T, a, b, c in rows:
    print(f"{T:>7} {a:>8.3f}m {b:>7.3f}m {c:>9.3f}m {a/b:>12.2f}x {b/c:>13.2f}x")
json.dump([{"ctx": T, "two_pass_ms": a, "fused_ms": b, "fp16_sdpa_ms": c,
            "speedup_vs_two_pass": a/b, "ratio_vs_fp16": b/c} for T, a, b, c in rows],
          open("/root/g/bench/context/fused_latency.json", "w"), indent=2)
print("\nPREREG SCORING")
s32 = rows[-1][1] / rows[-1][2]; r32 = rows[-1][2] / rows[-1][3]; r4 = rows[0][2] / rows[0][3]
print(f"  B1 speedup vs two-pass @32K  pred 1.8-3.0x   got {s32:.2f}x  "
      f"{'CONFIRMED' if 1.8 <= s32 <= 3.0 else 'FALSIFIED'}")
print(f"  B2 fused vs fp16 @32K        pred 0.8-1.4x   got {r32:.2f}x  "
      f"{'CONFIRMED' if 0.8 <= r32 <= 1.4 else ('FALSIFIED' if r32 > 2.0 else 'OUTSIDE-INTERVAL')}")
print(f"  B3 ratio improves with ctx   pred 32K<4K     4K {r4:.2f}x vs 32K {r32:.2f}x  "
      f"{'CONFIRMED' if r32 < r4 else 'FALSIFIED'}")
