import sys, json, statistics, torch
sys.path.insert(0, "/root/g/kernel")
from nf4_kv import (quantize_kv, attend_nf4_kv, attend_nf4_kv_split,
                    attend_nf4_kv_gqa, dequant_kv_ref)
free, total = torch.cuda.mem_get_info()
print(f"free {free/2**30:.2f} GB of {total/2**30:.2f} GB   {torch.cuda.get_device_name(0)}")
T, H_KV, D, BT, REPS = 32768, 4, 128, 128, 25
def bench(fn, reps=REPS):
    for _ in range(5): fn()
    torch.cuda.synchronize(); ts=[]
    for _ in range(reps):
        s,e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return statistics.median(ts)
g = torch.Generator(device="cpu").manual_seed(7)
k = (torch.randn(T, H_KV, D, generator=g)*.5).cuda().bfloat16()
v = (torch.randn(T, H_KV, D, generator=g)*.5).cuda().bfloat16()
kp, ka = quantize_kv(k); vp, va = quantize_kv(v)
kf = dequant_kv_ref(kp, ka, D).to(torch.float16).transpose(0,1).unsqueeze(0).contiguous()
vf = dequant_kv_ref(vp, va, D).to(torch.float16).transpose(0,1).unsqueeze(0).contiguous()
rows=[]
for H_q in (16, 64):
    q = torch.randn(H_q, D, device="cuda", dtype=torch.float32)
    rep = H_q // H_KV
    qf = q.to(torch.float16).unsqueeze(0).unsqueeze(2)
    kf2, vf2 = kf.repeat_interleave(rep,1), vf.repeat_interleave(rep,1)
    t_two = bench(lambda: attend_nf4_kv(q, kp, ka, vp, va, block_t=BT))
    t_spl = bench(lambda: attend_nf4_kv_split(q, kp, ka, vp, va, block_t=BT))
    t_gq  = bench(lambda: attend_nf4_kv_gqa(q, kp, ka, vp, va, block_t=BT, precision="ieee"))
    t_gqt = bench(lambda: attend_nf4_kv_gqa(q, kp, ka, vp, va, block_t=BT, precision="tf32"))
    t_f16 = bench(lambda: torch.nn.functional.scaled_dot_product_attention(qf, kf2, vf2))
    rows.append((H_q, rep, t_two, t_spl, t_gq, t_gqt, t_f16))
print(f"\nT={T}, H_kv={H_KV}, D={D}")
print(f"{'H_q':>4} {'GQA':>5} {'two-pass':>9} {'split':>8} {'gqa-ieee':>9} {'gqa-tf32':>9} "
      f"{'fp16':>7} {'gqa/two':>8} {'gqa/fp16':>9}")
for H_q, rep, a, s, gi, gt, f in rows:
    print(f"{H_q:>4} {rep:>4}:1 {a:>8.3f}m {s:>7.3f}m {gi:>8.3f}m {gt:>8.3f}m "
          f"{f:>6.3f}m {a/gi:>7.2f}x {gi/f:>8.2f}x")
json.dump([{"H_q":H_q,"gqa":rep,"two_pass_ms":a,"split_ms":s,"gqa_ieee_ms":gi,
            "gqa_tf32_ms":gt,"fp16_ms":f,"speedup_vs_two":a/gi,"ratio_vs_fp16":gi/f}
           for H_q,rep,a,s,gi,gt,f in rows],
          open("/root/g/bench/context/gqa_latency.json","w"), indent=2)
g16, g64 = rows[0], rows[1]
s6a, s6b = g64[2]/g64[4], g64[4]/g64[6]
gain64, gain16 = g64[2]/g64[4], g16[2]/g16[4]
print("\nPREREG SCORING — B6 (registered + STAMPED before this run)")
print(f"  B6a gqa vs two-pass @32K 16:1  pred 1.5-6.0x  got {s6a:.2f}x  "
      f"{'CONFIRMED' if 1.5 <= s6a <= 6.0 else ('FALSIFIED' if s6a <= 1.0 else 'OUTSIDE-INTERVAL')}")
print(f"  B6b gqa vs fp16 @32K           pred <=1.3x    got {s6b:.2f}x  "
      f"{'CONFIRMED' if s6b <= 1.3 else ('FALSIFIED' if s6b > 2.0 else 'OUTSIDE-INTERVAL')}")
print(f"  B6c gain tracks GQA ratio      16:1 {gain64:.2f}x vs 4:1 {gain16:.2f}x  "
      f"{'CONFIRMED' if gain64 > gain16*1.2 else 'FALSIFIED'}")
print(f"  tf32 vs ieee speed: {g64[4]/g64[5]:.2f}x (correctness cost of ieee)")
