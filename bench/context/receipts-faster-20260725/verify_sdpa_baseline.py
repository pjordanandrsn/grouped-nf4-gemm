import sys, statistics, torch, torch.nn.functional as F
sys.path.insert(0, "/root/g/kernel")
from experts4bit_qlora import NF4KVCache
from nf4_kv import attend_nf4_kv_gqa

T, H_KV, H_Q, D = 32768, 4, 64, 128
GQA = H_Q // H_KV
c = NF4KVCache()
g = torch.Generator(device="cpu").manual_seed(0)
for lo in range(0, T, 4096):
    n = min(4096, T - lo)
    k = (torch.randn(1, H_KV, n, D, generator=g) * 0.5).bfloat16().cuda()
    v = (torch.randn(1, H_KV, n, D, generator=g) * 0.5).bfloat16().cuda()
    c.update(k, v, 0)
ks, vs = c._k[0], c._v[0]
torch.manual_seed(0)
q = (torch.randn(H_Q, D) * 0.5).cuda().float()
qb = q.bfloat16().view(1, H_Q, 1, D)
scale = D ** -0.5
kb = c._load(ks, torch.bfloat16); vb = c._load(vs, torch.bfloat16)

# ground truth in fp32, GQA done explicitly, no shortcuts
k32 = kb.float().repeat_interleave(GQA, dim=1)
v32 = vb.float().repeat_interleave(GQA, dim=1)
truth = F.scaled_dot_product_attention(q.view(1,H_Q,1,D), k32, v32, scale=scale).view(H_Q, D)

gqa_out = F.scaled_dot_product_attention(qb, kb, vb, scale=scale, enable_gqa=True).view(H_Q, D)
rep_out = F.scaled_dot_product_attention(qb, kb.repeat_interleave(GQA,1),
                                         vb.repeat_interleave(GQA,1), scale=scale).view(H_Q, D)
fus_out = attend_nf4_kv_gqa(q, ks[1], ks[2], vs[1], vs[2], scale=scale)

def rel(a): return ((a.float()-truth).norm()/truth.norm()).item()
print(f"enable_gqa vs fp32 truth : {rel(gqa_out):.3e}   <- is the fast path CORRECT?")
print(f"repeat     vs fp32 truth : {rel(rep_out):.3e}")
print(f"fused nf4  vs fp32 truth : {rel(fus_out):.3e}")
print(f"enable_gqa vs repeat     : {((gqa_out.float()-rep_out.float()).norm()/rep_out.float().norm()).item():.3e}")

def timed(fn, reps=50):
    s,e = torch.cuda.Event(True), torch.cuda.Event(True); ts=[]
    for i in range(reps+10):
        torch.cuda.synchronize(); s.record(); r=fn(); e.record(); torch.cuda.synchronize()
        if i>=10: ts.append(s.elapsed_time(e))
    return statistics.median(ts), r
t_gqa,_ = timed(lambda: F.scaled_dot_product_attention(qb, kb, vb, scale=scale, enable_gqa=True))
krep, vrep = kb.repeat_interleave(GQA,1).contiguous(), vb.repeat_interleave(GQA,1).contiguous()
t_rep,_ = timed(lambda: F.scaled_dot_product_attention(qb, krep, vrep, scale=scale))
t_fus,_ = timed(lambda: attend_nf4_kv_gqa(q, ks[1], ks[2], vs[1], vs[2], scale=scale))
kv_bf16 = kb.numel()*2*2
print(f"\nbf16 SDPA enable_gqa : {t_gqa:7.3f} ms  ({kv_bf16/t_gqa*1e3/1e9:6.1f} GB/s of KV)")
print(f"bf16 SDPA repeat_int : {t_rep:7.3f} ms   <- what #12 most likely measured")
print(f"fused nf4            : {t_fus:7.3f} ms  ({kv_bf16/3.5556/t_fus*1e3/1e9:6.1f} GB/s of KV)")
print(f"\nA2000 HBM is ~288 GB/s; bf16 KV here is {kv_bf16/1e6:.0f} MB")
