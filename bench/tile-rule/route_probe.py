"""Real-router group sizes through e4b's fused training step (4-layer slice of Qwen3-30B-A3B, TC1's alpaca token rows, micro-batch 2),
then replay of every recorded forward call under each M-tile height and both rules, and an ABBA step timing of the two rules."""
import json, os, sys, time, math, statistics, warnings
import torch
warnings.filterwarnings("ignore")
tok = json.load(open(sys.argv[1])); out_path = sys.argv[2]; steps = int(sys.argv[3]) if len(sys.argv) > 3 else 6
import nf4_grouped as ng
from experts4bit_qlora import load_moe_4bit_streaming, enable_fast_train
from experts4bit_qlora.lora import add_attention_lora, quantize_attention_projections_4bit
import bitsandbytes as bnb
model, cfg = load_moe_4bit_streaming("/workspace/qwen3_l4_real", "cuda", torch.bfloat16, 16, 16, offload=False, pin=True, prefetch=False, quant_type="nf4")
model.to("cuda"); quantize_attention_projections_4bit(model)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); model.config.use_cache = False
add_attention_lora(model, 16, 16, torch.float32); enable_fast_train(model, dgrad=True); model.train()
opt = bnb.optim.AdamW8bit([p for p in model.parameters() if p.requires_grad], lr=2e-4, weight_decay=0.001)
PAD = tok["pad_id"]; rows = tok["train"]
def mb(i):
    a, b = rows[2 * i], rows[2 * i + 1]; n = max(len(a), len(b))
    ids = torch.tensor([a + [PAD] * (n - len(a)), b + [PAD] * (n - len(b))], device="cuda")
    lab = ids.clone(); lab[ids == PAD] = -100
    att = (ids != PAD).long()
    return ids, lab, att
REC = []; _orig = ng.gemm_4bit_grouped
def rec(a_cat, B, absmax, sizes, expert_ids, *args, **kw):
    if REC is not None and max(sizes) > 1:
        REC.append({"sizes": [int(s) for s in sizes], "N": int(B.shape[1]), "K": int(B.shape[2]) * 2})
    return _orig(a_cat, B, absmax, sizes, expert_ids, *args, **kw)
ng.gemm_4bit_grouped = rec
def step(i0):
    for j in range(4):                                   # accum 4
        ids, lab, att = mb(i0 * 4 + j)
        model(input_ids=ids, attention_mask=att, labels=lab).loss.backward()
    opt.step(); opt.zero_grad(set_to_none=True)
for s in range(steps):
    step(s)
torch.cuda.synchronize()
calls = REC; REC = None
st = [dict(G=len(c["sizes"]), T=sum(c["sizes"]), mean=sum(c["sizes"]) / len(c["sizes"]), max=max(c["sizes"])) for c in calls]
print("recorded forward calls", len(calls), "mean rows/group median", statistics.median(x["mean"] for x in st), "max median", statistics.median(x["max"] for x in st),
      "max p90", sorted(x["max"] for x in st)[int(0.9 * len(st))], flush=True)
# replay: time each recorded call at every height (a sample of 64 calls for time), plus the two rules' picks
W = {}
def weights(N, K):
    if (N, K) not in W:
        from nf4_pack_ref import quantize_pack_nf4
        p, am = quantize_pack_nf4(torch.randn(128 * N, K) * 0.02)
        W[(N, K)] = (p.reshape(128, N, K // 2).cuda(), am.reshape(128, N, K // 64).float().cuda())
    return W[(N, K)]
sample = calls[:: max(1, len(calls) // 64)]
rep = []
for c in sample:
    B, am = weights(c["N"], c["K"]); sizes = c["sizes"]; eids = list(range(len(sizes)))
    x = torch.randn(sum(sizes), c["K"], device="cuda", dtype=torch.bfloat16)
    t = {}
    for bm in (16, 32, 64, 128):
        for _ in range(2): _orig(x, B, am, sizes, eids, block_m=bm)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(5): _orig(x, B, am, sizes, eids, block_m=bm)
        torch.cuda.synchronize(); t[bm] = (time.perf_counter() - t0) * 1e3 / 5
    pm, pc = ng._prefill_block_m(max(sizes)), ng._prefill_block_m_cost(sizes)
    rep.append(dict(N=c["N"], K=c["K"], G=len(sizes), mean=round(sum(sizes) / len(sizes), 1), max=max(sizes), t={k: round(v, 3) for k, v in t.items()},
                    max_bm=pm, cost_bm=pc, best=min(t, key=t.get)))
tot = {r: sum(x["t"][x[f"{r}_bm"]] for x in rep) for r in ("max", "cost")}; tot["best"] = sum(min(x["t"].values()) for x in rep)
print("replay over", len(rep), "calls: total ms max-rule %.1f cost-rule %.1f best %.1f" % (tot["max"], tot["cost"], tot["best"]), flush=True)
# step A/B, ABBA, 3 timed steps each after 1 warm
ab = {}
for rule in ("max", "cost", "cost", "max"):
    os.environ["GNF4_PREFILL_TILE_RULE"] = rule
    step(50); torch.cuda.synchronize()
    ts = []
    for s in range(3):
        torch.cuda.synchronize(); t0 = time.perf_counter(); step(60 + s); torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    ab.setdefault(rule, []).append(statistics.median(ts))
print("step s (ABBA medians):", {k: [round(v, 4) for v in vs] for k, vs in ab.items()}, flush=True)
json.dump(dict(stats=st, replay=rep, replay_total_ms=tot, step_ab=ab), open(out_path, "w"))
