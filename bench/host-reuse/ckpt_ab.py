"""Gradient-checkpointing policy A/B on e4b's fused training step (4-layer Qwen3-30B-A3B slice, TC1 rows, mb2 x accum 4).

--mode layer: what TC1's e4b arm runs (HF gradient_checkpointing_enable, non-reentrant: each decoder layer recomputed in backward).
--mode attn : decoder-layer checkpointing off; each layer's self_attn forward wrapped in a non-reentrant checkpoint, so the MoE block's
              activations are kept by autograd and the MoE forward is NOT recomputed.
--mode none : no checkpointing (memory upper bound).
Prints median step time over --steps (no profiler), peak allocated memory, and with --check saves every trainable gradient after one
micro-batch under deterministic mode for a bitwise comparison across modes.
"""
import argparse, json, time
import torch
import torch.utils.checkpoint as cp
ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["layer", "attn", "none"], required=True)
ap.add_argument("--ckpt", default="/workspace/qwen3_l4_real"); ap.add_argument("--tokens", default="/workspace/tokens_qwen3prof945.json")
ap.add_argument("--batch", type=int, default=2); ap.add_argument("--accum", type=int, default=4); ap.add_argument("--steps", type=int, default=8)
ap.add_argument("--check", default=None)
a = ap.parse_args()
if a.check:
    torch.use_deterministic_algorithms(True, warn_only=True)
from experts4bit_qlora import load_moe_4bit_streaming, enable_fast_train
from experts4bit_qlora.lora import add_attention_lora, quantize_attention_projections_4bit
import bitsandbytes as bnb
torch.manual_seed(0)
model, cfg = load_moe_4bit_streaming(a.ckpt, "cuda", torch.bfloat16, 16, 16, offload=False, pin=True, prefetch=False, quant_type="nf4")
model.to("cuda")
quantize_attention_projections_4bit(model)
if a.mode != "none":
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.config.use_cache = False
torch.manual_seed(1)
add_attention_lora(model, 16, 16, torch.float32)
assert enable_fast_train(model, dgrad=True) > 0
n_wrapped = 0
if a.mode == "attn":
    for mod in model.modules():
        if mod.__class__.__name__.endswith("DecoderLayer"):
            mod.gradient_checkpointing = False
            attn = mod.self_attn
            orig = attn.forward
            def fwd(*args, _orig=orig, **kw):
                return cp.checkpoint(_orig, *args, use_reentrant=False, **kw)
            attn.forward = fwd
            n_wrapped += 1
model.train()
params = [p for p in model.parameters() if p.requires_grad]
names = [n for n, p in model.named_parameters() if p.requires_grad]
opt = bnb.optim.AdamW8bit(params, lr=2e-4, weight_decay=0.001)
TOK = json.load(open(a.tokens))
def batch(i):
    rows = TOK["train"]; PAD = TOK["pad_id"]
    r = [rows[(a.batch * i + j) % len(rows)] for j in range(a.batch)]
    n = max(len(x) for x in r)
    ids = torch.tensor([x + [PAD] * (n - len(x)) for x in r], device="cuda")
    lab = ids.clone(); lab[ids == PAD] = -100
    return ids, lab, (ids != PAD).long()
if a.check:
    ids, lab, att = batch(0)
    model(input_ids=ids, labels=lab, attention_mask=att).loss.backward()
    torch.cuda.synchronize()
    torch.save({n: p.grad.detach().cpu() for n, p in zip(names, params)}, a.check)
    print("check written", a.check, "params", len(names), "wrapped", n_wrapped)
    raise SystemExit
def step(i):
    for j in range(a.accum):
        ids, lab, att = batch(i * a.accum + j)
        model(input_ids=ids, labels=lab, attention_mask=att).loss.backward()
    opt.step(); opt.zero_grad(set_to_none=True)
for i in range(3):
    step(i)
torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
ts = []
for i in range(a.steps):
    torch.cuda.synchronize(); t0 = time.perf_counter(); step(10 + i); torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
ts.sort()
print(json.dumps({"mode": a.mode, "wrapped": n_wrapped, "step_s_median": round(ts[len(ts) // 2], 4), "step_s_min": round(ts[0], 4),
                  "step_s_all": [round(t, 4) for t in ts], "peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3)}))
