#!/usr/bin/env python3
"""Measured routing histograms: per-expert token counts (group sizes) per layer.

Phase-0's access counter recorded only the distinct-expert UNION (occupancy);
the GEMM harness needs the group-size DISTRIBUTION — how many token-rows each
expert receives — because skewed routing (hot experts, cold/empty experts)
changes grouped-GEMM cost in a way the uniform-M assumption hides. This hooks
the router gate (a bias-free Linear whose out_features == n_experts), takes
top-k per token at seq 2048 (single stream — Phase-0 showed long-context
single-stream reads a tighter union than batched at matched eff_tokens), and
bincounts assignments per expert per layer.

Load dtype: OLMoE bf16 (tiny, clean). Qwen3-30B in 4-bit — Phase-0 established
NF4 perturbs expert SELECTION negligibly, and it fits a 24 GB card directly
instead of the slow CPU-offload path. Recorded per run.

  MODEL=allenai/OLMoE-1B-7B-0924 LOAD=bf16 OUT=routing_olmoe.json python routing_hist.py
  MODEL=Qwen/Qwen3-30B-A3B LOAD=nf4 OUT=routing_qwen.json python routing_hist.py
"""

import json
import os

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ["MODEL"]
OUT = os.environ["OUT"]
LOAD = os.environ.get("LOAD", "bf16")
SEQ = int(os.environ.get("SEQ", "2048"))


def _cfg_int(cfg, keys):
    t = getattr(cfg, "text_config", cfg)
    for k in keys:
        if getattr(t, k, None):
            return getattr(t, k)
    raise SystemExit(f"none of {keys} on config")


cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
E = _cfg_int(cfg, ("num_experts", "num_local_experts", "n_routed_experts"))
k = _cfg_int(
    cfg, ("num_experts_per_tok", "experts_per_token", "top_k_experts", "moe_top_k")
)
print(f"model={MODEL} E={E} k={k} load={LOAD}", flush=True)

kw = dict(trust_remote_code=True, low_cpu_mem_usage=True, device_map="auto")
if LOAD == "nf4":
    from transformers import BitsAndBytesConfig

    kw["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
else:
    kw["torch_dtype"] = torch.bfloat16
model = AutoModelForCausalLM.from_pretrained(MODEL, **kw).eval()

# per-layer per-expert assignment counts, keyed by module registration order
counts: dict[int, torch.Tensor] = {}
order: dict[int, int] = {}


def hook(mod, inp, out):
    logits = out[0] if isinstance(out, tuple) else out
    if logits.dim() == 3:
        logits = logits.reshape(-1, logits.shape[-1])
    if logits.shape[-1] != E:
        return
    mid = id(mod)
    L = order.setdefault(mid, len(order))
    idx = torch.topk(logits.float(), k, dim=-1).indices.reshape(-1)  # [tokens*k]
    c = torch.bincount(idx.cpu(), minlength=E)
    counts[L] = counts.get(L, torch.zeros(E, dtype=torch.long)) + c


n = 0
for m in model.modules():
    if isinstance(m, torch.nn.Linear) and m.out_features == E and m.bias is None:
        m.register_forward_hook(hook)
        n += 1
print(f"hooked {n} router gates", flush=True)
assert n > 0, "no router gate matched (out_features==E, bias=None)"

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
from datasets import load_dataset  # noqa: E402  (heavy import deferred past the hook setup)

ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
buf = []
for r in ds:
    t = r["text"].strip()
    if len(t) > 50:
        buf.append(t)
    if len(buf) >= 4000:
        break
ids = tok("\n".join(buf), return_tensors="pt").input_ids[0][:SEQ]
print(f"corpus tokens: {ids.numel()} (want {SEQ})", flush=True)
in_dev = next(model.parameters()).device
with torch.no_grad():
    model(ids.reshape(1, -1).to(in_dev), use_cache=False)

per_layer = [counts[L].tolist() for L in sorted(counts)]
tokens = SEQ


def summarize(vec):
    v = sorted(vec, reverse=True)
    nz = [x for x in v if x > 0]
    import statistics

    return {
        "occupancy": len(nz) / E,
        "empty_groups": E - len(nz),
        "max": v[0],
        "mean_nonzero": (sum(nz) / len(nz)) if nz else 0,
        "p50": statistics.median(v),
        "p95": v[max(0, int(0.05 * E)) - 1] if E else 0,
        "cv": (statistics.pstdev(v) / (sum(v) / E))
        if sum(v)
        else 0,  # skew: coeff of variation
    }


layer_summ = [summarize(v) for v in per_layer]
occ = [s["occupancy"] for s in layer_summ]
# representative layer = the one whose occupancy is the median across layers
med_occ = sorted(occ)[len(occ) // 2]
rep = min(range(len(occ)), key=lambda i: abs(occ[i] - med_occ))
out = {
    "model": MODEL,
    "E": E,
    "k": k,
    "seq": SEQ,
    "load": LOAD,
    "tokens": tokens,
    "expected_assignments": tokens * k,
    "per_layer_counts": per_layer,
    "layer_summary": layer_summ,
    "representative_layer": rep,
    "representative_counts": per_layer[rep],
    "occupancy_mean": sum(occ) / len(occ),
    "occupancy_range": [min(occ), max(occ)],
}
with open(OUT, "w") as f:
    json.dump(out, f, indent=1)
print(
    f"wrote {OUT}: {len(per_layer)} layers, occupancy mean {out['occupancy_mean']:.3f} "
    f"range {out['occupancy_range']}, rep-layer {rep} cv {layer_summ[rep]['cv']:.2f} "
    f"max-group {layer_summ[rep]['max']} vs uniform {tokens * k // E}",
    flush=True,
)
