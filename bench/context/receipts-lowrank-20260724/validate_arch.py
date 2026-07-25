import os, sys, json, torch
sys.path.insert(0, "/root/g/kernel")
from experts4bit_qlora import NF4KVCache
from transformers import AutoTokenizer, AutoConfig
M = os.environ["VAL_MODEL"]; N = int(os.environ.get("VAL_TOKENS", "1024"))
tok = AutoTokenizer.from_pretrained(M)
cfg = AutoConfig.from_pretrained(M)
t = getattr(cfg, "text_config", cfg)
is_moe = any(k in str(type(cfg)).lower() or getattr(t, k, None)
             for k in ("num_experts", "num_local_experts"))
if is_moe:
    from experts4bit_qlora import load_moe_4bit_streaming
    model, _ = load_moe_4bit_streaming(M, device="cuda:0", dtype=torch.bfloat16, r=8,
                                       alpha=16, offload=True, pin=True, quant_type="nf4")
else:
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16).cuda()
model.eval()
from datasets import load_dataset
ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
ids = tok("\n\n".join(x for x in ds["text"] if x.strip()),
          return_tensors="pt").input_ids[:, :N].cuda()

def measure(cache):
    """Return (ppl, argmax[T]) WITHOUT retaining fp32 logits — Gemma's 262k
    vocab makes a single fp32 logit tensor ~1 GB, and holding two OOMs a 12 GB
    card for reasons that have nothing to do with the KV cache under test."""
    with torch.no_grad():
        lg = model(ids, use_cache=True, past_key_values=cache).logits
        am = lg[0].argmax(-1).clone()
        nll, n = 0.0, 0
        for lo in range(0, lg.shape[1] - 1, 128):       # chunked, never full fp32
            hi = min(lo + 128, lg.shape[1] - 1)
            lp = torch.log_softmax(lg[0, lo:hi].float(), -1)
            nll += float(-lp.gather(1, ids[0, lo + 1:hi + 1, None]).sum())
            n += hi - lo
        del lg
    torch.cuda.empty_cache()
    return float(torch.exp(torch.tensor(nll / n))), am

CFG = [("fp16 DynamicCache (ref)", "default"),
       ("NF4KVCache raw (control)", dict(quantize_keys=False, quantize_values=False)),
       ("K4 V4",                    dict(quantize_keys=True,  quantize_values=True)),
       ("K4 V16 (keys only)",       dict(quantize_keys=True,  quantize_values=False)),
       ("K16 V4 (values only)",     dict(quantize_keys=False, quantize_values=True))]
ref_am = None; rows = []
for name, kw in CFG:
    c = None if kw == "default" else NF4KVCache(**kw)
    p, am = measure(c)
    if ref_am is None: ref_am = am
    rows.append((name, p, float((ref_am == am).float().mean()),
                 None if c is None else c.memory_bytes()))
    del am; torch.cuda.empty_cache()
b = rows[0][1]
print(f"\n=== {M}  ({N} tokens, gqa {t.num_attention_heads}:{t.num_key_value_heads}) ===")
print(f"{'config':<26} {'ppl':>9} {'d-ppl':>9} {'agree':>8} {'cache MB':>9}")
for n, p, a, mb in rows:
    print(f"{n:<26} {p:>9.3f} {p-b:>+9.3f} {a*100:>7.2f}% "
          f"{'-' if mb is None else f'{mb/2**20:>8.2f}'}")
print(f"\ncontrol: raw vs fp16  d-ppl {rows[1][1]-b:+.4f} -> "
      f"{'SEMANTICS MATCH' if abs(rows[1][1]-b) < 0.002 else 'CACHE SEMANTICS DIFFER'}")
json.dump({"model": M, "tokens": N, "rows": [
    {"config": n, "ppl": p, "agree_ref": a, "cache_bytes": mb} for n, p, a, mb in rows]},
    open(f"/root/g/bench/context/validate_{M.split('/')[-1]}.json", "w"), indent=2)
