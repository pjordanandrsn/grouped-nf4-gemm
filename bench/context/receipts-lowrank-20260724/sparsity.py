import os, sys, json, torch
sys.path.insert(0, "/root/g/kernel")
from experts4bit_qlora import load_moe_4bit_streaming, NF4KVCache
from transformers import AutoTokenizer
M = os.environ.get("SP_MODEL", "allenai/OLMoE-1B-7B-0924")
N, CH = int(os.environ.get("SP_TOKENS", "1024")), 128
tok = AutoTokenizer.from_pretrained(M)
model, _ = load_moe_4bit_streaming(M, device="cuda:0", dtype=torch.bfloat16, r=8,
                                   alpha=16, offload=True, pin=True, quant_type="nf4")
model.eval()
from datasets import load_dataset
ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
ids = tok("\n\n".join(x for x in ds["text"] if x.strip()),
          return_tensors="pt").input_ids[:, :N].cuda()

def chunked(cache):
    """Teacher-forced in chunks so eviction happens BETWEEN forwards. The
    reference arm runs the identical protocol, so chunking is not a confound."""
    nll, n, peak = 0.0, 0, 0
    for lo in range(0, N, CH):
        L = min(CH, N - lo)
        with torch.no_grad():
            lg = model(ids[:, lo:lo + L], past_key_values=cache, use_cache=True).logits
        last = L - 1 if lo + L >= N else L          # final position has no target
        if last > 0:
            lp = torch.log_softmax(lg[0, :last].float(), -1)
            nll += float(-lp.gather(1, ids[0, lo + 1:lo + 1 + last, None]).sum())
            n += last
        cache.evict()          # between forwards: the mask for THIS forward was
                               # already built against the pre-eviction length
        peak = max(peak, cache.memory_bytes())
        del lg; torch.cuda.empty_cache()
    return float(torch.exp(torch.tensor(nll / n))), peak

ARMS = [("full        fp16", dict(quantize_keys=False, quantize_values=False)),
        ("full        nf4 ", dict(quantize_keys=True,  quantize_values=True))]
for w in (512, 256, 128):
    ARMS.append((f"sink4+rec{w:<4} fp16", dict(quantize_keys=False, quantize_values=False,
                                               keep_sink=4, keep_recent=w)))
    ARMS.append((f"sink4+rec{w:<4} nf4 ", dict(quantize_keys=True, quantize_values=True,
                                               keep_sink=4, keep_recent=w)))
rows = []
for name, kw in ARMS:
    p, mb = chunked(NF4KVCache(**kw))
    rows.append((name, p, mb))
b, b_mb = rows[0][1], rows[0][2]
print(f"\n=== {M}  {N} tokens, chunk {CH} ===")
print(f"{'arm':<20} {'ppl':>8} {'d-ppl':>8} {'cache MB':>9} {'vs fp16':>8}")
for n, p, mb in rows:
    print(f"{n:<20} {p:>8.3f} {p-b:>+8.3f} {mb/2**20:>8.2f} {b_mb/mb:>7.2f}x")
json.dump([{"arm": n, "ppl": p, "cache_bytes": mb} for n, p, mb in rows],
          open("/root/g/bench/context/sparsity.json", "w"), indent=2)
