# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Post-hoc diagnostic for Experiment A. NOT a registered test.

Does H2O track the sink-allocation curve, or does it pull ahead when it gets to
select more often? The keep-set dump says H2O is spending its budget on early
tokens; if that is all it is doing, shrinking the chunk should hurt it the way
it hurts sink-heavy static splits (sink128+rec4: best at chunk 128, worst at
chunk 8). If it is really selecting by importance, finer boundaries were
pre-registered as FAVOURING it (prereg confound iii) and it should pull ahead.
"""
import json, sys, torch
sys.path.insert(0, "/root/g/kernel"); sys.path.insert(0, "/root/g/bench/context")
import attn_select as A
from experts4bit_qlora import load_moe_4bit_streaming
from transformers import AutoTokenizer
from datasets import load_dataset

model, _ = load_moe_4bit_streaming(A.MODEL, device="cuda:0", dtype=torch.bfloat16,
                                   r=8, alpha=16, offload=True, pin=True, quant_type="nf4")
model.eval(); model.set_attn_implementation("eager")
n_kv = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
tok = AutoTokenizer.from_pretrained(A.MODEL)
ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
wiki = tok("\n\n".join(x for x in ds["text"] if x.strip()),
           return_tensors="pt").input_ids[:, :A.WIKI_N].cuda()
rows = []
for chunk in (128, 32, 8):
    A.CH = chunk
    r = A.run(model, wiki, A.H2O(A.SINK, A.H2O_RECENT, A.H2O_TOPK, n_kv), False, None)
    rows.append(dict(r, chunk=chunk, arm="h2o-132", dtype="fp16", registered=False))
    print(f"chunk={chunk:3d}  h2o-132  ppl={r['ppl']['all']:8.3f}  held={r['held_tokens']}", flush=True)
    json.dump(rows, open("/root/g/bench/context/attn_select_h2o_chunk.json", "w"), indent=2)
print("wrote")
