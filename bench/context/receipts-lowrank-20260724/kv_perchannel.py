import sys, json, torch
sys.path.insert(0,"/root/g/kernel")
from experts4bit_qlora import load_moe_4bit_streaming, NF4KVCache
from transformers import AutoTokenizer
M="allenai/OLMoE-1B-7B-0924"; N=1024
tok=AutoTokenizer.from_pretrained(M)
model,_=load_moe_4bit_streaming(M,device="cuda:0",dtype=torch.bfloat16,r=8,alpha=16,offload=True,pin=True,quant_type="nf4")
model.eval()
from datasets import load_dataset
ds=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="test")
ids=tok("\n\n".join(t for t in ds["text"] if t.strip()),return_tensors="pt").input_ids[:,:N].cuda()
def run(c):
    with torch.no_grad(): return model(ids,use_cache=True,past_key_values=c).logits.float()
def ppl(lg):
    lp=torch.log_softmax(lg[0,:-1],-1); return float(torch.exp(-lp.gather(1,ids[0,1:,None]).mean()))
PC=dict(key_scaling="per_channel",group=64)
CFG=[("fp16 cache (baseline)",None),
     ("K4    V4   per-token", dict(quantize_keys=True,quantize_values=True)),
     ("K4pc  V4   per-chan K",dict(quantize_keys=True,quantize_values=True,**PC)),
     ("K4    V16  per-token", dict(quantize_keys=True,quantize_values=False)),
     ("K4pc  V16  per-chan K",dict(quantize_keys=True,quantize_values=False,**PC))]
base=None; rows=[]
for name,kw in CFG:
    c=None if kw is None else NF4KVCache(**kw)
    lg=run(c)
    if base is None: base=lg; mb=None
    else: mb=c.memory_bytes()
    ag=float((base[0].argmax(-1)==lg[0].argmax(-1)).float().mean())
    rows.append((name,ppl(lg),ag,mb)); del lg; torch.cuda.empty_cache()
b=rows[0][1]
print(f"{'config':<24} {'ppl':>7} {'d-ppl':>7} {'argmax':>8} {'cache MB':>9}")
for n,p,a,mb in rows:
    print(f"{n:<24} {p:>7.3f} {p-b:>+7.3f} {a*100:>7.2f}% {'-' if mb is None else f'{mb/2**20:>8.2f}'}")
json.dump([{"config":n,"ppl":p,"argmax_agree":a,"cache_bytes":mb} for n,p,a,mb in rows],
          open("/root/g/bench/context/kv_fidelity_perchannel.json","w"),indent=2)
