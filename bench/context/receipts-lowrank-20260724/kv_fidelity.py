import sys, json, math, torch
sys.path.insert(0,"/root/g/kernel")
from experts4bit_qlora import load_moe_4bit_streaming, NF4KVCache
from transformers import AutoTokenizer
M="allenai/OLMoE-1B-7B-0924"; N=1024; GEN=96
tok=AutoTokenizer.from_pretrained(M)
model,_=load_moe_4bit_streaming(M,device="cuda:0",dtype=torch.bfloat16,r=8,alpha=16,offload=True,pin=True,quant_type="nf4")
model.eval()
from datasets import load_dataset
ds=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="test")
ids=tok("\n\n".join(t for t in ds["text"] if t.strip()),return_tensors="pt").input_ids[:,:N].cuda()

def run(cache):
    with torch.no_grad():
        o=model(ids,use_cache=True,past_key_values=cache)
    return o.logits.float()

def ppl(lg):
    lp=torch.log_softmax(lg[0,:-1],-1)
    return float(torch.exp(-lp.gather(1,ids[0,1:,None]).mean()))

CFG=[("fp16 cache (baseline)",None),
     ("K4 V4  (both nf4)",dict(quantize_keys=True ,quantize_values=True )),
     ("K4 V16 (keys only)", dict(quantize_keys=True ,quantize_values=False)),
     ("K16 V4 (values only)",dict(quantize_keys=False,quantize_values=True ))]
base_lg=None; rows=[]
for name,kw in CFG:
    lg=run(None if kw is None else NF4KVCache(**kw))
    if base_lg is None: base_lg=lg
    am_b, am = base_lg[0].argmax(-1), lg[0].argmax(-1)
    agree=float((am_b==am).float().mean())
    dlog=float((lg-base_lg).norm()/base_lg.norm())
    rows.append((name,ppl(lg),agree,dlog))
    del lg; torch.cuda.empty_cache()

print(f"{'config':<22} {'ppl':>7} {'d-ppl':>7} {'argmax agree':>13} {'logit rel':>10}")
b=rows[0][1]
for n,p,a,d in rows:
    print(f"{n:<22} {p:>7.3f} {p-b:>+7.3f} {a*100:>12.2f}% {d:>10.4f}")

# free-running greedy: what a user actually sees
prompt=ids[:,:256]
def gen(cache):
    with torch.no_grad():
        return model.generate(prompt,max_new_tokens=GEN,do_sample=False,
                              use_cache=True,past_key_values=cache,
                              pad_token_id=tok.eos_token_id)[0,256:]
ref=gen(None)
print("\nfree-running greedy, 96 new tokens from a 256-token prompt:")
for name,kw in CFG[1:]:
    got=gen(NF4KVCache(**kw))
    n=min(len(ref),len(got))
    same=(ref[:n]==got[:n])
    first=int((~same).nonzero()[0]) if (~same).any() else -1
    print(f"  {name:<22} identical={bool(same.all())}  first divergence: "
          f"{'none' if first<0 else f'token {first}'}  match {int(same.sum())}/{n}")
json.dump([{"config":n,"ppl":p,"argmax_agree":a,"logit_rel":d} for n,p,a,d in rows],
          open("/root/g/bench/context/kv_fidelity.json","w"),indent=2)
