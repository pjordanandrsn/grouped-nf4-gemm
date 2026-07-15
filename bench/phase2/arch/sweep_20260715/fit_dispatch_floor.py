import json, glob, os
from collections import defaultdict
SM={"2kada":22,"a4000":48,"4kada":48,"a4500":56,"l4":58,"pro45bw":82,"a100":108,"h200":132,"l40":142,"b200":148}
# also fold in the A2000 off-window (sm 26) if present
recs=[]
for d in sorted(glob.glob(os.path.expanduser("~/gnf4-hwsweep/*/hw_harness.json"))):
    name=d.split("/")[-2]
    if name not in SM: continue
    h=json.load(open(d)); cells=defaultdict(dict)
    for c in h.get("cells",[]):
        if c.get("status")=="ok": cells[(c["regime"],c["model"],c["proj"],c["N"],c["K"])][c["backend"]]=c["ms_median"]
    for (reg,m,p,N,K),b in cells.items():
        if reg=="prefill_s2048" and "dequant_grouped" in b and "fused_nf4" in b:
            recs.append((SM[name],N,K,p,b["dequant_grouped"]/b["fused_nf4"]))
# candidate predicate: loser iff sm < SMT. Find SMT that best separates.
print("Per-(sm,proj) prefill fused/dequant — is there a clean SM threshold?")
bysm=defaultdict(list)
for sm,N,K,p,r in recs: bysm[sm].append((p,r))
for sm in sorted(bysm):
    losers=[f"{p}:{r:.2f}" for p,r in sorted(bysm[sm]) if r<1.0]
    nlose=len(losers); n=len(bysm[sm])
    print(f"sm {sm:>3}: {nlose}/{n} lose | " + (" ".join(losers) if losers else "none <1.0"))
print()
# test threshold SMT: route to dequant iff sm<SMT AND fused<1.0 predicted.
# The DISPATCH can only see (N,K,M,sm) not the ratio. Does 'sm<SMT AND proj==gate_up-shape' work?
# gate_up shape heuristic: N > K (up-proj widens); down: N < K. Check separation.
print("Separator test: predicate = (sm < SMT) AND (N >= K)  [gate_up-ish widening]")
for SMT in [40, 64]:
    tp=fp=tn=fn=0
    for sm,N,K,p,r in recs:
        pred_dequant = (sm < SMT and N >= K)
        actually_loses = r < 1.0
        if pred_dequant and actually_loses: tp+=1
        elif pred_dequant and not actually_loses: fp+=1  # wrongly gave up a win
        elif not pred_dequant and actually_loses: fn+=1  # missed a loss (stays fused, loses)
        else: tn+=1
    print(f" SMT={SMT}: correctly-routed-losers={tp} wrongly-gave-up-wins={fp} missed-losses={fn} correct-keeps={tn}")
# list the actual losers to see their N,K,proj
print("\nAll prefill losers (ratio<1.0):")
for sm,N,K,p,r in sorted(recs):
    if r<1.0: print(f"  sm{sm:>3} {p:<8} N={N:<5} K={K:<5} N>=K={N>=K} -> {r:.2f}")
