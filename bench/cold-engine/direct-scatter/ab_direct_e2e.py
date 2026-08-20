"""End-to-end A/B: does the preadv scatter show up in a SERVED step?

gnf4#119 measured the fill path in isolation (~43% faster). This asks the
different, harder question: how much of a decode step does that recover,
with a model, a router, attention, and a GPU all competing for the same
wall.

The only variable is cold_direct. Same manifest, same routing, same tier
sizing, same cold destination -- the arms differ in whether a cold row's
segments are DMA'd straight into the kernel-shaped stacks or copied there
from an arena row.

A/B/A per instrument law 6, and each arm ASSERTS which landing it actually
took, because a silent fallback to the copy path would make both arms the
same measurement wearing different labels.
"""
import argparse
import json
import statistics
import time

import torch
from transformers import AutoTokenizer

from experts4bit_qlora.engines import hybrid as hy
from experts4bit_qlora.loader import load_moe_4bit_streaming

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/root/models/olmoe")
ap.add_argument("--arena", default="/root/models/olmoe.arena")
ap.add_argument("--manifest", default="/root/man_cold05.json")
ap.add_argument("--hot-rows", type=int, default=384)
ap.add_argument("--steps", type=int, default=128)
ap.add_argument("--out", default="/root/ab_direct.json")
a = ap.parse_args()

PROSE = ("The question of how memory works has occupied philosophers and "
         "scientists for centuries. When we recall an event, we do not "
         "replay a recording; we reconstruct it, and the reconstruction "
         "is shaped by everything we have learned since. This is why "
         "eyewitness testimony is less reliable than juries assume. ")
tk = AutoTokenizer.from_pretrained(a.model)
ids = tk(PROSE * 4, return_tensors="pt").input_ids[:, :64].to("cuda")
man = json.load(open(a.manifest))


def arm(tag, direct):
    model, _ = load_moe_4bit_streaming(a.model, device="cuda",
                                       dtype=torch.bfloat16, r=8, alpha=16,
                                       quant_type="nf4", arena=a.arena)
    n = hy.enable_hybrid_tier(model, a.arena, man, hot_rows=a.hot_rows,
                              cold_dest="cpu", cold_direct=direct)
    assert n == 16, n
    st = hy.cold_stats(model)
    want = "direct-scatter" if direct else "copy"
    assert st["cold_landing"] == want, (
        "arm %s asked for %s but took %s (%s) -- the arms would be the same "
        "measurement with different labels"
        % (tag, want, st["cold_landing"], st.get("cold_landing_fallback")))
    toks, per = [], []
    with torch.no_grad():
        for _ in range(3):
            o = model(ids, use_cache=True)
            c, p = o.logits[:, -1:].argmax(-1), o.past_key_values
            for _ in range(4):
                o = model(c, past_key_values=p, use_cache=True)
                c, p = o.logits[:, -1:].argmax(-1), o.past_key_values
        torch.cuda.synchronize()
        pre = hy.cold_stats(model)
        o = model(ids, use_cache=True)
        p, c = o.past_key_values, o.logits[:, -1:].argmax(-1)
        for _ in range(a.steps):
            t0 = time.perf_counter_ns()
            o = model(c, past_key_values=p, use_cache=True)
            torch.cuda.synchronize()
            per.append(time.perf_counter_ns() - t0)
            p, c = o.past_key_values, o.logits[:, -1:].argmax(-1)
            toks.append(int(c.item()))
    post = hy.cold_stats(model)
    r = {"landing": st["cold_landing"], "median_ns": statistics.median(per),
         "total_ns": sum(per),
         "reads_in_window": post.get("disk_reads", 0) - pre.get("disk_reads", 0),
         "cold_rows_cpu": post.get("cold_rows_cpu"), "tokens": toks}
    print("%-10s landing=%-14s median %7.2f ms | win_reads %5d | cold_cpu %6d"
          % (tag, r["landing"], r["median_ns"] / 1e6, r["reads_in_window"],
             r["cold_rows_cpu"]))
    hy.disable_hybrid_tier(model)
    del model
    torch.cuda.empty_cache()
    return r


res = {"schema": "e4b-direct-e2e/1", "config": vars(a), "arms": {}}
for tag, d in (("copy-A", False), ("direct", True), ("copy-B", False)):
    res["arms"][tag] = arm(tag, d)

cA, cB, dd = (res["arms"][k] for k in ("copy-A", "direct", "copy-B"))
base = min(cA["median_ns"], dd["median_ns"])
sp = abs(cA["median_ns"] - dd["median_ns"]) / base
res["self_pair_spread"] = sp
res["direct_vs_copy"] = (cB["median_ns"] - base) / base
res["tokens_identical"] = cA["tokens"] == cB["tokens"] == dd["tokens"]
print("\nself-pair (copy vs copy): %.2f%%   direct vs copy: %+.2f%%"
      % (sp * 100, res["direct_vs_copy"] * 100))
print("tokens identical across arms:", res["tokens_identical"])
if abs(res["direct_vs_copy"]) <= sp:
    print("VERDICT: inside the instrument's own spread -- no measurable "
          "end-to-end difference at this cold mass")
json.dump(res, open(a.out, "w"), indent=2)
