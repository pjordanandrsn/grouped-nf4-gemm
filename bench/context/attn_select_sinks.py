# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).

"""Post-hoc diagnostic for Experiment A. NOT a registered test.

The keep-set dump (``attn_select_oracle.json``) shows H2O and the future-knowing
oracle landing on nearly the same tokens, and that set is ~50 CONTIGUOUS
positions from the start of the sequence plus a thin scattered tail. Accumulated
attention mass is confounded by opportunity: summing over query positions
rewards a token for how many queries *could* attend to it, and that count falls
linearly with position. Position 5 is scored by ~500 queries, position 400 by
~100.

So the H2O arm may not be doing importance selection at all — it may be a
"spend more of the budget on sinks" arm wearing a different name. That is
testable without any attention: sweep the static sink/recent split at a FIXED
budget. If H2O sits inside the static family rather than above it, A3's
confirmation is a StreamingLLM result, not an H2O one, and it should be reported
that way.
"""
import json
import os
import sys

import torch

sys.path.insert(0, "/root/g/kernel")
sys.path.insert(0, "/root/g/bench/context")
import attn_select as A  # noqa: E402
from experts4bit_qlora import load_moe_4bit_streaming  # noqa: E402

OUT = os.environ.get("S_OUT", "/root/g/bench/context/attn_select_sinks.json")
SEEDS = [int(s) for s in os.environ.get("S_SEEDS", "0,1,2").split(",")]
SPLITS_132 = [4, 16, 32, 68, 100, 128]           # sink; recent = 132 - sink
SPLITS_260 = [4, 68, 132, 196, 256]              # sink; recent = 260 - sink


def main():
    model, _ = load_moe_4bit_streaming(A.MODEL, device="cuda:0", dtype=torch.bfloat16,
                                       r=8, alpha=16, offload=True, pin=True,
                                       quant_type="nf4")
    model.eval()
    model.set_attn_implementation("eager")

    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(A.MODEL)
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    wiki = tok("\n\n".join(x for x in ds["text"] if x.strip()),
               return_tensors="pt").input_ids[:, :A.WIKI_N].cuda()

    rows = []
    for budget, splits in ((132, SPLITS_132), (260, SPLITS_260)):
        for sink in splits:
            recent = budget - sink
            r = A.run(model, wiki, A.Recency(sink, recent), False, None)
            r.update(fixture="wikitext", arm=f"sink{sink}+rec{recent}", dtype="fp16",
                     budget=budget, registered=False)
            rows.append(r)
            print(f"wikitext  budget={budget:3d}  sink{sink:<4}+rec{recent:<4} "
                  f"ppl={r['ppl']['all']:8.3f}  held={r['held_tokens']}", flush=True)
            json.dump(rows, open(OUT, "w"), indent=2)

    for sink in SPLITS_132:
        recent = 132 - sink
        for seed in SEEDS:
            g = torch.Generator().manual_seed(seed)
            base = torch.randint(1000, 20000, (A.IND_HALF,), generator=g)
            ids = torch.cat([base, base]).unsqueeze(0).cuda()
            r = A.run(model, ids, A.Recency(sink, recent), False, A.IND_HALF)
            r.update(fixture=f"induction-s{seed}", arm=f"sink{sink}+rec{recent}",
                     dtype="fp16", budget=132, registered=False, seed=seed)
            rows.append(r)
            print(f"induction-s{seed}  sink{sink:<4}+rec{recent:<4} "
                  f"2nd={r['ppl']['second']:11.3f}", flush=True)
            json.dump(rows, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
