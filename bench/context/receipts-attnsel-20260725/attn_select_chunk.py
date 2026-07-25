# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).

"""Post-hoc diagnostic for Experiment A. NOT a registered test.

The static sweep says perplexity improves monotonically as budget moves from
`recent` to `sink`, all the way to sink128+rec4 — which contradicts the usual
result that a few sinks plus a large recent window is the good operating point.
Before that goes in a finding as "sinks beat recency", the obvious alternative
explanation gets tested rather than asserted.

The protocol teacher-forces in chunks of 128 and evicts BETWEEN forwards, so
every query already sees its own chunk in full: a query at index i attends to
all held tokens plus chunk positions 0..i, up to 127 tokens of local context
that arrive free of the budget. `keep_recent` therefore only buys context across
a chunk boundary, and shrinking it to 4 should be nearly free — for reasons that
have nothing to do with sinks being valuable.

If that is the mechanism, shrinking the chunk shrinks the free context and the
optimum must move back toward `recent`. If the optimum stays pinned at
sink-heavy regardless of chunk, the explanation is wrong and sinks really are
carrying it.

The full-cache arm runs at both chunk sizes as a control: teacher forcing with
no eviction is chunk-invariant, so if that number moves, the protocol change did
something unintended and the comparison is void.
"""
import json
import os
import sys

import torch

sys.path.insert(0, "/root/g/kernel")
sys.path.insert(0, "/root/g/bench/context")
import attn_select as A  # noqa: E402
from experts4bit_qlora import load_moe_4bit_streaming  # noqa: E402

OUT = os.environ.get("C_OUT", "/root/g/bench/context/attn_select_chunk.json")
SPLITS = [4, 32, 68, 100, 128]
CHUNKS = [128, 32, 8]


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
    for chunk in CHUNKS:
        A.CH = chunk
        r = A.run(model, wiki, A.Full(), False, None)
        rows.append(dict(r, chunk=chunk, arm="full", dtype="fp16", registered=False))
        print(f"chunk={chunk:3d}  full          ppl={r['ppl']['all']:8.3f}"
              f"   <- control, must not move", flush=True)
        for sink in SPLITS:
            recent = 132 - sink
            r = A.run(model, wiki, A.Recency(sink, recent), False, None)
            rows.append(dict(r, chunk=chunk, arm=f"sink{sink}+rec{recent}",
                             dtype="fp16", budget=132, registered=False))
            print(f"chunk={chunk:3d}  sink{sink:<4}+rec{recent:<4} "
                  f"ppl={r['ppl']['all']:8.3f}  held={r['held_tokens']}", flush=True)
            json.dump(rows, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
