# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Concatenate capture stream dirs into one auditable dir — mechanical only.

For the ceiling data-extension: capture set A (294k) and set B (+294k) are
separate pod runs; the audit wants one dataset. This concatenates the streams
with NO math: arrays stack along axis 0, and each subsequent capture's
record_token indices are offset past the previous maximum, so the loader's
cross-token Δ mask can never pair records across captures (different token
ids ⇒ masked at join, same as any token boundary). E/k/family/decode_only
must match across inputs; meta records the composition transparently.

Usage:
    python combine_streams.py <out_dir> <stream_dir_1> <stream_dir_2> [...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

RP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RP))
from capture.streams import write_capture  # noqa: E402

ARRS = ("hidden_post_block_l", "router_logits_l", "token_embedding", "topk_set")


def main():
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    out = sys.argv[1]
    dirs = [Path(d) for d in sys.argv[2:]]
    metas = [json.loads((d / "meta.json").read_text()) for d in dirs]
    for key in ("E", "k", "family"):
        vals = {m[key] for m in metas}
        if len(vals) != 1:
            raise SystemExit(f"mismatched {key} across inputs: {vals}")
    if not all(m.get("decode_only") for m in metas):
        raise SystemExit("contract violation: a non-decode-only input")

    parts = {a: [] for a in ARRS}
    toks, layers = [], []
    tok_base = 0
    for d in dirs:
        for a in ARRS:
            parts[a].append(np.load(d / f"{a}.npy"))
        t = np.load(d / "record_token.npy")
        toks.append(t + tok_base)
        tok_base = int(toks[-1].max()) + 1
        layers.append(np.load(d / "record_layer.npy"))

    n = sum(p.shape[0] for p in parts["topk_set"])
    meta = dict(metas[0])
    meta.update({
        "records": n,
        "prompts": sum(m.get("prompts", 0) for m in metas),
        "prompt_set": "+".join(str(m.get("prompt_set", "a")) for m in metas),
        "combined_from": [
            {k: m.get(k) for k in ("records", "prompts", "tokens_per_prompt",
                                   "prompt_set", "load", "model")}
            for m in metas
        ],
        "input_note": "concatenated captures (record_token offset per part; "
                      "cross-capture pairs masked like any token boundary)",
    })
    write_capture(out,
                  np.concatenate(parts["hidden_post_block_l"]),
                  np.concatenate(parts["router_logits_l"]),
                  np.concatenate(parts["token_embedding"]),
                  np.concatenate(parts["topk_set"]),
                  meta,
                  record_token=np.concatenate(toks),
                  record_layer=np.concatenate(layers))
    print(f"combined {len(dirs)} captures -> {out}: {n} records "
          f"(prompts={meta['prompts']}, sets={meta['prompt_set']})")


if __name__ == "__main__":
    main()
