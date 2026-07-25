#!/usr/bin/env python3
# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).

"""F1 of PREREG-kv-vs-bf16.md — the control this project never ran.

Sixteen findings of KV latency, and every control is another NF4 configuration.
This one is `DynamicCache`: transformers' own bf16 cache, what a user runs by
default. The gap is what NF4 KV actually costs in time, against the 3.56x memory
it actually buys — the two halves of the trade, measured together, on one model.

OLMoE is GQA 1:1, so none of D1's `enable_gqa` effect applies here. What is
measured is the DEQUANT — the thing this module ships — not the fused kernel it
does not use.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, "/root/g/kernel")
from experts4bit_qlora import NF4KVCache, load_moe_4bit_streaming  # noqa: E402
from transformers import DynamicCache  # noqa: E402

OUT = os.environ.get("VB_OUT", "/root/g/bench/context/kv_vs_bf16.json")
MODEL = os.environ.get("VB_MODEL", "allenai/OLMoE-1B-7B-0924")
CONTEXTS = [int(x) for x in os.environ.get("VB_CTX", "4096,16384").split(",")]
NEW = int(os.environ.get("VB_NEW", "32"))
REPS = int(os.environ.get("VB_REPS", "3"))


def decode(model, ids, cache, new=NEW):
    out_ids = []
    with torch.no_grad():
        o = model(ids, past_key_values=cache, use_cache=True)
        nxt = o.logits[:, -1:].argmax(-1)
        out_ids.append(int(nxt))
        del o
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(new):
            o = model(nxt, past_key_values=cache, use_cache=True)
            nxt = o.logits[:, -1:].argmax(-1)
            out_ids.append(int(nxt))
            del o
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / new
    return out_ids, dt


def make(kind, ctx):
    if kind == "bf16":
        return DynamicCache()
    if kind == "nf4":
        return NF4KVCache()
    return NF4KVCache(residence="host", max_context=ctx + NEW + 64)


def bytes_of(cache):
    if isinstance(cache, NF4KVCache):
        return cache.memory_bytes(), cache.device_bytes()
    total = 0
    for ly in getattr(cache, "layers", []):
        for t in (getattr(ly, "keys", None), getattr(ly, "values", None)):
            if torch.is_tensor(t):
                total += t.numel() * t.element_size()
    return total, total


def main():
    model, _ = load_moe_4bit_streaming(MODEL, device="cuda:0",
                                       dtype=torch.bfloat16, r=8, alpha=16,
                                       offload=False, pin=False, quant_type="nf4")
    model.eval()
    print(f"{MODEL}  contexts={CONTEXTS}  new={NEW}", flush=True)
    rows, toks = [], {}
    for ctx in CONTEXTS:
        ids = torch.randint(100, 20000, (1, ctx), device="cuda")
        for kind in ("bf16", "nf4", "nf4-host"):
            ts = []
            for _ in range(REPS):
                c = make(kind, ctx)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                out, dt = decode(model, ids, c)
                ts.append(dt)
                peak = torch.cuda.max_memory_allocated()
                cb, db = bytes_of(c)
                del c
            toks[(ctx, kind)] = out
            rows.append(dict(context=ctx, cache=kind, s_per_step=statistics.median(ts),
                             peak_bytes=peak, cache_bytes=cb, device_bytes=db))
            print(f"ctx={ctx:>6} {kind:<9} {statistics.median(ts) * 1e3:8.2f} ms/step  "
                  f"peak={peak / 2**20:7.1f} MB  cache={cb / 1e6:7.1f} MB "
                  f"(on device {db / 1e6:6.1f})", flush=True)
            json.dump(rows, open(OUT, "w"), indent=2)

    def g(ctx, kind):
        return next(r for r in rows if r["context"] == ctx and r["cache"] == kind)

    c0 = CONTEXTS[0]
    f1a = g(c0, "nf4")["s_per_step"] / g(c0, "bf16")["s_per_step"]
    f1d = g(c0, "nf4-host")["s_per_step"] / g(c0, "bf16")["s_per_step"]
    f1c = (g(c0, "bf16")["peak_bytes"] - g(c0, "nf4")["peak_bytes"]) / 2 ** 20
    print("\n=== scoring ===")
    print(f"F1a nf4/bf16 @{c0}      = {f1a:.3f}   [1.2,2.0], falsify outside [1.0,3.0]  "
          f"{'CONFIRMED' if 1.2 <= f1a <= 2.0 else ('FALSIFIED' if not (1.0 <= f1a <= 3.0) else 'outside interval')}")
    v = {"F1a": dict(measured=f1a, interval=[1.2, 2.0]),
         "F1c": dict(measured_mb=f1c, interval=[300, 450]),
         "F1d": dict(measured=f1d, interval=[1.4, 2.6])}
    if len(CONTEXTS) > 1:
        c1 = CONTEXTS[1]
        r1 = g(c1, "nf4")["s_per_step"] / g(c1, "bf16")["s_per_step"]
        v["F1b"] = dict(ratio_lo=f1a, ratio_hi=r1,
                        verdict="CONFIRMED" if r1 >= f1a - 0.05 else "FALSIFIED")
        print(f"F1b ratio {c0}->{c1}    = {f1a:.3f} -> {r1:.3f}   must not fall  "
              f"{v['F1b']['verdict']}")
    print(f"F1c bf16 peak - nf4 peak = {f1c:+.1f} MB  [300,450]  "
          f"{'CONFIRMED' if 300 <= f1c <= 450 else 'FALSIFIED'}")
    print(f"F1d nf4-host/bf16 @{c0}  = {f1d:.3f}   [1.4,2.6]  "
          f"{'CONFIRMED' if 1.4 <= f1d <= 2.6 else ('FALSIFIED' if not (1.0 <= f1d <= 3.5) else 'outside interval')}")
    a, b = toks[(c0, "bf16")], toks[(c0, "nf4")]
    div = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
    print(f"\nreported: greedy ids diverge at position {div} of {len(a)} "
          f"(bf16 vs nf4; lossy by design, see finding #10)")
    json.dump(dict(rows=rows, verdicts=v, divergence_at=div,
                   tokens={f"{k[0]}-{k[1]}": val for k, val in toks.items()}),
              open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
