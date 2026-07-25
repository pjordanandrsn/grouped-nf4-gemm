#!/usr/bin/env python3
# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).

"""M1 of PREREG-composition-scout.md — weights and KV on the same link.

A 2x2 over RESIDENCE with quantization held constant (NF4 KV in every arm), so
the only thing varying is where bytes live:

    neither   weights resident,  KV resident     <- floor
    W-only    weights streamed,  KV resident
    KV-only   weights resident,  KV streamed
    both      weights streamed,  KV streamed

Additivity is then `(both - neither)` against the sum of the two singles. That
decomposition is the entire point; measuring only `both` would say what it costs
and nothing about why.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.environ.get("GNF4_KERNEL", "/root/work"))
from experts4bit_qlora import NF4KVCache, load_moe_4bit_streaming  # noqa: E402

OUT = os.environ.get("MC_OUT", "/root/work/composition.json")
MODEL = os.environ.get("MC_MODEL", "Qwen/Qwen3-30B-A3B")
CONTEXTS = [int(x) for x in os.environ.get("MC_CTX", "4096,32768").split(",")]
NEW = int(os.environ.get("MC_NEW", "16"))
REPS = int(os.environ.get("MC_REPS", "3"))
SEED = int(os.environ.get("MC_SEED", "0"))


def decode(model, ids, cache, new=NEW):
    out = []
    with torch.no_grad():
        o = model(ids, past_key_values=cache, use_cache=True)
        nxt = o.logits[:, -1:].argmax(-1); out.append(int(nxt)); del o
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(new):
            o = model(nxt, past_key_values=cache, use_cache=True)
            nxt = o.logits[:, -1:].argmax(-1); out.append(int(nxt)); del o
        torch.cuda.synchronize()
        return out, (time.perf_counter() - t0) / new


def main():
    models = {}
    for w_streamed in (False, True):
        models[w_streamed], _ = load_moe_4bit_streaming(
            MODEL, device="cuda:0", dtype=torch.bfloat16, r=8, alpha=16,
            offload=w_streamed, pin=w_streamed, quant_type="nf4")
        models[w_streamed].eval()
        print(f"loaded weights_streamed={w_streamed}: "
              f"resident {torch.cuda.memory_allocated()/2**30:.2f} GiB", flush=True)

    cfg = models[False].config
    print(f"{MODEL}: L={cfg.num_hidden_layers} q={cfg.num_attention_heads} "
          f"kv={cfg.num_key_value_heads} (GQA {cfg.num_attention_heads//cfg.num_key_value_heads}:1)",
          flush=True)
    rows, toks = [], {}
    for ctx in CONTEXTS:
        g = torch.Generator(device="cuda").manual_seed(SEED)
        ids = torch.randint(100, 20000, (1, ctx), device="cuda", generator=g)
        for name, w_str, kv_str in (("neither", False, False), ("W-only", True, False),
                                    ("KV-only", False, True), ("both", True, True)):
            ts = []
            for _ in range(REPS):
                c = (NF4KVCache(residence="host", max_context=ctx + NEW + 64)
                     if kv_str else NF4KVCache())
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                o, dt = decode(models[w_str], ids, c)
                ts.append(dt); peak = torch.cuda.max_memory_allocated()
                dev = c.device_bytes(); del c
            toks[(ctx, name)] = o
            rows.append(dict(context=ctx, arm=name, weights_streamed=w_str,
                             kv_streamed=kv_str, s_per_step=statistics.median(ts),
                             peak_bytes=peak, kv_device_bytes=dev))
            print(f"ctx={ctx:>6} {name:<8} {statistics.median(ts)*1e3:8.1f} ms/step  "
                  f"peak={peak/2**30:6.2f} GiB  kv_on_device={dev/1e6:7.1f} MB", flush=True)
            json.dump(rows, open(OUT, "w"), indent=2)

    def g_(ctx, name):
        return next(r for r in rows if r["context"] == ctx and r["arm"] == name)

    print("\n=== M1 scoring ===")
    v = {}
    for ctx in CONTEXTS:
        n = g_(ctx, "neither")["s_per_step"]
        w = g_(ctx, "W-only")["s_per_step"] - n
        k = g_(ctx, "KV-only")["s_per_step"] - n
        b = g_(ctx, "both")["s_per_step"] - n
        add = b / (w + k) if (w + k) > 0 else float("nan")
        v[f"M1a_{ctx}"] = dict(w_cost_ms=w*1e3, kv_cost_ms=k*1e3, both_cost_ms=b*1e3,
                               additivity=add)
        print(f"ctx={ctx:>6}  W adds {w*1e3:7.1f} ms | KV adds {k*1e3:7.1f} ms | "
              f"both adds {b*1e3:7.1f} ms | additivity {add:.3f}")
    a0 = v[f"M1a_{CONTEXTS[0]}"]["additivity"]
    print(f"\nM1a @{CONTEXTS[0]} = {a0:.3f}   [0.85,1.20] confirms, outside [0.70,1.60] falsifies  "
          f"{'CONFIRMED' if 0.85<=a0<=1.20 else ('FALSIFIED' if not (0.70<=a0<=1.60) else 'outside interval')}")
    gate = all(toks[(c, 'neither')] == toks[(c, 'KV-only')] and
               toks[(c, 'W-only')] == toks[(c, 'both')] for c in CONTEXTS)
    print(f"M1b GATE ids identical across residence = {gate}   "
          f"{'CONFIRMED' if gate else 'FALSIFIED'}")
    pb = g_(CONTEXTS[-1], "both")["peak_bytes"] / 2**30
    print(f"M1c both-arm peak = {pb:.2f} GiB   <10 confirms, >14 falsifies  "
          f"{'CONFIRMED' if pb<10 else ('FALSIFIED' if pb>14 else 'outside interval')}")
    if len(CONTEXTS) > 1:
        a1 = v[f"M1a_{CONTEXTS[1]}"]["additivity"]
        print(f"M1d additivity {CONTEXTS[0]}->{CONTEXTS[1]}: {a0:.3f} -> {a1:.3f}   "
              f"{'CONFIRMED' if a1 >= a0-0.20 else 'FALSIFIED'}")
    json.dump(dict(rows=rows, scoring=v, gate=gate), open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
