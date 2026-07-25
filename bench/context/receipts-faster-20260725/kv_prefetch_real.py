#!/usr/bin/env python3
# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).

"""E1 of PREREG-kv-stream-faster.md (amendment 1) — B1 on a real model.

B1 measured 95.9% of the streamed transfer hidden, with one stated confound: the
harness's "compute" was dequantization, not attention. The argument that a real
decode has MORE to hide behind is plausible and untested, and this project has a
standing record of plausible mechanisms being falsified.

Weights are RESIDENT, not offloaded. Streaming them would contend for the same
link and make the KV term unattributable, and "weights fit, context does not" is
exactly the case the streamed tier is for.

Prefetch is driven by a forward pre-hook per decoder layer requesting layer i+1
-- the shape `offload.py` already uses -- rather than staging everything up
front, which would put the whole cache on the device and defeat the point.
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

OUT = os.environ.get("E1_OUT", "/root/g/bench/context/kv_prefetch_real.json")
MODEL = os.environ.get("E1_MODEL", "allenai/OLMoE-1B-7B-0924")
PROMPT_LEN = int(os.environ.get("E1_PROMPT", "4096"))
NEW = int(os.environ.get("E1_NEW", "24"))
REPS = int(os.environ.get("E1_REPS", "3"))


def attach_prefetch(model, cache):
    """Pre-hook per decoder layer: when layer i is about to run, start i+1's
    copy. Returns handles so the hooks come off again — a benchmark that leaves
    hooks installed measures the next arm too."""
    hooks = []
    layers = model.model.layers
    for i, layer in enumerate(layers):
        def pre(_mod, _args, idx=i):
            cache.prefetch(idx + 1)
            return None
        hooks.append(layer.register_forward_pre_hook(pre))
    return hooks


def decode(model, ids, cache, prefetch, new=NEW):
    """Greedy decode. Returns (token ids, seconds per step)."""
    hooks = attach_prefetch(model, cache) if prefetch else []
    try:
        out_ids = []
        with torch.no_grad():
            if prefetch:
                cache.prefetch(0)
            o = model(ids, past_key_values=cache, use_cache=True)
            nxt = o.logits[:, -1:].argmax(-1)
            out_ids.append(int(nxt))
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(new):
                if prefetch:
                    cache.prefetch(0)
                o = model(nxt, past_key_values=cache, use_cache=True)
                nxt = o.logits[:, -1:].argmax(-1)
                out_ids.append(int(nxt))
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / new
        return out_ids, dt
    finally:
        for h in hooks:
            h.remove()


def make_cache(kind, ctx):
    if kind == "resident":
        return NF4KVCache()
    return NF4KVCache(residence="host", max_context=ctx + NEW + 64)


def main():
    # 4-bit, and offload=False so the weights are RESIDENT: streamed weights
    # would contend for the same link and make the KV term unattributable.
    # bf16 would be 14 GB on a 12 GB card, which is why this is not a plain
    # from_pretrained.
    model, _ = load_moe_4bit_streaming(MODEL, device="cuda:0",
                                       dtype=torch.bfloat16, r=8, alpha=16,
                                       offload=False, pin=False, quant_type="nf4")
    model.eval()
    cfg = model.config
    ids = torch.randint(100, 20000, (1, PROMPT_LEN), device="cuda")
    packed_per_token = 2 * cfg.num_key_value_heads * (
        cfg.hidden_size // cfg.num_attention_heads) * (0.5 + 4.0 / 64.0)
    kv_bytes = packed_per_token * cfg.num_hidden_layers * PROMPT_LEN
    print(f"{MODEL}: L={cfg.num_hidden_layers} kv={cfg.num_key_value_heads} "
          f"prompt={PROMPT_LEN} new={NEW}", flush=True)
    print(f"packed KV at prompt: {kv_bytes / 1e6:.1f} MB "
          f"-> {kv_bytes / 6.20e9 * 1e3:.1f} ms/step at 6.20 GB/s", flush=True)

    rows, toks = [], {}
    for name, kind, pre in (("resident", "resident", False),
                            ("streamed", "host", False),
                            ("streamed+pre", "host", True)):
        ts = []
        for r in range(REPS):
            c = make_cache(kind, PROMPT_LEN)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            out, dt = decode(model, ids, c, pre)
            ts.append(dt)
            peak = torch.cuda.max_memory_allocated()
            dev = c.device_bytes()
            del c
        toks[name] = out
        rows.append(dict(arm=name, prefetch=pre, s_per_step=statistics.median(ts),
                         peak_bytes=peak, device_bytes=dev,
                         kv_packed_bytes=kv_bytes))
        print(f"{name:<13} {statistics.median(ts) * 1e3:8.2f} ms/step  "
              f"peak={peak / 2**20:7.1f} MB  kv_on_device={dev / 1e6:7.1f} MB",
              flush=True)
        json.dump(rows, open(OUT, "w"), indent=2)

    base = rows[0]["s_per_step"]
    plain = rows[1]["s_per_step"]
    pre = rows[2]["s_per_step"]
    e1a = pre / base
    hidden = 1 - (pre - base) / max(plain - base, 1e-12)
    e1c = toks["resident"] == toks["streamed+pre"]
    v = {
        "E1a": dict(measured=e1a, interval=[1.00, 1.20],
                    verdict="CONFIRMED" if e1a <= 1.35 else "FALSIFIED",
                    inside_interval=1.00 <= e1a <= 1.20),
        "E1b": dict(measured_hidden=hidden, threshold=0.90,
                    verdict="CONFIRMED" if hidden >= 0.90 else
                    ("FALSIFIED" if hidden < 0.80 else "outside interval")),
        "E1c": dict(identical=e1c,
                    verdict="CONFIRMED" if e1c else "FALSIFIED"),
    }
    print("\n=== scoring ===")
    print(f"E1a prefetched/resident = {e1a:.3f}  [1.00,1.20], falsify >1.35  "
          f"{v['E1a']['verdict']}")
    print(f"E1b transfer hidden     = {hidden * 100:.1f}%  >=90%  "
          f"{v['E1b']['verdict']}")
    print(f"E1c greedy ids identical= {e1c}   {v['E1c']['verdict']}")
    if not e1c:
        print(f"    resident: {toks['resident'][:12]}")
        print(f"    streamed: {toks['streamed+pre'][:12]}")
    print(f"E1d exposed transfer: {(plain - base) * 1e3:.2f} ms -> "
          f"{(pre - base) * 1e3:.2f} ms/step")
    json.dump(dict(rows=rows, verdicts=v, tokens=toks), open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
