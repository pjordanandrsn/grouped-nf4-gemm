#!/usr/bin/env python3
# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).

"""C4 derivation — what a STREAMED KV tier costs, before anything is built.

`kv_budget.py` answers "how much VRAM does the cache take". This answers the
question that decides whether C4 is worth building: **if the cache lives in
pinned host memory and is streamed per layer, what does that cost per decode
step, and where does it stop paying?**

The asymmetry that makes this its own problem. Streamed *weights* amortize over
a batch — one H2D copy of an expert serves every sequence in the step. Streamed
*KV does not*: each sequence owns its own cache, so KV bytes scale with batch
while weight bytes do not. Batching, the standard fix for a transfer-bound
decode, therefore buys less and less until it buys nothing:

    step_bytes = W + batch * KV(ctx)        W = weight stream, per step
    step_time  = L * c + step_bytes / B     L = layers, c = per-copy fixed cost
    tok/s      = batch / step_time

Two numbers fall straight out and they are the whole decision:

  * **ceiling = B / KV(ctx)** — the aggregate tok/s as batch goes to infinity.
    No amount of batching beats it, and it applies even when the weights are
    fully resident and KV is the *only* thing streaming. This is the number for
    the case C4 actually serves: weights fit, context does not.
  * **batch* = W / KV(ctx)** — where KV traffic equals weight traffic. Past it,
    the thing you increased to amortize the weights is what you are now paying
    for.

Quantization moves batch* by exactly its compression ratio, which is a
justification for the NF4 KV work that has nothing to do with fitting in VRAM:
in a streamed regime it buys **batch headroom**.

Honesty about the two sides. The **KV side is exact** — derived from each
model's own config by `kv_budget.derive`, no figure carried over. The **weight
side is a parameter**, because a model's cold-stream bytes/token depends on its
residency plan (the K dial) and no planner exists yet to compute it. It is swept
rather than guessed, with one anchored point: `docs/RESULTS-ikllama-ab.md`
measured Qwen3-235B at 2.29 tok/s on a 26.74 GB/s link, which back-solves to
~11.7 GB/token IF that arm was purely transfer-bound. That inference is labelled
wherever it is used.

Usage:
    python kv_stream_budget.py --link 6.5 --fixed-cost-us 20
    python kv_stream_budget.py --link 26.74          # the A100 transfer-law box
    python kv_stream_budget.py --probe pcie_probe.json   # use measured values
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kv_budget import derive, kv_bytes  # noqa: E402

#: bf16 KV costs 2 B/element. NF4 packs one nibble per element plus one fp32
#: absmax per 64-element block: 0.5 + 4/64 = 0.5625 B/element. The ratio is
#: therefore exactly 32/9 for any head_dim that the blocksize tiles — which is
#: every architecture measured (64/128/256, and MLA's 576). Derived here rather
#: than carried as "3.56x" so the side channel stays visible.
NF4_B_PER_ELEM = 0.5 + 4.0 / 64.0
BF16_B_PER_ELEM = 2.0
NF4_RATIO = BF16_B_PER_ELEM / NF4_B_PER_ELEM

CONTEXTS = (4096, 32768, 131072)
#: Back-solved from the one measured transfer-law point, NOT independently
#: verified: 26.74 GB/s / 2.29 tok/s. Only valid if that arm was purely
#: transfer-bound; treat as an anchor, not a constant.
ANCHOR_W_GB = 26.74 / 2.29
WEIGHT_SWEEP_GB = (2.0, 6.0, 12.0, 24.0)


def load_configs(path):
    with open(path) as f:
        return json.load(f)


def rows(cfgs):
    out = []
    for name, cfg in cfgs.items():
        d = derive(cfg)
        layers = cfg["num_hidden_layers"]          # derive() reports geometry, not depth
        for ctx in CONTEXTS:
            bf16 = kv_bytes(d, ctx)
            out.append(dict(model=name, n_layers=layers, kind=d["kind"],
                            context=ctx, kv_bf16_B=bf16,
                            kv_nf4_B=bf16 / NF4_RATIO))
    return out


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs",
                    default=os.path.join(here, "receipts-c0-20260724",
                                         "config-fields.json"))
    ap.add_argument("--link", type=float, default=None,
                    help="H2D GB/s; omit to use --probe or the 26.74 reference")
    ap.add_argument("--fixed-cost-us", type=float, default=0.0,
                    help="per-copy fixed cost. MEASURED to not accumulate when the "
                         "copies are queued rather than synced individually (see "
                         "pcie_probe: 94 slices cost 1.00x one blob of the same "
                         "bytes), so this defaults to 0 and applies only to a "
                         "sync-per-layer design.")
    ap.add_argument("--probe", default=None,
                    help="pcie_probe.json — takes link + fixed cost from measurement")
    ap.add_argument("--vram-free", type=float, default=8.0,
                    help="GB left for KV after weights (resident or streamed)")
    ap.add_argument("--target-batch", type=int, default=8)
    ap.add_argument("--target-tps", type=float, default=5.0,
                    help="aggregate tok/s the deployment needs")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    link, cost_us, src = a.link, a.fixed_cost_us, "argument"
    if a.probe:
        p = json.load(open(a.probe))
        link = p["fit"]["asymptotic_GBs"]
        cost_us = p["fit"]["fixed_cost_s"] * 1e6
        src = f"measured ({p.get('gpu', '?')})"
    if link is None:
        link, src = 26.74, "docs/RESULTS-ikllama-ab.md reference box"

    data = rows(load_configs(a.configs))
    print(f"NF4 KV compression: {NF4_RATIO:.4f}x  "
          f"({NF4_B_PER_ELEM} vs {BF16_B_PER_ELEM} B/element, derived)")
    print(f"link: {link:.2f} GB/s  per-copy fixed cost: {cost_us:.1f} us  [{src}]")

    print("\n=== KV per SEQUENCE, and the aggregate tok/s ceiling it imposes ===")
    print("ceiling = link / KV(ctx): the cap as batch -> inf, i.e. what you get")
    print("when the weights are RESIDENT and only the cache streams.\n")
    print(f"{'model':<32} {'ctx':>7} {'bf16 GB':>9} {'nf4 GB':>8} "
          f"{'bf16 tok/s':>11} {'nf4 tok/s':>10}")
    for r in data:
        b16, b4 = r["kv_bf16_B"] / 1e9, r["kv_nf4_B"] / 1e9
        floor_s = r["n_layers"] * cost_us / 1e6
        c16 = 1.0 / (b16 / link + floor_s)
        c4 = 1.0 / (b4 / link + floor_s)
        r.update(ceiling_bf16=c16, ceiling_nf4=c4)
        print(f"{r['model']:<32} {r['context']:>7} {b16:>9.3f} {b4:>8.3f} "
              f"{c16:>11.1f} {c4:>10.1f}")

    print("\n=== batch* = W / KV : where KV traffic overtakes weight traffic ===")
    print("Below batch*, batching amortizes the weight stream and KV rides along.")
    print("Above it, further batching mostly buys more KV traffic.\n")
    hdr = "  ".join(f"W={w:g}GB" for w in WEIGHT_SWEEP_GB)
    print(f"{'model':<32} {'ctx':>7} {'fmt':>5}  {hdr}")
    for r in data:
        for fmt, kv in (("bf16", r["kv_bf16_B"]), ("nf4", r["kv_nf4_B"])):
            cells = "  ".join(f"{w * 1e9 / kv:>7.1f}" for w in WEIGHT_SWEEP_GB)
            print(f"{r['model']:<32} {r['context']:>7} {fmt:>5}  {cells}")

    print(f"\n=== anchored point: Qwen3-235B, W = {ANCHOR_W_GB:.1f} GB/token ===")
    print("Back-solved from RESULTS-ikllama-ab.md (26.74 GB/s / 2.29 tok/s), which")
    print("assumes that arm was purely transfer-bound. Verify before building on it.\n")
    print(f"{'ctx':>7} {'fmt':>5} {'KV GB':>7} {'batch*':>7} "
          f"{'KV share @b=1':>14} {'tok/s @b=1':>11}")
    for r in data:
        if not r["model"].startswith("Qwen/Qwen3-235B"):
            continue
        for fmt, kv in (("bf16", r["kv_bf16_B"]), ("nf4", r["kv_nf4_B"])):
            kv_gb = kv / 1e9
            share = kv_gb / (ANCHOR_W_GB + kv_gb)
            tps = 1.0 / ((ANCHOR_W_GB + kv_gb) / link
                         + r["n_layers"] * cost_us / 1e6)
            print(f"{r['context']:>7} {fmt:>5} {kv_gb:>7.3f} "
                  f"{ANCHOR_W_GB / kv_gb:>7.1f} {share * 100:>13.1f}% {tps:>11.2f}")

    print(f"\n=== does C4 have a window? "
          f"(free VRAM {a.vram_free:g} GB, target batch {a.target_batch}, "
          f"target {a.target_tps:g} tok/s aggregate) ===")
    print("Streaming only earns its build if BOTH hold: the resident NF4 cache")
    print("does NOT fit, AND the streamed ceiling still meets the target. If the")
    print("cache fits, C3 already solved it; if the ceiling is below target, no")
    print("amount of engineering on this link gets there.\n")
    print(f"{'model':<32} {'ctx':>7} {'resident b':>10} {'streamed b':>10} "
          f"{'ceiling':>8}  verdict")
    windows = []
    for r in data:
        kv = r["kv_nf4_B"] / 1e9
        # Streaming holds two layer slices (double-buffered), not the whole cache
        slice_gb = 2.0 * kv / r["n_layers"]
        res_b = int(a.vram_free // kv)
        str_b = int(a.vram_free // slice_gb)
        ceiling = link / kv
        if res_b >= a.target_batch:
            v = "resident already fits — C3 is enough"
        elif ceiling < a.target_tps:
            v = f"IMPOSSIBLE — ceiling {ceiling:.1f} < target"
        elif str_b >= a.target_batch:
            v = "**WINDOW** — streaming is the only way to fit"
        else:
            v = "does not fit even streamed"
        r.update(resident_max_batch=res_b, streamed_max_batch=str_b,
                 ceiling_nf4_tps=ceiling, verdict=v)
        windows.append(v)
        print(f"{r['model']:<32} {r['context']:>7} {res_b:>10} {str_b:>10} "
              f"{ceiling:>8.1f}  {v}")
    n_win = sum(1 for v in windows if "WINDOW" in v)
    print(f"\n{n_win}/{len(windows)} cells are a genuine window for C4 at this "
          f"link and budget.")
    if not n_win:
        print("=> On these inputs C4 does not pay. Either the resident NF4 cache "
              "already fits\n   (C3 solved it) or the link cannot reach the target. "
              "Re-run with the real\n   target before building anything.")

    if a.json:
        json.dump(dict(link_GBs=link, fixed_cost_us=cost_us, source=src,
                       nf4_ratio=NF4_RATIO, anchor_W_GB=ANCHOR_W_GB,
                       vram_free_GB=a.vram_free, target_batch=a.target_batch,
                       target_tps=a.target_tps, rows=data),
                  open(a.json, "w"), indent=2)
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
