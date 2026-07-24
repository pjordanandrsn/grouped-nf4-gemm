#!/usr/bin/env python3
"""KV-cache budget derivation (Phase C0) — from config.json only.

Emits the KB/token slope, the bounded floor, and VRAM-at-context table that
docs/context-budgets.md publishes. No figure here is carried over from prior
docs or conversation: everything is computed from each model's own config.

The derivation handles four structures, because the naive
`2 * num_key_value_heads * head_dim * n_layers` formula is wrong for three of them:

  uniform  — every layer full-attention (Qwen3, OLMoE).
  hybrid   — interleaved sliding/full (gpt-oss, Gemma-4). Only FULL layers grow
             with context; sliding layers converge to a constant `window - 1`
             tokens. Reporting one blended KB/token for these is meaningless.
  per-type geometry — Gemma-4's full ("global") layers use
             num_global_key_value_heads x global_head_dim, which differs from the
             top-level pair its sliding layers use (measured: 2x512 vs 8x256).
  kv-shared — layers at/after (n_layers - num_kv_shared_layers) allocate no KV
             at all and reuse an earlier layer's cache.
  MLA      — a joint compressed latent + decoupled rope key, NOT separate K and
             V, so there is no factor of 2 (Kimi/DeepSeek-style).

Usage:
    python kv_budget.py --configs receipts-c0-20260724/config-fields.json
    python kv_budget.py --fetch Qwen/Qwen3-235B-A22B-Instruct-2507   # needs HF
"""
from __future__ import annotations

import argparse
import json
import os
import sys

BYTES_PER_ELEM = 2  # fp16/bf16 KV — the transformers default. C3 adds q8/q4.
CONTEXTS = (4096, 8192, 32768, 131072)


def _get(cfg, key, default=None):
    """Field access that works on both dicts and HF config objects."""
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def derive(cfg, n_layers: int | None = None, bytes_per_elem: int = BYTES_PER_ELEM) -> dict:
    """Derive KV cost from a config (dict or HF config object).

    Returns slope_b (unbounded bytes/token) and floor_b (bounded constant).
    Total KV bytes at context C = slope_b * C + floor_b.
    """
    layers = n_layers if n_layers is not None else _get(cfg, "num_hidden_layers")
    heads = _get(cfg, "num_attention_heads")
    kv_heads = _get(cfg, "num_key_value_heads", heads) or heads
    head_dim = _get(cfg, "head_dim") or (_get(cfg, "hidden_size") // heads)

    kv_lora = _get(cfg, "kv_lora_rank")
    if kv_lora:  # MLA — joint latent, no 2x
        per_layer = (kv_lora + _get(cfg, "qk_rope_head_dim")) * bytes_per_elem
        return dict(kind="mla", slope_b=per_layer * layers, floor_b=0,
                    per_layer_full=per_layer, per_layer_sliding=0, window=None,
                    detail=f"(r{kv_lora}+rope{_get(cfg,'qk_rope_head_dim')})"
                           f"x{bytes_per_elem}B x{layers}L")

    per_sliding = 2 * kv_heads * head_dim * bytes_per_elem
    g_kv = _get(cfg, "num_global_key_value_heads")
    g_hd = _get(cfg, "global_head_dim")
    per_full = 2 * (g_kv or kv_heads) * (g_hd or head_dim) * bytes_per_elem

    layer_types = list(_get(cfg, "layer_types", []) or [])[:layers]
    if not layer_types:
        return dict(kind="uniform", slope_b=per_full * layers, floor_b=0,
                    per_layer_full=per_full, per_layer_sliding=0, window=None,
                    detail=f"2x{kv_heads}x{head_dim}x{bytes_per_elem}B x{layers}L")

    window = _get(cfg, "sliding_window")
    shared = _get(cfg, "num_kv_shared_layers", 0) or 0
    first_shared = layers - shared
    n_full = sum(1 for i, x in enumerate(layer_types)
                 if x == "full_attention" and i < first_shared)
    n_slide = sum(1 for i, x in enumerate(layer_types)
                  if x == "sliding_attention" and i < first_shared)
    # measured: a sliding layer stores window-1 tokens, not window
    floor = per_sliding * n_slide * (window - 1) if window else 0
    return dict(kind="hybrid", slope_b=per_full * n_full, floor_b=floor,
                per_layer_full=per_full, per_layer_sliding=per_sliding,
                window=window, n_full=n_full, n_sliding=n_slide,
                kv_shared_layers=shared,
                detail=f"full {per_full}B/tok/L x{n_full}; "
                       f"sliding {per_sliding}B x{n_slide} x{window - 1 if window else 0} bounded"
                       + (f"; {shared} kv-shared allocate 0" if shared else ""))


def kv_bytes(d: dict, context: int) -> int:
    """Total KV bytes at a given context length."""
    return d["slope_b"] * context + d["floor_b"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", help="JSON map {model: config_fields}")
    ap.add_argument("--fetch", nargs="*", help="model ids to fetch from HF")
    ap.add_argument("--json", help="write the derived table here")
    args = ap.parse_args()

    configs: dict[str, dict] = {}
    if args.configs:
        configs.update(json.load(open(args.configs)))
    for name in args.fetch or []:
        from transformers import AutoConfig  # only needed for --fetch
        cfg = AutoConfig.from_pretrained(name)
        t = getattr(cfg, "text_config", cfg)
        configs[name] = t.to_dict() if hasattr(t, "to_dict") else vars(t)
    if not configs:
        default = os.path.join(os.path.dirname(__file__),
                               "receipts-c0-20260724", "config-fields.json")
        if os.path.exists(default):
            configs = json.load(open(default))
        else:
            print("give --configs or --fetch", file=sys.stderr)
            return 2

    rows = []
    hdr = f"{'model':<40} {'KB/token':>9} {'floor':>9}"
    for c in CONTEXTS:
        hdr += f" {(str(c // 1024) + 'K'):>8}"
    print(hdr)
    for name, cfg in configs.items():
        d = derive(cfg)
        line = f"{name:<40} {d['slope_b'] / 1024:>9.1f} {d['floor_b'] / 2**20:>7.1f}MB"
        for c in CONTEXTS:
            line += f" {kv_bytes(d, c) / 2**30:>7.2f}G"
        print(line)
        rows.append(dict(model=name, kind=d["kind"],
                         kb_per_token=d["slope_b"] / 1024,
                         floor_mb=d["floor_b"] / 2**20, detail=d["detail"],
                         vram_gb={str(c): kv_bytes(d, c) / 2**30 for c in CONTEXTS}))
    print("\narithmetic:")
    for r in rows:
        print(f"  {r['model']}\n    {r['kind']}: {r['detail']}")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rows, fh, indent=1)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
