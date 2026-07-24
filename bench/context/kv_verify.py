#!/usr/bin/env python3
"""Rung-one KV verification (Phase C0) — derived vs measured, on one small GPU.

Why a truncated-depth probe: the KV cache geometry is a function of the config
and the model's own attention code, not of weight *values*. So we instantiate
each architecture's REAL model class from its REAL config with depth cut to
`L_probe` and random weights, prefill at two context lengths, and diff the actual
cache tensors' bytes. That isolates two independent claims cheaply enough to run
on a 12 GB card:

  (a) per-layer per-token bytes == the config-derived figure, and
  (b) only full-attention layers grow with context (sliding layers are bounded
      at `window - 1`, which this also measures).

What it deliberately does NOT establish: the full-depth, real-weight slope for
models too large for the local card. That is rung two (cloud) — see
docs/context-budgets.md. A depth-dependent mechanism (e.g. Gemma's
`num_kv_shared_layers`) would be invisible here, which is precisely why rung two
exists and why rows are tiered.

Usage:  python kv_verify.py [--json out.json]
Runs offline against models already in the local HF cache.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kv_budget import derive  # single source of truth for the derivation

os.environ.setdefault("HF_HUB_OFFLINE", "1")

# (name, probe depth, two prefill lengths). Contexts straddle each model's
# window so the bounded/unbounded regimes separate in the marginal.
SUITE = [
    ("allenai/OLMoE-1B-7B-0924", 4, (512, 2048)),
    ("Qwen/Qwen3-235B-A22B-Instruct-2507", 4, (512, 2048)),
    ("Qwen/Qwen3-30B-A3B", 4, (512, 2048)),
    ("openai/gpt-oss-20b", 4, (512, 2048)),
    ("google/gemma-4-26B-A4B", 6, (2048, 4096)),
]
# Shrunk to keep the probe inside a small card. None of these touch the
# attention/KV geometry, which is what is under test.
SHRINK = (
    ("vocab_size", 512),
    ("vocab_size_per_layer_input", 512),
    ("intermediate_size", 128),
    ("moe_intermediate_size", 128),
    ("shared_expert_intermediate_size", 128),
    ("expert_dim", 128),
    ("num_experts", 4),
    ("num_local_experts", 4),
    ("num_experts_per_tok", 2),
    ("top_k_experts", 2),
    ("decoder_sparse_step", 1),
)


def cache_bytes(past_key_values) -> int:
    """Sum the unique cache tensors' bytes (dedup by data_ptr: views alias)."""
    total, seen = 0, set()

    def walk(o):
        nonlocal total
        if torch.is_tensor(o):
            ptr = o.data_ptr()
            if ptr and ptr not in seen:
                seen.add(ptr)
                total += o.numel() * o.element_size()
        elif isinstance(o, (list, tuple)):
            for x in o:
                walk(x)
        elif hasattr(o, "__dict__"):
            for x in vars(o).values():
                walk(x)

    walk(past_key_values)
    return total


def probe(name: str, l_probe: int, ctxs: tuple[int, int]) -> dict:
    cfg = AutoConfig.from_pretrained(name)
    t = getattr(cfg, "text_config", cfg)
    real_layers = t.num_hidden_layers
    full_model = derive(t)                       # what the docs table publishes
    t.num_hidden_layers = l_probe
    if getattr(t, "layer_types", None):
        t.layer_types = list(t.layer_types)[:l_probe]
    probe_derived = derive(t, l_probe)           # what this probe should measure

    for attr, val in SHRINK:
        if hasattr(t, attr):
            setattr(t, attr, val)
    for attr in ("pad_token_id", "bos_token_id", "eos_token_id"):
        for obj in (t, cfg):
            if getattr(obj, attr, None) is not None:
                setattr(obj, attr, 0)            # must stay < shrunk vocab

    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg).to("cuda:0", torch.bfloat16).eval()
    measured = {}
    shapes = []
    for ctx in ctxs:
        gc.collect()
        torch.cuda.empty_cache()
        ids = torch.randint(0, 256, (1, ctx), device="cuda")
        with torch.no_grad():
            out = model(ids, use_cache=True)
        torch.cuda.synchronize()
        measured[ctx] = cache_bytes(out.past_key_values)
        if ctx == ctxs[0]:
            layers = getattr(out.past_key_values, "layers", None) or []
            for i, ly in enumerate(layers):
                k = getattr(ly, "keys", None)
                if torch.is_tensor(k):
                    lt = (t.layer_types[i] if getattr(t, "layer_types", None) else "full_attention")
                    shapes.append(dict(layer=i, type=lt, k_shape=list(k.shape)))
        del out, ids
    del model
    gc.collect()
    torch.cuda.empty_cache()

    c1, c2 = ctxs[0], ctxs[-1]
    marginal = (measured[c2] - measured[c1]) / (c2 - c1)
    expected = probe_derived["slope_b"]          # only unbounded layers grow
    err = abs(marginal - expected) / max(expected, 1) * 100
    return dict(model=name, real_layers=real_layers, probe_layers=l_probe,
                contexts=list(ctxs), measured_bytes=measured,
                marginal_b_per_token=marginal, expected_b_per_token=expected,
                err_pct=err, passed=err < 1.0, layer0_shapes=shapes,
                probe_derived=probe_derived, full_model_derived=full_model,
                full_model_slope_kb_per_token=full_model["slope_b"] / 1024,
                full_model_floor_mb=full_model["floor_b"] / 2**20)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print("needs a CUDA/HIP device")
        return 2
    print(f"device: {torch.cuda.get_device_name(0)} | torch {torch.__version__}")
    results, ok = [], True
    for name, l_probe, ctxs in SUITE:
        try:
            r = probe(name, l_probe, ctxs)
        except Exception as e:  # a missing local model shouldn't kill the suite
            print(f"\n{name}: SKIP ({type(e).__name__}: {str(e)[:90]})")
            continue
        results.append(r)
        ok &= r["passed"]
        d = r["probe_derived"]
        desc = (f"{d.get('n_full', r['probe_layers'])}F+{d.get('n_sliding', 0)}S"
                f" win{d['window']}" if d["kind"] == "hybrid" else d["kind"])
        print(f"\n{name}  (real L={r['real_layers']}, probe L={r['probe_layers']}, {desc})")
        print(f"  marginal/token measured {r['marginal_b_per_token']:>8.1f} B"
              f" | expected {r['expected_b_per_token']:>8} B"
              f" | {'PASS' if r['passed'] else 'FAIL'} ({r['err_pct']:.2f}% err)")
        print(f"  -> full model: {r['full_model_slope_kb_per_token']:.1f} KB/token"
              f" + {r['full_model_floor_mb']:.1f} MB floor")
    # An empty run is a FAILURE, not a pass: if every model skipped (missing
    # cache, load error) there is no verification, and a caller that trusts the
    # exit code would treat "nothing ran" as "rung one is green".
    verified = bool(results) and ok
    if not results:
        print("\nrung-one: NO MODELS RAN (all skipped) — not a pass")
    else:
        print(f"\nrung-one: {'ALL PASS' if ok else 'FAILURES PRESENT'}"
              f" ({len(results)} models)")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=1, default=str)
        print(f"receipt: {args.json}")
    return 0 if verified else 1


if __name__ == "__main__":
    raise SystemExit(main())
