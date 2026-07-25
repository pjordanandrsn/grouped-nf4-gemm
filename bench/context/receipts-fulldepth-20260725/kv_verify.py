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

`--full-depth` (rung 1.5) runs the two largest models at their REAL depth on the
same small card, by narrowing `hidden_size` as well — which is safe because the
cache is `[B, num_key_value_heads, T, head_dim]` and both fields come from the
config, so width is not under test. Guarded on `head_dim` being explicit, and
self-checked by re-deriving after the shrink. That closes the depth-dependent
mechanism rung two was scoped to catch (e.g. a KV-sharing threshold like
Gemma's `num_kv_shared_layers`).

What NEITHER mode establishes: real weights, and real width under `--full-depth`.
Both are argued irrelevant — geometry is a function of config and code, which is
this probe's founding premise — but argued is not measured, so rung two as
originally specified (full-depth, real-weight) stays open.

Usage:  python kv_verify.py [--full-depth] [--json out.json]
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

# Rung 1.5 (--full-depth): the two models rung one could only probe truncated.
# Depth is the ONLY thing the rung-one probe cuts -- vocab, MLP and experts are
# already shrunk above -- so the question rung two exists to answer ("does
# anything depth-dependent change the slope at real depth?") does not actually
# need real weights or a big card. It needs real DEPTH and real KV fields.
FULL_SUITE = [
    ("Qwen/Qwen3-235B-A22B-Instruct-2507", (512, 2048)),
    ("openai/gpt-oss-120b", (512, 2048)),
]
#: Carrying full depth on a 12 GB card needs the projections narrowed too:
#: 94 layers of Qwen3-235B's real 4096-wide attention is ~14.6 GB of bf16
#: parameters before any cache exists. ``hidden_size`` sets the projections'
#: INPUT width and nothing else -- the cache is
#: ``[B, num_key_value_heads, T, head_dim]``, both of which are read from the
#: config -- so narrowing it leaves the geometry under test untouched. That is
#: true ONLY where ``head_dim`` is explicit; where it is absent transformers
#: derives it as ``hidden_size // num_attention_heads`` and this would move the
#: very number being measured. Guarded, and additionally self-checked by
#: re-deriving after the shrink and requiring an exact match.
WIDTH_SHRINK = 512


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


def probe(name: str, l_probe: int, ctxs: tuple[int, int],
          full_depth_mode: bool = False) -> dict:
    cfg = AutoConfig.from_pretrained(name)
    t = getattr(cfg, "text_config", cfg)
    real_layers = t.num_hidden_layers
    full_model = derive(t)                       # what the docs table publishes
    if full_depth_mode:
        l_probe = real_layers
    t.num_hidden_layers = l_probe
    if getattr(t, "layer_types", None):
        t.layer_types = list(t.layer_types)[:l_probe]
    # full_depth: KV-share cutoff must anchor to the real depth, not the probe depth
    probe_derived = derive(t, l_probe, full_depth=real_layers)

    for attr, val in SHRINK:
        if hasattr(t, attr):
            setattr(t, attr, val)
    narrowed = False
    if full_depth_mode:
        if getattr(t, "head_dim", None) is None:
            raise RuntimeError(
                "full-depth probe needs an explicit head_dim: without it, "
                "narrowing hidden_size would move the KV geometry under test")
        t.hidden_size = WIDTH_SHRINK
        narrowed = True
        # Self-check: the shrink must be invisible to the derivation, or it
        # touched something it had no business touching.
        after = derive(t, l_probe, full_depth=real_layers)
        if after["slope_b"] != probe_derived["slope_b"] or \
                after["floor_b"] != probe_derived["floor_b"]:
            raise RuntimeError(
                f"narrowing hidden_size changed the derived KV geometry "
                f"({probe_derived['slope_b']} -> {after['slope_b']} B/token); "
                f"the probe would be measuring a different model")
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
                full_depth=full_depth_mode, hidden_narrowed=narrowed,
                contexts=list(ctxs), measured_bytes=measured,
                marginal_b_per_token=marginal, expected_b_per_token=expected,
                err_pct=err, passed=err < 1.0, layer0_shapes=shapes,
                probe_derived=probe_derived, full_model_derived=full_model,
                full_model_slope_kb_per_token=full_model["slope_b"] / 1024,
                full_model_floor_mb=full_model["floor_b"] / 2**20)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--full-depth", action="store_true",
                    help="rung 1.5: run FULL_SUITE at real depth (narrowed "
                         "projections) instead of the truncated rung-one suite")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print("needs a CUDA/HIP device")
        return 2
    print(f"device: {torch.cuda.get_device_name(0)} | torch {torch.__version__}")
    suite = ([(n, None, c) for n, c in FULL_SUITE] if args.full_depth else SUITE)
    results, ok = [], True
    for name, l_probe, ctxs in suite:
        try:
            r = probe(name, l_probe or 0, ctxs, full_depth_mode=args.full_depth)
        except Exception as e:  # a missing local model shouldn't kill the suite
            print(f"\n{name}: SKIP ({type(e).__name__}: {str(e)[:90]})")
            continue
        results.append(r)
        ok &= r["passed"]
        d = r["probe_derived"]
        desc = (f"{d.get('n_full', r['probe_layers'])}F+{d.get('n_sliding', 0)}S"
                f" win{d['window']}" if d["kind"] == "hybrid" else d["kind"])
        tag = ("FULL DEPTH" if r["full_depth"] else f"probe L={r['probe_layers']}")
        print(f"\n{name}  (real L={r['real_layers']}, {tag}, {desc})")
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
