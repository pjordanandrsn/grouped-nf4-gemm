# Copyright (c) 2026 Cerin Amroth LLC. MIT.
"""Reproduce the offload ladder: bulk -> routed -> +grouped kernel -> +speculative.

This is the harness behind the README's headline number and findings #34/#40/#42.
The original was written on a pod and lost with it (#47) — the receipts kept
results but no `cmd`/`script`/`argv`, so the project's flagship claim had no
reproducible driver. This is that driver, rebuilt from the documented API.

The four rungs are applied CUMULATIVELY in one process, so the model is loaded
once:

  1. bulk        `load_moe_4bit_streaming(offload=True)` at defaults — the
                 pre-hook stages a layer's ENTIRE expert stack (E) although
                 decode routes to top_k.
  2. routed      `enable_routed_staging` — the copy follows the router.
  3. +grouped    `enable_fast` — the fused grouped-NF4 GEMM. Note this only
                 reaches the streaming path at all as of the #22 fix; before it,
                 the kernel was dead code on every offloaded model.
  4. +spec       `enable_speculative_staging` — layer i-distance's router
                 predicts layer i's routing so the copy can start early.

Every rung must be LOGIT-identical to rung 1 on a natural prompt. Greedy ids on
random tokens measure chaos amplification, not correctness (#24).

API gap worth knowing: `load_moe_4bit_streaming` returns `(model, config)` and
NOT the offload handles, but `enable_routed_staging` needs them. They are
stashed as `_offload` on each layer's MLP; `_collect_handles` below mirrors the
walk `enable_speculative_staging` does internally. If the loader ever returns
handles publicly, use that instead.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

import torch

from experts4bit_qlora import (
    enable_fast,
    enable_routed_staging,
    enable_speculative_staging,
    load_moe_4bit_streaming,
    speculative_stats,
)

PROMPT = (
    "The city had grown around the river for three hundred years, and every "
    "bridge told a story about the people who built it. When the flood came, "
    "the oldest span was the only one that held, because"
)


def _collect_handles(model):
    """Recover the offload handles the loader does not return.

    Mirrors `enable_speculative_staging`'s discovery: the handle lives at
    `layer.mlp._offload`, with a `layer.modules()` fallback for architectures
    that nest the MoE block deeper.
    """
    layers = model.model.layers if hasattr(model, "model") else model.layers
    handles = []
    for ly in layers:
        h = getattr(getattr(ly, "mlp", None), "_offload", None)
        if h is None:
            for m in ly.modules():
                h = getattr(m, "_offload", None)
                if h is not None:
                    break
        if h is not None:
            handles.append(h)
    return handles


@torch.no_grad()
def decode(model, ids, n_tokens, warmup):
    """True single-token decode; returns (median s/token, first-token logits).

    MUST use a KV cache and feed ONE token per step. An earlier draft of this
    harness re-ran a full forward over the whole prompt each step: that is a
    prefill-shaped call (M>1), which dispatches the M-tile GEMM, NOT the
    `_gemv_nf4_grouped` decode path the ladder is about. It would have measured
    the wrong kernel entirely.
    """
    out = model(ids, use_cache=True)
    past = out.past_key_values
    first_logits = out.logits[:, -1, :].float().clone()
    nxt = out.logits[:, -1:, :].argmax(-1)          # [1,1]

    times = []
    for step in range(n_tokens + warmup):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        o = model(nxt, past_key_values=past, use_cache=True)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        past = o.past_key_values
        nxt = o.logits[:, -1:, :].argmax(-1)
        if step >= warmup:
            times.append(dt)
    return statistics.median(times), first_logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-235B-A22B")
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--distance", type=int, default=2)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--rungs", default="bulk,routed,fast,spec",
                    help="comma list; a short run can stop early")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    dev = "cuda"
    want = [r.strip() for r in args.rungs.split(",") if r.strip()]
    print(f"# {torch.cuda.get_device_name(0)} x{torch.cuda.device_count()}  torch {torch.__version__}")
    print(f"# model={args.model}  tokens={args.tokens} (warmup {args.warmup})  rungs={want}")

    t0 = time.time()
    model, _ = load_moe_4bit_streaming(args.model, dev, torch.bfloat16, r=8, alpha=16,
                                       offload=True, pin=True)
    model.eval()
    print(f"# loaded in {time.time() - t0:.0f}s")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    ids = tok(PROMPT, return_tensors="pt").input_ids.to(dev)
    print(f"# prompt {ids.shape[-1]} tokens (natural text — logit gate, not greedy ids)")

    handles = _collect_handles(model)
    print(f"# recovered {len(handles)} offload handles")
    if not handles:
        raise SystemExit("no offload handles found — was offload=True honoured?")

    rows, base_logits, base_ms = [], None, None

    for rung in want:
        if rung == "bulk":
            pass                                        # loader defaults
        elif rung == "routed":
            enable_routed_staging(handles)
        elif rung == "fast":
            n = enable_fast(model)
            print(f"#   enable_fast patched {n} modules")
        elif rung == "spec":
            _, hooked = enable_speculative_staging(model, distance=args.distance, k=args.topk)
            print(f"#   speculative hooks on {hooked} layers")
        else:
            raise SystemExit(f"unknown rung {rung!r}")

        s_tok, logits = decode(model, ids, args.tokens, args.warmup)
        if base_logits is None:
            base_logits, base_ms = logits, s_tok
            dlog = 0.0
        else:
            dlog = (logits - base_logits).abs().max().item()
        rows.append({"rung": rung, "s_per_token": s_tok, "tok_per_s": 1.0 / s_tok,
                     "vs_bulk": base_ms / s_tok, "max_abs_dlogit": dlog})
        print(f"  {rung:8s} {s_tok:8.4f} s/tok  {1.0/s_tok:6.3f} tok/s  "
              f"{base_ms/s_tok:6.2f}x  max|dlogit|={dlog:.3e}")

    if "spec" in want:
        st = speculative_stats(handles)
        print(f"# speculation: {st}")

    bad = [r for r in rows if r["max_abs_dlogit"] != 0.0]
    print(f"\n# ladder {rows[-1]['vs_bulk']:.2f}x over bulk"
          f"   bit-identical: {'YES' if not bad else 'NO -> ' + str([r['rung'] for r in bad])}")

    if args.out:
        json.dump({"model": args.model, "device": torch.cuda.get_device_name(0),
                   "gpus": torch.cuda.device_count(), "rows": rows,
                   "spec": speculative_stats(handles) if "spec" in want else None},
                  open(args.out, "w"), indent=2, default=str)
        print(f"# wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
