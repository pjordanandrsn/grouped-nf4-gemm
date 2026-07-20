# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Native-only gpt-oss model build: meta-init + selective shard load + native
expert patch — NO dequant materialization at any point.

Why (the pod path): ``from_pretrained`` with the dequant config materializes
every expert in bf16 — ~128 GB host peak at 120b (Phase-6 spin-4 measurement).
A training pod has ~96 GB. This loader never touches the dequant path:

  1. instantiate the model skeleton on the meta device (config only),
  2. stream ONLY the non-expert tensors from the shards onto CPU,
  3. build ExpertsMxfp4LoRA per layer straight from the native shard bytes
     (per-tensor sha256 file-range verify — the T1 gate — happens here),
  4. hand back (model, wrappers); placement/pinning is the caller's step.

Host peak ≈ non-expert bf16 + native expert bytes (+ transient one shard's
worth of mmap), i.e. ~16 GB at 20b and ~67 GB at 120b.

Gated on the A2000 at 20b against the from_pretrained+patch path: identical
provenance hashes, identical step-0 canary within the module band, trains.
"""
from __future__ import annotations

import json
import os

import torch

from mxfp4_qlora import ExpertsMxfp4LoRA
from run_mxfp4_20b_qlora import (  # reuse the run's building blocks verbatim
    BLOCK_SUFFIXES, build_wrapper, expert_tensor_names, read_layer_native,
)


def _weight_map(snap: str) -> dict:
    with open(os.path.join(snap, "model.safetensors.index.json")) as f:
        return json.load(f)["weight_map"]


def build_native_qlora_model(snap: str, r: int = 8, alpha: int = 16,
                             dtype: torch.dtype = torch.bfloat16,
                             log=print):
    """(model, wrappers, file_hashes): gpt-oss with every mlp.experts replaced
    by ExpertsMxfp4LoRA over the shipped native bytes, non-expert weights
    loaded bf16 on CPU, dequant path never entered."""
    from transformers import AutoConfig, AutoModelForCausalLM
    from safetensors import safe_open

    cfg = AutoConfig.from_pretrained(snap)
    # Skeleton on meta: no allocation, no init. Quantization config is dropped
    # from the config copy so the skeleton builds plain GptOssExperts holders
    # (which we replace) instead of hunting for the mxfp4 kernels package.
    if hasattr(cfg, "quantization_config"):
        try:
            delattr(cfg, "quantization_config")
        except AttributeError:
            cfg.quantization_config = None
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg, dtype=dtype)
    model.eval()

    wmap = _weight_map(snap)
    n_layers = cfg.num_hidden_layers

    # Every checkpoint tensor that is NOT one of the six per-layer expert
    # tensors is a non-expert weight to stream in as-is.
    expert_names = set()
    for L in range(n_layers):
        expert_names.update(expert_tensor_names(L).values())
    nonexpert = {n: f for n, f in wmap.items() if n not in expert_names}

    by_shard: dict = {}
    for name, shard in nonexpert.items():
        by_shard.setdefault(shard, []).append(name)

    state = {}
    for shard, names in sorted(by_shard.items()):
        with safe_open(os.path.join(snap, shard), framework="pt", device="cpu") as f:
            for name in names:
                state[name] = f.get_tensor(name)
    log(f"non-expert tensors: {len(state)} loaded from {len(by_shard)} shards")

    # Materialize non-expert modules from the loaded tensors (assign=True moves
    # the real tensors in; meta params are replaced wholesale).
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    # Everything still missing must be expert tensors (about to be replaced) —
    # anything else is a real load failure.
    leftovers = [m for m in missing if ".mlp.experts." not in m]
    if leftovers:
        raise RuntimeError(f"non-expert tensors missing from shards: {leftovers[:8]}")
    if unexpected:
        raise RuntimeError(f"unexpected tensors: {unexpected[:8]}")

    layers = model.model.layers
    wrappers, file_hashes = [], {}
    for L in range(n_layers):
        native, hashes = read_layer_native(snap, wmap, L, verify=True)
        file_hashes.update(hashes)
        w = build_wrapper(native, r, alpha)
        layers[L].mlp.experts = w
        wrappers.append(w)
        if L % 6 == 0:
            log(f"layer {L} native-patched")
    log(f"all {n_layers} layers native-patched, {len(file_hashes)} tensors verified")

    _reinit_nonpersistent_buffers(model, log)

    # No meta tensors may remain anywhere.
    metas = [n for n, p in model.named_parameters() if p.is_meta]
    metas += [n for n, b in model.named_buffers() if b is not None and b.is_meta]
    if metas:
        raise RuntimeError(f"meta tensors remain after load: {metas[:8]}")
    return model, wrappers, file_hashes


def _reinit_nonpersistent_buffers(model, log=print):
    """Non-persistent buffers are never in the checkpoint, so meta-init leaves
    them on meta. gpt-oss has exactly the rotary embedding's ``inv_freq``
    (+ ``original_inv_freq``); recompute it from config via the module's own
    ``rope_init_fn`` — the same values eager init would have produced. The
    trailing meta-check in the caller catches any OTHER architecture growing
    a new non-persistent buffer (fail loudly, then extend this)."""
    rope = model.model.rotary_emb
    if rope.inv_freq.is_meta:
        # Version-proof: build a fresh rotary module of the same class from
        # the same config on a REAL device and take its buffers (v5.14 no
        # longer exposes rope_init_fn on the instance).
        fresh = type(rope)(rope.config)
        rope.register_buffer("inv_freq", fresh.inv_freq.clone(), persistent=False)
        if hasattr(fresh, "attention_scaling"):
            rope.attention_scaling = fresh.attention_scaling
        if hasattr(fresh, "original_inv_freq"):
            rope.original_inv_freq = fresh.original_inv_freq.clone()
        log("rotary inv_freq re-initialized from a fresh config-built module")
