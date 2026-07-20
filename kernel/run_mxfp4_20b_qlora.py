# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Phase-7 dev run: gpt-oss-20b QLoRA over NATIVE MXFP4 experts, on the A2000.

Plan §Phase 2.5.2 (dev target — 20b is proof-of-method, never a headline):
LoRA adapters train over the frozen native expert bytes; deliverables are the
loss curve (held-out eval), loaded/peak VRAM, per-step JSONL, a step-0
golden canary vs the transformers dequant path (the A4-oracle arm), and the
provenance hash table: sha256(file range) == sha256(loaded bytes), pre == post.

Method (the Phase-6 shard-read method, adapted to training):
  1. Load the model via transformers' dequant path on CPU (bf16) — this IS the
     reference arm; run the step-0 canary forward there.
  2. Per layer: read the four native expert tensors from the safetensors
     shards (safe_open), verify sha256(loaded) == sha256(file range), build
     ExpertsMxfp4 + ExpertsMxfp4LoRA, replace layer.mlp.experts (dense dequant
     weights freed).
  3. Move the model to CUDA with the native storage stubbed to 0-element
     tensors (the e4b offload placeholder trick), then restore the CPU-pinned
     storage: compute on GPU, packed bytes stream per expert visit, decode on
     device, recompute in backward.
  4. Train LoRA (experts only), JSONL telemetry, eval on held-out chunks.
  5. Re-hash everything; write the provenance artifact.

Stdout is a log; artifacts land in --out. Designed to run detached inside
gnf4-v6 (shared GPU: ~3 GB is the resident voice-tts neighbor — stay under
~9 GB peak).
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time

import torch

from mxfp4_loader import file_tensor_sha256, to_kernel_shapes
from mxfp4_qlora import (ExpertsMxfp4, ExpertsMxfp4LoRA, adapter_parameters,
                         lora_parameters)

BLOCK_SUFFIXES = ("gate_up_proj_blocks", "gate_up_proj_scales",
                  "down_proj_blocks", "down_proj_scales")
BIAS_SUFFIXES = ("gate_up_proj_bias", "down_proj_bias")
CANARY_TAIL = 32  # positions compared in the canary (was 8 — small-n top1)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def rss_gb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1e6
    return -1.0


def resolve_snapshot(model_id: str) -> str:
    # Resolve the cached snapshot dir directly: hub's local_files_only path
    # validates the FULL repo tree (README/LICENSE/metal/...) even under
    # allow_patterns, and this cache deliberately holds only the shards +
    # configs. The run needs a directory, not hub bookkeeping.
    import glob
    from huggingface_hub.constants import HF_HUB_CACHE
    base = os.path.join(HF_HUB_CACHE,
                        f"models--{model_id.replace('/', '--')}", "snapshots")
    cands = sorted(glob.glob(os.path.join(base, "*")), key=os.path.getmtime)
    if not cands:
        raise FileNotFoundError(f"no local snapshot under {base}")
    snap = cands[-1]
    if not os.path.exists(os.path.join(snap, "model.safetensors.index.json")):
        raise FileNotFoundError(f"snapshot {snap} lacks the safetensors index")
    return snap


def shard_of(snap: str) -> dict:
    with open(os.path.join(snap, "model.safetensors.index.json")) as f:
        return json.load(f)["weight_map"]


def expert_tensor_names(layer: int) -> dict:
    base = f"model.layers.{layer}.mlp.experts."
    return {s: base + s for s in BLOCK_SUFFIXES + BIAS_SUFFIXES}


def read_layer_native(snap: str, weight_map: dict, layer: int, verify: bool):
    """Read the six expert tensors for one layer from the shards; optionally
    verify sha256(loaded bytes) == sha256(file data-section range) for the four
    native uint8 tensors (the provenance receipt, at load time)."""
    from safetensors import safe_open
    names = expert_tensor_names(layer)
    got, hashes = {}, {}
    by_shard = {}
    for suffix, name in names.items():
        by_shard.setdefault(weight_map[name], []).append((suffix, name))
    for shard, items in by_shard.items():
        path = os.path.join(snap, shard)
        with safe_open(path, framework="pt", device="cpu") as f:
            for suffix, name in items:
                t = f.get_tensor(name)
                got[suffix] = t
                if verify and suffix in BLOCK_SUFFIXES:
                    want = file_tensor_sha256(path, name)
                    have = hashlib.sha256(
                        t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                    if want != have:
                        raise ValueError(f"PROVENANCE FAIL at load: {name}")
                    hashes[name] = want
    return got, hashes


def build_wrapper(native: dict, r: int, alpha: int) -> ExpertsMxfp4LoRA:
    gu_b, gu_s = to_kernel_shapes(native["gate_up_proj_blocks"],
                                  native["gate_up_proj_scales"])
    dn_b, dn_s = to_kernel_shapes(native["down_proj_blocks"],
                                  native["down_proj_scales"])
    base = ExpertsMxfp4(gu_b.contiguous(), gu_s.contiguous(),
                        dn_b.contiguous(), dn_s.contiguous(),
                        native["gate_up_proj_bias"], native["down_proj_bias"])
    return ExpertsMxfp4LoRA(base, r=r, alpha=alpha, mode="loop")


def cuda_move_with_storage_stubbed(model, wrappers, pin: bool):
    """model.to('cuda') would drag the ~10 GB of native uint8 storage onto the
    card. Stub each storage tensor to 0 elements, move, then restore the CPU
    (optionally pinned) originals — Parameters/buffers holding CPU data while
    the module computes on CUDA is exactly the streaming contract."""
    stash = []
    for w in wrappers:
        b = w.base
        rec = {}
        for name in ("gate_up_blocks", "down_blocks"):
            p = getattr(b, name)
            rec[name] = p.data
            p.data = torch.empty(0, dtype=torch.uint8)
        for name in ("gate_up_scales", "down_scales"):
            t = getattr(b, name)
            rec[name] = t
            setattr(b, name, torch.empty(0, dtype=torch.uint8))
        stash.append((b, rec))
    model.to("cuda")

    def _maybe_pin(t):
        # memlock-capped hosts (seen on a SECURE 3090: ulimit -Hl = 8 MB) make
        # pin_memory() throw — degrade to pageable staging instead of dying.
        if not pin:
            return t
        try:
            return t.pin_memory()
        except RuntimeError:
            return t

    for b, rec in stash:
        for name in ("gate_up_blocks", "down_blocks"):
            getattr(b, name).data = _maybe_pin(rec[name])
        for name in ("gate_up_scales", "down_scales"):
            setattr(b, name, _maybe_pin(rec[name]))


def hash_all_wrappers(wrappers) -> dict:
    return {i: w.base.expert_bytes_sha256() for i, w in enumerate(wrappers)}


def batches_from_wikitext(tok, seq_len: int, n_train: int, n_eval: int, seed: int):
    import datasets
    # datasets 5.x rejects the legacy bare repo id; Salesforce/ is canonical
    ds = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(t for t in ds["text"][:6000] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0]
    n_chunks = ids.shape[0] // seq_len
    chunks = ids[: n_chunks * seq_len].reshape(n_chunks, seq_len)
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(n_chunks, generator=g)
    train = chunks[order[:n_train]]
    evalc = chunks[order[n_train:n_train + n_eval]]
    return train, evalc


@torch.no_grad()
def gpu_shuttle_forward(model, ids):
    """One forward of a CPU-resident model on CUDA by shuttling each decoder
    layer on/off the card around its own forward (hooks). Peak device memory
    ~= perimeter (embed+norm+head+rotary) + one layer + activations. Returns
    (last-8-position logits fp32 on CPU, loss)."""
    dev = "cuda"
    core = model.model
    perim = [core.embed_tokens, core.norm, model.lm_head, core.rotary_emb]
    hooks = []

    def pre(mod, args, kwargs):
        mod.to(dev)
        return None

    def post(mod, args, kwargs, out):
        mod.to("cpu")
        return None

    try:
        for m in perim:
            m.to(dev)
        for lyr in core.layers:
            hooks.append(lyr.register_forward_pre_hook(pre, with_kwargs=True))
            hooks.append(lyr.register_forward_hook(post, with_kwargs=True))
        out = model(input_ids=ids.to(dev), labels=ids.to(dev))
        logits = out.logits[0, -CANARY_TAIL:].float().cpu().clone()
        loss = float(out.loss)
    finally:
        for h in hooks:
            h.remove()
        for m in perim:
            m.to("cpu")
        torch.cuda.empty_cache()
    return logits, loss


@torch.no_grad()
def exact_chunk_ppl(model, tok, text: str, device) -> float:
    """The Phase-6 P2 recipe, verbatim shape: tokenize the fixture, NCH full
    512-token chunks, per-chunk CE of logits[:-1] vs chunk[1:] (sum), ppl =
    exp(nll/cnt). At 120b the stamped band is [26.55, 27.05] (RESULTS-mxfp4-
    serve): the training-lane step-0 model must land in the same band."""
    import math
    import torch.nn.functional as F
    was = model.training
    model.eval()
    tids = tok(text, return_tensors="pt").input_ids.to(device)
    nch = tids.shape[1] // 512
    nll, cnt = 0.0, 0
    for c in range(nch):
        ch = tids[:, c * 512:(c + 1) * 512]
        lo = model(input_ids=ch).logits.float()
        nll += float(F.cross_entropy(lo[0, :-1], ch[0, 1:], reduction="sum"))
        cnt += ch.shape[1] - 1
    if was:
        model.train()
    return math.exp(nll / cnt)


@torch.no_grad()
def eval_loss(model, eval_chunks, device) -> float:
    was_training = model.training
    model.eval()
    tot = 0.0
    for i in range(eval_chunks.shape[0]):
        ids = eval_chunks[i:i + 1].to(device)
        tot += float(model(input_ids=ids, labels=ids).loss)
    if was_training:
        model.train()
    return tot / eval_chunks.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--out", default="/work/mx7/out20b")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--eval-chunks", type=int, default=8)
    ap.add_argument("--canary-tokens", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--no-pin", action="store_true")
    ap.add_argument("--cpu-threads", type=int, default=6)
    ap.add_argument("--native-load", action="store_true",
                    help="meta-init + shard load; NO dequant path (the pod "
                         "recipe — skips the dequant-arm CPU canary)")
    ap.add_argument("--ppl-text", default=None,
                    help="exact-chunk ppl fixture; evaluated at step 0 on CUDA")
    ap.add_argument("--ppl-band", default=None,
                    help="lo,hi — HARD gate on the step-0 exact-chunk ppl")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)

    from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config

    snap = resolve_snapshot(args.model)
    wmap = shard_of(snap)
    log(f"snapshot {snap}")

    tok = AutoTokenizer.from_pretrained(snap)
    train_chunks, eval_chunks = batches_from_wikitext(
        tok, args.seq, args.steps, args.eval_chunks, args.seed)
    log(f"data: {train_chunks.shape[0]} train + {eval_chunks.shape[0]} eval chunks of {args.seq}")

    canary_ids = eval_chunks[0:1, : args.canary_tokens]
    ref_logits = ref_loss = None

    if args.native_load:
        # Touch CUDA FIRST: surface a broken device context in seconds, not
        # after the 100+ s native build (pod 2l4ex0zwexn2jr failed lazy init
        # only at placement time).
        torch.zeros(1, device="cuda")
        # pod recipe: meta-init + shard load, no dequant materialization ever
        from mxfp4_native_load import build_native_qlora_model
        t0 = time.time()
        model, wrappers, pre_file_hashes = build_native_qlora_model(
            snap, r=args.r, alpha=args.alpha, log=log)
        log(f"native build {time.time()-t0:.0f}s rss={rss_gb():.1f}GB")
    else:
        # 1. dequant reference arm (the A4-oracle path), bf16 on CPU.
        # Canary lessons: CPU bf16 forward is oneDNN-emulated on this Xeon
        # (>35 min, smoke 1); model.float() spikes +~40 GB and host-OOMs
        # (smoke 2); fp32-from-load crashes transformers' grouped-mm (its
        # mxfp4 dequant emits bf16 experts regardless of dtype) AND would be
        # ~84 GB anyway (smoke 3). So: keep the model CPU-resident and run
        # the canary as a LAYER-SHUTTLE GPU forward — each decoder layer hops
        # to CUDA for its forward and back (~1.7 GB in flight), bf16 on the
        # same device our native path runs on: a tighter reference.
        t0 = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            snap, dtype=torch.bfloat16,
            quantization_config=Mxfp4Config(dequantize=True))
        model.eval()
        log(f"dequant model loaded on CPU in {time.time()-t0:.0f}s rss={rss_gb():.1f}GB")

        ref_logits, ref_loss = gpu_shuttle_forward(model, canary_ids)
        log(f"canary (dequant bf16, GPU layer-shuttle): loss={ref_loss:.4f} "
            f"rss={rss_gb():.1f}GB cuda_peak={torch.cuda.max_memory_allocated()/1e9:.2f}GB")

        # 2. patch every layer to native-mxfp4 QLoRA (per-layer provenance verify)
        layers = model.model.layers
        wrappers, pre_file_hashes = [], {}
        for L in range(len(layers)):
            native, hashes = read_layer_native(snap, wmap, L, verify=True)
            pre_file_hashes.update(hashes)
            w = build_wrapper(native, args.r, args.alpha)
            layers[L].mlp.experts = w
            wrappers.append(w)
            del native
            if L % 6 == 0:
                gc.collect()
                log(f"layer {L} patched rss={rss_gb():.1f}GB")
        gc.collect()
        log(f"all {len(wrappers)} layers patched + provenance-verified "
            f"({len(pre_file_hashes)} native tensors) rss={rss_gb():.1f}GB")

    pre_module_hashes = hash_all_wrappers(wrappers)

    # 3. placement: compute on CUDA, native storage CPU(-pinned)
    for p in model.parameters():
        p.requires_grad_(False)
    for w in wrappers:
        for p in adapter_parameters(w):
            p.requires_grad_(True)
    cuda_move_with_storage_stubbed(model, wrappers, pin=not args.no_pin)
    torch.cuda.reset_peak_memory_stats()
    loaded = torch.cuda.memory_allocated() / 1e9
    log(f"on cuda: loaded={loaded:.2f}GB rss={rss_gb():.1f}GB")

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    # step-0 canary on the native path (same tokens; GPU)
    with torch.no_grad():
        our_out = model(input_ids=canary_ids.to("cuda"), labels=canary_ids.to("cuda"))
        our_logits = our_out.logits[0, -CANARY_TAIL:].float().cpu()
        our_loss = float(our_out.loss)
    kl = top1 = None
    if ref_logits is not None:
        p_ref = torch.softmax(ref_logits, -1)
        kl = float((p_ref * (torch.log_softmax(ref_logits, -1)
                             - torch.log_softmax(our_logits, -1))).sum(-1).mean())
        top1 = float((ref_logits.argmax(-1) == our_logits.argmax(-1)).float().mean())
        log(f"canary (native, GPU): loss={our_loss:.4f} vs ref {ref_loss:.4f} "
            f"| KL={kl:.4f} top1={top1:.3f}")
    else:
        log(f"canary (native, GPU): loss={our_loss:.4f} (no dequant arm — native load)")

    ppl0 = None
    if args.ppl_text:
        ppl0 = exact_chunk_ppl(model, tok, open(args.ppl_text).read(), "cuda")
        log(f"step-0 exact-chunk ppl = {ppl0:.3f}")
        if args.ppl_band:
            lo_b, hi_b = (float(v) for v in args.ppl_band.split(","))
            if not (lo_b <= ppl0 <= hi_b):
                raise SystemExit(
                    f"PPL GATE FAIL: {ppl0:.3f} outside [{lo_b}, {hi_b}]")
            log(f"ppl gate PASS: in [{lo_b}, {hi_b}]")

    # 4. train
    params = [p for w in wrappers for p in lora_parameters(w)]
    n_p = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    log(f"training {len(params)} adapter tensors, {n_p/1e6:.1f}M params")

    jsonl = open(os.path.join(args.out, "steps.jsonl"), "w")
    model.train()
    ev0 = eval_loss(model, eval_chunks, "cuda")
    log(f"eval@0 {ev0:.4f}")
    evals = {0: ev0}
    for step in range(args.steps):
        t = time.time()
        ids = train_chunks[step:step + 1].to("cuda")
        opt.zero_grad(set_to_none=True)
        loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        opt.step()
        dt = time.time() - t
        rec = dict(step=step, loss=float(loss), dt=round(dt, 2),
                   cuda_peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 3),
                   rss_gb=round(rss_gb(), 2))
        jsonl.write(json.dumps(rec) + "\n")
        jsonl.flush()
        log(f"step {step} loss={rec['loss']:.4f} dt={dt:.1f}s "
            f"peak={rec['cuda_peak_gb']:.2f}GB")
        if (step + 1) % 10 == 0 and (step + 1) < args.steps:
            evals[step + 1] = eval_loss(model, eval_chunks, "cuda")
            log(f"eval@{step+1} {evals[step+1]:.4f}")
    ev1 = eval_loss(model, eval_chunks, "cuda")
    evals[args.steps] = ev1
    log(f"eval@{args.steps} {ev1:.4f} (was {ev0:.4f})")
    jsonl.close()

    # 5. provenance post-check + artifact
    post_module_hashes = hash_all_wrappers(wrappers)
    identical = post_module_hashes == pre_module_hashes
    log(f"post-training module hashes identical: {identical}")

    artifact = dict(
        model=args.model, snapshot=snap,
        config=dict(steps=args.steps, seq=args.seq, lr=args.lr, r=args.r,
                    alpha=args.alpha, seed=args.seed,
                    native_load=bool(args.native_load)),
        canary=dict(ref_loss=ref_loss, our_loss=our_loss, kl=kl, top1=top1),
        step0_exact_chunk_ppl=ppl0,
        eval_loss=evals,
        cuda=dict(loaded_gb=loaded,
                  peak_gb=torch.cuda.max_memory_allocated() / 1e9),
        adapters_m=n_p / 1e6,
        provenance=dict(
            n_file_tensors=len(pre_file_hashes),
            file_hashes=pre_file_hashes,
            pre_equals_post=identical,
            load_time_verified=True),
    )
    with open(os.path.join(args.out, "run_artifact.json"), "w") as f:
        json.dump(artifact, f, indent=1)
    if not identical:
        raise SystemExit("PROVENANCE FAIL: module bytes changed during training")
    log("DONE — artifact written")


if __name__ == "__main__":
    main()
