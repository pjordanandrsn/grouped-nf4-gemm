#!/usr/bin/env python3
"""Flagship Phase B: REAL Qwen3-235B-A22B decode on the offload pipeline —
actual checkpoint, actual router, actual text — experts NF4-quantized
layer-by-layer into host pinned RAM, streamed per token, MoE via the fused
kernel. Produces tokens/sec receipts AND verbatim generations.

Pipeline:
  1. Download the bf16 checkpoint (safetensors shards) to a RAM-backed dir.
  2. Stream-quantize: per layer, per expert, quantize_4bit(gate;up) and
     (down) on GPU, repack into pinned [E,N,K/2] + fp32 absmax host stacks
     (the #1949 layout the kernel consumes); attention/router/norm weights
     go GPU-resident bf16; embeddings stay CPU-pinned (row gather per
     token); lm_head GPU-resident.
  3. Generate: token-by-token through the double-buffered stream loop with
     the REAL router (softmax -> top-8 -> renormalize -> weighted sum) and
     Qwen3 attention details (QK-norm per head, rope theta 1e6, GQA 64/4).

Verbatim continuations are written into the receipts; degeneration is
mechanically flagged (distinct-bigram ratio). VRAM cap discipline as in
Phase A.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as Fn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "kernel"))

MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
H, I, E, K_TOP, L_TOT = 4096, 1536, 128, 8, 94
N_HEADS, N_KV, HD = 64, 4, 128
ROPE_THETA = 1_000_000.0


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def download(cache):
    from huggingface_hub import snapshot_download

    log(f"downloading {MODEL} -> {cache}")
    p = snapshot_download(MODEL, cache_dir=cache,
                          allow_patterns=["*.safetensors", "*.json", "tokenizer*"])
    log("download complete")
    return Path(p)


class Shards:
    def __init__(self, root: Path):
        from safetensors import safe_open

        self.safe_open = safe_open
        idx = json.loads((root / "model.safetensors.index.json").read_text())
        self.map = idx["weight_map"]
        self.root = root
        self.open_handles = {}

    def get(self, name, dtype=torch.bfloat16):
        shard = self.map[name]
        h = self.open_handles.get(shard)
        if h is None:
            h = self.safe_open(str(self.root / shard), framework="pt")
            self.open_handles[shard] = h
        return h.get_tensor(name).to(dtype)


def quantize_expert(w_bf16_gpu):
    """bf16 [N,K] on GPU -> (packed uint8 [N,K/2] cpu-pinned-ready, absmax fp32 [N,K/64])."""
    from bitsandbytes import functional as F

    q, st = F.quantize_4bit(w_bf16_gpu, blocksize=64, quant_type="nf4")
    N, K = w_bf16_gpu.shape
    packed = q.reshape(N, K // 2)
    am = st.absmax
    if getattr(st, "nested", False):
        am = F.dequantize_blockwise(st.absmax, st.state2) + st.offset
    absmax = am.to(torch.float32).reshape(N, K // 64)
    return packed, absmax


def build(shards, dev, layers):
    """Quantize experts into pinned host stacks; residents to GPU."""
    host, attn_w, router_w, norms = [], [], [], []
    t0 = time.time()
    for lay in range(layers):
        p = f"model.layers.{lay}."
        gu_b = torch.empty(E, 2 * I, H // 2, dtype=torch.uint8).pin_memory()
        gu_a = torch.empty(E, 2 * I, H // 64, dtype=torch.float32).pin_memory()
        dn_b = torch.empty(E, H, I // 2, dtype=torch.uint8).pin_memory()
        dn_a = torch.empty(E, H, I // 64, dtype=torch.float32).pin_memory()
        for e in range(E):
            ep = f"{p}mlp.experts.{e}."
            gate = shards.get(ep + "gate_proj.weight").to(dev)
            up = shards.get(ep + "up_proj.weight").to(dev)
            pb, pa = quantize_expert(torch.cat([gate, up], 0))
            gu_b[e].copy_(pb.cpu())
            gu_a[e].copy_(pa.cpu())
            del gate, up, pb, pa
            down = shards.get(ep + "down_proj.weight").to(dev)
            pb, pa = quantize_expert(down)
            dn_b[e].copy_(pb.cpu())
            dn_a[e].copy_(pa.cpu())
            del down, pb, pa
        host.append((gu_b, gu_a, dn_b, dn_a))
        attn_w.append({k: shards.get(f"{p}self_attn.{k}.weight").to(dev)
                       for k in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm")})
        router_w.append(shards.get(f"{p}mlp.gate.weight").to(dev, torch.float32))
        norms.append((shards.get(p + "input_layernorm.weight").to(dev),
                      shards.get(p + "post_attention_layernorm.weight").to(dev)))
        torch.cuda.empty_cache()
        if (lay + 1) % 8 == 0:
            log(f"  built layer {lay+1}/{layers} ({time.time()-t0:.0f}s)")
    embed = shards.get("model.embed_tokens.weight").pin_memory()  # CPU, row-gather
    final_norm = shards.get("model.norm.weight").to(dev)
    lm_head = shards.get("lm_head.weight").to(dev)
    log(f"build complete in {time.time()-t0:.0f}s")
    return host, attn_w, router_w, norms, embed, final_norm, lm_head


def rmsnorm(x, w, eps=1e-6):
    return (x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * w


def encode_prompt(tokenizer, prompt):
    """Robust chat-template encode -> flat list[int], across every return
    shape transformers has used (list[int], list[list[int]], tensor,
    BatchEncoding/dict). Falls back to plain tokenization if the template
    path misbehaves."""
    msgs = [{"role": "user", "content": prompt}]
    try:
        enc = tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True, return_dict=False)
    except Exception:
        enc = tokenizer(prompt)["input_ids"]
    if hasattr(enc, "input_ids"):          # BatchEncoding attribute access
        enc = enc.input_ids
    elif hasattr(enc, "keys"):             # mapping / dict-like
        enc = enc["input_ids"]
    if hasattr(enc, "tolist"):             # tensor
        enc = enc.tolist()
    while enc and isinstance(enc[0], (list, tuple)):  # unwrap batch dim
        enc = enc[0]
    toks = [int(x) for x in enc]
    assert toks and all(isinstance(x, int) for x in toks), f"bad encode: {toks[:5]}"
    return toks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/dev/shm/hf")
    ap.add_argument("--layers", type=int, default=L_TOT)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--paired-tokens", type=int, default=64,
                    help="tokens for the prefetch-off paired run + identity check")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = "cuda"
    layers = args.layers

    root = download(args.cache)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(root))
    prompts = [
        "The key difference between mixture-of-experts and dense transformer models is",
        "Write a haiku about memory bandwidth.",
        "Explain, in two sentences, why quantization reduces energy per token:",
    ]
    # Encode BEFORE the ~17 min build so any tokenizer quirk fails in seconds.
    prompt_toks = [encode_prompt(tokenizer, p) for p in prompts]
    log(f"prompts encoded: lengths {[len(t) for t in prompt_toks]}")

    # on-box link microbench (the Phase-B1 gap): pinned H2D, 1 GiB x 10
    pin = torch.empty(1 << 30, dtype=torch.uint8).pin_memory()
    gbuf = torch.empty(1 << 30, dtype=torch.uint8, device=dev)
    for _ in range(2):
        gbuf.copy_(pin, non_blocking=True)
    torch.cuda.synchronize()
    _ts = []
    for _ in range(10):
        _a = time.perf_counter()
        gbuf.copy_(pin, non_blocking=True)
        torch.cuda.synchronize()
        _ts.append(time.perf_counter() - _a)
    h2d_gbps = (1 << 30) / statistics.median(_ts) / 1e9
    del pin, gbuf
    torch.cuda.empty_cache()
    per_tok_bytes = layers * K_TOP * (2 * I * H // 2 + 2 * I * (H // 64) * 4
                                      + H * I // 2 + H * (I // 64) * 4)
    waterfall_toks = h2d_gbps * 1e9 / per_tok_bytes
    log(f"link {h2d_gbps:.1f} GB/s; {per_tok_bytes/1e9:.2f} GB/token -> waterfall {waterfall_toks:.2f} tok/s")

    shards = Shards(root)
    host, attn_w, router_w, norms, embed, final_norm, lm_head = build(shards, dev, layers)

    # rope tables + kv cache + staging (Phase A machinery, real-weights edition)
    MAXSEQ = 512
    pos = torch.arange(MAXSEQ)
    inv = 1.0 / (ROPE_THETA ** (torch.arange(0, HD, 2) / HD))
    ang = pos[:, None] * inv[None, :]
    COS = torch.cos(ang).to(dev, torch.bfloat16)
    SIN = torch.sin(ang).to(dev, torch.bfloat16)
    kc = torch.zeros(layers, N_KV, MAXSEQ, HD, dtype=torch.bfloat16, device=dev)
    vc = torch.zeros_like(kc)
    stage = [dict(
        gu_b=torch.empty(K_TOP, 2 * I, H // 2, dtype=torch.uint8, device=dev),
        gu_a=torch.empty(K_TOP, 2 * I, H // 64, dtype=torch.float32, device=dev),
        dn_b=torch.empty(K_TOP, H, I // 2, dtype=torch.uint8, device=dev),
        dn_a=torch.empty(K_TOP, H, I // 64, dtype=torch.float32, device=dev),
    ) for _ in range(2)]
    copy_stream = torch.cuda.Stream()
    copy_done = [torch.cuda.Event() for _ in range(2)]
    from nf4_grouped import gemm_4bit_grouped

    ids_dev = torch.arange(K_TOP, dtype=torch.int32, device=dev)
    sizes = [1] * K_TOP

    def rot(x, t):
        x1, x2 = x[:, 0::2], x[:, 1::2]
        c, s = COS[t], SIN[t]
        out = torch.empty_like(x)
        out[:, 0::2] = x1 * c - x2 * s
        out[:, 1::2] = x1 * s + x2 * c
        return out

    def attention(lay, h, t):
        w = attn_w[lay]
        q = (h @ w["q_proj"].t()).view(N_HEADS, HD)
        k = (h @ w["k_proj"].t()).view(N_KV, HD)
        v = (h @ w["v_proj"].t()).view(N_KV, HD)
        q = rmsnorm(q, w["q_norm"])
        k = rmsnorm(k, w["k_norm"])
        q, k = rot(q, t), rot(k, t)
        kc[lay, :, t] = k
        vc[lay, :, t] = v
        rep = N_HEADS // N_KV
        keys = kc[lay, :, : t + 1].repeat_interleave(rep, 0)
        vals = vc[lay, :, : t + 1].repeat_interleave(rep, 0)
        att = torch.einsum("hd,htd->ht", q.float(), keys.float()) / HD ** 0.5
        ctx = torch.einsum("ht,htd->hd", torch.softmax(att, -1), vals.float()).to(torch.bfloat16)
        return ctx.reshape(1, -1) @ w["o_proj"].t()

    moe_done = [torch.cuda.Event() for _ in range(2)]
    buf_ids = [[-1] * K_TOP, [-1] * K_TOP]   # expert id resident in each slot
    prev_eids = [list(range(K_TOP)) for _ in range(layers)]  # last token's routing
    hits_total = [0, 0]  # (hits, opportunities) under prefetch

    def copy_slot(buf, j, lay, e):
        s = stage[buf]
        gu_b, gu_a, dn_b, dn_a = host[lay]
        s["gu_b"][j].copy_(gu_b[e], non_blocking=True)
        s["gu_a"][j].copy_(gu_a[e], non_blocking=True)
        s["dn_b"][j].copy_(dn_b[e], non_blocking=True)
        s["dn_a"][j].copy_(dn_a[e], non_blocking=True)
        buf_ids[buf][j] = e

    def issue_copy(buf, eids, lay, guard_moe=False):
        with torch.cuda.stream(copy_stream):
            if guard_moe:
                copy_stream.wait_event(moe_done[buf])  # don't clobber an in-flight MoE read
            for j, e in enumerate(eids):
                copy_slot(buf, j, lay, e)
            copy_done[buf].record(copy_stream)

    def resolve(buf, lay, eids, wts, prefetch):
        """Make slots hold exactly `eids`; return slot-aligned weights + hit count."""
        torch.cuda.current_stream().wait_event(copy_done[buf])
        cur = buf_ids[buf]
        if not prefetch:
            return wts, K_TOP  # issue_copy already placed eids in order
        need = list(eids)
        hit_slots = {}
        for j, e in enumerate(cur):
            if e in need:
                hit_slots[e] = j
        misses = [e for e in need if e not in hit_slots]
        free = [j for j in range(K_TOP) if cur[j] not in need]
        if misses:
            with torch.cuda.stream(copy_stream):
                for e, j in zip(misses, free):
                    copy_slot(buf, j, lay, e)
                    hit_slots[e] = j
                copy_done[buf].record(copy_stream)
            torch.cuda.current_stream().wait_event(copy_done[buf])
        slot_w = torch.zeros(K_TOP, dtype=torch.bfloat16, device=wts.device)
        for e, w in zip(eids, wts):
            slot_w[hit_slots[e]] = w
        return slot_w, K_TOP - len(misses)

    def route(lay, h):
        logits = h.float() @ router_w[lay].t()
        probs = torch.softmax(logits, -1)[0]
        w, idx = torch.topk(probs, K_TOP)
        w = (w / w.sum()).to(torch.bfloat16)
        return idx.tolist(), w

    def moe(buf, h, weights):
        s = stage[buf]
        a_cat = h.expand(K_TOP, -1).contiguous()
        up = gemm_4bit_grouped(a_cat, s["gu_b"], s["gu_a"], sizes, ids_dev)
        act = (Fn.silu(up[:, :I].float()) * up[:, I:].float()).to(torch.bfloat16)
        down = gemm_4bit_grouped(act, s["dn_b"], s["dn_a"], sizes, ids_dev)
        return (weights[:, None] * down).sum(0, keepdim=True)

    def step(tok_id, t, prefetch):
        """One decode step. prefetch=False: copy after each layer's router
        resolves (the serialized Phase-B baseline). prefetch=True: while
        layer L computes, speculatively stream layer L+1's experts using
        LAST TOKEN'S routing for L+1 (real routers are sticky); on resolve,
        only mispredicted experts are fetched. Slot permutation is absorbed
        by slot-aligned router weights, so outputs are bit-identical to the
        non-speculative path (asserted by the paired identity check)."""
        h = embed[int(tok_id)].to(dev, torch.bfloat16).view(1, -1)
        if prefetch:
            issue_copy(0, prev_eids[0], 0, guard_moe=True)
        for lay in range(layers):
            buf = lay % 2
            h = h + attention(lay, rmsnorm(h, norms[lay][0]), t)
            hn = rmsnorm(h, norms[lay][1])
            eids, wts = route(lay, hn)
            if prefetch and lay + 1 < layers:
                issue_copy((lay + 1) % 2, prev_eids[lay + 1], lay + 1, guard_moe=True)
            if not prefetch:
                issue_copy(buf, eids, lay)
            slot_w, hits = resolve(buf, lay, eids, wts, prefetch)
            if prefetch:
                hits_total[0] += hits
                hits_total[1] += K_TOP
                prev_eids[lay] = list(eids)
            h = h + moe(buf, hn, slot_w)
            moe_done[buf].record()
        h = rmsnorm(h, final_norm)
        return (h @ lm_head.t()).float()

    def generate(toks, n_new, prefetch):
        """Prompt pass + greedy generation; returns (ids, median s/tok, hit_rate)."""
        for lay in range(layers):
            prev_eids[lay] = list(range(K_TOP))  # reset speculation state
        hits_total[0] = hits_total[1] = 0
        t = 0
        for tid in toks:
            logits = step(tid, t, prefetch)
            t += 1
        gen, times = [], []
        cur = int(logits.argmax())
        for _ in range(n_new):
            gen.append(cur)
            a = time.perf_counter()
            logits = step(cur, t, prefetch)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - a)
            t += 1
            cur = int(logits.argmax())
            if cur == tokenizer.eos_token_id:
                break
        hr = hits_total[0] / max(hits_total[1], 1) if prefetch else None
        return gen, statistics.median(times), hr

    torch.cuda.reset_peak_memory_stats()
    results = []
    for prompt, toks in zip(prompts, prompt_toks):
        g_off, med_off, _ = generate(toks, args.paired_tokens, prefetch=False)
        g_on, med_on, hr = generate(toks, args.max_new, prefetch=True)
        identical = g_off == g_on[: len(g_off)]
        text = tokenizer.decode(g_on, skip_special_tokens=True)
        bigrams = list(zip(g_on, g_on[1:]))
        d2 = len(set(bigrams)) / max(len(bigrams), 1)
        results.append({
            "prompt": prompt, "text": text, "n_tokens": len(g_on),
            "toks_per_s_off": 1.0 / med_off, "toks_per_s_on": 1.0 / med_on,
            "prefetch_speedup": med_off / med_on, "hit_rate": hr,
            "greedy_identical": identical, "distinct2": d2,
        })
        log(f"PROMPT: {prompt!r}\n  off {1.0/med_off:.2f} -> on {1.0/med_on:.2f} tok/s "
            f"(x{med_off/med_on:.3f}, hit {hr:.2f}, identical={identical})\n  {text[:220]}")

    out = {
        "model": MODEL, "layers": layers,
        "h2d_gbps": h2d_gbps, "per_token_gb": per_tok_bytes / 1e9,
        "waterfall_toks": waterfall_toks,
        "results": results,
        "vram_peak_gb": torch.cuda.max_memory_allocated() / 1e9,
        "gpu": torch.cuda.get_device_name(0),
    }
    Path(args.out).write_text(json.dumps(out, indent=1))
    log(f"wrote {args.out}; VRAM peak {out['vram_peak_gb']:.1f} GB")


if __name__ == "__main__":
    main()
