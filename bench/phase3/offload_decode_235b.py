#!/usr/bin/env python3
"""Qwen3-235B-A22B-class end-to-end offload decode: experts in host pinned
RAM, streamed per token, MoE compute via the fused NF4 kernel — the
tokens/sec demo for "a 235B MoE on a 24 GB-class VRAM budget".

Phase A: SYNTHETIC weights (decode timing is data-independent for a fixed
codebook gather; the property suite pins numerics separately). Attention is
REAL math (bf16 GQA + rotary + RMSNorm on resident weights, KV cache
appended) so the non-expert path costs what it costs; the router is random
top-k with a fixed seed (documented: gather locality at k=8/128 random is
the conservative case vs a learned router's reuse).

Working-set discipline: everything GPU-resident (attention weights, KV,
staging buffers, workspace) is budgeted to stay under --vram-cap (default
20 GB) so the number transfers to 24 GB consumer cards regardless of what
the host GPU is. VRAM peak is measured and reported.

Modes (--moe): fused (the kernel) | dequant (bnb dequant + matmul baseline)
| none (pure-stream ceiling: H2D only, no MoE compute). The waterfall
prediction is bytes_per_token / measured_H2D_GBps; the registered question
is whether the pipelined loop achieves it and whether fused compute stays
hidden under the copy stream.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "kernel"))


def build_layer_host_stacks(E, N_gu, K_gu, N_dn, K_dn, gen):
    """One layer's expert stacks in pinned host memory (synthetic NF4 bytes)."""
    gu_b = torch.randint(0, 256, (E, N_gu, K_gu // 2), dtype=torch.uint8,
                         generator=gen).pin_memory()
    gu_a = (torch.rand(E, N_gu, K_gu // 64, generator=gen) * 0.1 + 0.01).pin_memory()
    dn_b = torch.randint(0, 256, (E, N_dn, K_dn // 2), dtype=torch.uint8,
                         generator=gen).pin_memory()
    dn_a = (torch.rand(E, N_dn, K_dn // 64, generator=gen) * 0.1 + 0.01).pin_memory()
    return gu_b, gu_a, dn_b, dn_a


class Attention:
    """Minimal real GQA attention at bs1: resident bf16 weights, rotary, KV."""

    def __init__(self, hidden, n_heads, n_kv, head_dim, layers, max_seq, dev):
        g = torch.Generator(device="cpu").manual_seed(1)
        s = 0.02
        mk = lambda *shape: (torch.randn(*shape, generator=g) * s).to(dev, torch.bfloat16)
        self.q = [mk(n_heads * head_dim, hidden) for _ in range(layers)]
        self.k = [mk(n_kv * head_dim, hidden) for _ in range(layers)]
        self.v = [mk(n_kv * head_dim, hidden) for _ in range(layers)]
        self.o = [mk(hidden, n_heads * head_dim) for _ in range(layers)]
        self.kc = torch.zeros(layers, n_kv, max_seq, head_dim, dtype=torch.bfloat16, device=dev)
        self.vc = torch.zeros_like(self.kc)
        self.n_heads, self.n_kv, self.hd = n_heads, n_kv, head_dim
        pos = torch.arange(max_seq)
        inv = 1.0 / (10000 ** (torch.arange(0, head_dim, 2) / head_dim))
        ang = pos[:, None] * inv[None, :]
        self.cos = torch.cos(ang).to(dev, torch.bfloat16)
        self.sin = torch.sin(ang).to(dev, torch.bfloat16)

    def rot(self, x, t):  # x [heads, hd]
        h = x.shape[0]
        x1, x2 = x[:, 0::2], x[:, 1::2]
        c, s = self.cos[t], self.sin[t]
        out = torch.empty_like(x)
        out[:, 0::2] = x1 * c - x2 * s
        out[:, 1::2] = x1 * s + x2 * c
        return out

    def __call__(self, L, h, t):  # h [1, hidden]
        q = (h @ self.q[L].t()).view(self.n_heads, self.hd)
        k = (h @ self.k[L].t()).view(self.n_kv, self.hd)
        v = (h @ self.v[L].t()).view(self.n_kv, self.hd)
        q, k = self.rot(q, t), self.rot(k, t)
        self.kc[L, :, t] = k
        self.vc[L, :, t] = v
        rep = self.n_heads // self.n_kv
        keys = self.kc[L, :, : t + 1].repeat_interleave(rep, 0)   # [H, t+1, hd]
        vals = self.vc[L, :, : t + 1].repeat_interleave(rep, 0)
        att = torch.einsum("hd,htd->ht", q.float(), keys.float()) / self.hd ** 0.5
        w = torch.softmax(att, -1)
        ctx = torch.einsum("ht,htd->hd", w, vals.float()).to(torch.bfloat16)
        return ctx.reshape(1, -1) @ self.o[L].t()


def rmsnorm(x, eps=1e-6):
    return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps).to(x.dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=94)
    ap.add_argument("--experts", type=int, default=128)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--inter", type=int, default=1536)
    ap.add_argument("--heads", type=int, default=64)
    ap.add_argument("--kv-heads", type=int, default=4)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--warmup-tokens", type=int, default=4)
    ap.add_argument("--moe", choices=["fused", "dequant", "none"], default="fused")
    ap.add_argument("--vram-cap-gb", type=float, default=20.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dev = "cuda"
    E, k = args.experts, args.topk
    N_gu, K_gu = 2 * args.inter, args.hidden          # gate_up [E, 2I, H]
    N_dn, K_dn = args.hidden, args.inter              # down    [E, H, I]
    L = args.layers
    per_tok_bytes = L * k * (N_gu * K_gu // 2 + N_gu * (K_gu // 64) * 4
                             + N_dn * K_dn // 2 + N_dn * (K_dn // 64) * 4)

    from nf4_grouped import gemm_4bit_grouped, dequant_ref

    # --- link microbench (pinned H2D, 1 GiB x 10) --------------------------
    pin = torch.empty(1 << 30, dtype=torch.uint8).pin_memory()
    gbuf = torch.empty(1 << 30, dtype=torch.uint8, device=dev)
    for _ in range(2):
        gbuf.copy_(pin, non_blocking=True)
    torch.cuda.synchronize()
    ts = []
    for _ in range(10):
        a = time.perf_counter()
        gbuf.copy_(pin, non_blocking=True)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - a)
    h2d_gbps = (1 << 30) / statistics.median(ts) / 1e9
    del pin, gbuf
    torch.cuda.empty_cache()
    waterfall_toks = 1.0 / (per_tok_bytes / (h2d_gbps * 1e9))
    print(f"link: {h2d_gbps:.1f} GB/s pinned H2D; {per_tok_bytes/1e9:.2f} GB/token "
          f"-> waterfall ceiling {waterfall_toks:.2f} tok/s", flush=True)

    # --- host expert stacks (the 128 GB) -----------------------------------
    gen = torch.Generator().manual_seed(7)
    host = []
    t0 = time.time()
    for i in range(L):
        host.append(build_layer_host_stacks(E, N_gu, K_gu, N_dn, K_dn, gen))
        if (i + 1) % 10 == 0:
            print(f"  host stacks {i+1}/{L} ({time.time()-t0:.0f}s)", flush=True)
    host_gb = sum(sum(t.numel() * t.element_size() for t in lay) for lay in host) / 1e9
    print(f"host expert store: {host_gb:.1f} GB pinned in {time.time()-t0:.0f}s", flush=True)

    # --- GPU residents ------------------------------------------------------
    attn = Attention(args.hidden, args.heads, args.kv_heads, args.head_dim,
                     L, args.tokens + 8, dev)
    # double-buffered staging for one layer's active experts
    stage = [
        dict(
            gu_b=torch.empty(k, N_gu, K_gu // 2, dtype=torch.uint8, device=dev),
            gu_a=torch.empty(k, N_gu, K_gu // 64, dtype=torch.float32, device=dev),
            dn_b=torch.empty(k, N_dn, K_dn // 2, dtype=torch.uint8, device=dev),
            dn_a=torch.empty(k, N_dn, K_dn // 64, dtype=torch.float32, device=dev),
        )
        for _ in range(2)
    ]
    copy_stream = torch.cuda.Stream()
    copy_done = [torch.cuda.Event() for _ in range(2)]

    router_gen = torch.Generator().manual_seed(11)
    eids_all = torch.stack([
        torch.stack([torch.randperm(E, generator=router_gen)[:k] for _ in range(L)])
        for _ in range(args.tokens + args.warmup_tokens)
    ])  # [T, L, k] host

    ids_dev = torch.arange(k, dtype=torch.int32, device=dev)
    sizes = [1] * k

    def issue_copy(buf, tok, lay):
        s = stage[buf]
        eids = eids_all[tok, lay]
        with torch.cuda.stream(copy_stream):
            for j, e in enumerate(eids.tolist()):
                gu_b, gu_a, dn_b, dn_a = host[lay]
                s["gu_b"][j].copy_(gu_b[e], non_blocking=True)
                s["gu_a"][j].copy_(gu_a[e], non_blocking=True)
                s["dn_b"][j].copy_(dn_b[e], non_blocking=True)
                s["dn_a"][j].copy_(dn_a[e], non_blocking=True)
            copy_done[buf].record(copy_stream)

    def moe(buf, h):
        s = stage[buf]
        if args.moe == "none":
            return h
        a_cat = h.expand(k, -1).contiguous()  # token replicated per expert
        if args.moe == "fused":
            up = gemm_4bit_grouped(a_cat, s["gu_b"], s["gu_a"], sizes, ids_dev)
            gate, upv = up[:, : args.inter], up[:, args.inter:]
            act = (torch.nn.functional.silu(gate.float()) * upv.float()).to(torch.bfloat16)
            down = gemm_4bit_grouped(act, s["dn_b"], s["dn_a"], sizes, ids_dev)
        else:  # dequant baseline: torch LUT decode + bf16 matmul per expert
            outs = []
            for j in range(k):
                w_gu = dequant_ref(s["gu_b"][j], s["gu_a"][j], N_gu, K_gu).to(torch.bfloat16)
                u = h @ w_gu.t()
                gate, upv = u[:, : args.inter], u[:, args.inter:]
                a = (torch.nn.functional.silu(gate.float()) * upv.float()).to(torch.bfloat16)
                w_dn = dequant_ref(s["dn_b"][j], s["dn_a"][j], N_dn, K_dn).to(torch.bfloat16)
                outs.append(a @ w_dn.t())
            down = torch.cat(outs)
        return h + down.mean(0, keepdim=True)  # uniform router weights (synthetic)

    # --- pipelined decode loop ----------------------------------------------
    torch.cuda.reset_peak_memory_stats()
    h = torch.randn(1, args.hidden, device=dev, dtype=torch.bfloat16) * 0.1
    times = []
    T = args.tokens + args.warmup_tokens
    for tok in range(T):
        t_start = time.perf_counter()
        issue_copy(0, tok, 0)
        for lay in range(L):
            buf, nxt = lay % 2, (lay + 1) % 2
            if lay + 1 < L:
                issue_copy(nxt, tok, lay + 1)
            h = h + attn(lay, rmsnorm(h), tok)
            torch.cuda.current_stream().wait_event(copy_done[buf])
            h = moe(buf, rmsnorm(h))
        torch.cuda.synchronize()
        dt = time.perf_counter() - t_start
        if tok >= args.warmup_tokens:
            times.append(dt)
        if tok % 8 == 0:
            print(f"  tok {tok}/{T}: {dt*1e3:.0f} ms", flush=True)

    med = statistics.median(times)
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    out = {
        "config": vars(args),
        "per_token_gb": per_tok_bytes / 1e9,
        "h2d_gbps": h2d_gbps,
        "waterfall_ceiling_toks": waterfall_toks,
        "token_ms": [round(t * 1e3, 1) for t in times],
        "median_s_per_tok": med,
        "toks_per_s": 1.0 / med,
        "achieved_fraction_of_waterfall": (1.0 / med) / waterfall_toks,
        "vram_peak_gb": peak_gb,
        "gpu": torch.cuda.get_device_name(0),
    }
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"MODE={args.moe}: {1.0/med:.2f} tok/s (waterfall {waterfall_toks:.2f}, "
          f"{out['achieved_fraction_of_waterfall']*100:.0f}%), VRAM peak {peak_gb:.1f} GB "
          f"-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
