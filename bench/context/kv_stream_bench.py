#!/usr/bin/env python3
# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).

"""Registered harness for `PREREG-kv-streaming.md` (S1-S5). Stamped before it ran.

Tests whether finding #14's model -- `step_time = bytes / link` -- describes the
host-resident KV tier that actually exists, rather than the one the arithmetic
imagined. Synthetic geometry, no model and no weights: KV transfer cost is a
function of geometry, which is what lets a 235B-shaped cache be measured on a
12 GB card (same rationale as `kv_verify.py`'s rung-one probe).

Two measurements, split on purpose:

  load-only    every layer loaded once. The transfer term ALONE -- both arms
               dequantize identically, and only the host arm crosses PCIe
               first. S1-S3 are scored on this.
  load+append  a full decode step. Separated because the arms have structurally
               different append costs (S5: `torch.cat` reallocates the whole
               packed store per layer per step; the arena writes one token at an
               offset), which would otherwise contaminate the transfer number.
"""
from __future__ import annotations

import json
import os
import statistics
import sys

import torch

sys.path.insert(0, "/root/g/kernel")
from experts4bit_qlora import NF4KVCache  # noqa: E402

OUT = os.environ.get("SB_OUT", "/root/g/bench/context/kv_stream_bench.json")
REPS = int(os.environ.get("SB_REPS", "10"))
LINK_GBS = float(os.environ.get("SB_LINK", "6.20"))     # measured, pcie_probe.json

# Qwen3-235B's KV geometry. 576 packed B/layer/token = 2 x 4 x 128 x 0.5625.
LAYERS, H_KV, D = 94, 4, 128
CONTEXTS = [4096, 8192, 32768]
NF4_B_PER_ELEM = 0.5 + 4.0 / 64.0


def packed_bytes(ctx, quant):
    per = 2 * H_KV * D * (NF4_B_PER_ELEM if quant else 2.0)
    return int(per * LAYERS * ctx)


def build(ctx, quant, residence, chunk=2048):
    """Prefill a cache to `ctx` tokens. Untimed: arena page-locking and the
    initial quantization are setup, not steady state."""
    kw = dict(quantize_keys=quant, quantize_values=quant)
    if residence == "host":
        kw.update(residence="host", max_context=ctx + 4 * REPS + 64)
    c = NF4KVCache(**kw)
    g = torch.Generator(device="cpu").manual_seed(0)
    for lo in range(0, ctx, chunk):
        n = min(chunk, ctx - lo)
        k = (torch.randn(1, H_KV, n, D, generator=g) * 0.5).bfloat16().cuda()
        v = (torch.randn(1, H_KV, n, D, generator=g) * 0.5).bfloat16().cuda()
        for li in range(LAYERS):
            c.update(k, v, li)
        del k, v
    torch.cuda.synchronize()
    return c


def check_pinned(c):
    """Confound 3: a prefix view that lost its pinning would silently stage
    through a bounce buffer and inflate S1 for reasons unrelated to the law."""
    if c.residence != "host":
        return
    slot = c._k[0]
    t = slot[1]
    assert not t.is_cuda, "host arm is holding the cache on the GPU"
    assert t.is_pinned(), (
        "arena view is NOT pinned -- the copy would stage through a bounce "
        "buffer and S1 would measure the wrong thing")


def time_loop(fn, reps=REPS):
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    ts = []
    for i in range(reps + 2):
        torch.cuda.synchronize()
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        if i >= 2:
            ts.append(start.elapsed_time(end) / 1e3)
    return statistics.median(ts)


def measure(ctx, quant, residence):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    c = build(ctx, quant, residence)
    check_pinned(c)
    base_peak = torch.cuda.max_memory_allocated()

    def load_only():
        for li in range(LAYERS):
            k = c._load(c._k[li], torch.bfloat16)
            v = c._load(c._v[li], torch.bfloat16)
            del k, v

    g = torch.Generator(device="cpu").manual_seed(1)
    k1 = (torch.randn(1, H_KV, 1, D, generator=g) * 0.5).bfloat16().cuda()
    v1 = (torch.randn(1, H_KV, 1, D, generator=g) * 0.5).bfloat16().cuda()

    def step():
        for li in range(LAYERS):
            k, v = c.update(k1, v1, li)
            del k, v

    t_load = time_loop(load_only)
    t_step = time_loop(step)
    free, total = torch.cuda.mem_get_info()
    r = dict(context=ctx, dtype="nf4" if quant else "bf16", residence=residence,
             load_s=t_load, step_s=t_step, append_s=max(t_step - t_load, 0.0),
             peak_bytes=base_peak, cache_bytes=c.memory_bytes(),
             device_bytes=c.device_bytes(), packed_bytes=packed_bytes(ctx, quant),
             free_vram_mb=free // 2 ** 20, total_vram_mb=total // 2 ** 20)
    del c, k1, v1
    torch.cuda.empty_cache()
    return r


def main():
    print(f"geometry: {LAYERS}L x {H_KV}kv x {D}d (Qwen3-235B)   "
          f"link {LINK_GBS} GB/s   reps {REPS}", flush=True)
    rows = []
    jobs = [(c, True) for c in CONTEXTS] + [(8192, False), (4096, False)]
    for ctx, quant in jobs:
        for residence in ("gpu", "host"):
            r = measure(ctx, quant, residence)
            rows.append(r)
            print(f"ctx={ctx:>6} {r['dtype']:>4} {residence:>4}  "
                  f"load={r['load_s'] * 1e3:8.1f}ms  step={r['step_s'] * 1e3:8.1f}ms  "
                  f"append={r['append_s'] * 1e3:7.1f}ms  "
                  f"peak={r['peak_bytes'] / 2 ** 20:7.1f}MB  "
                  f"free={r['free_vram_mb']}MB", flush=True)
            json.dump(rows, open(OUT, "w"), indent=2)

    def get(ctx, dt, res):
        return next(r for r in rows if r["context"] == ctx
                    and r["dtype"] == dt and r["residence"] == res)

    print("\n=== scoring ===")
    verdicts = {}
    for ctx in CONTEXTS:
        g, h = get(ctx, "nf4", "gpu"), get(ctx, "nf4", "host")
        over = h["load_s"] - g["load_s"]
        pred = h["packed_bytes"] / (LINK_GBS * 1e9)
        ratio = over / pred
        tag = "S1" if ctx == 32768 else "  "
        print(f"{tag} ctx={ctx:>6} overhead={over * 1e3:8.1f}ms "
              f"predicted={pred * 1e3:8.1f}ms  ratio={ratio:.3f}")
        if ctx == 32768:
            verdicts["S1"] = dict(measured=ratio, interval=[0.85, 1.25],
                                  verdict="CONFIRMED" if 0.85 <= ratio <= 1.25
                                  else "FALSIFIED")

    o_nf4 = get(8192, "nf4", "host")["load_s"] - get(8192, "nf4", "gpu")["load_s"]
    o_bf16 = get(8192, "bf16", "host")["load_s"] - get(8192, "bf16", "gpu")["load_s"]
    s2 = o_bf16 / o_nf4
    verdicts["S2"] = dict(measured=s2, interval=[3.0, 4.1],
                          verdict="CONFIRMED" if 3.0 <= s2 <= 4.1 else "FALSIFIED")
    print(f"S2 bf16/nf4 overhead @8192 = {s2:.3f}  (exact byte ratio 3.5556)")

    o32 = get(32768, "nf4", "host")["load_s"] - get(32768, "nf4", "gpu")["load_s"]
    s3 = o32 / o_nf4
    verdicts["S3"] = dict(measured=s3, interval=[3.4, 4.6],
                          verdict="CONFIRMED" if 3.4 <= s3 <= 4.6 else "FALSIFIED")
    print(f"S3 overhead 32K/8K = {s3:.3f}  (exact context ratio 4.0)")

    s4 = get(32768, "nf4", "host")["peak_bytes"] / get(32768, "nf4", "gpu")["peak_bytes"]
    verdicts["S4"] = dict(measured=s4, threshold=0.10,
                          verdict="CONFIRMED" if s4 < 0.10 else "FALSIFIED")
    print(f"S4 peak GPU streamed/resident @32K = {s4:.4f}  (< 0.10)")

    r_app = get(32768, "nf4", "gpu")["append_s"] / get(4096, "nf4", "gpu")["append_s"]
    h_app = get(32768, "nf4", "host")["append_s"] / get(4096, "nf4", "host")["append_s"]
    ok = r_app >= 4.0 and 0.5 <= h_app <= 2.0
    verdicts["S5"] = dict(resident_ratio=r_app, host_ratio=h_app,
                          verdict="CONFIRMED" if ok else "FALSIFIED")
    print(f"S5 append 32K/4K: resident={r_app:.2f} (>=4.0)  host={h_app:.2f} "
          f"(0.5-2.0)")

    for k, v in verdicts.items():
        print(f"  {k}: {v['verdict']}")
    json.dump(dict(rows=rows, verdicts=verdicts, link_GBs=LINK_GBS,
                   geometry=dict(layers=LAYERS, h_kv=H_KV, head_dim=D)),
              open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
