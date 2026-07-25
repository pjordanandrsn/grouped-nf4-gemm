#!/usr/bin/env python3
# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).

"""D1 of PREREG-kv-stream-faster.md (amendment 3) — fused attend on packed KV.

Stamped before the integration existed. The question is not whether the kernel
is fast (#12 measured 4.975 ms against fp16's 6.055 on this card at GQA 16:1) but
whether feeding it the cache's OWN tensors delivers that — `attend_nf4_kv_gqa`
takes `[T, H_kv, D/2]` packed and `[T, H_kv, D/64]` absmax, which is exactly what
an nf4 slot holds, so there is no repacking to lose it in.

Three arms, one decode step, GQA 16:1:

  dequant+SDPA   what NF4KVCache does today: materialize a bf16 layer, then SDPA
  fused          attend_nf4_kv_gqa straight off the slot tensors
  fp16 SDPA      a bf16 cache with no quantization at all — the reference

D1c is a gate: this is a different ARITHMETIC path, not a different schedule, so
unlike the prefetch work a wrong answer here would be silent and permanent.
"""
from __future__ import annotations

import json
import os
import statistics
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "/root/g/kernel")
from experts4bit_qlora import NF4KVCache  # noqa: E402
from nf4_kv import attend_nf4_kv_gqa  # noqa: E402

OUT = os.environ.get("FU_OUT", "/root/g/bench/context/kv_fused_bench.json")
REPS = int(os.environ.get("FU_REPS", "25"))
T = int(os.environ.get("FU_T", "32768"))
H_KV, H_Q, D = 4, 64, 128
GQA = H_Q // H_KV


def build_cache(t):
    c = NF4KVCache()
    g = torch.Generator(device="cpu").manual_seed(0)
    for lo in range(0, t, 4096):
        n = min(4096, t - lo)
        k = (torch.randn(1, H_KV, n, D, generator=g) * 0.5).bfloat16().cuda()
        v = (torch.randn(1, H_KV, n, D, generator=g) * 0.5).bfloat16().cuda()
        c.update(k, v, 0)
    torch.cuda.synchronize()
    return c


def timed(fn, reps=REPS):
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    ts = []
    for i in range(reps + 5):
        torch.cuda.synchronize()
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        if i >= 5:
            ts.append(start.elapsed_time(end) / 1e3)
    return statistics.median(ts)


def peak_of(fn):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() - before


def main():
    print(f"T={T} H_kv={H_KV} H_q={H_Q} (GQA {GQA}:1) D={D}  reps {REPS}",
          flush=True)
    c = build_cache(T)
    kslot, vslot = c._k[0], c._v[0]
    kp, ka = kslot[1], kslot[2]
    vp, va = vslot[1], vslot[2]
    torch.manual_seed(0)
    q = (torch.randn(H_Q, D) * 0.5).cuda().float()
    qb = q.bfloat16().view(1, H_Q, 1, D)
    scale = D ** -0.5

    # enable_gqa broadcasts the kv heads INSIDE the kernel. The obvious
    # alternative, repeat_interleave, materializes a 16x replicated cache --
    # 1.07 GB at this shape -- which no real attention path does and which would
    # make the baseline a strawman. Same reason the arms are bf16 and not fp32:
    # fp32 doubles the bytes and gives up the tensor-core path, and a reference
    # nobody would actually run is not a reference.
    def dequant_sdpa():
        k = c._load(kslot, torch.bfloat16)            # [1, H_kv, T, D]
        v = c._load(vslot, torch.bfloat16)
        return F.scaled_dot_product_attention(
            qb, k, v, scale=scale, enable_gqa=True).view(H_Q, D)

    def fused():
        return attend_nf4_kv_gqa(q, kp, ka, vp, va, scale=scale)

    # the reference a user would otherwise run: a bf16 cache, never quantized
    kb = c._load(kslot, torch.bfloat16)
    vb = c._load(vslot, torch.bfloat16)

    def fp16_sdpa():
        return F.scaled_dot_product_attention(
            qb, kb, vb, scale=scale, enable_gqa=True).view(H_Q, D)

    ref, got = dequant_sdpa(), fused()
    rel = ((got.float() - ref.float()).norm() / ref.float().norm()).item()

    t_deq = timed(dequant_sdpa)
    t_fus = timed(fused)
    t_f16 = timed(fp16_sdpa)
    del kb, vb
    torch.cuda.empty_cache()
    p_deq = peak_of(dequant_sdpa)
    p_fus = peak_of(fused)

    d1a, d1b, d1d = t_fus / t_deq, t_fus / t_f16, p_fus / max(p_deq, 1)
    print(f"\ndequant+SDPA {t_deq * 1e3:8.3f} ms   peak {p_deq / 2**20:7.2f} MB")
    print(f"fused        {t_fus * 1e3:8.3f} ms   peak {p_fus / 2**20:7.2f} MB")
    print(f"fp16 SDPA    {t_f16 * 1e3:8.3f} ms")
    print(f"\n=== scoring ===")
    print(f"D1a fused/(dequant+SDPA) = {d1a:.3f}   [0.25,0.50]  "
          f"{'CONFIRMED' if 0.25 <= d1a <= 0.50 else 'FALSIFIED'}")
    print(f"D1b fused/fp16 SDPA      = {d1b:.3f}   [0.70,1.10]  "
          f"{'CONFIRMED' if 0.70 <= d1b <= 1.10 else ('FALSIFIED' if not (0.60 <= d1b <= 1.40) else 'outside interval')}")
    print(f"D1c relative error       = {rel:.3e}  < 2e-3       "
          f"{'CONFIRMED' if rel < 2e-3 else 'FALSIFIED'}")
    print(f"D1d peak fused/dequant   = {d1d:.3f}   < 0.40       "
          f"{'CONFIRMED' if d1d < 0.40 else 'FALSIFIED'}")
    json.dump(dict(T=T, h_kv=H_KV, h_q=H_Q, gqa=GQA, d=D, reps=REPS,
                   s_dequant_sdpa=t_deq, s_fused=t_fus, s_fp16_sdpa=t_f16,
                   peak_dequant=p_deq, peak_fused=p_fus, rel_err=rel,
                   D1a=d1a, D1b=d1b, D1d=d1d), open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
