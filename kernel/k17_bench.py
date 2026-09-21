#!/usr/bin/env python3
"""Lane K17 bench (kernel/PREREG-k17-fused-splitk-gemv.md): the int4-b32 GEMV with its split-K reduce as a
second launch (today's path) against the same GEMV with the reduce folded into its own launch, per shape and
row count, under K14's instrument -- CUDA-graph replay medians with the launch floor beside every row.

    python k17_bench.py k17_rows.json

Arms per (shape, R): ``two_launch`` = gemv_int4_b32(..., fused_reduce=False); ``fused`` = fused_reduce=True.
Every row first asserts P1 -- the two arms are torch.equal -- and refuses to time a row that is not (a perf
number for a kernel that computes something else is the failure mode this repository does not report).
Rows carry: us per call for both arms, the saving in us and as a multiple of the launch floor, SK from the
planner, the relative error of each arm against the dequantised reference, and the raw bytes of the store.
"""
import json
import sys

import torch

from int4_b32 import _plan, gemv_int4_b32, quant_x_rows
from int4_pack_ref import dequant_int4_ref, pack_int4_b32
from k14_bench import launch_floor, timed_replay

dev = "cuda"
floor = launch_floor(dev)
print(f"launch floor {floor*1e3:.2f} us on {torch.cuda.get_device_name()}", flush=True)

# The registered shapes: Qwen3-30B-A3B's expert projections (per expert) and attention projections
# (unfused and the P54 fused qkv). R = activation rows per call: 1 (B=1 decode), 8/16 (small batches),
# 128 (B=16 x top-8 expert rows).
SHAPES = [("expert_gate_up", 1536, 2048), ("expert_down", 2048, 768),
          ("attn_q", 4096, 2048), ("attn_kv", 512, 2048), ("attn_o", 2048, 4096), ("attn_qkv_fused", 5120, 2048)]
ROWS = (1, 8, 16, 128)
E = 8            # experts in the store; eids cycle through them so several experts are touched at R > 1

rows = []
for name, N, K in SHAPES:
    torch.manual_seed(0)
    w = torch.randn(E, N, K) / (K ** 0.5)
    packed, scales = zip(*(pack_int4_b32(w[e].float()) for e in range(E)))
    packed = torch.stack([p.reshape(N, K // 2) for p in packed]).contiguous().to(dev)
    scales = torch.stack([s.reshape(N, K // 32) for s in scales]).contiguous().to(dev)
    deq = [dequant_int4_ref(packed[e].cpu(), scales[e].cpu(), N, K) for e in range(E)]
    for R in ROWS:
        torch.manual_seed(1)
        x = (torch.randn(R, K) / 8).to(torch.bfloat16).to(dev)
        eids = (torch.arange(R) % E).to(torch.int32).to(dev)
        xq, xs = quant_x_rows(x)
        bn, wp, sk, ku = _plan(N, K, R, torch.cuda.get_device_properties(dev).multi_processor_count)
        y0 = gemv_int4_b32(xq, xs, packed, scales, eids, N, K, fused_reduce=False)
        y1 = gemv_int4_b32(xq, xs, packed, scales, eids, N, K, fused_reduce=True)
        equal = bool(torch.equal(y0, y1))
        ref = torch.stack([x[r].float().cpu() @ deq[int(eids[r])].T for r in range(R)])
        denom = ref.abs().max().clamp_min(1e-30)
        err0 = float((y0.float().cpu() - ref).abs().max() / denom)
        err1 = float((y1.float().cpu() - ref).abs().max() / denom)
        row = {"shape": name, "N": N, "K": K, "R": R, "E": E, "sk": sk, "block_n": bn, "warps": wp, "ku": ku,
               "p1_equal": equal, "rel_err": {"two_launch": err0, "fused": err1},
               "store_bytes": int(packed[0].numel() + scales[0].numel() * scales.element_size()) * E}
        if not equal:
            row["ms"] = {"two_launch": "refused: P1 failed (fused != two_launch)", "fused": "refused"}
            print(f"   {name:16s} N={N:5d} K={K:5d} R={R:3d} sk={sk:2d} | P1 FAILED: max|delta| "
                  f"{(y0.float() - y1.float()).abs().max().item():g} -- not timed", flush=True)
            rows.append(row)
            continue
        # preallocate everything so both arms allocate nothing under capture (K14's contract)
        part = torch.empty(sk * R, N, dtype=torch.float32, device=dev)
        out0 = torch.empty(R, N, dtype=torch.bfloat16, device=dev)
        cnt = torch.zeros(R * ((N + bn - 1) // bn), dtype=torch.int32, device=dev)
        out1 = torch.empty(R, N, dtype=torch.bfloat16, device=dev)

        def two_launch(xq=xq, xs=xs, eids=eids, part=part, out0=out0):
            return gemv_int4_b32(xq, xs, packed, scales, eids, N, K, part=part, out=out0, fused_reduce=False)

        def fused(xq=xq, xs=xs, eids=eids, part=part, cnt=cnt, out1=out1):
            return gemv_int4_b32(xq, xs, packed, scales, eids, N, K, part=part, cnt=cnt, out=out1, fused_reduce=True)

        ms = {}
        for arm, fn in (("two_launch", two_launch), ("fused", fused), ("two_launch_b", two_launch), ("fused_b", fused)):
            try:
                ms[arm] = timed_replay(fn, iters=200)
            except Exception as e:                       # noqa: BLE001
                ms[arm] = f"error: {type(e).__name__}: {e}"
                print(f"   {name} R={R} {arm} FAILED {type(e).__name__}: {e}", flush=True)
        # the counter must read zero after 800+ captured replays, or the epilogue is not re-arming
        row["cnt_zero_after_replays"] = bool(int(cnt.abs().sum()) == 0)
        row["ms"] = ms
        if all(isinstance(ms.get(a), float) for a in ("two_launch", "fused", "two_launch_b", "fused_b")):
            t0 = min(ms["two_launch"], ms["two_launch_b"])
            t1 = min(ms["fused"], ms["fused_b"])
            row["us"] = {"two_launch": t0 * 1e3, "fused": t1 * 1e3, "saving": (t0 - t1) * 1e3,
                         "saving_over_floor": (t0 - t1) / floor, "fused_over_two_launch": t1 / t0,
                         "aa_spread": {"two_launch": abs(ms["two_launch"] - ms["two_launch_b"]) * 1e3,
                                       "fused": abs(ms["fused"] - ms["fused_b"]) * 1e3}}
            print(f"   {name:16s} N={N:5d} K={K:5d} R={R:3d} sk={sk:2d} | two-launch {t0*1e3:7.2f} us | fused {t1*1e3:7.2f} us | "
                  f"saving {(t0-t1)*1e3:+6.2f} us = {(t0-t1)/floor:+.2f} floors | ratio {t1/t0:.3f} | "
                  f"cnt re-armed {row['cnt_zero_after_replays']}", flush=True)
        rows.append(row)

json.dump({"device": torch.cuda.get_device_name(), "sm_count": torch.cuda.get_device_properties(dev).multi_processor_count,
           "launch_floor_us": floor * 1e3, "torch": torch.__version__, "rows": rows},
          open(sys.argv[1] if len(sys.argv) > 1 else "k17_rows.json", "w"), indent=1)
bad = [r for r in rows if not r["p1_equal"]]
print(f"rows: {len(rows)}; P1 failures: {len(bad)}", flush=True)
sys.exit(1 if bad else 0)
