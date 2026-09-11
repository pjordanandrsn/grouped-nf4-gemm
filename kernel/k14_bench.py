#!/usr/bin/env python3
"""K14 Stage A: at M=16, does an int4 path beat dequant-then-GEMM on the
attention projection shapes?

    python kernel/k14_bench.py [--out rows.json] [--ms M,M,...]

`Int4Linear` (experts4bit-qlora) serves M=1 with the int4 GEMV and EVERYTHING
above it by dequantising the weight once, caching the bf16 copy, and handing it
to cuBLAS. P42's census measured what that costs at B=16: the int4 arm and the
plain-bf16 arm land within 0.027 ms of each other on the same 241 calls, so the
store saves no time at batch, and it holds both representations resident
(1.812 GB bf16 + 0.453 GB int4 for this model) while reading only one.

The module's comment says a batched int4 attention that wins "needs a small-M
int4 GEMM with weight-tile reuse; not this module". **That kernel already
exists here** -- `gemm_int4_b32_grouped_captured`, int8 MMA through `tl.dot`
against prebuilt device tiles -- built for the expert path and never pointed at
a single projection. A projection is the E=1 case of it: one group, M rows,
local expert index 0, which is exactly the shape of `Int4Linear`'s own
`packed [1, N, K//2]` / `scales [1, N, K//32]` buffers.

So Stage A writes no kernel. It measures three shipped paths on the four real
projection shapes:

  bf16   x @ dequant(packed).t()              -- what ships above M=1
  gemv   quant_x_rows + gemv_int4_b32         -- what GEMV_ROWS_MAX=1 forbids
  gemm   quant_x_rows + grouped GEMM, E=1     -- the kernel nobody called here

plus `floor`, the graph-replay cost of one trivial kernel, because a shape whose
baseline already sits at the launch floor cannot be helped by reading fewer
bytes, however few.

Measurement follows int4_b32's own two load-bearing rules: CUDA-graph replay,
never eager (the eager host floor hides everything), and every arm's FULL cost
including its reduce and its activation quantisation. An arm timed without its
quantisation would be a kernel time, not a path time.

The bf16 arm dequantises the SAME packed bytes rather than using the original
weight, so all three arms compute the same underlying product and the numbers
are comparable. The int4 arms additionally quantise activations to int8, which
the shipped decode path already accepts at M=1; the relative error is reported,
not asserted away.
"""
import argparse
import json
import os
import sys

import torch
import triton

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from int4_b32 import (gemm_int4_b32_grouped_captured, gemv_int4_b32,  # noqa: E402
                      quant_x_rows)
from int4_pack_ref import dequant_int4_ref, pack_int4_b32  # noqa: E402

# Qwen/Qwen3-30B-A3B @ ad44e777: hidden 2048, 32 heads, 4 kv heads, head_dim 128.
# (N, K) as Int4Linear stores them: N = out_features, K = in_features.
# `gemm_int4_b32_grouped_captured`'s shipped default, and the M extent of the
# hardware MMA K11 read out of the PTX (`mma.sync.aligned.m16n8k16`).
BLOCK_M = 16

SHAPES = [("q_proj", 4096, 2048), ("k_proj", 512, 2048),
          ("v_proj", 512, 2048), ("o_proj", 2048, 4096)]


def timed_replay(fn, iters=200, warmup=20):
    """ms per replay of fn, captured into a graph. Buffers must be preallocated
    by the caller -- capture forbids allocation."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(iters):
        g.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters


def launch_floor(dev):
    """One trivial kernel under graph replay: the per-node floor this device
    charges for existing. Any arm within noise of this is launch-bound."""
    t = torch.zeros(1, device=dev)
    return timed_replay(lambda: t.add_(1.0))


def peak_bw_gbs(dev):
    """Theoretical device bandwidth, from the properties torch exposes. Reported
    as context for the GB/s columns; no verdict rests on it."""
    p = torch.cuda.get_device_properties(dev)
    bus, clk = getattr(p, "memory_bus_width", 0), getattr(p, "memory_clock_rate", 0)
    if not bus or not clk:
        return None
    return clk * 1e3 * (bus / 8) * 2 / 1e9      # kHz -> Hz, bits -> bytes, DDR


def one_shape(name, N, K, ms, dev="cuda"):
    torch.manual_seed(0)
    w = torch.randn(N, K, dtype=torch.bfloat16, device=dev) / (K ** 0.5)
    packed, scales = pack_int4_b32(w.float().cpu())
    packed = packed.reshape(1, N, K // 2).to(dev).contiguous()
    scales = scales.reshape(1, N, K // 32).to(dev).contiguous()
    w_deq = dequant_int4_ref(packed.reshape(N, K // 2),
                             scales.reshape(N, K // 32)).to(torch.bfloat16).to(dev)
    del w

    rows = []
    for M in ms:
        x = torch.randn(M, K, dtype=torch.bfloat16, device=dev) / 8
        eids = torch.zeros(M, dtype=torch.int32, device=dev)
        # The tile table for the degenerate E=1 case, written out rather than
        # built. A tile holds at most BLOCK_M rows -- the kernel's m_mask is
        # `tl.arange(0, BLOCK_M) < rows`, so a tile claiming more rows than that
        # would silently compute only the first BLOCK_M and drop the rest. The
        # correctness check below would catch it, but building it right is
        # better than relying on a tolerance to notice.
        t0 = list(range(0, M, BLOCK_M))
        t_row0 = torch.tensor(t0, dtype=torch.int32, device=dev)
        t_rows = torch.tensor([min(BLOCK_M, M - r) for r in t0],
                              dtype=torch.int32, device=dev)
        t_group = torch.zeros(len(t0), dtype=torch.int32, device=dev)
        assert int(t_rows.sum()) == M and int(t_rows.max()) <= BLOCK_M

        ref = (x @ w_deq.t()).float()
        out = {}
        err = {}

        def a_bf16(x=x):
            return x @ w_deq.t()

        def a_gemv(x=x, eids=eids):
            xq, xs = quant_x_rows(x)
            return gemv_int4_b32(xq, xs, packed, scales, eids, N, K)

        def a_gemm(x=x, t_row0=t_row0, t_rows=t_rows, t_group=t_group):
            xq, xs = quant_x_rows(x)
            return gemm_int4_b32_grouped_captured(xq, xs, packed, scales,
                                                  t_row0, t_rows, t_group,
                                                  block_m=BLOCK_M)

        for arm, fn in (("bf16", a_bf16), ("gemv", a_gemv), ("gemm", a_gemm)):
            try:
                y = fn().float()
                denom = ref.abs().max().clamp_min(1e-30)
                err[arm] = float((y - ref).abs().max() / denom)
                out[arm] = timed_replay(fn)
            except Exception as e:                      # noqa: BLE001
                out[arm] = f"error: {type(e).__name__}: {e}"
                err[arm] = None
                print(f"    {name} M={M} {arm} FAILED {type(e).__name__}: {e}",
                      flush=True)

        wb_bf16 = N * K * 2
        wb_int4 = N * K // 2 + N * (K // 32) * 2
        row = {"shape": name, "N": N, "K": K, "M": M, "ms": out, "rel_err": err,
               "weight_bytes": {"bf16": wb_bf16, "int4": wb_int4},
               "gbs": {a: (wb_bf16 if a == "bf16" else wb_int4) / (t / 1e3) / 1e9
                       for a, t in out.items() if isinstance(t, float)}}
        for a in ("gemv", "gemm"):
            if isinstance(out.get(a), float) and isinstance(out.get("bf16"), float):
                row.setdefault("vs_bf16", {})[a] = out[a] / out["bf16"]
        rows.append(row)
        cells = " ".join(f"{a}={out[a] * 1e3:.1f}us" if isinstance(out[a], float)
                         else f"{a}=ERR" for a in ("bf16", "gemv", "gemm"))
        ratios = " ".join(f"{a}/bf16={row['vs_bf16'][a]:.3f}"
                          for a in row.get("vs_bf16", {}))
        print(f"   {name:8s} N={N:5d} K={K:5d} M={M:3d} | {cells} | {ratios}",
              flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--ms", default="1,4,8,16,32")
    a = ap.parse_args()
    ms = [int(x) for x in a.ms.split(",")]

    p = torch.cuda.get_device_properties(0)
    floor = launch_floor("cuda")
    bw = peak_bw_gbs(0)
    print(f"# {p.name}, {p.multi_processor_count} SMs, torch {torch.__version__}, "
          f"triton {triton.__version__}")
    print(f"# graph-replay launch floor {floor * 1e3:.2f} us; "
          f"peak bandwidth {('%.0f GB/s' % bw) if bw else 'unknown'}\n")

    rows = []
    for name, N, K in SHAPES:
        rows += one_shape(name, N, K, ms)

    rep = {"gpu": p.name, "sms": p.multi_processor_count,
           "torch": torch.__version__, "triton": triton.__version__,
           "launch_floor_ms": floor, "peak_bw_gbs": bw, "rows": rows}
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rep, f, indent=1)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
