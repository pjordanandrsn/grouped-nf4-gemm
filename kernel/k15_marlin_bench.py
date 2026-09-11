#!/usr/bin/env python3
"""K15: what does a Marlin-class 4-bit GEMM cost at M=16 on our projection shapes?

    python kernel/k15_marlin_bench.py [--out rows.json] [--ms 1,16,32]

Runs in a **vLLM venv**, not ours: vLLM 0.28.0 brings torch 2.13.0+cu130 while
the e4b/gnf4 stack is on 2.8.0+cu128, so the two sides cannot share a process.
That is the same two-venv shape P37 used to put the engines side by side.

Why this lane exists. P42 and K14 established that at B=16 our attention
projections dequantise to bf16 and read 1.812 GB per step where vLLM's GPTQ-Int4
checkpoint reads 0.467 GB, and that no int4 arm WE ship beats that dequant path
-- our grouped GEMM runs at 22 % of the bandwidth ceiling against a bf16 path at
106 %. vLLM nonetheless completes the whole step in 7.882 ms against our 11.93.
Marlin is the kernel that makes their side of that possible, so before writing
one it is worth measuring what one actually costs on exactly our shapes.

Two group sizes, deliberately:

  g128  the comparator's own configuration (~4.125 bits/weight incl. scales)
  g32   OUR group size (~4.5 bits/weight)

g128 answers "what does vLLM get". g32 answers the engineering question --
whether Marlin's advantage is the weight FORMAT or the KERNEL -- by matching our
bytes exactly and leaving only the kernel different.

Correctness is checked in-process against the dequantised reference Marlin's own
quantiser returns, and reported per cell. It cannot be checked beforehand on the
house A2000: that host's driver is 12.9 and this torch build needs 13.0.
"""
import argparse
import json

import torch

# (name, N=out_features, K=in_features) -- Qwen/Qwen3-30B-A3B @ ad44e777.
# Marlin takes the weight as [K, N], so these are transposed at construction.
SHAPES = [("q_proj", 4096, 2048), ("k_proj", 512, 2048),
          ("v_proj", 512, 2048), ("o_proj", 2048, 4096)]


def timed_replay(fn, iters=200, warmup=20):
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


def launch_floor(dev="cuda"):
    t = torch.zeros(1, device=dev)
    return timed_replay(lambda: t.add_(1.0))


def streaming_gbs(dev="cuda"):
    """Measured, not read off a spec sheet -- the same 512 MB copy k14_bench uses,
    so the two lanes' GB/s columns are against comparable ceilings."""
    n = 512 * 1024 * 1024 // 4
    src = torch.empty(n, dtype=torch.float32, device=dev)
    dst = torch.empty_like(src)
    ms = timed_replay(lambda d=dst, s=src: d.copy_(s), iters=30, warmup=5)
    gbs = (src.numel() * 4 * 2) / (ms / 1e3) / 1e9
    del src, dst
    torch.cuda.empty_cache()
    return gbs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--ms", default="1,4,8,16,32")
    ap.add_argument("--groups", default="128,32")
    a = ap.parse_args()
    ms_list = [int(x) for x in a.ms.split(",")]
    groups = [int(x) for x in a.groups.split(",")]

    import vllm
    from vllm.model_executor.layers.quantization.utils import marlin_utils as mu
    from vllm.model_executor.layers.quantization.utils import marlin_utils_test as mt
    from vllm.scalar_type import scalar_types

    p = torch.cuda.get_device_properties(0)
    floor = launch_floor()
    bw = streaming_gbs()
    print(f"# {p.name}, {p.multi_processor_count} SMs, vllm {vllm.__version__}, "
          f"torch {torch.__version__}")
    print(f"# graph-replay launch floor {floor * 1e3:.2f} us; "
          f"measured streaming bandwidth {bw:.0f} GB/s")
    import os as _os
    atomic = _os.environ.get("VLLM_MARLIN_USE_ATOMIC_ADD", "")
    print(f"# marlin supported group sizes: {mu.MARLIN_SUPPORTED_GROUP_SIZES}")
    # vLLM's own log recommends this for small size_n, which is exactly k_proj/v_proj
    # (N=512). Measuring with it OFF only would understate the comparator on half the
    # shapes, so the runner invokes this bench twice and the receipts say which is which.
    print(f"# VLLM_MARLIN_USE_ATOMIC_ADD={atomic or '(unset)'}\n")

    rows = []
    for name, N, K in SHAPES:
        for gs in groups:
            if gs not in mu.MARLIN_SUPPORTED_GROUP_SIZES:
                print(f"   {name} g{gs}: unsupported, skipped", flush=True)
                continue
            torch.manual_seed(0)
            w = torch.randn(K, N, dtype=torch.half, device="cuda") / (K ** 0.5)
            try:
                q = mt.marlin_quantize(w, scalar_types.uint4b8, gs, act_order=False)
            except Exception as e:                       # noqa: BLE001
                print(f"   {name} g{gs} QUANTIZE FAILED {type(e).__name__}: {e}",
                      flush=True)
                continue
            w_ref, q_w, s = q[0], q[1], q[2]
            g_idx, sort_idx = q[3], q[4]
            # MarlinWorkspace sizes its scratch as N//min_thread_n * max_parallel, which
            # for N=512 is 128 -- and the kernel refuses below `min_workspace_size`, which
            # on this part is the SM count (170): "workspace.numel = 128 is below
            # min_workspace_size = 170". k_proj and v_proj lost every cell to that in
            # k15-marlin-4. Size it to cover both rules; a too-large scratch costs
            # kilobytes and is not on the timed path.
            need = max(N // mu.GPTQ_MARLIN_MIN_THREAD_N * mu.GPTQ_MARLIN_MAX_PARALLEL,
                       2 * torch.cuda.get_device_properties(0).multi_processor_count)
            ws = mt.MarlinWorkspace(N, mu.GPTQ_MARLIN_MIN_THREAD_N,
                                    mu.GPTQ_MARLIN_MAX_PARALLEL)
            if ws.scratch.numel() < need:
                ws.scratch = torch.zeros(need, dtype=torch.int, device="cuda")
            # 4 bits of weight + one fp16 scale per group, per output column
            wb = N * K // 2 + N * (K // gs) * 2
            for M in ms_list:
                x = torch.randn(M, K, dtype=torch.half, device="cuda") / 8

                def call(x=x, q_w=q_w, s=s, g_idx=g_idx, sort_idx=sort_idx, ws=ws):
                    return mu.apply_gptq_marlin_linear(
                        x, q_w, s, None, g_idx, sort_idx, ws.scratch,
                        scalar_types.uint4b8, N, K, True, bias=None)
                try:
                    y = call().float()
                    ref = (x @ w_ref).float()
                    err = float((y - ref).abs().max()
                                / ref.abs().max().clamp_min(1e-30))
                    t = timed_replay(call)
                except Exception as e:                   # noqa: BLE001
                    print(f"   {name} g{gs} M={M} FAILED {type(e).__name__}: {e}",
                          flush=True)
                    rows.append({"shape": name, "N": N, "K": K, "M": M,
                                 "group_size": gs, "ms": f"error: {e}"})
                    continue
                rows.append({"shape": name, "N": N, "K": K, "M": M,
                             "group_size": gs, "ms": t, "rel_err": err,
                             "weight_bytes": wb,
                             "gbs": wb / (t / 1e3) / 1e9,
                             "x_floor": t * 1e3 / (floor * 1e3)})
                print(f"   {name:8s} g{gs:<3d} M={M:3d} | {t * 1e3:7.2f} us "
                      f"| {wb / (t / 1e3) / 1e9:6.0f} GB/s "
                      f"({wb / (t / 1e3) / 1e9 / bw * 100:5.1f}% of ceiling) "
                      f"| {t * 1e3 / (floor * 1e3):5.2f}x floor "
                      f"| rel_err {err:.5f}", flush=True)
            del w, w_ref, q_w, s
            torch.cuda.empty_cache()

    rep = {"gpu": p.name, "sms": p.multi_processor_count, "vllm": vllm.__version__,
           "atomic_add": atomic or None,
           "torch": torch.__version__, "launch_floor_ms": floor,
           "streaming_gbs": bw, "rows": rows}
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rep, f, indent=1)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
