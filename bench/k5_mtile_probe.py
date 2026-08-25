# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""PREREG-k5-mtile-probe: time the PRODUCT M-tile kernel at exactly the
M=1 decode shapes (sizes=[1]*8, one 1-row tile per group via the
kernel's own m_mask) against the production GEMV at the K1 winners.
The launch section is replicated here; the kernel is the product's own
-- nothing to drift. Timing-only: outputs are discarded and no
correctness claim is made (a K5-B routing cycle would carry the
fidelity gates)."""

import argparse
import itertools
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "kernel"))
import nf4_grouped  # noqa: E402
from nf4_grouped import (BLOCKSIZE, _gemm_nf4_grouped, _lut,  # noqa: E402
                         build_group_tiles, triton)

CELLS = {"gate_up": (1536, 2048, 8, (64, 2), 16),
         "down": (2048, 768, 8, (32, 2), 1)}
BNS = (64, 128, 256)
WARPS = (4, 8)
STAGES = (2, 3)
GROUPS = (1, 2)  # BLOCK_K = 64 * groups; both product-supported
BLOCK_M = 16


def _mk(N, K, T, E=8, device="cuda"):
    g = torch.Generator(device="cpu").manual_seed(0)
    packed = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8,
                           generator=g).to(device)
    absmax = (torch.rand(E, N, K // 64, generator=g) + 0.5).to(device)
    a = torch.randn(T, K, generator=g).to(device=device,
                                          dtype=torch.bfloat16)
    eids = torch.arange(E, dtype=torch.int32, device=device)[:T]
    return a, packed, absmax, eids


def _time(fn, iters=200, warmup=50, chunks=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    per = max(1, iters // chunks)
    spans = []
    for _ in range(chunks):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(per):
            fn()
        e1.record()
        e1.synchronize()
        spans.append(e0.elapsed_time(e1) / per)
    spans.sort()
    return spans[len(spans) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="k5_probe.json")
    args = ap.parse_args()
    dev = "cuda"
    rep = {"gpu": torch.cuda.get_device_name(0), "cells": {}}
    gemv_sum = 0.0
    mtile_sum = 0.0
    for name, (N, K, T, cfg, sk) in CELLS.items():
        a, p, ax, eids = _mk(N, K, T)

        def gemv():
            return nf4_grouped.gemm_4bit_grouped(
                a, p, ax, [1] * T, eids, decode_config=cfg, split_k=sk)

        g_start = _time(gemv) * 1000.0
        t_row0, t_rows, t_group = build_group_tiles([1] * T, BLOCK_M, dev)
        out = torch.empty(T, N, dtype=torch.bfloat16, device=dev)
        lut = _lut(torch.device(dev))
        rows = []
        for bn, w, st, gr in itertools.product(BNS, WARPS, STAGES, GROUPS):
            if K % (BLOCKSIZE * gr):
                continue
            grid = (t_row0.numel(), triton.cdiv(N, bn))

            def mtile():
                _gemm_nf4_grouped[grid](
                    a, p, ax, out, lut, t_row0, t_rows, t_group, eids,
                    K, N, p.stride(0), p.stride(1), ax.stride(0),
                    ax.stride(1), BLOCK_M=BLOCK_M, BLOCK_N=bn,
                    BLOCK_K=BLOCKSIZE * gr, GROUPS=gr, VARIANT=1,
                    num_warps=w, num_stages=st)
            try:
                ms = _time(mtile) * 1000.0
            except Exception as e:                    # noqa: BLE001
                rows.append({"bn": bn, "warps": w, "stages": st,
                             "groups": gr, "error": str(e)[:80]})
                continue
            rows.append({"bn": bn, "warps": w, "stages": st,
                         "groups": gr, "us": ms})
        g_end = _time(gemv) * 1000.0
        drift = abs(g_end - g_start) / min(g_start, g_end) * 100
        ok = sorted((r for r in rows if "us" in r), key=lambda r: r["us"])
        rep["cells"][name] = {
            "gemv_us": min(g_start, g_end), "noise_drift_pct": drift,
            "noise_gate_pass": drift <= 5.0,
            "mtile_best": ok[0] if ok else None, "table": rows,
        }
        gemv_sum += min(g_start, g_end)
        if ok:
            mtile_sum += ok[0]["us"]
    rep["summary"] = {
        "gemv_sum_us": gemv_sum,
        "mtile_sum_us": mtile_sum if mtile_sum else None,
        "ratio_mtile_over_gemv": (mtile_sum / gemv_sum
                                  if mtile_sum else None),
        "noise_gate_pass": all(c["noise_gate_pass"]
                               for c in rep["cells"].values()),
    }
    Path(args.out).write_text(json.dumps(rep, indent=1))
    s = rep["summary"]
    print(f"K5PROBE gemv={gemv_sum:.1f}us mtile={s['mtile_sum_us']:.1f}us "
          f"ratio={s['ratio_mtile_over_gemv']:.3f} "
          f"noise={'PASS' if s['noise_gate_pass'] else 'FAIL'}")
    for n, c in rep["cells"].items():
        b = c["mtile_best"]
        print(f"  {n}: gemv={c['gemv_us']:.1f} mtile_best={b['us']:.1f} "
              f"(bn={b['bn']} w={b['warps']} s={b['stages']} "
              f"g={b['groups']})")


if __name__ == "__main__":
    main()
