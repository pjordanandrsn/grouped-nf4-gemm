# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""PREREG-k6-bespoke-gemv: the M-padded tl.dot GEMV with K4's wide
loads, benched against the production K1 winners and the wide-scalar
kernel (V0) at the census cells.

The dot-pad kernel's dequant mirrors `_gemv_nf4_grouped`'s WIDE_LOADS
branch VERBATIM (same words, same shift table, same gather LUT); only
the reduce differs -- `tl.dot` with x in M-row 0 and 15 zero rows,
because a GEMV has no free M dimension (registered in the prereg). So
V1 - V0 isolates the reduce structure, which is the question K4 left.

Bench-local kernel: no product change in this cycle."""

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "kernel"))
import nf4_grouped  # noqa: E402
from nf4_grouped import BLOCKSIZE, _lut, tl, triton  # noqa: E402

CELLS = {"gate_up": (1536, 2048, 8, (64, 2), 16),
         "down": (2048, 768, 8, (32, 2), 1)}
BNS = (16, 32, 64, 128)
WARPS = (2, 4, 8)
STAGES = (2, 3, 4)


@triton.jit
def _k6_dot_pad(a_ptr, b_ptr, amax_ptr, out_ptr, lut_ptr, eids_ptr,
                K, N, stride_be, stride_bn, stride_ae, stride_an,
                BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    g = tl.program_id(0)
    pid_n = tl.program_id(1)
    eid = tl.load(eids_ptr + g).to(tl.int64)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    offs_k = tl.arange(0, BLOCK_K)
    b_base = b_ptr + eid * stride_be + offs_n[:, None] * stride_bn
    a_base = a_ptr + g * K
    offs_kw = tl.arange(0, BLOCK_K // 8)
    sh = tl.arange(0, 8)
    shift = (sh // 2) * 8 + tl.where(sh % 2 == 0, 4, 0)
    offs_m = tl.arange(0, 16)
    acc = tl.zeros((16, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + offs_k
        words = tl.load(b_base + (k0 // 8) + offs_kw[None, :],
                        mask=n_mask[:, None], other=0)
        nib = (words[:, :, None] >> shift[None, None, :]) & 0xF
        nib = tl.reshape(nib, (BLOCK_N, BLOCK_K))
        w = tl.load(lut_ptr + nib)
        am = tl.load(amax_ptr + eid * stride_ae + offs_n * stride_an
                     + (k0 // BLOCK_K), mask=n_mask, other=0.0)
        # absmax folds into W before the dot; the scalar kernel scales
        # after its sum -- same per-group math, different fp order (the
        # K1-class gate, not a bitwise one, is registered for this)
        wt = tl.trans((w * am[:, None]).to(tl.bfloat16))
        a = tl.load(a_base + kk).to(tl.float32)
        av = tl.where(offs_m[:, None] == 0, a[None, :], 0.0)
        acc = tl.dot(av.to(tl.bfloat16), wt, acc)
    y = tl.sum(tl.where(offs_m[:, None] == 0, acc, 0.0), axis=0)
    tl.store(out_ptr + g * N + offs_n, y.to(tl.bfloat16), mask=n_mask)


def dot_pad(a, packed, absmax, eids, N, K, bn, warps, stages):
    T = a.shape[0]
    out = torch.empty(T, N, dtype=torch.bfloat16, device=a.device)
    _k6_dot_pad[(T, triton.cdiv(N, bn))](
        a, packed, absmax, out, _lut(a.device), eids, K, N,
        packed.stride(0), packed.stride(1), absmax.stride(0),
        absmax.stride(1), BLOCK_N=bn, BLOCK_K=BLOCKSIZE,
        num_warps=warps, num_stages=stages)
    return out


def _mk(N, K, T, E=8, device="cuda", seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
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


def gate(N, K, cfg, sk, bn, warps, stages, rows=4096):
    """K1-class gate: max|delta| <= 1e-2 on bf16 outputs AND exact
    per-row argmax agreement over `rows` random rows (prereg)."""
    maxd, agree, done = 0.0, 0, 0
    batch = 0
    while done < rows:
        batch += 1
        a, p, ax, eids = _mk(N, K, 8, seed=batch)
        ref = nf4_grouped.gemm_4bit_grouped(a, p, ax, [1] * 8, eids,
                                            decode_config=cfg, split_k=sk)
        got = dot_pad(a, p, ax, eids, N, K, bn, warps, stages)
        maxd = max(maxd, (ref.float() - got.float()).abs().max().item())
        agree += int((ref.argmax(-1) == got.argmax(-1)).sum().item())
        done += 8
    return {"rows": done, "max_abs_delta": maxd,
            "argmax_agree": agree, "argmax_total": done,
            "pass": maxd <= 1e-2 and agree == done}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="k6_bench.json")
    ap.add_argument("--gate-rows", type=int, default=4096)
    args = ap.parse_args()
    rep = {"gpu": torch.cuda.get_device_name(0),
           "torch": torch.__version__, "cells": {}}
    base_sum = v0_sum = best_sum = 0.0
    for name, (N, K, T, cfg, sk) in CELLS.items():
        a, p, ax, eids = _mk(N, K, T)

        def baseline():
            return nf4_grouped.gemm_4bit_grouped(
                a, p, ax, [1] * T, eids, decode_config=cfg, split_k=sk)

        os.environ["GNF4_GEMV_WIDE_LOADS"] = "0"
        b_start = _time(baseline) * 1000.0
        os.environ["GNF4_GEMV_WIDE_LOADS"] = "1"
        v0 = _time(baseline) * 1000.0
        os.environ["GNF4_GEMV_WIDE_LOADS"] = "0"
        rows = []
        for bn, w, st in itertools.product(BNS, WARPS, STAGES):
            try:
                ms = _time(lambda: dot_pad(a, p, ax, eids, N, K,
                                           bn, w, st)) * 1000.0
            except Exception as e:                    # noqa: BLE001
                rows.append({"bn": bn, "warps": w, "stages": st,
                             "error": str(e)[:100]})
                continue
            rows.append({"bn": bn, "warps": w, "stages": st, "us": ms})
        b_end = _time(baseline) * 1000.0
        drift = abs(b_end - b_start) / min(b_start, b_end) * 100
        ok = sorted((r for r in rows if "us" in r), key=lambda r: r["us"])
        best = ok[0] if ok else None
        g = None
        if best:
            g = gate(N, K, cfg, sk, best["bn"], best["warps"],
                     best["stages"], rows=args.gate_rows)
        rep["cells"][name] = {
            "baseline_us": min(b_start, b_end),
            "wide_scalar_us": v0,
            "noise_drift_pct": drift, "noise_gate_pass": drift <= 5.0,
            "dot_pad_best": best, "gate": g, "table": rows,
        }
        base_sum += min(b_start, b_end)
        v0_sum += v0
        if best and g and g["pass"]:
            best_sum += best["us"]
        elif best:
            best_sum += float("inf")
    rep["summary"] = {
        "baseline_pair_us": base_sum, "wide_scalar_pair_us": v0_sum,
        "dot_pad_pair_us": (best_sum if best_sum != float("inf")
                            else None),
        "noise_gate_pass": all(c["noise_gate_pass"]
                               for c in rep["cells"].values()),
    }
    Path(args.out).write_text(json.dumps(rep, indent=1))
    s = rep["summary"]
    dp = s["dot_pad_pair_us"]
    print(f"K6BENCH baseline={base_sum:.1f}us wide_scalar={v0_sum:.1f}us "
          f"dot_pad={f'{dp:.1f}us' if dp else 'GATE-FAILED'} "
          f"noise={'PASS' if s['noise_gate_pass'] else 'FAIL'}")
    for n, c in rep["cells"].items():
        b, g = c["dot_pad_best"], c["gate"]
        bs = (f"{b['us']:.1f}us (bn={b['bn']} w={b['warps']} "
              f"s={b['stages']})") if b else "none"
        gs = (f"gate max|d|={g['max_abs_delta']:.2e} "
              f"argmax {g['argmax_agree']}/{g['argmax_total']} "
              f"{'PASS' if g['pass'] else 'FAIL'}") if g else "gate n/a"
        print(f"  {n}: base={c['baseline_us']:.1f} wide={c['wide_scalar_us']:.1f} "
              f"dot={bs} | {gs}")


if __name__ == "__main__":
    main()
