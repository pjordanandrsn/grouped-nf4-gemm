# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""PREREG-k3-attribution instruments.

Peel battery: REPLICA kernels of the decode GEMV mainloop (the product
kernel is never touched) with one component removed per variant; every
peel keeps a data dependency from its loads to the store so DCE cannot
fake a result. Attribution by subtraction with the residual reported.

--one-shot <cell> runs a single launch of the full replica (for `ncu`
to wrap). The split-K contribution is measured on the PRODUCT kernel
(sk16 vs sk1 at gate_up) — the peel structure itself runs sk=1."""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "kernel"))
import nf4_grouped  # noqa: E402
from _triton_shim import tl, triton  # noqa: E402

CELLS = {"gate_up": (1536, 2048, 8, (64, 2)),
         "down": (2048, 768, 8, (32, 2))}
PEELS = ("full", "no_lut", "no_absmax", "no_act", "loads_only",
         "loads_only_wide")


@triton.jit
def _peel_gemv(a_ptr, b_ptr, bw_ptr, amax_ptr, out_ptr, lut_ptr, eids_ptr, K, N,
               stride_be, stride_bn, stride_ae, stride_an,
               BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
               PEEL: tl.constexpr):
    g = tl.program_id(0)
    pid_n = tl.program_id(1)
    eid = tl.load(eids_ptr + g).to(tl.int64)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    offs_k = tl.arange(0, BLOCK_K)
    b_base = b_ptr + eid * stride_be + offs_n[:, None] * stride_bn
    bw_ptr_base = bw_ptr + eid * (stride_be // 4) + offs_n[:, None] * (stride_bn // 4)
    a_base = a_ptr + g * K
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + offs_k
        bytes_ = tl.load(b_base + (kk[None, :] // 2),
                         mask=n_mask[:, None], other=0).to(tl.int32)
        if PEEL == 4:                       # loads-only streaming floor
            acc += tl.sum(bytes_.to(tl.float32), axis=1)
        elif PEEL == 5:                     # K4: WIDE loads-only floor
            offs_kw = tl.arange(0, BLOCK_K // 8)
            words = tl.load(bw_ptr_base + (k0 // 8) + offs_kw[None, :],
                            mask=n_mask[:, None], other=0)
            acc += tl.sum(words.to(tl.float32), axis=1)
        else:
            nib = tl.where((kk[None, :] % 2) == 0,
                           (bytes_ >> 4) & 0xF, bytes_ & 0xF)
            if PEEL == 1:                   # no LUT gather
                w = nib.to(tl.float32)
            else:
                w = tl.load(lut_ptr + nib)
            if PEEL == 2:                   # no absmax
                am = 1.0
            else:
                am = tl.load(amax_ptr + eid * stride_ae
                             + offs_n * stride_an + (k0 // BLOCK_K),
                             mask=n_mask, other=0.0)
            if PEEL == 3:                   # no activation load
                acc += tl.sum(w, axis=1) * am
            else:
                a = tl.load(a_base + kk).to(tl.float32)
                acc += tl.sum(w * a[None, :], axis=1) * am
    tl.store(out_ptr + g * N + offs_n, acc.to(tl.bfloat16), mask=n_mask)


def _mk(N, K, T, E=8, device="cuda"):
    g = torch.Generator(device="cpu").manual_seed(0)
    packed = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8,
                           generator=g).to(device)
    absmax = (torch.rand(E, N, K // 64, generator=g) + 0.5).to(device)
    a = torch.randn(T, K, generator=g).to(device=device,
                                          dtype=torch.bfloat16)
    eids = torch.arange(E, dtype=torch.int32, device=device)[:T]
    return a, packed, absmax, eids


def _lut(device):
    return torch.tensor(nf4_grouped.NF4_LUT, dtype=torch.float32,
                        device=device)


def _peel_call(name, peel, device="cuda"):
    N, K, T, (bn, warps) = CELLS[name]
    a, p, ax, eids = _mk(N, K, T)
    out = torch.empty(T, N, dtype=torch.bfloat16, device=device)
    lut = _lut(device)
    grid = (T, -(-N // bn))

    def call():
        _peel_gemv[grid](a, p, p.view(torch.int32), ax, out, lut, eids,
                         K, N,
                         p.stride(0), p.stride(1), ax.stride(0),
                         ax.stride(1), BLOCK_N=bn, BLOCK_K=64,
                         PEEL=peel, num_warps=warps, num_stages=3)
    return call


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
    ap.add_argument("--one-shot", choices=list(CELLS), default=None)
    ap.add_argument("--out", default="k3_attr.json")
    args = ap.parse_args()
    if args.one_shot:
        call = _peel_call(args.one_shot, 0)
        call()
        torch.cuda.synchronize()
        print(f"ONE-SHOT {args.one_shot} done")
        return

    rep = {"gpu": torch.cuda.get_device_name(0), "cells": {}}
    for name, (N, K, T, (bn, warps)) in CELLS.items():
        a, p, ax, eids = _mk(N, K, T)

        def product(sk=1):
            def call():
                return nf4_grouped.gemm_4bit_grouped(
                    a, p, ax, [1] * T, eids, decode_config=(bn, warps),
                    split_k=sk)
            return call

        cell = {"peel_us": {}}
        full_start = _time(_peel_call(name, 0)) * 1000.0
        for i, peel in enumerate(PEELS):
            cell["peel_us"][peel] = _time(_peel_call(name, i)) * 1000.0
        full_end = _time(_peel_call(name, 0)) * 1000.0
        cell["noise_drift_pct"] = (abs(full_end - full_start)
                                   / min(full_start, full_end) * 100)
        cell["product_sk1_us"] = _time(product(1)) * 1000.0
        cell["replica_vs_product_pct"] = (
            abs(cell["peel_us"]["full"] - cell["product_sk1_us"])
            / cell["product_sk1_us"] * 100)
        if name == "gate_up":
            cell["product_sk16_us"] = _time(product(16)) * 1000.0
        f = cell["peel_us"]["full"]
        cell["components_us"] = {
            "lut_gather": f - cell["peel_us"]["no_lut"],
            "absmax": f - cell["peel_us"]["no_absmax"],
            "activation": f - cell["peel_us"]["no_act"],
            "loads_floor": cell["peel_us"]["loads_only"],
        }
        acct = sum(max(v, 0.0) for v in cell["components_us"].values())
        cell["residual_us"] = f - acct
        rep["cells"][name] = cell
    Path(args.out).write_text(json.dumps(rep, indent=1))
    for name, c in rep["cells"].items():
        comp = {k: round(v, 1) for k, v in c["components_us"].items()}
        print(f"{name}: full={c['peel_us']['full']:.1f}us {comp} "
              f"residual={c['residual_us']:.1f}us "
              f"replica_vs_product={c['replica_vs_product_pct']:.1f}%")


if __name__ == "__main__":
    main()
