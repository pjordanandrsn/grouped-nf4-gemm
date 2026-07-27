# Copyright (c) 2026 Cerin Amroth LLC. MIT.
"""P-A / P-B from PREREG-prefill-singleload.md against the shipped prefill GEMM.

`_gemm_nf4_grouped` has the same duplicated-byte load #43 fixed in the decode
GEMV, but here the loaded `w[BN,BK]` feeds tl.dot against `a[BM,BK]`, so every
weight element is reused BLOCK_M times. The registered expectation is that this
is a NULL result -- prefill is compute-bound and the MMA should dominate.

All three arms use the v5 LUT-LOAD codebook path (VARIANT 0) and GROUPS==1.
tl.gather cannot appear in the source at all: Triton 3.0 resolves it while
walking the AST even inside a dead constexpr branch. Using the load form on
every card also isolates the byte-load fix from the v6 codebook optimisation,
which is the cleaner comparison.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys

import torch
import triton
import triton.language as tl

BLOCKSIZE = 64
NF4 = [
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
    0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0,
]


# --------------------------------------------------------------------------
# P0 -- the shipped prefill mainloop (VARIANT 1 / GROUPS 1), copied verbatim
# --------------------------------------------------------------------------
@triton.jit
def _p0_shipped(a_ptr, b_ptr, amax_ptr, out_ptr, lut_ptr, t_row0_ptr, t_rows_ptr,
                t_group_ptr, expert_ids_ptr, K, N, stride_be, stride_bn,
                stride_ae, stride_an, BLOCK_M: tl.constexpr,
                BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    row0 = tl.load(t_row0_ptr + pid_m)
    rows = tl.load(t_rows_ptr + pid_m)
    grp = tl.load(t_group_ptr + pid_m)
    eid = tl.load(expert_ids_ptr + grp)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < rows
    n_mask = offs_n < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    offs_k = tl.arange(0, BLOCK_K)
    b_base = b_ptr + eid * stride_be + offs_n[:, None] * stride_bn
    a_base = a_ptr + (row0 + offs_m)[:, None] * K
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + offs_k
        bytes_ = tl.load(b_base + (kk[None, :] // 2), mask=n_mask[:, None],
                         other=0).to(tl.int32)
        nib = tl.where((kk[None, :] % 2) == 0, (bytes_ >> 4) & 0xF, bytes_ & 0xF)
        w = tl.load(lut_ptr + nib)
        am = tl.load(amax_ptr + eid * stride_ae + offs_n * stride_an + (k0 // 64),
                     mask=n_mask, other=0.0)
        w = w * am[:, None]
        a = tl.load(a_base + kk[None, :], mask=m_mask[:, None], other=0.0).to(tl.float32)
        acc += tl.dot(a, tl.trans(w))
    out_ptrs = out_ptr + (row0 + offs_m)[:, None] * N + offs_n[None, :]
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=m_mask[:, None] & n_mask[None, :])


# --------------------------------------------------------------------------
# P-A -- load bytes ONCE, then tl.join+reshape back to the original K layout.
#        A operand and the single tl.dot are untouched.
# --------------------------------------------------------------------------
@triton.jit
def _pa_join(a_ptr, b_ptr, amax_ptr, out_ptr, lut_ptr, t_row0_ptr, t_rows_ptr,
             t_group_ptr, expert_ids_ptr, K, N, stride_be, stride_bn,
             stride_ae, stride_an, BLOCK_M: tl.constexpr,
             BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    row0 = tl.load(t_row0_ptr + pid_m)
    rows = tl.load(t_rows_ptr + pid_m)
    grp = tl.load(t_group_ptr + pid_m)
    eid = tl.load(expert_ids_ptr + grp)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < rows
    n_mask = offs_n < N
    HALF: tl.constexpr = BLOCK_K // 2
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    offs_k = tl.arange(0, BLOCK_K)
    offs_b = tl.arange(0, HALF)
    b_base = b_ptr + eid * stride_be + offs_n[:, None] * stride_bn
    a_base = a_ptr + (row0 + offs_m)[:, None] * K
    for k0 in range(0, K, BLOCK_K):
        bytes_ = tl.load(b_base + ((k0 // 2) + offs_b)[None, :],
                         mask=n_mask[:, None], other=0).to(tl.int32)
        nib_hi = (bytes_ >> 4) & 0xF          # k = k0 + 2i
        nib_lo = bytes_ & 0xF                 # k = k0 + 2i + 1
        w_hi = tl.load(lut_ptr + nib_hi)
        w_lo = tl.load(lut_ptr + nib_lo)
        # join -> [BN, HALF, 2] -> reshape [BN, BK] reproduces w[:,2i]=hi, [:,2i+1]=lo
        w = tl.reshape(tl.join(w_hi, w_lo), [BLOCK_N, BLOCK_K])
        am = tl.load(amax_ptr + eid * stride_ae + offs_n * stride_an + (k0 // 64),
                     mask=n_mask, other=0.0)
        w = w * am[:, None]
        a = tl.load(a_base + (k0 + offs_k)[None, :], mask=m_mask[:, None],
                    other=0.0).to(tl.float32)
        acc += tl.dot(a, tl.trans(w))
    out_ptrs = out_ptr + (row0 + offs_m)[:, None] * N + offs_n[None, :]
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=m_mask[:, None] & n_mask[None, :])


# --------------------------------------------------------------------------
# P-B -- split the contraction into two k=BK/2 dots. No tl.join, but the A
#        loads become stride-2 and each MMA loses half its k-depth.
# --------------------------------------------------------------------------
@triton.jit
def _pb_split(a_ptr, b_ptr, amax_ptr, out_ptr, lut_ptr, t_row0_ptr, t_rows_ptr,
              t_group_ptr, expert_ids_ptr, K, N, stride_be, stride_bn,
              stride_ae, stride_an, BLOCK_M: tl.constexpr,
              BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    row0 = tl.load(t_row0_ptr + pid_m)
    rows = tl.load(t_rows_ptr + pid_m)
    grp = tl.load(t_group_ptr + pid_m)
    eid = tl.load(expert_ids_ptr + grp)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < rows
    n_mask = offs_n < N
    HALF: tl.constexpr = BLOCK_K // 2
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    offs_b = tl.arange(0, HALF)
    b_base = b_ptr + eid * stride_be + offs_n[:, None] * stride_bn
    a_base = a_ptr + (row0 + offs_m)[:, None] * K
    for k0 in range(0, K, BLOCK_K):
        bytes_ = tl.load(b_base + ((k0 // 2) + offs_b)[None, :],
                         mask=n_mask[:, None], other=0).to(tl.int32)
        nib_hi = (bytes_ >> 4) & 0xF
        nib_lo = bytes_ & 0xF
        w_hi = tl.load(lut_ptr + nib_hi)
        w_lo = tl.load(lut_ptr + nib_lo)
        am = tl.load(amax_ptr + eid * stride_ae + offs_n * stride_an + (k0 // 64),
                     mask=n_mask, other=0.0)
        w_hi = w_hi * am[:, None]
        w_lo = w_lo * am[:, None]
        a_hi = tl.load(a_base + (k0 + 2 * offs_b)[None, :], mask=m_mask[:, None],
                       other=0.0).to(tl.float32)
        a_lo = tl.load(a_base + (k0 + 2 * offs_b + 1)[None, :], mask=m_mask[:, None],
                       other=0.0).to(tl.float32)
        acc += tl.dot(a_hi, tl.trans(w_hi)) + tl.dot(a_lo, tl.trans(w_lo))
    out_ptrs = out_ptr + (row0 + offs_m)[:, None] * N + offs_n[None, :]
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=m_mask[:, None] & n_mask[None, :])


VARIANTS = {"p0_shipped": _p0_shipped, "pa_join": _pa_join, "pb_split": _pb_split}

SHAPES = [("olmoe_gu", 2048, 2048, 64), ("qwen_gu", 1536, 2048, 128),
          ("qwen_dn", 2048, 768, 128), ("gemma_gu", 1408, 2816, 128),
          ("gptoss_dn", 2880, 2880, 128)]


def make(E, N, K, T, groups, dev, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    B = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, generator=g).to(dev)
    am = (torch.rand(E, N, K // BLOCKSIZE, generator=g) * 0.5 + 0.05).float().to(dev)
    a = (torch.randn(T, K, generator=g) * 0.1).bfloat16().to(dev)
    eids = (torch.arange(groups) % E).to(torch.int32).to(dev)
    lut = torch.tensor(NF4, dtype=torch.float32, device=dev)
    return B, am, a, eids, lut


def tiles(sizes, BM):
    r0, rr, gg, off = [], [], [], 0
    for gi, s in enumerate(sizes):
        for t in range(0, s, BM):
            r0.append(off + t); rr.append(min(BM, s - t)); gg.append(gi)
        off += s
    return r0, rr, gg


def run_one(fn, B, am, a, eids, lut, N, K, sizes, BM, BN, BK, warps, iters):
    dev = a.device
    r0, rr, gg = tiles(sizes, BM)
    t_row0 = torch.tensor(r0, dtype=torch.int32, device=dev)
    t_rows = torch.tensor(rr, dtype=torch.int32, device=dev)
    t_grp = torch.tensor(gg, dtype=torch.int32, device=dev)
    out = torch.empty(a.shape[0], N, dtype=torch.bfloat16, device=dev)
    grid = (len(r0), triton.cdiv(N, BN))

    def go():
        fn[grid](a, B, am, out, lut, t_row0, t_rows, t_grp, eids, K, N,
                 B.stride(0), B.stride(1), am.stride(0), am.stride(1),
                 BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,  num_warps=warps, num_stages=3)

    for _ in range(6):
        go()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); go(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens-per-group", type=int, default=32)
    ap.add_argument("--groups", type=int, default=8)
    ap.add_argument("--bm", type=int, default=64)
    ap.add_argument("--bn", type=int, default=64)
    ap.add_argument("--bk", type=int, default=64)
    ap.add_argument("--warps", type=int, default=4)
    ap.add_argument("--iters", type=int, default=7)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    dev = "cuda"
    print(f"# {torch.cuda.get_device_name(0)}  torch {torch.__version__}  triton {triton.__version__}")
    print(f"# PREFILL  tokens/group={args.tokens_per_group} groups={args.groups} "
          f"BM={args.bm} BN={args.bn} BK={args.bk} warps={args.warps}")
    sizes = [args.tokens_per_group] * args.groups
    T = sum(sizes)
    payload = {}
    for name, K, N, E in SHAPES:
        B, am, a, eids, lut = make(E, N, K, T, args.groups, dev)
        row = {}
        base_out = None
        for vn, fn in VARIANTS.items():
            try:
                ms, out = run_one(fn, B, am, a, eids, lut, N, K, sizes,
                                  args.bm, args.bn, args.bk, args.warps, args.iters)
            except Exception as exc:
                row[vn] = {"error": str(exc)[:120]}
                print(f"  {name:11s} {vn:11s} FAILED: {str(exc)[:90]}")
                continue
            if vn == "p0_shipped":
                base_out = out.float().clone()
                agree = 0.0
            else:
                agree = ((out.float() - base_out).abs().max().item()
                         / base_out.abs().max().item())
            row[vn] = {"ms": ms, "agree": agree}
        if "ms" in row.get("p0_shipped", {}):
            b = row["p0_shipped"]["ms"]
            parts = []
            for vn in VARIANTS:
                r = row.get(vn, {})
                if "ms" not in r:
                    parts.append(f"{vn}=ERR"); continue
                g = "OK" if r["agree"] <= 8e-03 else "BAD"
                parts.append(f"{vn} {b / r['ms']:5.3f}x[{g}]")
            print(f"  {name:11s} base {b:7.4f} ms   " + "  ".join(parts))
        payload[name] = row
    if args.out:
        json.dump(payload, open(args.out, "w"), indent=2)
        print(f"# wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
