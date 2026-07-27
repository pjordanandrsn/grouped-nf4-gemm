# Copyright (c) 2026 Cerin Amroth LLC. MIT.
"""Price what a GEMV must actually do. See PREREG-reachable-target.md.

#39's 487 GB/s "decode floor" already includes nibble unpack, the LUT gather and
the absmax scale — its ladder was read 685.5 -> decode 487.0 -> shipped 99.2. It
omits the activation multiply, the K-reduction and the store, none of which a
GEMV can skip. Every "headroom" number in the project measures distance to that
omission-laden figure, which the project's own prereg calls unreachable.

Five kernels over identical bytes, each adding exactly one term:

    R1 read    load packed bytes
    R2 decode  + unpack + LUT + scale          (should reproduce ~#39's 487)
    R3 mul     + multiply by the activation
    R4 reduce  + the per-iteration K-reduction (the shipped kernel's pattern)
    R5 gemv    + store [BLOCK_N] instead of a scalar  == a complete GEMV

**Dead-code elimination is the trap.** A rung whose result is unused compiles to
nothing and looks infinitely fast. Every rung therefore ends in a store. R1-R3
store one scalar per program (a single final `tl.sum` sink); R4 uses the
per-iteration reduction the real kernel uses; R5 stores the full vector.

Because R1-R3 still pay ONE final reduction for their sink, they are slightly
pessimistic — which makes R5 a slightly *conservative* (easier) target. That
biases the conclusion against "there is no headroom", i.e. against the outcome
this experiment is most likely to be accused of wanting.
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


def _prologue():
    """Shared index setup, kept identical across rungs so only the term differs."""


@triton.jit
def _r1_read(a_ptr, b_ptr, amax_ptr, out_ptr, lut_ptr, eids_ptr, K, N,
             sbe, sbn, sae, san, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    g, pid_n = tl.program_id(0), tl.program_id(1)
    eid = tl.load(eids_ptr + g)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    offs_b = tl.arange(0, 32)
    b_base = b_ptr + eid * sbe + offs_n[:, None] * sbn
    acc = tl.zeros((BLOCK_N, 32), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        by = tl.load(b_base + ((k0 // 2) + offs_b)[None, :], mask=n_mask[:, None],
                     other=0).to(tl.int32)
        acc += by.to(tl.float32)                       # touch it so it cannot vanish
    tl.store(out_ptr + g * N + pid_n, tl.sum(tl.sum(acc, axis=1), axis=0))


@triton.jit
def _r2_decode(a_ptr, b_ptr, amax_ptr, out_ptr, lut_ptr, eids_ptr, K, N,
               sbe, sbn, sae, san, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    g, pid_n = tl.program_id(0), tl.program_id(1)
    eid = tl.load(eids_ptr + g)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    offs_b = tl.arange(0, 32)
    b_base = b_ptr + eid * sbe + offs_n[:, None] * sbn
    am_base = amax_ptr + eid * sae + offs_n * san
    acc = tl.zeros((BLOCK_N, 32), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        by = tl.load(b_base + ((k0 // 2) + offs_b)[None, :], mask=n_mask[:, None],
                     other=0).to(tl.int32)
        w_hi = tl.load(lut_ptr + ((by >> 4) & 0xF))
        w_lo = tl.load(lut_ptr + (by & 0xF))
        am = tl.load(am_base + (k0 // 64), mask=n_mask, other=0.0)
        acc += (w_hi + w_lo) * am[:, None]             # unpack + LUT + scale
    tl.store(out_ptr + g * N + pid_n, tl.sum(tl.sum(acc, axis=1), axis=0))


@triton.jit
def _r3_mul(a_ptr, b_ptr, amax_ptr, out_ptr, lut_ptr, eids_ptr, K, N,
            sbe, sbn, sae, san, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    g, pid_n = tl.program_id(0), tl.program_id(1)
    eid = tl.load(eids_ptr + g)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    offs_b = tl.arange(0, 32)
    b_base = b_ptr + eid * sbe + offs_n[:, None] * sbn
    a_base = a_ptr + g * K
    am_base = amax_ptr + eid * sae + offs_n * san
    acc = tl.zeros((BLOCK_N, 32), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        by = tl.load(b_base + ((k0 // 2) + offs_b)[None, :], mask=n_mask[:, None],
                     other=0).to(tl.int32)
        w_hi = tl.load(lut_ptr + ((by >> 4) & 0xF))
        w_lo = tl.load(lut_ptr + (by & 0xF))
        a_hi = tl.load(a_base + k0 + 2 * offs_b).to(tl.float32)
        a_lo = tl.load(a_base + k0 + 2 * offs_b + 1).to(tl.float32)
        am = tl.load(am_base + (k0 // 64), mask=n_mask, other=0.0)
        acc += (w_hi * a_hi[None, :] + w_lo * a_lo[None, :]) * am[:, None]
    tl.store(out_ptr + g * N + pid_n, tl.sum(tl.sum(acc, axis=1), axis=0))


@triton.jit
def _r4_reduce(a_ptr, b_ptr, amax_ptr, out_ptr, lut_ptr, eids_ptr, K, N,
               sbe, sbn, sae, san, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """The shipped kernel's per-iteration K-reduction, but a scalar sink."""
    g, pid_n = tl.program_id(0), tl.program_id(1)
    eid = tl.load(eids_ptr + g)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    offs_b = tl.arange(0, 32)
    b_base = b_ptr + eid * sbe + offs_n[:, None] * sbn
    a_base = a_ptr + g * K
    am_base = amax_ptr + eid * sae + offs_n * san
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        by = tl.load(b_base + ((k0 // 2) + offs_b)[None, :], mask=n_mask[:, None],
                     other=0).to(tl.int32)
        w_hi = tl.load(lut_ptr + ((by >> 4) & 0xF))
        w_lo = tl.load(lut_ptr + (by & 0xF))
        a_hi = tl.load(a_base + k0 + 2 * offs_b).to(tl.float32)
        a_lo = tl.load(a_base + k0 + 2 * offs_b + 1).to(tl.float32)
        am = tl.load(am_base + (k0 // 64), mask=n_mask, other=0.0)
        acc += (tl.sum(w_hi * a_hi[None, :], axis=1)
                + tl.sum(w_lo * a_lo[None, :], axis=1)) * am
    tl.store(out_ptr + g * N + pid_n, tl.sum(acc, axis=0))     # scalar sink


@triton.jit
def _r5_gemv(a_ptr, b_ptr, amax_ptr, out_ptr, lut_ptr, eids_ptr, K, N,
             sbe, sbn, sae, san, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """R4 + the real [BLOCK_N] store == a complete GEMV. THE REACHABLE TARGET."""
    g, pid_n = tl.program_id(0), tl.program_id(1)
    eid = tl.load(eids_ptr + g)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    offs_b = tl.arange(0, 32)
    b_base = b_ptr + eid * sbe + offs_n[:, None] * sbn
    a_base = a_ptr + g * K
    am_base = amax_ptr + eid * sae + offs_n * san
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        by = tl.load(b_base + ((k0 // 2) + offs_b)[None, :], mask=n_mask[:, None],
                     other=0).to(tl.int32)
        w_hi = tl.load(lut_ptr + ((by >> 4) & 0xF))
        w_lo = tl.load(lut_ptr + (by & 0xF))
        a_hi = tl.load(a_base + k0 + 2 * offs_b).to(tl.float32)
        a_lo = tl.load(a_base + k0 + 2 * offs_b + 1).to(tl.float32)
        am = tl.load(am_base + (k0 // 64), mask=n_mask, other=0.0)
        acc += (tl.sum(w_hi * a_hi[None, :], axis=1)
                + tl.sum(w_lo * a_lo[None, :], axis=1)) * am
    tl.store(out_ptr + g * N + offs_n, acc.to(tl.bfloat16), mask=n_mask)


RUNGS = [("R1_read", _r1_read), ("R2_decode", _r2_decode), ("R3_mul", _r3_mul),
         ("R4_reduce", _r4_reduce), ("R5_gemv", _r5_gemv)]


def make(E, N, K, T, dev, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    B = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, generator=g).to(dev)
    am = (torch.rand(E, N, K // BLOCKSIZE, generator=g) * .5 + .05).float().to(dev)
    a = (torch.randn(T, K, generator=g) * .1).bfloat16().to(dev)
    eids = (torch.arange(T) % E).to(torch.int32).to(dev)
    lut = torch.tensor(NF4, dtype=torch.float32, device=dev)
    return B, am, a, eids, lut


def time_fn(fn, B, am, a, eids, lut, N, K, T, bn, bk, warps, iters, bf16_out):
    dev = a.device
    out = (torch.empty(T, N, dtype=torch.bfloat16, device=dev) if bf16_out
           else torch.empty(T, N, dtype=torch.float32, device=dev))
    grid = (T, triton.cdiv(N, bn))

    def go():
        fn[grid](a, B, am, out, lut, eids, K, N, B.stride(0), B.stride(1),
                 am.stride(0), am.stride(1), BLOCK_N=bn, BLOCK_K=bk,
                 num_warps=warps, num_stages=3)
    for _ in range(8):
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
    ap.add_argument("--E", type=int, default=8)
    ap.add_argument("--N", type=int, default=3072)
    ap.add_argument("--K", type=int, default=4096)
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--bn", type=int, default=64)
    ap.add_argument("--warps", type=int, default=2)
    ap.add_argument("--iters", type=int, default=7)
    ap.add_argument("--pairs", type=int, default=3)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    dev = "cuda"
    p = torch.cuda.get_device_properties(0)
    print(f"# {p.name} sm_{p.major}{p.minor} SMs={p.multi_processor_count} "
          f"triton {triton.__version__}")
    B, am, a, eids, lut = make(args.E, args.N, args.K, args.T, dev)
    mb = B.numel() / 1e6
    print(f"# E={args.E} N={args.N} K={args.K} T={args.T}  packed={mb:.2f} MB  "
          f"BLOCK_N={args.bn} warps={args.warps}")

    # paired against R1 so drift cancels; a self-pair must read ~1.000x
    def paired(fn, bf16_out):
        rs, ms = [], None
        for _ in range(args.pairs):
            b, _ = time_fn(_r1_read, B, am, a, eids, lut, args.N, args.K, args.T,
                           args.bn, BLOCKSIZE, args.warps, 3, False)
            ms, o = time_fn(fn, B, am, a, eids, lut, args.N, args.K, args.T,
                            args.bn, BLOCKSIZE, args.warps, 3, bf16_out)
            rs.append(ms / b)                    # >1 means slower than R1
        return statistics.median(rs), ms, o

    rows, prev = [], None
    for name, fn in RUNGS:
        rel, ms, out = paired(fn, name == "R5_gemv")
        step = (ms / prev) if prev else 1.0
        rows.append({"rung": name, "ms": ms, "vs_R1": rel, "step_vs_prev": step})
        print(f"  {name:10s} {ms:8.4f} ms   {rel:6.3f}x R1   step {step:6.3f}x")
        prev = ms

    print(f"\n# self-pair sanity (R1 vs R1): {rows[0]['vs_R1']:.3f}x "
          f"(must be ~1.000; otherwise VOID)")

    # the shipped kernel, same shape
    sys.path.insert(0, "kernel")
    from nf4_grouped import _gemv_nf4_grouped
    sh, _ = time_fn(_gemv_nf4_grouped, B, am, a, eids, lut, args.N, args.K,
                    args.T, args.bn, BLOCKSIZE, args.warps, args.iters, True)
    r5 = rows[-1]["ms"]
    print(f"\n  shipped    {sh:8.4f} ms")
    print(f"  R5 target  {r5:8.4f} ms")
    print(f"\n# REACHABLE HEADROOM  shipped/R5 = {sh / r5:.3f}x")
    print(f"#   (<1.3x closes the structural line; >2.0x justifies a CUDA GEMV)")

    if args.out:
        json.dump({"device": p.name, "rows": rows, "shipped_ms": sh,
                   "r5_ms": r5, "reachable_headroom": sh / r5},
                  open(args.out, "w"), indent=2)
        print(f"# wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
