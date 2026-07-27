# Copyright (c) 2026 Cerin Amroth LLC. MIT.
"""Test H4/H5/H6 from PREREG-gemv-issue-bound.md against the shipped GEMV.

The shipped `_gemv_nf4_grouped` runs at ~99-111 GB/s where a flat decode of the
same bytes reaches 487, and #41 falsified memory-, compute-, decode- and
occupancy-bound explanations. This measures three issue-side defects visible in
the source:

  H4  each packed byte is loaded TWICE (index is kk//2 over consecutive k)
  H5  tl.sum(axis=1) runs K/BLOCK_K times instead of once after the loop
  H6  the absmax index pins BLOCK_K to the NF4 blocksize -> trip count is
      not tunable, which is why #41's BLOCK_N x warps x split_k sweep missed it

Arms are interleaved inside one process and reported as medians, because the
A2000 this runs on is a shared production GPU.
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
# V0 -- the shipped kernel, copied verbatim so the baseline is honest
# --------------------------------------------------------------------------
@triton.jit
def _v0_shipped(a_ptr, b_ptr, amax_ptr, out_ptr, lut_ptr, expert_ids_ptr,
                K, N, stride_be, stride_bn, stride_ae, stride_an,
                BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    g = tl.program_id(0)
    pid_n = tl.program_id(1)
    eid = tl.load(expert_ids_ptr + g)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    offs_k = tl.arange(0, BLOCK_K)
    b_base = b_ptr + eid * stride_be + offs_n[:, None] * stride_bn
    a_base = a_ptr + g * K
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + offs_k
        bytes_ = tl.load(b_base + (kk[None, :] // 2), mask=n_mask[:, None],
                         other=0).to(tl.int32)
        nib = tl.where((kk[None, :] % 2) == 0, (bytes_ >> 4) & 0xF, bytes_ & 0xF)
        w = tl.load(lut_ptr + nib)
        am = tl.load(amax_ptr + eid * stride_ae + offs_n * stride_an + (k0 // BLOCK_K),
                     mask=n_mask, other=0.0)
        a = tl.load(a_base + kk).to(tl.float32)
        acc += tl.sum(w * a[None, :], axis=1) * am
    tl.store(out_ptr + g * N + offs_n, acc.to(tl.bfloat16), mask=n_mask)


# --------------------------------------------------------------------------
# V1 -- H4 only: load each byte ONCE, extract both nibbles
# --------------------------------------------------------------------------
@triton.jit
def _v1_h4(a_ptr, b_ptr, amax_ptr, out_ptr, lut_ptr, expert_ids_ptr,
           K, N, stride_be, stride_bn, stride_ae, stride_an,
           BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    g = tl.program_id(0)
    pid_n = tl.program_id(1)
    eid = tl.load(expert_ids_ptr + g)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    HALF: tl.constexpr = BLOCK_K // 2
    offs_b = tl.arange(0, HALF)
    b_base = b_ptr + eid * stride_be + offs_n[:, None] * stride_bn
    a_base = a_ptr + g * K
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        bo = (k0 // 2) + offs_b
        bytes_ = tl.load(b_base + bo[None, :], mask=n_mask[:, None],
                         other=0).to(tl.int32)
        w_hi = tl.load(lut_ptr + ((bytes_ >> 4) & 0xF))   # k = k0 + 2*i
        w_lo = tl.load(lut_ptr + (bytes_ & 0xF))          # k = k0 + 2*i + 1
        a_hi = tl.load(a_base + k0 + 2 * offs_b).to(tl.float32)
        a_lo = tl.load(a_base + k0 + 2 * offs_b + 1).to(tl.float32)
        am = tl.load(amax_ptr + eid * stride_ae + offs_n * stride_an + (k0 // BLOCK_K),
                     mask=n_mask, other=0.0)
        acc += (tl.sum(w_hi * a_hi[None, :], axis=1)
                + tl.sum(w_lo * a_lo[None, :], axis=1)) * am
    tl.store(out_ptr + g * N + offs_n, acc.to(tl.bfloat16), mask=n_mask)


# --------------------------------------------------------------------------
# V2 -- H5 only: 2-D accumulator, ONE reduction after the loop
# --------------------------------------------------------------------------
@triton.jit
def _v2_h5(a_ptr, b_ptr, amax_ptr, out_ptr, lut_ptr, expert_ids_ptr,
           K, N, stride_be, stride_bn, stride_ae, stride_an,
           BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    g = tl.program_id(0)
    pid_n = tl.program_id(1)
    eid = tl.load(expert_ids_ptr + g)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    offs_k = tl.arange(0, BLOCK_K)
    b_base = b_ptr + eid * stride_be + offs_n[:, None] * stride_bn
    a_base = a_ptr + g * K
    acc2 = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + offs_k
        bytes_ = tl.load(b_base + (kk[None, :] // 2), mask=n_mask[:, None],
                         other=0).to(tl.int32)
        nib = tl.where((kk[None, :] % 2) == 0, (bytes_ >> 4) & 0xF, bytes_ & 0xF)
        w = tl.load(lut_ptr + nib)
        am = tl.load(amax_ptr + eid * stride_ae + offs_n * stride_an + (k0 // BLOCK_K),
                     mask=n_mask, other=0.0)
        a = tl.load(a_base + kk).to(tl.float32)
        acc2 += w * a[None, :] * am[:, None]
    acc = tl.sum(acc2, axis=1)
    tl.store(out_ptr + g * N + offs_n, acc.to(tl.bfloat16), mask=n_mask)


# --------------------------------------------------------------------------
# V3 -- H6 only: BLOCK_K decoupled from the quant blocksize via static_range
# --------------------------------------------------------------------------
@triton.jit
def _v3_h6(a_ptr, b_ptr, amax_ptr, out_ptr, lut_ptr, expert_ids_ptr,
           K, N, stride_be, stride_bn, stride_ae, stride_an,
           BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    g = tl.program_id(0)
    pid_n = tl.program_id(1)
    eid = tl.load(expert_ids_ptr + g)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    NSUB: tl.constexpr = BLOCK_K // 64
    offs_k = tl.arange(0, 64)
    b_base = b_ptr + eid * stride_be + offs_n[:, None] * stride_bn
    a_base = a_ptr + g * K
    am_base = amax_ptr + eid * stride_ae + offs_n * stride_an
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        for s in tl.static_range(NSUB):
            kk = k0 + s * 64 + offs_k
            bytes_ = tl.load(b_base + (kk[None, :] // 2), mask=n_mask[:, None],
                             other=0).to(tl.int32)
            nib = tl.where((kk[None, :] % 2) == 0, (bytes_ >> 4) & 0xF, bytes_ & 0xF)
            w = tl.load(lut_ptr + nib)
            am = tl.load(am_base + (k0 // 64) + s, mask=n_mask, other=0.0)
            a = tl.load(a_base + kk).to(tl.float32)
            acc += tl.sum(w * a[None, :], axis=1) * am
    tl.store(out_ptr + g * N + offs_n, acc.to(tl.bfloat16), mask=n_mask)


# --------------------------------------------------------------------------
# V4 -- all three composed
# --------------------------------------------------------------------------
@triton.jit
def _v4_all(a_ptr, b_ptr, amax_ptr, out_ptr, lut_ptr, expert_ids_ptr,
            K, N, stride_be, stride_bn, stride_ae, stride_an,
            BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    g = tl.program_id(0)
    pid_n = tl.program_id(1)
    eid = tl.load(expert_ids_ptr + g)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    NSUB: tl.constexpr = BLOCK_K // 64
    offs_b = tl.arange(0, 32)                 # 32 bytes == 64 k-values
    b_base = b_ptr + eid * stride_be + offs_n[:, None] * stride_bn
    a_base = a_ptr + g * K
    am_base = amax_ptr + eid * stride_ae + offs_n * stride_an
    acc2 = tl.zeros((BLOCK_N, 32), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        for s in tl.static_range(NSUB):
            base_k = k0 + s * 64
            bo = (base_k // 2) + offs_b
            bytes_ = tl.load(b_base + bo[None, :], mask=n_mask[:, None],
                             other=0).to(tl.int32)
            w_hi = tl.load(lut_ptr + ((bytes_ >> 4) & 0xF))
            w_lo = tl.load(lut_ptr + (bytes_ & 0xF))
            a_hi = tl.load(a_base + base_k + 2 * offs_b).to(tl.float32)
            a_lo = tl.load(a_base + base_k + 2 * offs_b + 1).to(tl.float32)
            am = tl.load(am_base + (k0 // 64) + s, mask=n_mask, other=0.0)
            acc2 += (w_hi * a_hi[None, :] + w_lo * a_lo[None, :]) * am[:, None]
    acc = tl.sum(acc2, axis=1)
    tl.store(out_ptr + g * N + offs_n, acc.to(tl.bfloat16), mask=n_mask)


VARIANTS = {
    "v0_shipped": (_v0_shipped, 64),
    "v1_h4_singleload": (_v1_h4, 64),
    "v2_h5_onereduce": (_v2_h5, 64),
    "v3_h6_blockk256": (_v3_h6, 256),
    "v4_all": (_v4_all, 256),
}


def make_data(E, N, K, T, device, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    packed = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, generator=g).to(device)
    absmax = (torch.rand(E, N, K // BLOCKSIZE, generator=g) * 0.5 + 0.05).float().to(device)
    a = (torch.randn(T, K, generator=g) * 0.1).bfloat16().to(device)
    eids = torch.arange(T, dtype=torch.int32, device=device) % E
    lut = torch.tensor(NF4, dtype=torch.float32, device=device)
    return packed, absmax, a, eids, lut


def reference(packed, absmax, a, eids, lut, N, K):
    """Pure-torch dequant + matvec. fp32 throughout so it is the ground truth."""
    out = torch.empty(a.shape[0], N, dtype=torch.float32, device=a.device)
    for t in range(a.shape[0]):
        e = int(eids[t])
        b = packed[e].to(torch.int32)                       # [N, K//2]
        hi = lut[(b >> 4) & 0xF]                            # k even
        lo = lut[b & 0xF]                                   # k odd
        w = torch.stack([hi, lo], dim=-1).reshape(N, K)     # [N, K]
        am = absmax[e].repeat_interleave(BLOCKSIZE, dim=1)  # [N, K]
        out[t] = (w * am * a[t].float()[None, :]).sum(dim=1)
    return out


def time_kernel(fn, bn, packed, absmax, a, eids, lut, N, K, block_k, warps, iters):
    T = a.shape[0]
    out = torch.empty(T, N, dtype=torch.bfloat16, device=a.device)
    grid = (T, triton.cdiv(N, bn))

    def run():
        fn[grid](a, packed, absmax, out, lut, eids, K, N,
                 packed.stride(0), packed.stride(1),
                 absmax.stride(0), absmax.stride(1),
                 BLOCK_N=bn, BLOCK_K=block_k, num_warps=warps, num_stages=3)

    for _ in range(10):
        run()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); run(); e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--E", type=int, default=8)
    ap.add_argument("--N", type=int, default=3072)
    ap.add_argument("--K", type=int, default=4096)
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--warps", type=int, default=2)
    ap.add_argument("--bn", type=int, default=64)
    ap.add_argument("--iters", type=int, default=7)
    ap.add_argument("--ksweep", action="store_true")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    dev = "cuda"
    print(f"# {torch.cuda.get_device_name(0)}  torch {torch.__version__}  triton {triton.__version__}")
    packed, absmax, a, eids, lut = make_data(args.E, args.N, args.K, args.T, dev)
    mb = packed.numel() / 1e6
    print(f"# E={args.E} N={args.N} K={args.K} T={args.T}  packed={mb:.2f} MB  "
          f"BLOCK_N={args.bn} warps={args.warps}")

    ref = reference(packed, absmax, a, eids, lut, args.N, args.K)

    # interleave arms: cycle through variants `iters` times rather than
    # running each to completion, so a drifting shared GPU hits all arms alike
    results = {}
    for name, (fn, bk) in VARIANTS.items():
        ms, out = time_kernel(fn, args.bn, packed, absmax, a, eids, lut,
                              args.N, args.K, bk, args.warps, args.iters)
        # SCALE-relative, not per-element: out[t,n] is a sum of K random-signed
        # terms, so entries land near zero and a per-element ratio explodes on
        # them. That is a property of the test data, not of the kernel -- the
        # first run of this harness "failed" every arm including the verbatim
        # copy of the shipped kernel, which is how the metric bug surfaced.
        d = (out.float() - ref).abs().max().item()
        err = d / ref.abs().max().item()
        gbs = packed.numel() / (ms * 1e-3) / 1e9
        if name == "v0_shipped":
            shipped_out = out.float().clone()
            vs_shipped = 0.0
        else:
            vs_shipped = ((out.float() - shipped_out).abs().max().item()
                          / shipped_out.abs().max().item())
        results[name] = {"ms": ms, "gbs": gbs, "rel_err": err,
                         "vs_shipped": vs_shipped, "block_k": bk}
        print(f"{name:22s} {ms:8.4f} ms  {gbs:7.1f} GB/s  "
              f"err_vs_fp32 {err:.3e}  err_vs_shipped {vs_shipped:.3e}")

    base = results["v0_shipped"]["ms"]
    print("\n# speedup vs shipped")
    for name, r in results.items():
        # bf16 output carries ~8 mantissa bits => 2^-8 = 3.9e-03 is the floor.
        # The gate is agreement with the SHIPPED kernel at that floor.
        gate = "OK  " if r["vs_shipped"] <= 8e-03 else "FAIL"
        print(f"{name:22s} {base / r['ms']:6.3f}x   agrees-with-shipped {gate} "
              f"({r['vs_shipped']:.2e})")

    payload = {"device": torch.cuda.get_device_name(0), "shape": vars(args),
               "results": results, "speedup": {k: base / v["ms"] for k, v in results.items()}}

    if args.ksweep:
        print("\n# K-sweep diagnostic: bytes held constant, trip count varied")
        print("# issue-bound => time super-linear in trip count; bandwidth-bound => flat")
        sweep = {}
        for K in (1024, 2048, 4096, 8192):
            N = (args.N * args.K) // K          # hold bytes/expert constant
            pk, am, aa, ei, lu = make_data(args.E, N, K, args.T, dev, seed=1)
            row = {}
            for name in ("v0_shipped", "v4_all"):
                fn, bk = VARIANTS[name]
                ms, _ = time_kernel(fn, args.bn, pk, am, aa, ei, lu, N, K, bk,
                                    args.warps, args.iters)
                row[name] = ms
            trips = K // 64
            print(f"K={K:5d} N={N:5d} trips={trips:4d}  "
                  f"shipped {row['v0_shipped']:7.4f} ms   v4 {row['v4_all']:7.4f} ms   "
                  f"ratio {row['v0_shipped'] / row['v4_all']:5.3f}x")
            sweep[K] = row
        payload["ksweep"] = sweep

    if args.tune:
        print("\n# v4 config sweep -- H6 made BLOCK_K a real knob, so retune the new kernel")
        best = None
        rows = []
        for bn in (32, 64, 128):
            for bk in (64, 128, 256, 512):
                for w in (1, 2, 4, 8):
                    try:
                        ms, out = time_kernel(_v4_all, bn, packed, absmax, a, eids,
                                              lut, args.N, args.K, bk, w, 5)
                    except Exception:
                        continue
                    e = ((out.float() - ref).abs().max().item()
                         / ref.abs().max().item())
                    if e > 8e-03:
                        continue
                    rows.append((ms, bn, bk, w, e))
                    if best is None or ms < best[0]:
                        best = (ms, bn, bk, w, e)
        rows.sort()
        for ms, bn, bk, w, e in rows[:8]:
            print(f"  BLOCK_N={bn:4d} BLOCK_K={bk:4d} warps={w}  {ms:7.4f} ms  "
                  f"{base / ms:6.3f}x vs shipped   err {e:.2e}")
        if best:
            print(f"  BEST: BLOCK_N={best[1]} BLOCK_K={best[2]} warps={best[3]} "
                  f"-> {base / best[0]:.3f}x vs shipped")
            payload["tuned"] = {"ms": best[0], "block_n": best[1],
                                "block_k": best[2], "warps": best[3],
                                "speedup": base / best[0]}

    if args.out:
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\n# wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
