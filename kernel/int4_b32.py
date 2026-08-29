# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Uniform symmetric int4 (blocksize 32) pack + decode-GEMV kernels.

The serve-lane format the sm_120 census licensed: a UNIFORM grid unpacks
arithmetically -- no codebook, no gather -- so the inner loop is integer
multiply-accumulate over int8-quantised activations, exact in int32,
with one fp32 scale product per (row, k-block). Measured on the census
cells (RTX 5090, graph-replay): dense M=1 qkv-shape 5.65 us at
1,044 GB/s -- 6.9x over the NF4 register-LUT GEMV and 2.6x over the
bf16 dense baseline; grouped top-8 expert cells 3.8-4.3x over the NF4
grouped GEMV. rel err vs an fp32 reference of the same int4 values is
~1e-7: the accumulation is exact, only output rounding remains.

This module is DECODE-ONLY (M = 1 rows). Prefill and training stay on
their existing paths: at large M dequant-then-matmul wins the fused/
dequant crossover, and NF4 remains the canonical training format (the
int4 grid costs +0.007 ppl on experts and -0.006 on attention at the
8k-token gate; the lm_head measured +0.18 and must NOT use this format).

Two measurement rules are load-bearing for anyone touching configs here
(receipts: audit int4port P1):
  - time under CUDA-graph replay, never eager -- the eager host floor
    (~28 us/call) hides everything;
  - sweep configs UNDER that metric -- eager sweeps anti-select split-K
    because the partials reduce pays a launch the replay does not.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from int4_pack_ref import BLOCK, dequant_int4_ref, pack_int4_b32  # noqa: F401


# ------------------------------------------------- activation quantise --
@triton.jit
def _quant_x_rows(x_ptr, xq_ptr, xs_ptr, K: tl.constexpr):
    """Per-row, per-32-block int8 symmetric quantise (Q8-style). One
    program per row; 32-wide inner blocks so K need not be a power of 2."""
    r = tl.program_id(0)
    o32 = tl.arange(0, BLOCK)
    for kb in range(0, K // BLOCK):
        x = tl.load(x_ptr + r * K + kb * BLOCK + o32).to(tl.float32)
        s = tl.max(tl.abs(x)) / 127.0 + 1e-12
        q = tl.floor(x / s + 0.5)
        q = tl.minimum(tl.maximum(q, -127.0), 127.0)
        tl.store(xq_ptr + r * K + kb * BLOCK + o32, q.to(tl.int8))
        tl.store(xs_ptr + r * (K // BLOCK) + kb, s)


def quant_x_rows(x: torch.Tensor):
    """``x [R, K]`` -> ``(xq [R, K] int8, xs [R, K//32] fp32)``."""
    R, K = x.shape
    xq = torch.empty(R, K, dtype=torch.int8, device=x.device)
    xs = torch.empty(R, K // BLOCK, dtype=torch.float32, device=x.device)
    _quant_x_rows[(R,)](x.contiguous(), xq, xs, K=K)
    return xq, xs


# ----------------------------------------------------------- the kernel --
@triton.jit
def _gemv_int4_b32(xq_ptr, xs_ptr, w_ptr, ws_ptr, eid_ptr, out_ptr,
                   N, K: tl.constexpr, R: tl.constexpr,
                   BLOCK_N: tl.constexpr, SK: tl.constexpr,
                   KU: tl.constexpr):
    """One program computes BLOCK_N output rows of expert ``eids[e]`` for
    activation row ``e``, over its split-K span. Grid (cdiv(N,BN), R, SK).
    Dense callers pass R=1 with ``eids=[0]``. Stores fp32 partials at
    ``out[(sk*R + e)*N + n]``; the wrapper reduces (exact in fp32 --
    the int32 block dots are exact, so only the scale sums reorder)."""
    pid = tl.program_id(0)
    e = tl.program_id(1)
    sk = tl.program_id(2)
    eid = tl.load(eid_ptr + e).to(tl.int64)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    KB: tl.constexpr = K // 32
    span: tl.constexpr = (KB + SK * KU - 1) // (SK * KU)
    ku = tl.arange(0, KU)
    pair = tl.arange(0, 16)
    wbase = w_ptr + eid * N * (K // 2)
    sbase = ws_ptr + eid * N * KB
    for kbi in range(0, span):
        kb0 = (sk * span + kbi) * KU
        if kb0 < KB:
            xoff = e * K + kb0 * 32 + ku[:, None] * 32 + 2 * pair[None, :]
            xe = tl.load(xq_ptr + xoff).to(tl.int32)
            xo = tl.load(xq_ptr + xoff + 1).to(tl.int32)
            xsv = tl.load(xs_ptr + e * KB + kb0 + ku)
            wb = tl.load(wbase + offs_n[:, None] * (K // 2) + kb0 * 16
                         + tl.arange(0, KU * 16)[None, :],
                         mask=n_mask[:, None], other=0).to(tl.int32)
            wb = tl.reshape(wb, (BLOCK_N, KU, 16))
            lo = (wb & 0xF) - 8
            hi = ((wb >> 4) & 0xF) - 8
            d = tl.sum(lo * xe[None, :, :], axis=2) \
              + tl.sum(hi * xo[None, :, :], axis=2)
            ws = tl.load(sbase + offs_n[:, None] * KB + kb0 + ku[None, :],
                         mask=n_mask[:, None], other=0.0).to(tl.float32)
            acc += tl.sum(d.to(tl.float32) * (ws * xsv[None, :]), axis=1)
    tl.store(out_ptr + (sk * R + e) * N + offs_n, acc, mask=n_mask)


def _plan(N: int, K: int):
    """Config from the graph-metric sweep on sm_120 (receipts int4port):
    bn128/w4-8/sk8 class won every census cell; sk fills the grid to
    2+ waves. Kept simple until a second box class is measured."""
    sk = 8 if (triton.cdiv(N, 128) * 8) >= 256 else 16
    return 128, 4, sk, 4                      # BLOCK_N, warps, SK, KU


def gemv_int4_b32(xq, xs, packed, scales, eids, N: int, K: int,
                  part: torch.Tensor | None = None):
    """Grouped decode GEMV: ``R = eids.numel()`` activation rows, row ``e``
    against expert ``eids[e]``. ``packed [E, N, K//2]``, ``scales
    [E, N, K//32]`` (fp16). Returns ``[R, N]`` bf16. ``part`` may be a
    preallocated ``[SK*R, N]`` fp32 buffer (pass it under capture)."""
    R = eids.numel()
    bn, wp, sk, ku = _plan(N, K)
    if part is None:
        part = torch.empty(sk * R, N, dtype=torch.float32, device=xq.device)
    _gemv_int4_b32[(triton.cdiv(N, bn), R, sk)](
        xq, xs, packed, scales, eids, part,
        N, K=K, R=R, BLOCK_N=bn, SK=sk, KU=ku, num_warps=wp)
    return part.reshape(sk, R, N).sum(0).to(torch.bfloat16)
