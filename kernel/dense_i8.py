# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Int8-weight dense projections at decode: one row, and 2 to 16 rows (lane K37, kernel/PREREG-k37-int8-attention-projections.md).

``out[m, n] = s[n] * sum_k q[n, k] * x[m, k]`` for an int8 weight ``q [N, K]`` with contiguous rows, an fp32 per-row
scale ``s [N]`` and bf16 activations, accumulated in fp32 with one bf16 downcast. This is W8A16 for Mixtral-8x7B's
attention projections (q and o: 4096 x 4096; k and v: 1024 x 4096), whose int8 copies halve the bytes a decode step
reads.

The kernel bodies, :func:`_gemv_i8_kernel` (M == 1) and :func:`_gemm_i8_small_kernel` (2 <= M <= 16), are
experts4bit-qlora's P149 prototypes VERBATIM (``bench/p149/p149_kernels.py`` at ``ffb1632c``; tests pin both sources'
sha256). P149's box K timed them against cuBLAS bf16 in a CUDA-graph microbench on an RTX 5090 (``p149k-5090-8``), and
the plans below are that box's choices. This module ships only the unsplit path behind a frozen per-shape plan: no
plan splits K, so the kernels' ``SPLITK`` branches are never taken.

The weights' grid is K36's :func:`dense_gemv.quantize_rows_int8`, re-exported here and never redefined: it is P149's
quantizer to the bit (the tests compare them), so ``(q * s)`` is P147's I8ATTN weights.

M > 16 is refused: a prefill-sized call keeps the caller's own path. Nothing here routes a model's calls or quantizes a
model; a consumer decides where, and its served read is the timing of record. No speed is claimed here.
"""
from __future__ import annotations

import os

from _triton_shim import tl, triton  # noqa: F401  (re-exported names)
from dense_gemv import quantize_rows_int8  # noqa: F401  (K36's grid, re-exported: P149's to the bit)

# Triton's interpreter runs tl.dot on bf16 operands' raw bits, so under TRITON_INTERPRET=1 the small-M kernel's dot runs in
# fp32 (IEEE) instead; compiled code always runs the bf16 dot. P149's switch, carried as it is.
DOT_F32 = os.environ.get("TRITON_INTERPRET") == "1"
#: the small-M kernel's one row tile (P149's): 2 <= M <= 16 rows, the rows past M masked.
BLOCK_M = 16

#: P149 box K's pinned choices (``p149k-5090-8``, RTX 5090), (BLOCK_N, KC, num_warps) per (N, K) for one row. Frozen: a
#: new shape joins only after a read of its own.
PLANS_GEMV = {(4096, 4096): (4, 512, 4), (1024, 4096): (4, 2048, 8)}
#: (BLOCK_N, BK, num_warps) per (N, K) for 2 to 16 rows: both shapes chose the same.
PLANS_SMALL = {(4096, 4096): (32, 256, 4), (1024, 4096): (32, 256, 4)}
#: the plans for a shape the tables do not list: q and o's GEMV plan, and the shared small-M plan. Such a shape runs
#: UNTIMED: no read licenses it, and its speed there is unknown (the contract still holds; its tests cover ragged shapes).
PLAN_GEMV_DEFAULT = (4, 512, 4)
PLAN_SMALL_DEFAULT = (32, 256, 4)


@triton.jit
def _gemv_i8_kernel(q_ptr, s_ptr, x_ptr, out_ptr, N, K, KSPAN, stride_qn,
                    BLOCK_N: tl.constexpr, KC: tl.constexpr, SPLITK: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    offs_c = tl.arange(0, KC)
    k_lo = pid_k * KSPAN
    k_hi = tl.minimum(k_lo + KSPAN, K)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    q_rows = q_ptr + offs_n.to(tl.int64)[:, None] * stride_qn
    for k0 in range(k_lo, k_hi, KC):
        offs_k = k0 + offs_c
        k_mask = offs_k < k_hi
        x = tl.load(x_ptr + offs_k, mask=k_mask, other=0.0).to(tl.float32)
        q = tl.load(q_rows + offs_k[None, :], mask=n_mask[:, None] & k_mask[None, :], other=0).to(tl.float32)
        acc += tl.sum(q * x[None, :], axis=1)
    if SPLITK:
        tl.store(out_ptr + pid_k * N + offs_n, acc, mask=n_mask)
    else:
        s = tl.load(s_ptr + offs_n, mask=n_mask, other=0.0)
        tl.store(out_ptr + offs_n, (acc * s).to(tl.bfloat16), mask=n_mask)


@triton.jit
def _gemm_i8_small_kernel(q_ptr, s_ptr, x_ptr, out_ptr, M, N, K, KSPAN, stride_qn, stride_xm, stride_om,
                          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BK: tl.constexpr, SPLITK: tl.constexpr,
                          DOT_F32: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < M
    n_mask = offs_n < N
    offs_c = tl.arange(0, BK)
    k_lo = pid_k * KSPAN
    k_hi = tl.minimum(k_lo + KSPAN, K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    q_rows = q_ptr + offs_n.to(tl.int64)[:, None] * stride_qn
    x_rows = x_ptr + offs_m.to(tl.int64)[:, None] * stride_xm
    for k0 in range(k_lo, k_hi, BK):
        offs_k = k0 + offs_c
        k_mask = offs_k < k_hi
        x = tl.load(x_rows + offs_k[None, :], mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        q = tl.load(q_rows + offs_k[None, :], mask=n_mask[:, None] & k_mask[None, :], other=0).to(tl.float32)
        if DOT_F32:
            acc = tl.dot(x.to(tl.float32), tl.trans(q), acc, input_precision="ieee")
        else:
            acc = tl.dot(x, tl.trans(q.to(tl.bfloat16)), acc)   # the codes are exact in bf16 (|q| <= 127)
    if SPLITK:
        ws = out_ptr + pid_k * M * N + offs_m[:, None] * N + offs_n[None, :]
        tl.store(ws, acc, mask=m_mask[:, None] & n_mask[None, :])
    else:
        s = tl.load(s_ptr + offs_n, mask=n_mask, other=0.0)
        o = out_ptr + offs_m.to(tl.int64)[:, None] * stride_om + offs_n[None, :]
        tl.store(o, (acc * s[None, :]).to(tl.bfloat16), mask=m_mask[:, None] & n_mask[None, :])


def plan(N: int, K: int, M: int = 1) -> tuple:
    """The frozen launch plan for one (N, K) at M rows: (BLOCK_N, KC, num_warps) for M == 1, else (BLOCK_N, BK,
    num_warps)."""
    if M == 1:
        return PLANS_GEMV.get((int(N), int(K)), PLAN_GEMV_DEFAULT)
    return PLANS_SMALL.get((int(N), int(K)), PLAN_SMALL_DEFAULT)


def _span(k, kc):
    """P149's K span for one split: whole chunks covering K (``_span(k, 1, kc)`` there)."""
    return -(-k // kc) * kc


def linear_i8(x, q, s, out=None):
    """``x @ (q * s[:, None]).T`` for 1 to 16 rows: ``x`` is bf16 ``[K]`` or ``[M, K]`` (rows contiguous), ``q`` an int8
    ``[N, K]`` with contiguous rows and ``s`` a contiguous fp32 ``[N]`` (``quantize_rows_int8``'s pair, made once by the
    caller: nothing here quantizes). The result is bf16 with x's leading shape (``[N]`` or ``[M, N]``), accumulated in
    fp32. ``out`` may be preallocated (a CUDA-graph capture must not allocate) and is returned. One row launches the
    GEMV, 2 to 16 rows the small-M kernel, each exactly as P149's ``gemv_i8`` / ``gemm_i8_small`` launch with one split.
    M > 16, M == 0, another dtype, a shape mismatch or a non-contiguous ``q``, ``s`` or row is refused."""
    import torch
    if x.dtype != torch.bfloat16 or q.dtype != torch.int8 or s.dtype != torch.float32:
        raise ValueError(f"linear_i8 takes bf16 x, int8 q and fp32 s; got {x.dtype}, {q.dtype}, {s.dtype}")
    if q.dim() != 2 or q.stride(1) != 1:
        raise ValueError("q must be [N, K] with contiguous rows (quantize_rows_int8's layout)")
    n, k = q.shape
    if s.shape != (n,) or not s.is_contiguous():
        raise ValueError(f"s must be a contiguous fp32 [N] = [{n}]; got {tuple(s.shape)}")
    if x.dim() == 1:
        m, lead, xs = 1, (n,), x.reshape(1, -1)
    elif x.dim() == 2:
        m, lead, xs = x.shape[0], (x.shape[0], n), x
    else:
        raise ValueError(f"x must be [K] or [M, K]; got {tuple(x.shape)}")
    if not 1 <= m <= BLOCK_M:
        raise ValueError(f"linear_i8 serves 1 to {BLOCK_M} rows; got M = {m} (the caller keeps its own path past that)")
    if xs.shape[1] != k or xs.stride(1) != 1:
        raise ValueError(f"x's rows must be contiguous with K = {k}; got {tuple(x.shape)}")
    if out is None:
        out = torch.empty(lead, dtype=torch.bfloat16, device=q.device)
    elif out.shape != lead or out.dtype != torch.bfloat16 or not out.is_contiguous():
        raise ValueError(f"out must be a contiguous bf16 {lead}; got {tuple(out.shape)} {out.dtype}")
    if m == 1:
        bn, kc, warps = plan(n, k, 1)
        grid = (triton.cdiv(n, bn), 1)
        _gemv_i8_kernel[grid](q, s, xs.reshape(-1), out.reshape(-1), n, k, _span(k, kc), q.stride(0),
                              BLOCK_N=bn, KC=kc, SPLITK=False, num_warps=warps)
        return out
    bn, bk, warps = plan(n, k, m)
    o2 = out.reshape(m, n)
    grid = (triton.cdiv(n, bn), 1)
    _gemm_i8_small_kernel[grid](q, s, xs, o2, m, n, k, _span(k, bk), q.stride(0), xs.stride(0), o2.stride(0),
                                BLOCK_M=BLOCK_M, BLOCK_N=bn, BK=bk, SPLITK=False, DOT_F32=DOT_F32, num_warps=warps)
    return out
