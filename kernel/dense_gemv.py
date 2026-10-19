# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""K35 (kernel/PREREG-k35-dense-bf16-decode-gemv.md): ``gemv_bf16_bw``, a bandwidth-targeted dense bf16 GEMV for ONE
row (a decode step at T == 1).

``out[n] = sum_k W[n, k] * x[k]`` for one bf16 row ``x [K]`` and bf16 ``W [N, K]``, accumulated in fp32 and stored in
bf16. It is meant for a model's dense bf16 projections at decode (Qwen3-30B-A3B's fused q/k/v 5120 x 2048 and o
2048 x 4096), where experts4bit-qlora's P142 census found cuBLAS's GEMV short of the card's DRAM rate.

The kernel body, :func:`_gemv_bf16_kernel`, is experts4bit-qlora's P148 prototype VERBATIM (``bench/p148/p148_kernels.py``
at ``caf5a28c``; ``tests`` pin its source's sha256). P148 timed that prototype against cuBLAS in a CUDA-graph
microbench on an RTX 5090. Its unsplit 2-row blocks over full-K chunks of 1024 won at both shapes, and split-K lost
at every config. This module ships only that unsplit path, behind a launch wrapper with a frozen per-shape plan; it
never splits K, so the kernel's ``SPLITK`` branch is never taken.

M > 1 is refused: a multi-row call keeps cuBLAS (``torch.nn.functional.linear``). Nothing here routes a model's calls;
a consumer decides where, and its served read is the timing of record.
"""
from __future__ import annotations

from _triton_shim import tl, triton  # noqa: F401  (re-exported names)

#: P148's measured winners (``p148-5090-2``, RTX 5090): (BLOCK_N, KC, num_warps) per (N, K). Frozen: a new shape joins
#: only after a read of its own.
PLANS = {(5120, 2048): (2, 1024, 4), (2048, 4096): (2, 1024, 4)}
#: the plan for a shape :data:`PLANS` does not list: the same config (both registered shapes chose it). Such a shape
#: runs UNTIMED: no read licenses it, and its speed there is unknown (the contract still holds; its tests cover ragged
#: shapes).
PLAN_DEFAULT = (2, 1024, 4)


@triton.jit
def _gemv_bf16_kernel(w_ptr, x_ptr, out_ptr, N, K, KSPAN, stride_wn,
                      BLOCK_N: tl.constexpr, KC: tl.constexpr, SPLITK: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    offs_c = tl.arange(0, KC)
    k_lo = pid_k * KSPAN
    k_hi = tl.minimum(k_lo + KSPAN, K)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    w_rows = w_ptr + offs_n.to(tl.int64)[:, None] * stride_wn
    for k0 in range(k_lo, k_hi, KC):
        offs_k = k0 + offs_c
        k_mask = offs_k < k_hi
        x = tl.load(x_ptr + offs_k, mask=k_mask, other=0.0).to(tl.float32)
        w = tl.load(w_rows + offs_k[None, :], mask=n_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)
    if SPLITK:
        tl.store(out_ptr + pid_k * N + offs_n, acc, mask=n_mask)
    else:
        tl.store(out_ptr + offs_n, acc.to(tl.bfloat16), mask=n_mask)


def plan(N: int, K: int) -> tuple:
    """(BLOCK_N, KC, num_warps) for one (N, K): :data:`PLANS`, else :data:`PLAN_DEFAULT`."""
    return PLANS.get((int(N), int(K)), PLAN_DEFAULT)


def gemv_bf16_bw(x, w, out=None):
    """``w [N, K] (bf16) @ x`` for ONE row: ``x`` is ``[K]`` or ``[1, K]`` (bf16, contiguous), and the result has x's
    leading shape (``[N]`` or ``[1, N]``), bf16, fp32-accumulated. ``out`` may be preallocated (a CUDA-graph capture
    must not allocate) and is returned. ``M > 1`` is refused, as is any dtype other than bf16 or a weight whose rows are
    not contiguous."""
    import torch
    if x.dim() == 2:
        if x.shape[0] != 1:
            raise ValueError(f"gemv_bf16_bw serves one row; got M = {x.shape[0]} (the caller keeps cuBLAS for M > 1)")
        lead = (1,)
    elif x.dim() == 1:
        lead = ()
    else:
        raise ValueError(f"x must be [K] or [1, K], got {tuple(x.shape)}")
    if w.dim() != 2 or x.dtype != torch.bfloat16 or w.dtype != torch.bfloat16:
        raise ValueError(f"bf16 x [K] and bf16 w [N, K] only, got x {x.dtype} {tuple(x.shape)}, w {w.dtype} {tuple(w.shape)}")
    n, k = w.shape
    xv = x.reshape(-1)
    if xv.shape[0] != k or not xv.is_contiguous() or w.stride(1) != 1:
        raise ValueError(f"x has {xv.shape[0]} elements for K = {k}, or x is not contiguous, or w's rows are not "
                         f"(stride {tuple(w.stride())})")
    if out is None:
        out = torch.empty(*lead, n, dtype=torch.bfloat16, device=w.device)
    if tuple(out.shape) != (*lead, n) or out.dtype != torch.bfloat16 or not out.is_contiguous():
        raise ValueError(f"out must be a contiguous bf16 {(*lead, n)}, got {out.dtype} {tuple(out.shape)}")
    bn, kc, warps = plan(n, k)
    span = -(-k // kc) * kc                          # one split covering whole KC chunks: P148's split_k == 1 launch
    _gemv_bf16_kernel[(triton.cdiv(n, bn), 1)](w, xv, out, n, k, span, w.stride(0), BLOCK_N=bn, KC=kc, SPLITK=False,
                                               num_warps=warps)
    return out
