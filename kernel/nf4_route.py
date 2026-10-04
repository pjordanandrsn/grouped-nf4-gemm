# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The grouped_mm training route (GNF4_TRAIN_GEMM=grouped_mm, opt-in, sm_90): dequantize the present experts' NF4 stacks to bf16
with one Triton kernel, then run the grouped GEMM through ``torch._grouped_mm`` -- forward ``a_cat @ W_e^T`` and dgrad
``grad_out @ W_e`` -- instead of grouped-nf4-gemm's fused decode-in-the-mainloop kernels.

Why it exists: on an H100 NVL experts4bit-qlora's fused training step is device-bound, and its comparator's route on that card is
exactly this one (dequantize, then a dense grouped GEMM that sm_90 runs natively). experts4bit-qlora's TC1c amendment 3 replays
recorded real-router calls on an H100 to decide whether the route is worth taking; this module is the route it would take.

Values: the dequantized stack is bit-equal to ``dequant_ref(...).to(bfloat16)`` (the test asserts it); the GEMM accumulates bf16
products in fp32 in cuBLAS's order, so outputs differ from the fused kernels' in the last bits (an A2000 per-group check put the
relative Frobenius difference at <= 2.4e-3). Not bit-identical: a training A/B decides it, never a silent default.

Refuses, with the reason, where ``torch._grouped_mm`` has no kernel for the device (torch 2.8: compute capability 9.0 only),
rather than falling back to the fused kernels -- the same rule as nf4_qlora's grouped_mm LoRA path.
"""
from __future__ import annotations

import os

import torch

from _triton_shim import tl, triton
from nf4_grouped import BLOCKSIZE, _lut, to_device_i32

ROUTE_STATS = {"fwd": 0, "dgrad": 0}


def train_gemm_route() -> str:
    """``GNF4_TRAIN_GEMM`` = ``fused`` (default) | ``grouped_mm``."""
    v = os.environ.get("GNF4_TRAIN_GEMM", "fused").strip().lower()
    if v not in ("fused", "grouped_mm"):
        raise ValueError(f"GNF4_TRAIN_GEMM must be 'fused' or 'grouped_mm', got {v!r}")
    return v


def _refuse_unless_supported(dev):
    if not hasattr(torch, "_grouped_mm"):
        raise RuntimeError("GNF4_TRAIN_GEMM=grouped_mm needs torch._grouped_mm (torch >= 2.8); this torch has none -- unset it")
    cap = torch.cuda.get_device_capability(dev)
    if cap != (9, 0):
        raise RuntimeError(f"GNF4_TRAIN_GEMM=grouped_mm: torch._grouped_mm runs on compute capability 9.0 only (torch 2.8); this device "
                           f"is {cap[0]}.{cap[1]} -- unset it (the fused kernels are the route here)")


@triton.jit
def _dequant_groups_kernel(b_ptr, am_ptr, eid_ptr, lut_ptr, out_ptr, N, K,
                           s_be, s_bn, s_ae, s_an, s_og, s_on,
                           BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, QBLOCK: tl.constexpr):
    g = tl.program_id(0)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.program_id(2) * BLOCK_K + tl.arange(0, BLOCK_K)
    e = tl.load(eid_ptr + g).to(tl.int64)
    n_mask = offs_n < N
    k_mask = offs_k < K
    mask = n_mask[:, None] & k_mask[None, :]
    byt = tl.load(b_ptr + e * s_be + offs_n[:, None].to(tl.int64) * s_bn + (offs_k[None, :] // 2), mask=mask, other=0).to(tl.int32)
    # bnb packs element 2j into the HIGH nibble, 2j+1 into the LOW nibble (dequant_ref's order)
    nib = tl.where((offs_k[None, :] % 2) == 0, (byt >> 4) & 0xF, byt & 0xF)
    val = tl.load(lut_ptr + nib)
    am = tl.load(am_ptr + e * s_ae + offs_n[:, None].to(tl.int64) * s_an + (offs_k[None, :] // QBLOCK), mask=mask, other=0.0)
    w = (val * am).to(tl.bfloat16)
    tl.store(out_ptr + g.to(tl.int64) * s_og + offs_n[:, None].to(tl.int64) * s_on + offs_k[None, :], w, mask=mask)


def dequant_groups(B: torch.Tensor, absmax: torch.Tensor, eids_dev: torch.Tensor, N: int, K: int) -> torch.Tensor:
    """The experts ``eids_dev`` (int32 device ids, one per group) of an NF4 stack ``B [E, N, K//2]`` / ``absmax [E, N, K//64]``,
    decoded to a contiguous bf16 ``[G, N, K]`` -- bit-equal to ``dequant_ref(B[e], absmax[e], N, K).to(bfloat16)`` per group."""
    G = eids_dev.numel()
    out = torch.empty(G, N, K, dtype=torch.bfloat16, device=B.device)
    if G == 0:
        return out
    am = absmax if absmax.dtype == torch.float32 else absmax.float()
    BN, BK = 32, 128
    _dequant_groups_kernel[(G, triton.cdiv(N, BN), triton.cdiv(K, BK))](
        B, am, eids_dev, _lut(B.device), out, N, K,
        B.stride(0), B.stride(1), am.stride(0), am.stride(1), out.stride(0), out.stride(1),
        BLOCK_N=BN, BLOCK_K=BK, QBLOCK=BLOCKSIZE, num_warps=4)
    return out


def _plan(sizes, expert_ids, dev):
    """(eids int32 device, offs int32 device = inclusive cumsum of sizes) for one call; host lists go through to_device_i32."""
    if torch.is_tensor(expert_ids) and expert_ids.is_cuda:
        eids = expert_ids.to(torch.int32)
        (sz,) = to_device_i32((list(sizes),), dev)
    else:
        sz, eids = to_device_i32((list(sizes), [int(e) for e in expert_ids]), dev)
    return eids, sz.cumsum(0, dtype=torch.int32)


def grouped_mm_forward(a_cat, B, absmax, sizes, expert_ids):
    """``gemm_4bit_grouped``'s contract (``a_cat [T, K]`` group-sorted, returns ``[T, N]`` bf16) through dequant + torch._grouped_mm."""
    dev = a_cat.device
    _refuse_unless_supported(dev)
    E, N, half = B.shape
    K = half * 2
    eids, offs = _plan(sizes, expert_ids, dev)
    W = dequant_groups(B, absmax, eids, N, K)                     # [G, N, K]
    ROUTE_STATS["fwd"] += 1
    return torch._grouped_mm(a_cat.contiguous().to(torch.bfloat16), W.transpose(1, 2), offs=offs)


def grouped_mm_dgrad(grad_out, B, absmax, sizes, expert_ids):
    """``dgrad_4bit_grouped``'s contract (``grad_out [T, N]``, returns ``grad_a [T, K]`` bf16) through dequant + torch._grouped_mm."""
    dev = grad_out.device
    _refuse_unless_supported(dev)
    E, N, half = B.shape
    K = half * 2
    eids, offs = _plan(sizes, expert_ids, dev)
    W = dequant_groups(B, absmax, eids, N, K)
    ROUTE_STATS["dgrad"] += 1
    return torch._grouped_mm(grad_out.contiguous().to(torch.bfloat16), W, offs=offs)
