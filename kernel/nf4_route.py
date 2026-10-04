# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The grouped_mm training route (GNF4_TRAIN_GEMM=auto, the default, takes it on sm_90; =grouped_mm forces it): dequantize the
present experts' NF4 stacks to bf16 with one Triton kernel, then run the grouped GEMM through ``torch._grouped_mm`` -- forward
``a_cat @ W_e^T`` and dgrad ``grad_out @ W_e`` -- instead of grouped-nf4-gemm's fused decode-in-the-mainloop kernels.

Why it exists: on an H100 NVL experts4bit-qlora's fused training step is device-bound, and its comparator's route on that card is
exactly this one (dequantize, then a dense grouped GEMM that sm_90 runs natively). experts4bit-qlora's TC1c amendment 3 replays
recorded real-router calls on an H100 to decide whether the route is worth taking; this module is the route it would take.

Values: the dequantized stack is bit-equal to ``dequant_ref(...).to(bfloat16)`` (the test asserts it); the GEMM accumulates bf16
products in fp32 in cuBLAS's order, so outputs differ from the fused kernels' in the last bits (an A2000 per-group check put the
relative Frobenius difference at <= 2.4e-3). Not bit-identical, so a training A/B decided it: experts4bit-qlora's TC1c amendment 6
measured the full Qwen3-30B-A3B step on an H100 NVL faster than the fused kernels' with the matched set EQUIVALENT, and ``auto``
takes the route on sm_90 since. ``GNF4_TRAIN_GEMM=fused`` keeps the fused kernels.

``auto`` never takes the route where ``torch._grouped_mm`` has no kernel for the device (torch 2.8: compute capability 9.0 only);
an explicit ``grouped_mm`` refuses there, with the reason, rather than falling back to the fused kernels -- the same rule as
nf4_qlora's grouped_mm LoRA path.
"""
from __future__ import annotations

import os

import torch

from _triton_shim import tl, triton
from nf4_grouped import BLOCKSIZE, _lut, to_device_i32

ROUTE_STATS = {"fwd": 0, "dgrad": 0}


_AUTO_ROUTE: dict = {}                                     # device index -> the route ``auto`` resolved to there


def train_gemm_route(dev=None) -> str:
    """``GNF4_TRAIN_GEMM`` = ``auto`` (default) | ``fused`` | ``grouped_mm``. ``auto`` is ``grouped_mm`` on a CUDA device of compute
    capability 9.0 when this torch has ``_grouped_mm``, and ``fused`` everywhere else (other cards, CPU). ``dev`` defaults to the
    current CUDA device."""
    v = os.environ.get("GNF4_TRAIN_GEMM", "auto").strip().lower()
    if v not in ("auto", "fused", "grouped_mm"):
        raise ValueError(f"GNF4_TRAIN_GEMM must be 'auto', 'fused' or 'grouped_mm', got {v!r}")
    if v != "auto":
        return v
    if dev is None:
        if not torch.cuda.is_available():
            return "fused"
        dev = torch.device("cuda", torch.cuda.current_device())
    dev = torch.device(dev)
    if dev.type != "cuda":
        return "fused"
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    r = _AUTO_ROUTE.get(idx)
    if r is None:
        ok = hasattr(torch, "_grouped_mm") and torch.cuda.get_device_capability(idx) == (9, 0)
        r = _AUTO_ROUTE[idx] = "grouped_mm" if ok else "fused"
    return r


def _refuse_unless_supported(dev):
    if not hasattr(torch, "_grouped_mm"):
        raise RuntimeError("GNF4_TRAIN_GEMM=grouped_mm needs torch._grouped_mm (torch >= 2.8); this torch has none -- unset it")
    cap = torch.cuda.get_device_capability(dev)
    if cap != (9, 0):
        raise RuntimeError(f"GNF4_TRAIN_GEMM=grouped_mm: torch._grouped_mm runs on compute capability 9.0 only (torch 2.8); this device "
                           f"is {cap[0]}.{cap[1]} -- unset it (the fused kernels are the route here)")


@triton.jit
def _dequant_groups_kernel(b_ptr, am_ptr, eid_ptr, lut_ptr, out_ptr, N, KB,
                           s_be, s_bn, s_ae, s_an, s_og, s_on,
                           BLOCK_N: tl.constexpr, BLOCK_KB: tl.constexpr, QB: tl.constexpr):
    """One [BLOCK_N rows x BLOCK_KB packed bytes] tile of one group's expert: bytes loaded coalesced, both nibbles decoded through a
    16-entry register LUT (tl.gather), interleaved with tl.join (element 2j = high nibble, 2j+1 = low -- dequant_ref's order), scaled
    by one absmax per quant block via reshape-broadcast, stored as contiguous bf16 rows. fp32 multiply then one bf16 rounding, so it
    is bit-equal to dequant_ref(...).to(bfloat16)."""
    g = tl.program_id(0)
    rn = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    rb = tl.program_id(2) * BLOCK_KB + tl.arange(0, BLOCK_KB)
    e = tl.load(eid_ptr + g).to(tl.int64)
    n_mask = rn < N
    byt = tl.load(b_ptr + e * s_be + rn[:, None].to(tl.int64) * s_bn + rb[None, :],
                  mask=n_mask[:, None] & (rb[None, :] < KB), other=0).to(tl.int32)
    lut = tl.load(lut_ptr + tl.arange(0, 16))
    hi = tl.reshape(tl.gather(lut, tl.reshape((byt >> 4) & 0xF, [BLOCK_N * BLOCK_KB]), 0), [BLOCK_N, BLOCK_KB])
    lo = tl.reshape(tl.gather(lut, tl.reshape(byt & 0xF, [BLOCK_N * BLOCK_KB]), 0), [BLOCK_N, BLOCK_KB])
    v = tl.reshape(tl.join(hi, lo), [BLOCK_N, BLOCK_KB // QB, 2 * QB])
    rq = tl.program_id(2) * (BLOCK_KB // QB) + tl.arange(0, BLOCK_KB // QB)
    am = tl.load(am_ptr + e * s_ae + rn[:, None].to(tl.int64) * s_an + rq[None, :],
                 mask=n_mask[:, None] & (rq[None, :] < KB // QB), other=0.0)
    w = tl.reshape(v * am[:, :, None], [BLOCK_N, 2 * BLOCK_KB]).to(tl.bfloat16)
    rk = tl.program_id(2) * (2 * BLOCK_KB) + tl.arange(0, 2 * BLOCK_KB)
    tl.store(out_ptr + g.to(tl.int64) * s_og + rn[:, None].to(tl.int64) * s_on + rk[None, :], w,
             mask=n_mask[:, None] & (rk[None, :] < 2 * KB))


# BLOCK_N 16 x BLOCK_KB 256 bytes at 8 warps: on an RTX A2000 this runs a Qwen3-30B-A3B gate_up stack at 233 GB/s (0.76x the time of
# bitsandbytes' dequantize_4bit, 5.1x faster than this module's first kernel, which gathered the LUT and the absmax per element and is
# what made experts4bit-qlora's TC1c amendment 4 boxes slower) and down at 0.93x bitsandbytes; bit-equal throughout.
_DQ_BLOCK_N, _DQ_BLOCK_KB, _DQ_WARPS = 16, 256, 8


def dequant_groups(B: torch.Tensor, absmax: torch.Tensor, eids_dev: torch.Tensor, N: int, K: int) -> torch.Tensor:
    """The experts ``eids_dev`` (int32 device ids, one per group) of an NF4 stack ``B [E, N, K//2]`` / ``absmax [E, N, K//64]``,
    decoded to a contiguous bf16 ``[G, N, K]`` -- bit-equal to ``dequant_ref(B[e], absmax[e], N, K).to(bfloat16)`` per group."""
    G = eids_dev.numel()
    out = torch.empty(G, N, K, dtype=torch.bfloat16, device=B.device)
    if G == 0:
        return out
    am = absmax if absmax.dtype == torch.float32 else absmax.float()
    KB = K // 2
    _dequant_groups_kernel[(G, triton.cdiv(N, _DQ_BLOCK_N), triton.cdiv(KB, _DQ_BLOCK_KB))](
        B, am, eids_dev, _lut(B.device), out, N, KB,
        B.stride(0), B.stride(1), am.stride(0), am.stride(1), out.stride(0), out.stride(1),
        BLOCK_N=_DQ_BLOCK_N, BLOCK_KB=_DQ_BLOCK_KB, QB=BLOCKSIZE // 2, num_warps=_DQ_WARPS)
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
