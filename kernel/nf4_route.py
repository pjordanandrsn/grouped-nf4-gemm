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

from _triton_shim import prebind, prebind_requested, tl, triton
from nf4_grouped import BLOCKSIZE, _host_reuse_enabled, _lists_of_ints_shape, _lut, _raw_stream_key, _ValueMemo, to_device_i32

#: GNF4_TRITON_PREBIND=1 (opt-in, read at import): the dequant kernel launches without Triton's per-call argument binding, the device
#: capability the route checks is read once per device, and one grouping's device plan is reused by value. Values identical.
_PREBIND = prebind_requested()

ROUTE_STATS = {"fwd": 0, "dgrad": 0}


_AUTO_ROUTE: dict = {}                                     # device index -> the route ``auto`` resolved to there


#: ``auto`` takes the dense route off sm_90 for a call with at most this many present groups. experts4bit-qlora's TC1 amendment 22 read it
#: on the full training step on an RTX 5090: Mixtral-8x7B (2 of 8 experts per token) stepped at 0.651x the fused kernels' time, while
#: Qwen3-30B-A3B (up to 128 present experts per call, a launch-bound step) stepped at 2.947x, so calls with many groups stay fused.
DENSE_AUTO_MAX_GROUPS = 16

#: The fused NF4 kernels' documented floor (README "Environment": an NVIDIA GPU of sm_80 or newer). Stated here as data so a
#: caller can ask before it launches; nothing below this is tested, and nothing here refuses on it at launch.
MIN_CAPABILITY = (8, 0)
#: The one compute capability ``torch._grouped_mm`` has a kernel for (torch 2.8), so the only place ``auto`` takes that route.
GROUPED_MM_CAPABILITY = (9, 0)
ROUTES = ("auto", "fused", "grouped_mm", "dense")


def route_for(capability, *, has_grouped_mm: bool, requested: str = "auto", n_groups=None) -> tuple:
    """The training route for a device, decided from facts about it rather than from a live device: ``(route, reason)``.

    The same decision :func:`train_gemm_route` makes, as a pure function, so it can be answered before CUDA is initialised, for a
    device that is not this one, or in a test without a GPU. ``capability`` is the ``(major, minor)`` compute capability, or
    ``None`` for a non-CUDA device; ``has_grouped_mm`` is whether the torch that will run has ``torch._grouped_mm``;
    ``requested`` is what ``GNF4_TRAIN_GEMM`` would say; ``n_groups`` is the call's number of present groups when known (``auto``
    takes the dense route off sm_90 for 1 to :data:`DENSE_AUTO_MAX_GROUPS` of them). ``route`` is ``"fused"``, ``"grouped_mm"`` or
    ``"dense"``, or ``None`` when no route of this module can train there (below :data:`MIN_CAPABILITY`, an explicit
    ``grouped_mm`` the device cannot run, or no CUDA device): ``reason`` then says why, in the words the launch-time refusal
    would use.
    """
    requested = str(requested).strip().lower()
    if requested not in ROUTES:
        raise ValueError(f"requested must be one of {ROUTES}, got {requested!r}")
    if capability is None:
        return None, "no CUDA device: the NF4 training kernels are CUDA-only"
    cap = (int(capability[0]), int(capability[1]))
    if cap < MIN_CAPABILITY:
        return None, (f"compute capability {cap[0]}.{cap[1]} is below the documented floor "
                      f"{MIN_CAPABILITY[0]}.{MIN_CAPABILITY[1]} of the fused NF4 kernels")
    if requested == "fused":
        return "fused", "GNF4_TRAIN_GEMM=fused"
    if requested == "dense":
        return "dense", "GNF4_TRAIN_GEMM=dense: per-expert dequant + torch.mm"
    grouped_ok = has_grouped_mm and cap == GROUPED_MM_CAPABILITY
    if requested == "grouped_mm":
        if not has_grouped_mm:
            return None, "GNF4_TRAIN_GEMM=grouped_mm needs torch._grouped_mm (torch >= 2.8); this torch has none"
        if not grouped_ok:
            return None, (f"GNF4_TRAIN_GEMM=grouped_mm: torch._grouped_mm runs on compute capability 9.0 only (torch 2.8); "
                          f"this device is {cap[0]}.{cap[1]}")
        return "grouped_mm", "GNF4_TRAIN_GEMM=grouped_mm"
    if grouped_ok:
        return "grouped_mm", "auto: compute capability 9.0 with torch._grouped_mm"
    why = (f"auto: compute capability {cap[0]}.{cap[1]} is not 9.0" if has_grouped_mm
           else "auto: this torch has no torch._grouped_mm")
    if n_groups is not None and 0 < int(n_groups) <= DENSE_AUTO_MAX_GROUPS:
        return "dense", f"{why}; {int(n_groups)} present groups <= {DENSE_AUTO_MAX_GROUPS} take the dense route"
    if n_groups is not None:
        return "fused", f"{why}; {int(n_groups)} present groups > {DENSE_AUTO_MAX_GROUPS} stay fused"
    return "fused", f"{why} (a call with 1 to {DENSE_AUTO_MAX_GROUPS} present groups takes the dense route)"


def train_gemm_route(dev=None, n_groups=None) -> str:
    """``GNF4_TRAIN_GEMM`` = ``auto`` (default) | ``fused`` | ``grouped_mm`` | ``dense``. ``auto`` is ``grouped_mm`` on a CUDA device
    of compute capability 9.0 when this torch has ``_grouped_mm``. On any other CUDA device it is ``dense`` for a call with 1 to
    :data:`DENSE_AUTO_MAX_GROUPS` present groups (``n_groups``) and ``fused`` above that or when ``n_groups`` is not given; CPU is
    ``fused``. ``dense`` dequantizes one present expert at a time and runs its GEMM through ``torch.mm``. An explicit value is used as
    given. ``dev`` defaults to the current CUDA device. The device-level decision is :func:`route_for`."""
    v = os.environ.get("GNF4_TRAIN_GEMM", "auto").strip().lower()
    if v not in ROUTES:
        raise ValueError(f"GNF4_TRAIN_GEMM must be 'auto', 'fused', 'grouped_mm' or 'dense', got {v!r}")
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
        route, _ = route_for(torch.cuda.get_device_capability(idx), has_grouped_mm=hasattr(torch, "_grouped_mm"))
        # Below the floor route_for answers None; ``auto`` has always said "fused" there and the kernel launch is what fails.
        r = _AUTO_ROUTE[idx] = route or "fused"
    if r == "fused" and n_groups is not None and 0 < int(n_groups) <= DENSE_AUTO_MAX_GROUPS:
        return "dense"
    return r


_CAPABILITY: dict = {}


def _capability(dev):
    """``torch.cuda.get_device_capability(dev)``; under GNF4_TRITON_PREBIND=1 read once per indexed device (a constant of the card,
    and 7-11 us a call on an RTX A2000 host, paid twice per projection)."""
    if not _PREBIND or getattr(dev, "index", None) is None:
        return torch.cuda.get_device_capability(dev)
    cap = _CAPABILITY.get(dev)
    if cap is None:
        cap = _CAPABILITY[dev] = torch.cuda.get_device_capability(dev)
    return cap


def _refuse_unless_supported(dev):
    route, why = route_for(_capability(dev), has_grouped_mm=hasattr(torch, "_grouped_mm"),
                           requested="grouped_mm")
    if route != "grouped_mm":
        raise RuntimeError(f"{why} -- unset it (the fused kernels are the route here)")


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


# GNF4_TRITON_PREBIND=1 (opt-in): the same kernel, launched without Triton's per-call argument binding (_triton_shim.prebind)
_dequant_groups_launch = prebind(_dequant_groups_kernel)

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
    _dequant_groups_launch[(G, triton.cdiv(N, _DQ_BLOCK_N), triton.cdiv(KB, _DQ_BLOCK_KB))](
        B, am, eids_dev, _lut(B.device), out, N, KB,
        B.stride(0), B.stride(1), am.stride(0), am.stride(1), out.stride(0), out.stride(1),
        BLOCK_N=_DQ_BLOCK_N, BLOCK_KB=_DQ_BLOCK_KB, QB=BLOCKSIZE // 2, num_warps=_DQ_WARPS)
    return out


_PLAN_FAST = _ValueMemo()


def _plan(sizes, expert_ids, dev):
    """(eids int32 device, offs int32 device = inclusive cumsum of sizes) for one call; host lists go through to_device_i32.

    Under GNF4_TRITON_PREBIND=1 the plan of a grouping already seen on this device and stream is reused by value -- a layer's
    gate_up and down calls, forward and dgrad, share one -- instead of rebuilding its upload key and relaunching the cumsum."""
    if _PREBIND and _lists_of_ints_shape((sizes, expert_ids)) and dev.type == "cuda" and _host_reuse_enabled() \
            and not torch.cuda.is_current_stream_capturing():
        ctx = _raw_stream_key(dev)
        hit = _PLAN_FAST.get((sizes, expert_ids), ctx)
        return hit if hit is not None else _PLAN_FAST.put((sizes, expert_ids), ctx, _build_plan(sizes, expert_ids, dev))
    return _build_plan(sizes, expert_ids, dev)


def _build_plan(sizes, expert_ids, dev):
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


ROUTE_STATS.setdefault("dense_fwd", 0)
ROUTE_STATS.setdefault("dense_dgrad", 0)


def _host_plan(sizes, expert_ids, dev):
    """(host sizes, device eids int32) for the per-expert loop. Device-side sizes cost one host read here: the loop slices by them."""
    sz = [int(v) for v in (sizes.tolist() if torch.is_tensor(sizes) else sizes)]
    if torch.is_tensor(expert_ids) and expert_ids.is_cuda:
        eids = expert_ids.to(torch.int32)
    else:
        (eids,) = to_device_i32(([int(e) for e in expert_ids],), dev)
    return sz, eids


def dense_forward(a_cat, B, absmax, sizes, expert_ids):
    """``gemm_4bit_grouped``'s contract (``a_cat [T, K]`` group-sorted, returns ``[T, N]`` bf16) as a per-expert loop: each present
    expert is dequantized alone (:func:`dequant_groups` on one id, bit-equal to ``dequant_ref`` in bf16) and multiplied with ``torch.mm``.
    One expert's bf16 weight is the only transient (Mixtral's gate_up: 235 MB), and any CUDA card runs it. Not bit-identical to the
    fused kernel (cuBLAS's accumulation order). Measured on an RTX A2000: 3.4x the fused forward at Mixtral-8x7B's shapes with 1,024
    rows per expert, 1.7x at Qwen3-30B-A3B's with 256; the fused kernel wins at Qwen3's down projection with 64 rows."""
    dev = a_cat.device
    if dev.type != "cuda":
        raise RuntimeError("GNF4_TRAIN_GEMM=dense needs a CUDA device")
    E, N, half = B.shape
    K = half * 2
    sz, eids = _host_plan(sizes, expert_ids, dev)
    a = a_cat.contiguous().to(torch.bfloat16)
    out = torch.empty(a.shape[0], N, device=dev, dtype=torch.bfloat16)
    r0 = 0
    for g, n in enumerate(sz):
        if n:
            W = dequant_groups(B, absmax, eids[g:g + 1], N, K)[0]          # [N, K]
            torch.mm(a[r0:r0 + n], W.t(), out=out[r0:r0 + n])
        r0 += n
    ROUTE_STATS["dense_fwd"] += 1
    return out


def dense_dgrad(grad_out, B, absmax, sizes, expert_ids):
    """``dgrad_4bit_grouped``'s contract (``grad_out [T, N]``, returns ``grad_a [T, K]`` bf16) as :func:`dense_forward`'s per-expert loop:
    ``grad_out_g @ W_g``. Measured on an RTX A2000: 7.7x the fused dgrad at Mixtral's gate_up shape, 4.6x at Qwen3-30B-A3B's."""
    dev = grad_out.device
    if dev.type != "cuda":
        raise RuntimeError("GNF4_TRAIN_GEMM=dense needs a CUDA device")
    E, N, half = B.shape
    K = half * 2
    sz, eids = _host_plan(sizes, expert_ids, dev)
    g_out = grad_out.contiguous().to(torch.bfloat16)
    out = torch.empty(g_out.shape[0], K, device=dev, dtype=torch.bfloat16)
    r0 = 0
    for g, n in enumerate(sz):
        if n:
            W = dequant_groups(B, absmax, eids[g:g + 1], N, K)[0]
            torch.mm(g_out[r0:r0 + n], W, out=out[r0:r0 + n])
        r0 += n
    ROUTE_STATS["dense_dgrad"] += 1
    return out

