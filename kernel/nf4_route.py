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

#: GNF4_TRITON_PREBIND (on by default, =0 off, read at import): the dequant kernel launches without Triton's per-call argument binding, the device
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
ROUTES = ("auto", "fused", "grouped_mm", "dense", "decoded")

#: ``GNF4_DECODED_MAX_BYTES``'s default: the decoded route's bf16 decode transient per chunk of groups (256 MiB, the cap
#: experts4bit-qlora's RD1 probe read). One expert larger than the cap is decoded alone.
DECODED_MAX_BYTES_DEFAULT = 256 * 2**20


def route_for(capability, *, has_grouped_mm: bool, requested: str = "auto", n_groups=None) -> tuple:
    """The training route for a device, decided from facts about it rather than from a live device: ``(route, reason)``.

    The same decision :func:`train_gemm_route` makes, as a pure function, so it can be answered before CUDA is initialised, for a
    device that is not this one, or in a test without a GPU. ``capability`` is the ``(major, minor)`` compute capability, or
    ``None`` for a non-CUDA device; ``has_grouped_mm`` is whether the torch that will run has ``torch._grouped_mm``;
    ``requested`` is what ``GNF4_TRAIN_GEMM`` would say; ``n_groups`` is the call's number of present groups when known (``auto``
    takes the dense route off sm_90 for 1 to :data:`DENSE_AUTO_MAX_GROUPS` of them). ``route`` is ``"fused"``, ``"grouped_mm"``,
    ``"dense"`` or ``"decoded"``, or ``None`` when no route of this module can train there (below :data:`MIN_CAPABILITY`, an
    explicit ``grouped_mm`` the device cannot run, or no CUDA device): ``reason`` then says why, in the words the launch-time
    refusal would use. ``decoded`` is opt-in only: ``auto`` never answers it.
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
    if requested == "decoded":
        return "decoded", "GNF4_TRAIN_GEMM=decoded: dequant_groups + one grouped bf16 GEMM launch per chunk of groups (opt-in)"
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
    """``GNF4_TRAIN_GEMM`` = ``auto`` (default) | ``fused`` | ``grouped_mm`` | ``dense`` | ``decoded``. ``auto`` is ``grouped_mm`` on a
    CUDA device of compute capability 9.0 when this torch has ``_grouped_mm``. On any other CUDA device it is ``dense`` for a call
    with 1 to :data:`DENSE_AUTO_MAX_GROUPS` present groups (``n_groups``) and ``fused`` above that or when ``n_groups`` is not given;
    CPU is ``fused``. ``dense`` dequantizes one present expert at a time and runs its GEMM through ``torch.mm``; ``decoded`` (opt-in,
    never ``auto``) is :func:`decoded_forward`. An explicit value is used as given. ``dev`` defaults to the current CUDA device. The
    device-level decision is :func:`route_for`."""
    v = os.environ.get("GNF4_TRAIN_GEMM", "auto").strip().lower()
    if v not in ROUTES:
        raise ValueError(f"GNF4_TRAIN_GEMM must be 'auto', 'fused', 'grouped_mm', 'dense' or 'decoded', got {v!r}")
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
    """``torch.cuda.get_device_capability(dev)``; unless GNF4_TRITON_PREBIND=0 read once per indexed device (a constant of the card,
    otherwise queried twice per projection)."""
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


# GNF4_TRITON_PREBIND (on unless =0): the same kernel, launched without Triton's per-call argument binding (_triton_shim.prebind)
_dequant_groups_launch = prebind(_dequant_groups_kernel)

# BLOCK_N 16 x BLOCK_KB 256 bytes at 8 warps; bit-equal to dequant_ref in bf16 throughout. This module's first kernel gathered the LUT
# and the absmax per element, and is what made experts4bit-qlora's TC1c amendment 4 boxes slower. Speed is read on rented cards only:
# on an RTX 5090 this decoder runs 0.94-1.01x the time of bitsandbytes' dequantize_4bit on large shapes and 0.65x on kv-sized ones
# (experts4bit-qlora bench/dq1/RESULTS-dq1.md, DQ1 run 2).
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

    Unless GNF4_TRITON_PREBIND=0, the plan of a grouping already seen on this device and stream is reused by value -- a layer's
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
    fused kernel (cuBLAS's accumulation order). Its speed against the fused kernel depends on rows per expert and is not quoted
    here: the RTX A2000 that timed it is a correctness-only testbed."""
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
    ``grad_out_g @ W_g``."""
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



# ---------------------------------------------------------------------------------------------------------------- the decoded route
# GNF4_TRAIN_GEMM=decoded (opt-in; ``auto`` never takes it -- route_for). Per chunk of present groups: ONE dequant_groups launch
# decodes every expert of the chunk to bf16, then ONE Triton grouped bf16 GEMM launch runs every group of the chunk -- forward
# ``a_g @ W_g^T`` and dgrad ``grad_out_g @ W_g`` through the same kernel (W read by strides). A chunk's decode transient is
# ``groups x N x K x 2`` bytes, at most GNF4_DECODED_MAX_BYTES (default 256 MiB) and at least one expert, so the chunk count is
# ceil(groups / max(1, cap // (N*K*2))). Two launches per chunk whatever the number of groups -- against `dense`'s two per group
# and the fused kernels' decode once per M-tile.
#
# This is experts4bit-qlora's RD1 probe arm `decoded_cap` (bench/moegen/rd1/rd_probe.py), moved here unchanged in arithmetic:
# the dequant is bit-equal to dequant_ref in bf16, the GEMM accumulates bf16 products in fp32 over BLOCK_K = 64 slices and rounds
# once to bf16. Not bit-identical to the fused kernels (TF32, decode in the loop) nor to `dense` (cuBLAS's order). Its gate is
# RD1's: relative error against an fp32 reference at most 2x dense's, on every call (kernel/test_nf4_decoded_interp.py in the
# interpreter, kernel/test_nf4_route.py compiled).
#
# Memory, measured, not guaranteed: on one RTX 5090 (RD1's licensed reading, rd1-rp-5090-2) the route's peak above its inputs at
# the 256 MiB cap was 180-448 MiB over eight families' expert shapes at seq 512 and 2048 (448 MiB: Mixtral-8x7B's down and
# gate_up at seq 2048, output buffer included), against 180-2016 MiB uncapped. A cap below one expert's decoded size still
# decodes that expert whole.
#
# No speed is claimed here. RD1 read the route per call on one RTX 5090. The training step is experts4bit-qlora's TC1 amendment 46
# (one RTX 5090, read in experts4bit-qlora#1221): decoded/fused 1.005 [0.979, 1.031] on OLMoE-1B-7B, 1.066 on Qwen3-30B-A3B -- no
# measurable step-time saving, so `auto` does not take this route anywhere.

ROUTE_STATS.setdefault("decoded_fwd", 0)
ROUTE_STATS.setdefault("decoded_dgrad", 0)

#: The grouped bf16 GEMM's tiles, (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages): BLOCK_M 32 below
#: :data:`DECODED_ROWS_SPLIT` mean rows per present group, 64 at or above. Taken from RD1's receipt, where of the probe's three
#: configs (32, 128, 64) won most calls up to ~75 mean rows and (64, 128, 64) most from ~90: a fixed rule, so a call's tiling (and
#: its values) never depends on a timing race. BLOCK_K is 64 in both, so the two accumulate each output in the same K order.
DECODED_TILES_SMALL = (32, 128, 64, 4, 3)
DECODED_TILES_LARGE = (64, 128, 64, 4, 3)
DECODED_ROWS_SPLIT = 80


def decoded_max_bytes() -> int:
    """``GNF4_DECODED_MAX_BYTES``: the decoded route's per-chunk decode transient in bytes, read per call (default
    :data:`DECODED_MAX_BYTES_DEFAULT`, 256 MiB). A positive integer; anything else raises."""
    v = os.environ.get("GNF4_DECODED_MAX_BYTES", "").strip()
    if not v:
        return DECODED_MAX_BYTES_DEFAULT
    try:
        n = int(v)
    except ValueError:
        n = 0
    if n <= 0:
        raise ValueError(f"GNF4_DECODED_MAX_BYTES must be a positive integer number of bytes, got {v!r}")
    return n


def decoded_chunks(n_groups: int, N: int, K: int, max_bytes: int) -> list:
    """The decoded route's chunks of groups, ``[(g0, g1), ...]``: consecutive groups whose bf16 decode (``N * K * 2`` bytes each)
    fits ``max_bytes``, at least one group per chunk. Pure arithmetic, so a test can ask it without a GPU."""
    step = max(1, int(max_bytes) // (int(N) * int(K) * 2))
    return [(g0, min(int(n_groups), g0 + step)) for g0 in range(0, int(n_groups), step)]


def decoded_tiles(n_rows: int, n_present: int) -> tuple:
    """The grouped bf16 GEMM's tile config for a call of ``n_rows`` rows over ``n_present`` non-empty groups."""
    return DECODED_TILES_SMALL if n_rows < DECODED_ROWS_SPLIT * max(1, n_present) else DECODED_TILES_LARGE


@triton.jit
def _grouped_bf16_gemm_kernel(a_ptr, w_ptr, out_ptr, t_row0_ptr, t_rows_ptr, t_grp_ptr,
                              N_OUT, R, s_am, s_wg, s_wr, s_wj, s_om,
                              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, DOT_BF16: tl.constexpr):
    """``out[row0 + i, j] = sum_r a[row0 + i, r] * W[g, r, j]`` for one (M-tile, N-tile) of one group: bf16 operands into tl.dot,
    fp32 accumulation over BK slices of R, one bf16 rounding at the store. W is read by strides, so the same kernel is the
    forward (``W[g, r, j] = W_g[j, r]``, R = K) and the dgrad (``W[g, r, j] = W_g[r, j]``, R = N). One launch covers every group
    of a chunk; ``a`` and ``out`` are row-major with unit column stride. ``DOT_BF16`` is off only under TRITON_INTERPRET=1, whose
    tl.dot does not take bf16 operands: there the same bf16 values enter as fp32 (as nf4_smallm's interpreter path does)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    row0 = tl.load(t_row0_ptr + pid_m)
    rows = tl.load(t_rows_ptr + pid_m)
    g = tl.load(t_grp_ptr + pid_m).to(tl.int64)
    rm = tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    mmask = rm < rows
    nmask = rn < N_OUT
    a_ptrs = a_ptr + (row0 + rm).to(tl.int64)[:, None] * s_am + rk[None, :]
    w_ptrs = w_ptr + g * s_wg + rk[:, None].to(tl.int64) * s_wr + rn[None, :].to(tl.int64) * s_wj
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, R, BK):
        kmask = (k0 + rk) < R
        a = tl.load(a_ptrs, mask=mmask[:, None] & kmask[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=kmask[:, None] & nmask[None, :], other=0.0)
        if not DOT_BF16:
            a = a.to(tl.float32)
            w = w.to(tl.float32)
        acc = tl.dot(a, w, acc)
        a_ptrs += BK
        w_ptrs += BK * s_wr
    o_ptrs = out_ptr + (row0 + rm).to(tl.int64)[:, None] * s_om + rn[None, :]
    tl.store(o_ptrs, acc.to(tl.bfloat16), mask=mmask[:, None] & nmask[None, :])


_grouped_bf16_gemm_launch = prebind(_grouped_bf16_gemm_kernel)

_DEC_PLAN: dict = {}                                       # (sizes, N, K, BLOCK_M, cap, device) -> the chunk plan
_DEC_PLAN_MAX = 64


def _m_tiles(sizes, bm):
    """``build_group_tiles``' expansion, relative to the first row of ``sizes``: (row0, rows, group) per M-tile."""
    t_row0, t_rows, t_grp, row = [], [], [], 0
    for g, m in enumerate(sizes):
        left = m
        while left > 0:
            take = min(bm, left)
            t_row0.append(row + (m - left))
            t_rows.append(take)
            t_grp.append(g)
            left -= take
        row += m
    return t_row0, t_rows, t_grp


def _decoded_plan(sz, N, K, bm, cap, dev):
    """Per chunk ``(g0, g1, r0, r1, (row0, rows, group) device int32)``, built once per grouping and shape and reused by value --
    a layer's forward and dgrad share it. A chunk whose groups are all empty is left out (it would launch nothing)."""
    key = (tuple(sz), N, K, bm, cap, str(dev))
    plan = _DEC_PLAN.get(key)
    if plan is None:
        plan, r0 = [], 0
        for g0, g1 in decoded_chunks(len(sz), N, K, cap):
            r1 = r0 + sum(sz[g0:g1])
            if r1 > r0:
                plan.append((g0, g1, r0, r1, to_device_i32(_m_tiles(sz[g0:g1], bm), dev)))
            r0 = r1
        if len(_DEC_PLAN) >= _DEC_PLAN_MAX:
            _DEC_PLAN.clear()
        _DEC_PLAN[key] = plan
    return plan


def _decoded(x, B, absmax, sizes, expert_ids, mode):
    dev = x.device
    # CUDA-only in real use; TRITON_INTERPRET=1 runs both kernels on CPU tensors (kernel/test_nf4_decoded_interp.py).
    if dev.type != "cuda" and os.environ.get("TRITON_INTERPRET") != "1":
        raise RuntimeError("GNF4_TRAIN_GEMM=decoded needs a CUDA device")
    E, N, half = B.shape
    K = half * 2
    sz, eids = _host_plan(sizes, expert_ids, dev)
    T = x.shape[0]
    assert sum(sz) == T, (sum(sz), T)
    bm, bn, bk, warps, stages = decoded_tiles(T, sum(1 for n in sz if n))
    n_out = N if mode == "fwd" else K
    x = x.contiguous().to(torch.bfloat16)
    out = torch.empty(T, n_out, device=dev, dtype=torch.bfloat16)
    for g0, g1, r0, r1, (row0, rows, grp) in _decoded_plan(sz, N, K, bm, decoded_max_bytes(), dev):
        W = dequant_groups(B, absmax, eids[g0:g1], N, K)                     # [g1 - g0, N, K] bf16, one launch
        R, s_wr, s_wj = (K, W.stride(2), W.stride(1)) if mode == "fwd" else (N, W.stride(1), W.stride(2))
        xc, oc = x[r0:r1], out[r0:r1]
        _grouped_bf16_gemm_launch[(row0.numel(), triton.cdiv(n_out, bn))](
            xc, W, oc, row0, rows, grp, n_out, R, xc.stride(0), W.stride(0), s_wr, s_wj, oc.stride(0),
            BM=bm, BN=bn, BK=bk, DOT_BF16=os.environ.get("TRITON_INTERPRET") != "1", num_warps=warps, num_stages=stages)
        del W
    return out


def decoded_forward(a_cat, B, absmax, sizes, expert_ids):
    """``gemm_4bit_grouped``'s contract (``a_cat [T, K]`` group-sorted, returns ``[T, N]`` bf16) through the decoded route: per chunk
    of present groups (``GNF4_DECODED_MAX_BYTES``), :func:`dequant_groups` and one grouped bf16 GEMM launch. Device-side sizes cost
    one host read (the chunks slice by them), as in :func:`dense_forward`."""
    out = _decoded(a_cat, B, absmax, sizes, expert_ids, "fwd")
    ROUTE_STATS["decoded_fwd"] += 1
    return out


def decoded_dgrad(grad_out, B, absmax, sizes, expert_ids):
    """``dgrad_4bit_grouped``'s contract (``grad_out [T, N]``, returns ``grad_a [T, K]`` bf16) through the decoded route:
    ``grad_out_g @ W_g`` per group, :func:`decoded_forward`'s chunks and kernel."""
    out = _decoded(grad_out, B, absmax, sizes, expert_ids, "dgrad")
    ROUTE_STATS["decoded_dgrad"] += 1
    return out
