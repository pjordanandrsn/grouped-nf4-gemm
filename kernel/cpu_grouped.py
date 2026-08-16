# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""cpu_grouped — torch-facing wrappers for the native CPU grouped GEMV
(hybrid tier Phase 2), plus the executable spec of the locked summation
order.

The native kernels (gnf4_native/cpu_kernels.c, compiled at first use with
``-march=native``) consume the SAME packed bytes as the GPU kernels:
NF4 ``B [E, N, K//2] u8`` + ``absmax [E, N, K//64] f32`` (high nibble =
even element), MXFP4 ``blocks`` + e8m0 ``scales`` (low nibble = even
element). fp32 activations in, fp32 out, batch 1–8 rows per routed group.

Bit-exactness contract: the kernel's summation tree is fixed (16-lane
groups ascending, four round-robin accumulators, mul+add — deliberately no
FMA — then a fixed combine; see the C header) and `ordered_gemv_ref` below
is its numpy mirror. Tests require EXACT equality between the two. Against
the repo's torch oracles (`dequant_ref` + fp32 matmul) agreement is
tolerance-level only, because torch's matmul does not define a summation
order — that is the documented FMA/ordering caveat.

No torch/triton import at module top (house rule: importable everywhere);
torch loads lazily inside the wrappers.
"""

from __future__ import annotations

import numpy as np

# Local copies of the codebooks and block sizes: importing them from
# nf4_grouped would pull triton at module top (the exact importability
# defect the README flags on nf4_pack_ref). test_cpu_grouped pins these
# against the canonical sources, so drift fails loudly.
BLOCKSIZE = 64          # nf4_grouped.BLOCKSIZE
MX_BLOCK = 32           # mxfp4_pack_ref.MX_BLOCK
_NF4_LUT32 = np.asarray([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
    0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0,
], dtype=np.float32)
_FP4_LUT32 = np.asarray([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=np.float32)


# --------------------------------------------------------------------------- #
# the executable spec (numpy mirror of the locked tree)
# --------------------------------------------------------------------------- #

def ordered_dot(a32: np.ndarray, w32: np.ndarray) -> np.float32:
    """The locked summation order for one output element.

    16-lane groups in ascending order; lane-group g accumulates into
    accumulator g % 4; lanes never leave their chain; final combine is
    (acc0+acc1)+(acc2+acc3) then a sequential scalar sum of the 16 lanes.
    K must be a multiple of 16 (both formats' packed strides guarantee it).
    """
    assert a32.dtype == np.float32 and w32.dtype == np.float32
    k = a32.shape[0]
    assert k % 16 == 0
    acc = np.zeros((4, 16), dtype=np.float32)
    for g in range(k // 16):
        lo = g * 16
        acc[g % 4] += w32[lo:lo + 16] * a32[lo:lo + 16]
    comb = (acc[0] + acc[1]) + (acc[2] + acc[3])
    s = np.float32(0.0)
    for i in range(16):
        s = np.float32(s + comb[i])
    return s


def dequant_row_nf4(packed_row: np.ndarray, absmax_row: np.ndarray) -> np.ndarray:
    """One weight row to fp32, elementwise-identical to the kernel:
    w = LUT[code] * absmax (single rounding). High nibble first."""
    hi = (packed_row >> 4) & 0x0F
    lo = packed_row & 0x0F
    codes = np.empty(packed_row.shape[0] * 2, dtype=np.uint8)
    codes[0::2] = hi
    codes[1::2] = lo
    scale = np.repeat(absmax_row.astype(np.float32), BLOCKSIZE)
    return _NF4_LUT32[codes] * scale


def dequant_row_mxfp4(packed_row: np.ndarray, scales_row: np.ndarray) -> np.ndarray:
    """One MXFP4 row to fp32 via ldexp(value, e-127) — the oracle's exact
    semantics (0xFF -> 2^128, zeros stay zero). Low nibble first."""
    hi = (packed_row >> 4) & 0x0F
    lo = packed_row & 0x0F
    codes = np.empty(packed_row.shape[0] * 2, dtype=np.uint8)
    codes[0::2] = lo
    codes[1::2] = hi
    exp = np.repeat(scales_row.astype(np.int32) - 127, MX_BLOCK)
    return np.ldexp(_FP4_LUT32[codes], exp).astype(np.float32)


def ref_gemv_grouped(a32, packed, scales, sizes, expert_ids, *, fmt):
    """The full numpy reference: exact mirror of the native kernel's output
    for either format. Slow — test sizes only."""
    assert fmt in ("nf4", "mxfp4")
    n = packed.shape[1]
    rows = a32.shape[0]
    out = np.empty((rows, n), dtype=np.float32)
    r = 0
    for g, e in enumerate(expert_ids):
        for _ in range(sizes[g]):
            for col in range(n):
                if fmt == "nf4":
                    w32 = dequant_row_nf4(packed[e, col], scales[e, col])
                else:
                    w32 = dequant_row_mxfp4(packed[e, col], scales[e, col])
                out[r, col] = ordered_dot(a32[r], w32)
            r += 1
    return out


def ordered_dgrad_ref(g32, packed, scales, sizes, expert_ids, *, fmt):
    """Executable spec for the grouped dgrad (hybrid Phase 5).

    ``gi[t, k] = sum_n g[t, n] * w[n, k]`` with the LOCKED order the native
    kernel implements: for each (t, k) the fold runs over rows n STRICTLY
    ASCENDING, one mul+add per n (``w = LUT[code]*scale; p = w*g; acc += p``
    — numpy's elementwise ops apply exactly that chain per k). Slow — test
    sizes only.
    """
    assert fmt in ("nf4", "mxfp4")
    n_cols = packed.shape[1]
    k = packed.shape[2] * 2
    rows = g32.shape[0]
    assert g32.shape[1] == n_cols
    out = np.zeros((rows, k), dtype=np.float32)
    r = 0
    for g, e in enumerate(expert_ids):
        for _ in range(sizes[g]):
            for n in range(n_cols):
                if fmt == "nf4":
                    w32 = dequant_row_nf4(packed[e, n], scales[e, n])
                else:
                    w32 = dequant_row_mxfp4(packed[e, n], scales[e, n])
                out[r] += w32 * np.float32(g32[r, n])
            r += 1
    return out


# --------------------------------------------------------------------------- #
# native wrappers
# --------------------------------------------------------------------------- #

def cpu_kernels_available() -> bool:
    try:
        import gnf4_native
    except ImportError:
        return False
    return gnf4_native.available()


def _check_common(a, packed, scales, sizes, expert_ids):
    import torch
    if a.dtype != torch.float32:
        raise TypeError(f"activations must be fp32, got {a.dtype} — the CPU "
                        f"tier converts once at the router, not per kernel")
    for t, name in ((a, "a"), (packed, "packed"), (scales, "scales")):
        if t.device.type != "cpu":
            raise ValueError(f"{name} must be a CPU tensor (device {t.device})")
        if not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if len(sizes) != len(expert_ids):
        raise ValueError("sizes and expert_ids length mismatch")
    if sum(sizes) != a.shape[0]:
        raise ValueError(f"sum(sizes)={sum(sizes)} != rows={a.shape[0]}")
    for s in sizes:
        if not (1 <= s <= 8):
            raise ValueError(f"group size {s} outside the decode contract 1..8")


def _run(fn_name, a, packed, scales, sizes, expert_ids, threads):
    import ctypes

    import torch

    import gnf4_native

    lib = gnf4_native.load()
    e_ids = torch.as_tensor(expert_ids, dtype=torch.int64).contiguous()
    if int(e_ids.max()) >= packed.shape[0] or int(e_ids.min()) < 0:
        raise ValueError("expert id out of range")
    sz = torch.as_tensor(sizes, dtype=torch.int32).contiguous()
    n = packed.shape[1]
    out = torch.empty(a.shape[0], n, dtype=torch.float32)
    rc = getattr(lib, fn_name)(
        ctypes.c_void_p(a.data_ptr()), ctypes.c_void_p(packed.data_ptr()),
        ctypes.c_void_p(scales.data_ptr()), ctypes.c_void_p(e_ids.data_ptr()),
        ctypes.c_void_p(sz.data_ptr()), len(sizes), n, a.shape[1],
        ctypes.c_void_p(out.data_ptr()), threads,
    )
    if rc != 0:
        raise ValueError(f"{fn_name} rejected the call (rc={rc}) — shape "
                         f"or group-size contract violated")
    return out


def gemv_nf4_grouped_cpu(a, packed, absmax, sizes, expert_ids, *, threads=0):
    """Grouped NF4 GEMV/small-GEMM on packed bytes, fp32 out.

    a [R, K] fp32 rows sorted by group · packed [E, N, K//2] u8 ·
    absmax [E, N, K//64] f32 · sizes per-group row counts (1..8) ·
    expert_ids [G]. Raises when the native library is unavailable — the
    exact-but-slow path is `ref_gemv_grouped`, deliberately explicit.
    """
    import torch
    _check_common(a, packed, absmax, sizes, expert_ids)
    if absmax.dtype != torch.float32:
        raise TypeError("absmax must be fp32")
    if a.shape[1] % 64 or packed.shape[2] * 2 != a.shape[1]:
        raise ValueError("K mismatch or K % 64 != 0")
    return _run("gnf4_gemv_nf4_grouped", a, packed, absmax, sizes,
                expert_ids, threads)


def gemv_mxfp4_grouped_cpu(a, packed, scales, sizes, expert_ids, *, threads=0):
    """MXFP4 variant: scales [E, N, K//32] u8 e8m0."""
    import torch
    _check_common(a, packed, scales, sizes, expert_ids)
    if scales.dtype != torch.uint8:
        raise TypeError("scales must be u8 e8m0")
    if a.shape[1] % 32 or packed.shape[2] * 2 != a.shape[1]:
        raise ValueError("K mismatch or K % 32 != 0")
    return _run("gnf4_gemv_mxfp4_grouped", a, packed, scales, sizes,
                expert_ids, threads)


def _check_dgrad(g, packed, scales, sizes, expert_ids):
    import torch
    if g.dtype != torch.float32:
        raise TypeError(f"grad rows must be fp32, got {g.dtype}")
    for t, name in ((g, "g"), (packed, "packed"), (scales, "scales")):
        if t.device.type != "cpu":
            raise ValueError(f"{name} must be a CPU tensor (device {t.device})")
        if not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if len(sizes) != len(expert_ids):
        raise ValueError("sizes and expert_ids length mismatch")
    if sum(sizes) != g.shape[0]:
        raise ValueError(f"sum(sizes)={sum(sizes)} != rows={g.shape[0]}")
    if g.shape[1] != packed.shape[1]:
        raise ValueError(f"g has {g.shape[1]} columns, stack has "
                         f"{packed.shape[1]} rows per expert")
    for s in sizes:
        # deliberately NO 1..8 cap: a training microbatch parks many rows
        # on one expert, and scratch reuse makes large groups cheap
        if s < 1:
            raise ValueError(f"group size {s} < 1")


def _run_dgrad(fn_name, g, packed, scales, sizes, expert_ids, threads):
    import ctypes

    import torch

    import gnf4_native

    lib = gnf4_native.load()
    e_ids = torch.as_tensor(expert_ids, dtype=torch.int64).contiguous()
    if int(e_ids.max()) >= packed.shape[0] or int(e_ids.min()) < 0:
        raise ValueError("expert id out of range")
    sz = torch.as_tensor(sizes, dtype=torch.int32).contiguous()
    k = packed.shape[2] * 2
    # the kernel ACCUMULATES into caller-zeroed rows (per-(t,k) chains span
    # every n-tile pass) — zeros here are part of the contract
    out = torch.zeros(g.shape[0], k, dtype=torch.float32)
    rc = getattr(lib, fn_name)(
        ctypes.c_void_p(g.data_ptr()), ctypes.c_void_p(packed.data_ptr()),
        ctypes.c_void_p(scales.data_ptr()), ctypes.c_void_p(e_ids.data_ptr()),
        ctypes.c_void_p(sz.data_ptr()), len(sizes), packed.shape[1], k,
        ctypes.c_void_p(out.data_ptr()), threads,
    )
    if rc != 0:
        raise ValueError(f"{fn_name} rejected the call (rc={rc}) — shape "
                         f"contract violated")
    return out


def dgrad_nf4_grouped_cpu(g, packed, absmax, sizes, expert_ids, *, threads=0):
    """Grouped NF4 dgrad on packed bytes: ``gi = g @ W`` per group, fp32.

    g [R, N] fp32 grad rows sorted by group · packed [E, N, K//2] u8 ·
    absmax [E, N, K//64] f32 · sizes per-group row counts (>= 1, no upper
    cap) · expert_ids [G]. Returns [R, K] fp32. The exact-but-slow path is
    ``ordered_dgrad_ref``.
    """
    import torch
    _check_dgrad(g, packed, absmax, sizes, expert_ids)
    if absmax.dtype != torch.float32:
        raise TypeError("absmax must be fp32")
    if (packed.shape[2] * 2) % 64:
        raise ValueError("K % 64 != 0")
    return _run_dgrad("gnf4_dgrad_nf4_grouped", g, packed, absmax, sizes,
                      expert_ids, threads)


def dgrad_mxfp4_grouped_cpu(g, packed, scales, sizes, expert_ids, *,
                            threads=0):
    """MXFP4 variant: scales [E, N, K//32] u8 e8m0."""
    import torch
    _check_dgrad(g, packed, scales, sizes, expert_ids)
    if scales.dtype != torch.uint8:
        raise TypeError("scales must be u8 e8m0")
    if (packed.shape[2] * 2) % 32:
        raise ValueError("K % 32 != 0")
    return _run_dgrad("gnf4_dgrad_mxfp4_grouped", g, packed, scales, sizes,
                      expert_ids, threads)


def pool_start(nthreads: int = 0) -> int:
    """Start the executor-owned worker pool (pinned, spin-then-sleep).
    While running, the grouped GEMVs dispatch on it instead of OpenMP —
    same static partition, bit-identical output. Returns worker count."""
    import gnf4_native
    return gnf4_native.load().gnf4_pool_start(nthreads)


def pool_stop() -> None:
    import gnf4_native
    gnf4_native.load().gnf4_pool_stop()


def route_epilogue_bf16(logits32, k, mode, norm, idx_out, wts_out):
    """Single-call deterministic top-k + softmax, writing into the caller's
    (possibly strided) int64 / bf16-as-uint16 landing rows. numpy arrays in;
    strides must be whole elements. mode 0 = softmax-then-topk (olmoe/
    qwen3), 1 = topk-then-softmax (gpt_oss)."""
    import ctypes

    import gnf4_native

    lib = gnf4_native.load()
    t, e = logits32.shape
    assert logits32.dtype == np.float32 and logits32.flags.c_contiguous
    assert idx_out.dtype == np.int64 and wts_out.dtype == np.uint16
    assert idx_out.strides[1] == 8 and wts_out.strides[1] == 2
    lib.gnf4_route_epilogue_bf16(
        ctypes.c_void_p(logits32.ctypes.data), t, e, k, mode, int(norm),
        ctypes.c_void_p(idx_out.ctypes.data), idx_out.strides[0] // 8,
        ctypes.c_void_p(wts_out.ctypes.data), wts_out.strides[0] // 2,
    )
