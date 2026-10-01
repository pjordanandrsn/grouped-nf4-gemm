# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Small-M int4-b32 GEMM for the attention projections (lane K16, PREREG-k16-smallm-int4-gemm.md).

``y[M, N] = x[M, K] @ dequant(packed[N, K//2], scales[N, K//32]).T`` for ``M <= 16``, ONE launch.

Why a second int4 GEMM exists beside ``int4_b32._gemm_int4_b32_grouped`` (K14): that kernel does exact
integer MMA with ONE ``tl.dot`` per 32-wide k-block, because its per-(row, k-block) fp32 scale product
must be applied after each dot -- at M=16 on a single projection that is 64 tiny dots per program and no
expert-count parallelism, 22% of the streaming ceiling on the 5090 (K14). This kernel does what Marlin
does (K15): dequantise the int4 tile IN REGISTERS, scale it per 32-block inside the tile, and run bf16
tensor-core MMA over a fat K-chunk (KC = 128..256), with split-K across programs and the reduction fused
into the same launch (the last program to finish a column block sums the fp32 partials in split order).
Activations stay bf16 -- no int8 activation quantisation on this path -- so the arithmetic is the
dequant-then-GEMM path's, without materialising the bf16 weight.

Contract (registered before any perf number): deterministic for a fixed config; within one bf16 output
ulp of ``x @ dequant_int4_ref(packed, scales).T`` on every shape; within one bf16 ulp across SK.
"""
from __future__ import annotations

import torch

from _triton_shim import triton, tl

BLOCK = 32          # the int4-b32 scale block, as int4_pack_ref.BLOCK
_SUPPORTED_KC = (256, 128, 64, 32)


@triton.jit
def _gemm_int4_b32_smallm(x_ptr, w_ptr, ws_ptr, part_ptr, cnt_ptr, out_ptr,
                          M, N, K: tl.constexpr,
                          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                          KC: tl.constexpr, SK: tl.constexpr, DOT_BF16: tl.constexpr):
    """Grid ``(cdiv(N, BLOCK_N), SK)``. Program (pid_n, pid_k) accumulates rows [0, M) x its N block
    over its K span in fp32, then either stores (SK == 1) or writes an fp32 partial and lets the LAST
    arriving program of the column block reduce all SK partials in split order and store bf16."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = tl.arange(0, BLOCK_M)
    m_mask = offs_m < M
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    KB: tl.constexpr = K // 32                    # scale blocks along K
    NKC: tl.constexpr = K // KC                   # K chunks
    PER_SPLIT: tl.constexpr = NKC // SK           # chunks per program
    NSC: tl.constexpr = KC // 32                  # scale blocks per chunk
    offs_kc = tl.arange(0, KC)
    offs_kh = tl.arange(0, KC // 2)               # packed bytes per row per chunk
    offs_sc = tl.arange(0, NSC)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for c in range(pid_k * PER_SPLIT, (pid_k + 1) * PER_SPLIT):
        k0 = c * KC
        a = tl.load(x_ptr + offs_m[:, None] * K + k0 + offs_kc[None, :],
                    mask=m_mask[:, None], other=0.0)                                 # [BM, KC]
        if DOT_BF16:
            a = a.to(tl.bfloat16)                                                    # tensor-core operand
        else:
            a = a.to(tl.float32)                                                     # interpreter / CPU contract path
        wb = tl.load(w_ptr + offs_n[:, None] * (K // 2) + (k0 // 2) + offs_kh[None, :],
                     mask=n_mask[:, None], other=0).to(tl.int32)                     # [BN, KC/2] bytes
        lo = ((wb & 0xF) - 8).to(tl.float32)                                         # even k = LOW nibble
        hi = (((wb >> 4) & 0xF) - 8).to(tl.float32)                                  # odd k = HIGH nibble
        w = tl.interleave(lo, hi)                                                    # [BN, KC] in k order
        sc = tl.load(ws_ptr + offs_n[:, None] * KB + (k0 // 32) + offs_sc[None, :],
                     mask=n_mask[:, None], other=0.0).to(tl.float32)                 # [BN, NSC]
        w3 = tl.reshape(w, (BLOCK_N, NSC, 32)) * sc[:, :, None]                     # scale per 32-block, in-tile
        wsc = tl.reshape(w3, (BLOCK_N, KC))
        if DOT_BF16:
            wsc = wsc.to(tl.bfloat16)                                                # the dequant-then-GEMM path rounds here too
        acc += tl.dot(a, tl.trans(wsc), out_dtype=tl.float32)                        # [BM, BN]
    ooff = offs_m[:, None] * N + offs_n[None, :]
    omask = m_mask[:, None] & n_mask[None, :]
    if SK == 1:
        tl.store(out_ptr + ooff, acc.to(tl.bfloat16), mask=omask)
    else:
        # fp32 partial [SK, BLOCK_M, N]; then the last arriver reduces IN SPLIT ORDER (deterministic)
        tl.store(part_ptr + pid_k * (BLOCK_M * N) + ooff, acc, mask=omask)
        tl.debug_barrier()
        prev = tl.atomic_add(cnt_ptr + pid_n, 1, sem="acq_rel")
        if prev == SK - 1:
            total = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for s in range(0, SK):
                total += tl.load(part_ptr + s * (BLOCK_M * N) + ooff, mask=omask, other=0.0)
            tl.store(out_ptr + ooff, total.to(tl.bfloat16), mask=omask)
            tl.atomic_xchg(cnt_ptr + pid_n, 0)                                       # armed for the next launch


def plan_smallm(N: int, K: int, *, block_n: int = 64, kc: int = 128, sk: int = 4):
    """Legalise a config for (N, K): KC must divide K and be a multiple of 32; SK must divide K // KC.
    Falls back downward (never silently to a different arithmetic). Returns (block_n, kc, sk)."""
    if K % 32:
        raise ValueError(f"K={K} is not a multiple of the int4-b32 scale block ({BLOCK})")
    kc = next((c for c in _SUPPORTED_KC if c <= kc and K % c == 0), None)
    if kc is None:
        raise ValueError(f"no supported K chunk divides K={K} (supported {_SUPPORTED_KC})")
    nkc = K // kc
    while sk > 1 and nkc % sk:
        sk //= 2
    return int(block_n), int(kc), int(max(1, sk))


def smallm_workspace(N: int, *, block_m: int = 16, block_n: int = 64, sk: int = 4, device="cuda"):
    """Preallocated (part, cnt) so the launch is capture-legal: ``part [SK, 16, N] fp32``, ``cnt [cdiv(N, BN)] int32``
    zeroed once; the kernel leaves ``cnt`` zeroed after every launch."""
    part = torch.empty(sk, block_m, N, dtype=torch.float32, device=device)
    cnt = torch.zeros(triton.cdiv(N, block_n), dtype=torch.int32, device=device)
    return part, cnt


def _interpreting() -> bool:
    import os
    return os.environ.get("TRITON_INTERPRET", "0") == "1"


def gemm_int4_b32_smallm(x: torch.Tensor, packed: torch.Tensor, scales: torch.Tensor, *,
                         block_n: int = 64, kc: int = 128, sk: int = 4, warps: int = 4, stages: int = 2,
                         workspace=None, dot_bf16: bool | None = None) -> torch.Tensor:
    """``x [M, K]`` (bf16/fp16/fp32; M <= 16), ``packed [N, K//2] uint8``, ``scales [N, K//32]`` (fp16/bf16/fp32)
    in the int4-b32 layout (``int4_pack_ref.pack_int4_b32``). Returns ``[M, N]`` bf16. One launch.

    ``dot_bf16``: the MMA operands are bf16 (the shipped tensor-core arithmetic; also what the dequant-then-GEMM
    path rounds to). Under ``TRITON_INTERPRET=1`` it defaults to False -- the interpreter has no bf16 dot (numpy),
    so the CPU contract suite runs the same indexing / dequant / scaling / split-K logic with fp32 operands; the
    bf16 numerics are owned by the compiled suite on a GPU. Never silently mixed: the choice is recorded by the
    caller's flag, and a test that wants bf16 under the interpreter gets the interpreter's failure, not a fallback."""
    M, K = x.shape
    N, kh = packed.shape
    if M > 16:
        raise ValueError(f"gemm_int4_b32_smallm serves M <= 16 rows (got {M}); the grouped M-tile kernel serves more")
    if kh * 2 != K or tuple(scales.shape) != (N, K // 32):
        raise ValueError(f"layout mismatch: x K={K}, packed {tuple(packed.shape)}, scales {tuple(scales.shape)}")
    block_n, kc, sk = plan_smallm(N, K, block_n=block_n, kc=kc, sk=sk)
    if workspace is None:
        part, cnt = smallm_workspace(N, block_n=block_n, sk=sk, device=x.device)
    else:
        part, cnt = workspace
        if part.shape[0] < sk or part.shape[2] != N or cnt.numel() < triton.cdiv(N, block_n):
            raise ValueError("workspace does not fit this plan")
    if dot_bf16 is None:
        dot_bf16 = not _interpreting()
    out = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
    xc = x.contiguous()
    _gemm_int4_b32_smallm[(triton.cdiv(N, block_n), sk)](
        xc, packed, scales, part, cnt, out, M, N, K=K,
        BLOCK_M=16, BLOCK_N=block_n, KC=kc, SK=sk, DOT_BF16=bool(dot_bf16), num_warps=warps, num_stages=stages)
    return out


# ---------------------------------------------------------------------------------------------- lane K19 --
@triton.jit
def _gemm_int4_b32_grouped_smallm(x_ptr, ord_ptr, w_ptr, ws_ptr, row0_ptr, rows_ptr, grp_ptr, out_ptr,
                                  N, K: tl.constexpr,
                                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, KC: tl.constexpr,
                                  GATHER: tl.constexpr, DOT_BF16: tl.constexpr):
    """K19: K16's arithmetic over K14's expert-major tiles. Grid ``(tiles, cdiv(N, BLOCK_N))``; program (g, pid_n)
    computes ``rows[g]`` (<= BLOCK_M) sorted rows x BLOCK_N outputs of expert ``grp[g]`` over the WHOLE K (no split:
    at decode the tile count times the N blocks already fills the card, so there are no partials, counters or a
    separate reduce). Activations stay bf16 (no int8 quantise); the int4 tile is dequantised and scaled in registers
    per 32-block and multiplied as bf16 on the tensor cores, one ``tl.dot`` per KC chunk, exactly as K16.

    ``GATHER``: sorted row r reads input row ``order[row0 + r]`` (the expert-major gather folded into the load, for
    the unsorted first projection); otherwise input row ``row0 + r`` (already sorted, e.g. the epilogue output).
    Output rows are written in SORTED order, K14's convention, so this is a drop-in for its captured call.
    Zero-row tiles (the static grid's padding) exit before the K loop, K14's zero-tile lesson."""
    g = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = tl.load(rows_ptr + g)
    if rows == 0:
        return
    row0 = tl.load(row0_ptr + g).to(tl.int64)
    eid = tl.load(grp_ptr + g).to(tl.int64)
    offs_m = tl.arange(0, BLOCK_M)
    m_mask = offs_m < rows
    if GATHER:
        src = tl.load(ord_ptr + row0 + offs_m, mask=m_mask, other=0).to(tl.int64)
    else:
        src = row0 + offs_m
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    KB: tl.constexpr = K // 32
    NKC: tl.constexpr = K // KC
    NSC: tl.constexpr = KC // 32
    offs_kc = tl.arange(0, KC)
    offs_kh = tl.arange(0, KC // 2)
    offs_sc = tl.arange(0, NSC)
    wbase = w_ptr + eid * N * (K // 2)
    sbase = ws_ptr + eid * N * KB
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for c in range(0, NKC):
        k0 = c * KC
        a = tl.load(x_ptr + src[:, None] * K + k0 + offs_kc[None, :], mask=m_mask[:, None], other=0.0)
        if DOT_BF16:
            a = a.to(tl.bfloat16)
        else:
            a = a.to(tl.float32)
        wb = tl.load(wbase + offs_n[:, None] * (K // 2) + (k0 // 2) + offs_kh[None, :],
                     mask=n_mask[:, None], other=0).to(tl.int32)
        lo = ((wb & 0xF) - 8).to(tl.float32)
        hi = (((wb >> 4) & 0xF) - 8).to(tl.float32)
        w = tl.interleave(lo, hi)
        sc = tl.load(sbase + offs_n[:, None] * KB + (k0 // 32) + offs_sc[None, :],
                     mask=n_mask[:, None], other=0.0).to(tl.float32)
        w3 = tl.reshape(w, (BLOCK_N, NSC, 32)) * sc[:, :, None]
        wsc = tl.reshape(w3, (BLOCK_N, KC))
        if DOT_BF16:
            wsc = wsc.to(tl.bfloat16)
        acc += tl.dot(a, tl.trans(wsc), out_dtype=tl.float32)
    ooff = (row0 + offs_m)[:, None] * N + offs_n[None, :]
    tl.store(out_ptr + ooff, acc.to(tl.bfloat16), mask=m_mask[:, None] & n_mask[None, :])


def gemm_int4_b32_grouped_smallm(x: torch.Tensor, packed: torch.Tensor, scales: torch.Tensor,
                                 t_row0: torch.Tensor, t_rows: torch.Tensor, t_group: torch.Tensor,
                                 order: torch.Tensor | None = None, *,
                                 block_n: int = 32, kc: int = 256, warps: int = 4, stages: int = 2,
                                 dot_bf16: bool | None = None) -> torch.Tensor:
    """K19, the grouped small-M int4-b32 GEMM: ``x [R, K]`` bf16 (unsorted when ``order`` is given, else already in
    expert-major order), ``packed [E, N, K//2]`` / ``scales [E, N, K//32]`` int4-b32 expert stacks, and the device
    tile table of ``int4_b32.build_group_tiles_fused`` (``t_row0``, ``t_rows`` <= 16, ``t_group`` = local expert ids;
    padding tiles rows=0). Returns ``[R, N]`` bf16 in the SORTED row order (K14's convention).

    Same contract as K16 per tile: each output row is within one bf16 ulp of ``x[src] @ dequant(packed[e]).T`` and is
    bit-identical to K16 (``gemm_int4_b32_smallm`` with ``sk=1`` and the same ``block_n``/``kc``) on that expert's
    rows, because a row's MMA output does not depend on the tile's other rows. Every launch parameter is static and
    every input a device tensor, so the call is legal inside CUDA-graph capture; the only allocation is ``out``.

    The default plan (BLOCK_N 32, KC 256, 4 warps, 2 stages) is K20's best on an RTX 5090 at Qwen3-30B-A3B's expert
    shapes on recorded B=16 routing: 5.20 ms/step for both projections + the tile build, against the shipped plan's 6.21
    (BLOCK_N 64, KC 128) and the served int8 GEMV route's 7.06 (``kernel/RESULTS-k20-k19-plan-sweep-5090.md``). Compiled,
    the plan changes no output bit: the MMA accumulates the same products in the same order whatever BLOCK_N and KC
    are (70 of 72 plans compared on the card, ``test_plans_are_bit_identical_compiled`` here). Under the interpreter
    the fp32 dot is numpy's and KC can move the last bit."""
    R, K = x.shape
    E, N, kh = packed.shape
    if kh * 2 != K or tuple(scales.shape) != (E, N, K // 32):
        raise ValueError(f"layout mismatch: x K={K}, packed {tuple(packed.shape)}, scales {tuple(scales.shape)}")
    block_n, kc, _sk = plan_smallm(N, K, block_n=block_n, kc=kc, sk=1)
    if dot_bf16 is None:
        dot_bf16 = not _interpreting()
    gather = order is not None
    out = torch.empty(R, N, dtype=torch.bfloat16, device=x.device)
    _gemm_int4_b32_grouped_smallm[(t_row0.numel(), triton.cdiv(N, block_n))](
        x.contiguous(), order if gather else t_row0, packed, scales, t_row0, t_rows, t_group, out,
        N, K=K, BLOCK_M=16, BLOCK_N=block_n, KC=kc, GATHER=bool(gather), DOT_BF16=bool(dot_bf16),
        num_warps=warps, num_stages=stages)
    return out
