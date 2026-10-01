# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""K25: K19's grouped small-M tensor-core GEMM on the NF4 expert store (``gemm_nf4_grouped_smallm``).

``out[r, :] = x[src(r), :] @ dequant_nf4(packed[e(r)], absmax[e(r)]).T`` over K14's expert-major 16-row tiles, ONE
launch, for decode batches. It is K19 (``int4_smallm._gemm_int4_b32_grouped_smallm``) with only the dequant swapped,
as K21 is on the MXFP4 store:
- the grid ``(tiles, cdiv(N, BLOCK_N))``, the tile table (``int4_b32.build_group_tiles_fused``), the in-kernel gather,
  the sorted-order output and K23's ``scatter`` / ``gather_div`` options are K19's;
- each NF4 nibble decodes through the 16-entry fp32 codebook (bnb's order: element 2j is the HIGH nibble), is scaled by
  its per-64 absmax in fp32, rounded to bf16 and multiplied on the tensor cores, one ``tl.dot`` per KC chunk.

Why it exists (lane P91, experts4bit-qlora#564): on the NF4 families' serving configs the served grouped GEMM
(``nf4_grouped._gemm_nf4_grouped``) is 62 % (Granite) and 72 % (OLMoE) of B=16 decode kernel time. That kernel gathers
its rows in a separate launch, steps K 64 at a time, and multiplies TF32 on fp32-dequantised weights.

Arithmetic. The weight operand is ``(codebook[nibble] * absmax).to(bf16)``, computed in fp32 exactly as
``nf4_grouped.dequant_ref`` does before the dequant-then-GEMM path's bf16 materialisation, so it equals
``dequant_ref(...).to(bfloat16)`` bit for bit; activations stay bf16 and the accumulation is fp32. That is NOT the
served kernel's arithmetic (TF32 on the fp32 weight, which rounds the weight less), so a consumer gates it on quality.

Codebook decode, two ways, the same fp32 values and the same outputs bit for bit (the compiled suite holds that):
``lut="pair"`` loads ONE int64 per packed byte from a 256-entry table holding both of the byte's fp32 codebook values
(half the load instructions of a per-nibble lookup); ``"load"`` loads one fp32 per nibble from the 16-entry table in L1.
On the A2000 (99 KB of shared memory per block) ``"load"`` overflows at BLOCK_N x KC of 32 x 256 and 64 x 128, and
``"pair"`` only at 128 x 256: the compiler stages the per-nibble decode through shared memory. A third decode, a
16-entry register codebook through ``tl.gather`` (the served kernel's VARIANT 1), was tried and left out: its weights
are exact, but on the A2000 its outputs moved a bit at some shapes (a different accumulation order), so choosing it
would not have been a pure speed choice.

Contract (registered before any perf number): within one bf16 output ulp of ``x[src] @ dequant_ref(...).to(bf16).T``
(compiled; the interpreter's fp32 operands compare against the fp32 ``dequant_ref``); the gather is bit-identical to
presorting; deterministic; compiled, the plan (BLOCK_N, KC, warps, stages) and the codebook decode move no output bit.
"""
from __future__ import annotations

import torch

from _triton_shim import triton, tl
from nf4_grouped import BLOCKSIZE, NF4_LUT, _TL_INTERLEAVE, _lut

_SUPPORTED_KC = (64, 128, 256)
_LUT_MODES = {"load": 0, "pair": 2}
_PAIR_CACHE: dict = {}


def pair_lut(device) -> torch.Tensor:
    """``[256] int64``: entry b holds the fp32 bits of ``NF4_LUT[b >> 4]`` (element 2j, the HIGH nibble) in its low
    word and ``NF4_LUT[b & 15]`` (element 2j + 1) in its high word. Cached per device."""
    key = str(device)
    if key not in _PAIR_CACHE:
        lut = torch.tensor(NF4_LUT, dtype=torch.float32)
        b = torch.arange(256)
        ev = lut[(b >> 4) & 0xF].view(torch.int32).to(torch.int64) & 0xFFFFFFFF
        od = lut[b & 0xF].view(torch.int32).to(torch.int64) & 0xFFFFFFFF
        _PAIR_CACHE[key] = (ev | (od << 32)).to(device)
    return _PAIR_CACHE[key]


@triton.jit
def _gemm_nf4_grouped_smallm(x_ptr, ord_ptr, w_ptr, am_ptr, lut_ptr, row0_ptr, rows_ptr, grp_ptr, out_ptr,
                             N, stride_we, stride_wn, stride_ae, stride_an,
                             K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, KC: tl.constexpr,
                             LUT: tl.constexpr, GATHER: tl.constexpr, DOT_BF16: tl.constexpr, EVEN_K: tl.constexpr,
                             sct_ptr=None, SCATTER: tl.constexpr = False, GDIV: tl.constexpr = 1):
    """Grid ``(tiles, cdiv(N, BLOCK_N))``; program (g, pid_n) computes ``rows[g]`` (<= BLOCK_M) sorted rows x BLOCK_N
    outputs of expert ``grp[g]`` over the whole K, K19's structure. ``GATHER`` / ``GDIV`` / ``SCATTER`` are K19's and
    K23's (the gather folded into the activation load, token rows read as ``order // GDIV``, the unsort folded into the
    store). ``EVEN_K`` False (KC does not divide K; K is a multiple of 64): the last chunk is masked, K21's tail, and the
    padded columns add exact zeros. Zero-row tiles exit before the K loop."""
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
        if GDIV != 1:
            src = src // GDIV
    else:
        src = row0 + offs_m
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    KB: tl.constexpr = K // 64
    NKC: tl.constexpr = (K + KC - 1) // KC
    NSC: tl.constexpr = KC // 64
    offs_kc = tl.arange(0, KC)
    offs_kh = tl.arange(0, KC // 2)
    offs_sc = tl.arange(0, NSC)
    wbase = w_ptr + eid * stride_we + offs_n.to(tl.int64)[:, None] * stride_wn
    abase = am_ptr + eid * stride_ae + offs_n.to(tl.int64)[:, None] * stride_an
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for c in range(0, NKC):
        k0 = c * KC
        if EVEN_K:
            a = tl.load(x_ptr + src[:, None] * K + k0 + offs_kc[None, :], mask=m_mask[:, None], other=0.0)
            wb = tl.load(wbase + (k0 // 2) + offs_kh[None, :], mask=n_mask[:, None], other=0).to(tl.int32)
            am = tl.load(abase + (k0 // 64) + offs_sc[None, :], mask=n_mask[:, None], other=0.0).to(tl.float32)
        else:
            a = tl.load(x_ptr + src[:, None] * K + k0 + offs_kc[None, :],
                        mask=m_mask[:, None] & ((k0 + offs_kc) < K)[None, :], other=0.0)
            wb = tl.load(wbase + (k0 // 2) + offs_kh[None, :],
                         mask=n_mask[:, None] & (((k0 // 2) + offs_kh) < (K // 2))[None, :], other=0).to(tl.int32)
            am = tl.load(abase + (k0 // 64) + offs_sc[None, :],
                         mask=n_mask[:, None] & (((k0 // 64) + offs_sc) < KB)[None, :], other=0.0).to(tl.float32)
        if DOT_BF16:
            a = a.to(tl.bfloat16)
        else:
            a = a.to(tl.float32)
        if LUT == 2:
            pv = tl.load(lut_ptr + wb)                                       # [BN, KC/2] int64: both values of a byte
            ev = pv.to(tl.int32).to(tl.float32, bitcast=True)                # low word = element 2j (HIGH nibble)
            od = (pv >> 32).to(tl.int32).to(tl.float32, bitcast=True)        # high word = element 2j + 1
            w = _TL_INTERLEAVE(ev, od)                                       # [BN, KC] in k order
        else:
            nib = _TL_INTERLEAVE((wb >> 4) & 0xF, wb & 0xF)                  # element 2j = HIGH nibble
            w = tl.load(lut_ptr + nib)
        w3 = tl.reshape(w, (BLOCK_N, NSC, 64)) * am[:, :, None]                # per-64 absmax, fp32, in-tile
        wsc = tl.reshape(w3, (BLOCK_N, KC))
        if DOT_BF16:
            wsc = wsc.to(tl.bfloat16)                                        # the dequant-then-GEMM path rounds here
        acc += tl.dot(a, tl.trans(wsc), out_dtype=tl.float32)
    if SCATTER:
        dst = tl.load(sct_ptr + row0 + offs_m, mask=m_mask, other=0).to(tl.int64)
    else:
        dst = row0 + offs_m
    ooff = dst[:, None] * N + offs_n[None, :]
    tl.store(out_ptr + ooff, acc.to(tl.bfloat16), mask=m_mask[:, None] & n_mask[None, :])


def gemm_nf4_grouped_smallm(x: torch.Tensor, packed: torch.Tensor, absmax: torch.Tensor,
                            t_row0: torch.Tensor, t_rows: torch.Tensor, t_group: torch.Tensor,
                            order: torch.Tensor | None = None, *,
                            block_n: int = 32, kc: int = 256, warps: int = 4, stages: int = 2, lut: str = "pair",
                            dot_bf16: bool | None = None,
                            scatter: torch.Tensor | None = None, gather_div: int = 1) -> torch.Tensor:
    """K25, the grouped small-M NF4 GEMM for decode-batch experts: K19's contract on the NF4 store.

    ``x [R, K]`` bf16 (unsorted when ``order`` is given, else already in expert-major order); ``packed [E, N, K//2]``
    uint8 in bitsandbytes' nibble order (element 2j in the HIGH nibble) and ``absmax [E, N, K//64]`` (fp32, fp16 or
    bf16), each contiguous along its last dimension (the expert and row strides are free, so a view of a fused stack
    serves); the device tile table of ``int4_b32.build_group_tiles_fused`` (``t_rows`` <= 16, ``t_group`` = local
    expert ids, padding tiles rows=0). Returns ``[R, N]`` bf16 in the SORTED row order, or with ``scatter`` (K23's
    option; the builder's ``order``) in the caller's row order. ``gather_div=k`` (K23's option; needs ``order``) reads
    ``x`` as the ``[R // k, K]`` token rows.

    KC is 64, 128 or 256; when it does not divide K the last chunk is masked (K21's tail). ``lut`` picks the codebook
    decode (``"pair"`` or ``"load"``), the same outputs bit for bit. Capture-legal: every launch parameter is static and
    every input a device tensor; the only allocation is ``out``. Opt-in, no consumer, no speed claim."""
    from int4_smallm import _interpreting
    R, K = x.shape
    if gather_div != 1:
        if order is None or gather_div < 1 or R * gather_div != order.numel():
            raise ValueError(f"gather_div={gather_div} needs order with x.shape[0] * gather_div rows "
                             f"(x has {R}, order has {None if order is None else order.numel()})")
        R = order.numel()
    if packed.dim() != 3 or absmax.dim() != 3:
        raise ValueError(f"expected packed [E, N, K//2] and absmax [E, N, K//64], got {tuple(packed.shape)} / "
                         f"{tuple(absmax.shape)}")
    E, N, kh = packed.shape
    if packed.dtype != torch.uint8:
        raise ValueError(f"NF4 store must be uint8 packed bytes, got {packed.dtype}")
    if absmax.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError(f"absmax must be fp32, fp16 or bf16 (de-nested), got {absmax.dtype}")
    if K % BLOCKSIZE:
        raise ValueError(f"K={K} is not a multiple of the NF4 block ({BLOCKSIZE})")
    if kh * 2 != K or tuple(absmax.shape) != (E, N, K // BLOCKSIZE):
        raise ValueError(f"layout mismatch: x K={K}, packed {tuple(packed.shape)}, absmax {tuple(absmax.shape)}")
    if packed.stride(2) != 1 or absmax.stride(2) != 1:
        raise ValueError("packed and absmax must be contiguous along their last dimension")
    if kc not in _SUPPORTED_KC:
        raise ValueError(f"kc={kc}: expected 64, 128 or 256")
    if lut not in _LUT_MODES:
        raise ValueError(f"lut={lut!r}: expected 'pair' or 'load'")
    if scatter is not None and scatter.numel() != R:
        raise ValueError(f"scatter has {scatter.numel()} rows, the call has {R}")
    if dot_bf16 is None:
        dot_bf16 = not _interpreting()
    gather = order is not None
    table = pair_lut(x.device) if lut == "pair" else _lut(x.device)
    out = torch.empty(R, N, dtype=torch.bfloat16, device=x.device)
    extra = {} if scatter is None else {"sct_ptr": scatter, "SCATTER": True}
    if gather_div != 1:
        extra["GDIV"] = int(gather_div)
    _gemm_nf4_grouped_smallm[(t_row0.numel(), triton.cdiv(N, block_n))](
        x.contiguous(), order if gather else t_row0, packed, absmax, table, t_row0, t_rows, t_group, out,
        N, packed.stride(0), packed.stride(1), absmax.stride(0), absmax.stride(1),
        K=K, BLOCK_M=16, BLOCK_N=block_n, KC=kc, LUT=_LUT_MODES[lut], GATHER=bool(gather), DOT_BF16=bool(dot_bf16),
        EVEN_K=K % kc == 0, num_warps=warps, num_stages=stages, **extra)
    return out
