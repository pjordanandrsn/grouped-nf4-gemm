"""The padded LoRA delta's glue in single kernels (``NF4_QLORA_FUSED_PAD=1``; opt-in, EXPLORE).

The lean padded delta (``nf4_qlora._lora_delta_padded_ladder``) builds its block around two ``bmm``s with a chain of small
launches. Per delta and forward: the adapters' gathers, ``F.pad`` on each (a fill and a copy), a full fp32 ``zeros`` block,
a cast and an ``index_copy_``. Its backward adds a full-size ``new_zeros`` for the delta's gradient. Each launch here writes
every element of its output exactly once, so no zero-fill pass precedes a copy pass:

* ``pad_rows(src, dst, sz, start, G, W)``: ``dst[g * W + s] = cast(src[start[g] + s])`` for ``g < G`` and ``s < sz[g]``,
  else zero. It builds the block from the group-sorted rows, and in the backward it builds the delta's padded gradient.
* ``unpad_rows(src, dst, flat)``: ``dst[r] = cast(src[flat[r]])``, the block back in caller order (the input's gradient).
* ``gather_pad(src, dst, idx, G)``: ``dst[g] = src[idx[g]]`` for ``g < G``, else zero (the adapters, padded to ``G'``).

Every value equals the ops it replaces: copies, exact widening casts and round-to-nearest-even narrowing, and zeros. The
``bmm``s are untouched, so the delta and every gradient are bit for bit the lean ladder's. On a tensor that is neither on
CUDA nor under ``TRITON_INTERPRET=1`` each function runs the torch ops it replaces, so the CPU suite checks the autograd
wiring against the path it stands in for.
"""
from __future__ import annotations

import os

import torch

from _triton_shim import tl, triton


def _use_triton(t: torch.Tensor) -> bool:
    return t.is_cuda or os.environ.get("TRITON_INTERPRET") == "1"


@triton.jit
def _pad_rows_kernel(src, dst, sz, start, G, W, K, s_src, s_dst, BLOCK_K: tl.constexpr):
    p = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    mk = offs < K
    g = p // W
    s = p - g * W
    valid = g < G
    n = tl.load(sz + g, mask=valid, other=0)
    st = tl.load(start + g, mask=valid, other=0)
    real = valid & (s < n)
    x = tl.load(src + (st + s).to(tl.int64) * s_src + offs, mask=mk & real, other=0.0)
    tl.store(dst + p.to(tl.int64) * s_dst + offs, x.to(dst.dtype.element_ty), mask=mk)


@triton.jit
def _unpad_rows_kernel(src, dst, flat, N, s_src, s_dst, BLOCK_N: tl.constexpr, F32_TO_BF16: tl.constexpr):
    r = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mn = offs < N
    p = tl.load(flat + r).to(tl.int64)
    x = tl.load(src + p * s_src + offs, mask=mn)
    if F32_TO_BF16:
        # torch's fp32 -> bf16 rounding (round to nearest even; a NaN becomes 0x7FC0), in integer arithmetic. The GPU's
        # cvt.rn gives the same bits; the Triton interpreter's cast truncates, so this keeps both bit for bit.
        b = x.to(tl.uint32, bitcast=True)
        h = ((b + 0x7FFF + ((b >> 16) & 1)) >> 16).to(tl.uint16)
        h = tl.where(x != x, 0x7FC0, h).to(tl.uint16)
        y = h.to(tl.bfloat16, bitcast=True)
    else:
        y = x.to(dst.dtype.element_ty)
    tl.store(dst + r.to(tl.int64) * s_dst + offs, y, mask=mn)


@triton.jit
def _gather_pad_kernel(src, dst, idx, G, M, BLOCK: tl.constexpr):
    g = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mm = offs < M
    valid = g < G
    e = tl.load(idx + g, mask=valid, other=0).to(tl.int64)
    x = tl.load(src + e * M + offs, mask=mm & valid, other=0.0)
    tl.store(dst + g.to(tl.int64) * M + offs, x, mask=mm)


def _block(n: int) -> int:
    return max(16, min(1024, triton.next_power_of_2(int(n))))


def pad_rows(src: torch.Tensor, dst: torch.Tensor, sz: torch.Tensor, start: torch.Tensor, G: int, W: int) -> torch.Tensor:
    """``dst[g * W + s] = src[start[g] + s]`` (cast to ``dst``'s dtype) for real rows, zero elsewhere. ``dst`` is
    ``[G' * W, K]`` and contiguous; ``src`` is ``[T, K]`` with unit stride along K."""
    rows, K = dst.shape
    if _use_triton(dst):
        bk = _block(K)
        _pad_rows_kernel[(rows, triton.cdiv(K, bk))](src, dst, sz, start, G, W, K, src.stride(0), dst.stride(0), BLOCK_K=bk)
        return dst
    dst.zero_()
    if G:
        flat = (torch.arange(int(sz.sum()), device=src.device)
                + torch.repeat_interleave(torch.arange(G, device=src.device) * W - start, sz))
        dst.index_copy_(0, flat, src.to(dst.dtype))
    return dst


def unpad_rows(src: torch.Tensor, dst: torch.Tensor, flat: torch.Tensor) -> torch.Tensor:
    """``dst[r] = src[flat[r]]``, cast to ``dst``'s dtype."""
    T, N = dst.shape
    if _use_triton(dst):
        bn = _block(N)
        narrow = src.dtype == torch.float32 and dst.dtype == torch.bfloat16
        _unpad_rows_kernel[(T, triton.cdiv(N, bn))](src, dst, flat, N, src.stride(0), dst.stride(0), BLOCK_N=bn,
                                                     F32_TO_BF16=narrow)
        return dst
    dst.copy_(src.index_select(0, flat).to(dst.dtype))
    return dst


def gather_pad(src: torch.Tensor, idx: torch.Tensor, G: int, Gp: int) -> torch.Tensor:
    """``[G', *src.shape[1:]]``: ``src[idx[g]]`` for ``g < G``, zero for the ``G' - G`` padded groups."""
    src = src.contiguous()
    dst = torch.empty((Gp,) + tuple(src.shape[1:]), dtype=src.dtype, device=src.device)
    M = src[0].numel() if src.shape[0] else 0
    if _use_triton(dst):
        b = _block(M)
        _gather_pad_kernel[(Gp, triton.cdiv(M, b))](src, dst, idx, G, M, BLOCK=b)
        return dst
    dst.zero_()
    dst[:G] = src.index_select(0, idx)
    return dst


class PadRows(torch.autograd.Function):
    """``a_cat`` (group-sorted, ``[T, K]``) into the zero-padded ``[G' * W, K]`` block in ``out_dtype``. The lean ladder's
    ``zeros`` + ``to`` + ``index_copy_``, as one launch. The backward is the input's gradient in its own dtype: the
    ``index_select`` + cast that autograd would run, as one launch."""

    @staticmethod
    def forward(ctx, a_cat, sz, start, flat, G, Gp, W, out_dtype):
        x = torch.empty(Gp * W, a_cat.shape[1], dtype=out_dtype, device=a_cat.device)
        pad_rows(a_cat, x, sz, start, G, W)
        ctx.save_for_backward(flat)
        ctx.in_dtype, ctx.rows = a_cat.dtype, a_cat.shape[0]
        return x

    @staticmethod
    def backward(ctx, gx):
        (flat,) = ctx.saved_tensors
        ga = torch.empty(ctx.rows, gx.shape[1], dtype=ctx.in_dtype, device=gx.device)
        unpad_rows(gx.contiguous(), ga, flat)
        return ga, None, None, None, None, None, None, None


class UnpadRows(torch.autograd.Function):
    """The block's rows back in caller order (``index_select``, as the lean ladder's ``_GatherRows``). The backward writes
    the zero-padded gradient in one launch, where ``_GatherRows`` ran a full ``new_zeros`` and an ``index_copy_``."""

    @staticmethod
    def forward(ctx, d, flat, sz, start, G, Gp, W):
        ctx.save_for_backward(sz, start)
        ctx.G, ctx.Gp, ctx.W = G, Gp, W
        return d.index_select(0, flat)

    @staticmethod
    def backward(ctx, g):
        sz, start = ctx.saved_tensors
        gd = torch.empty(ctx.Gp * ctx.W, g.shape[1], dtype=g.dtype, device=g.device)
        pad_rows(g.contiguous(), gd, sz, start, ctx.G, ctx.W)
        return gd, None, None, None, None, None, None


class GatherPad(torch.autograd.Function):
    """``src[idx]`` for DISTINCT ``idx``, zero-padded from ``G`` to ``G'`` groups: ``_GatherRows`` + ``F.pad``, as one
    launch. The backward drops the padded groups (``F.pad``'s) and scatters the rest into a zero gradient
    (``_GatherRows``'): ``new_zeros`` + ``index_copy_``, the same ops and values."""

    @staticmethod
    def forward(ctx, src, idx, G, Gp):
        ctx.save_for_backward(idx)
        ctx.shape, ctx.G = src.shape, G
        return gather_pad(src, idx, G, Gp)

    @staticmethod
    def backward(ctx, g):
        (idx,) = ctx.saved_tensors
        gs = g.new_zeros(ctx.shape)
        gs.index_copy_(0, idx, g[:ctx.G])
        return gs, None, None, None
