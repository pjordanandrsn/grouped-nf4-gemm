# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""FP8 KV storage primitives (hybrid Stage 2, Phase 7).

E4M3 for keys and values with a **per-token-per-head** scale — one fp32
scale per (token, head) row of ``head_dim`` values. This is the executable
spec half of Phase 7: quantize-on-write here, and a reference dequant that
materializes bf16 so quality can be measured *before* any fused kernel
exists. The shipping read path never calls the reference — it dequantizes
in registers inside the attention kernel (invariant 2 applies to KV: no
dequantized KV tensor is ever materialized in any memory tier).

Why per-token-per-head, stated so the choice is falsifiable: a KV row is
one head's view of one token, and its magnitude varies far more across
tokens than within a row. The in-tree NF4 KV work measured this axis
directly — per-token scaling cost +0.083 perplexity against per-channel's
+0.275 on the same model — so the finer axis is not a guess, it is the
measured winner carried forward to a wider format.

Format, per (token, head):

    scale = amax(|x|) / E4M3_MAX          fp32, exact
    q     = to_e4m3(x / scale)            8 bits, saturating by construction
    x'    = to_bf16(q) * scale

``E4M3_MAX`` is 448.0, the largest finite e4m3 value. Because the scale is
derived from the row's own amax, ``x / scale`` lands in [-448, 448] by
construction and cannot overflow to inf/NaN — torch's fp8 cast does NOT
saturate, so a format that relied on clamping would be one bad scale away
from NaN in the cache. An all-zero row keeps ``scale = 1.0`` (dividing by
its amax would be 0/0), which round-trips zeros exactly.

Bytes are opaque to the tier that stores them: :class:`row_pool.RowPool`
holds packed blocks and never interprets them, so this format is a block
layout decision, not a tiering change.
"""
from __future__ import annotations

import torch

E4M3_MAX = 448.0
FP8_DTYPE = torch.float8_e4m3fn


def quantize_kv_fp8(x: torch.Tensor, group: int | None = None):
    """``[..., T, H, D]`` (any float dtype) -> ``(q, scale)``.

    ``q`` is e4m3 with x's shape. ``scale`` is fp32 with one entry per
    quantization group: with the default ``group=None`` that is one per
    (token, head), i.e. ``x.shape[:-1]``; with ``group=g`` dividing ``D``
    it is ``x.shape[:-1] + (D // g,)``, one scale per ``g`` consecutive
    values of a row.

    Sub-row groups exist because the sensitivity is not symmetric between
    keys and values, measured rather than assumed (Phase 7): full-row
    scaling costs one model +0.574% perplexity through its KEYS while
    costing another +0.080%, and the byte price of a finer key grid is 4
    bytes per group against ``g`` bytes of payload — 1.88x compression at
    g=64 and 1.78x at g=32, against 1.94x for the full row at D=128.

    Runs on whatever device x is on; needs no FP8 tensor cores (this is
    storage, not arithmetic).
    """
    d = x.shape[-1]
    if d < 1:
        raise ValueError("head_dim must be >= 1")
    if group is None:
        group = d
    if group < 1 or d % group:
        raise ValueError(f"group {group} must divide head_dim {d}")
    xf = x.float()
    xg = xf.reshape(*xf.shape[:-1], d // group, group)
    amax = xg.abs().amax(dim=-1)
    # amax == 0 -> scale 1.0: the group is all zeros and round-trips
    # exactly, where amax/E4M3_MAX would make dequant 0*0 and quant 0/0.
    scale = torch.where(amax > 0, amax / E4M3_MAX, torch.ones_like(amax))
    q = (xg / scale.unsqueeze(-1)).reshape(xf.shape).to(FP8_DTYPE)
    return q, (scale.squeeze(-1) if group == d else scale)


def dequant_kv_fp8_ref(q: torch.Tensor, scale: torch.Tensor,
                       dtype=torch.bfloat16) -> torch.Tensor:
    """Reference dequant: ``[..., T, H, D]`` back to ``dtype``.

    Accepts either scale layout — one per row (``scale.ndim == q.ndim-1``)
    or one per sub-row group (``scale.ndim == q.ndim``).

    MATERIALIZES the result — this is the oracle a fused kernel is checked
    against and the path the quality harness measures, never the serving
    read path.
    """
    qf = q.to(torch.float32)
    if scale.ndim == qf.ndim - 1:
        return (qf * scale.unsqueeze(-1)).to(dtype)
    d, n_groups = qf.shape[-1], scale.shape[-1]
    qg = qf.reshape(*qf.shape[:-1], n_groups, d // n_groups)
    return (qg * scale.unsqueeze(-1)).reshape(qf.shape).to(dtype)


def kv_roundtrip_error(x: torch.Tensor):
    """``(max_abs, rel_fro)`` for one quantize/dequant round trip.

    Reported per call rather than asserted: e4m3 has 3 mantissa bits, so
    the relative error floor is ~2^-4 per element by construction. A caller
    comparing formats wants the number, not a threshold someone guessed.
    """
    q, s = quantize_kv_fp8(x)
    back = dequant_kv_fp8_ref(q, s, dtype=torch.float32)
    xf = x.float()
    diff = (back - xf).abs()
    denom = xf.norm()
    return (diff.max().item(),
            (diff.norm() / denom).item() if denom > 0 else 0.0)


def kv_block_bytes(block_tokens: int, n_kv_heads: int, head_dim: int) -> int:
    """Bytes one packed KV block occupies: fp8 payload + fp32 scales.

    The scale tail is what makes the honest compression ratio less than 2x
    against bf16 — at head_dim 128 it is 4 bytes per 128, i.e. 3.1% — and
    callers sizing a pool should budget from this function rather than from
    the payload alone.
    """
    return block_tokens * n_kv_heads * head_dim + block_tokens * n_kv_heads * 4


def pack_kv_block(q: torch.Tensor, scale: torch.Tensor,
                  out: torch.Tensor, *, layout: str = "tokens"
                  ) -> torch.Tensor:
    """Write ``q``/``scale`` for one block into a flat uint8 row.

    Two layouts, both measured — under DIFFERENT kernels, which is the
    point of keeping both:

    * ``"tokens"``: payload ``[T, H, D]`` (caller order), scales
      ``[T, H(, NG)]``. Won the bake-off under the DECODE-path kernel
      (86.3 vs 74.2 GB/s on the dev card): the H_kv programs of one
      sequence walk the same 512 B lines together and L2 serves the
      sibling quarters.
    * ``"heads"``: payload ``[H, T, D]``, scales ``[H, T(, NG)]`` — one
      private contiguous ``T*D``-byte run per (block, head). Rebuilt for
      the fp8-tensor-core kernel, whose bottleneck is the memory system
      rather than decode ALU; there per-CTA contiguity competes again
      (the tokens-major verdict was measured under a bottleneck that no
      longer exists on that path).

    Row SIZE is identical either way (``kv_block_bytes``); readers must
    agree on the layout — the paged-attention wrapper takes the same
    flag.
    """
    n = q.numel()
    if out.numel() < n + scale.numel() * 4:
        raise ValueError(f"row of {out.numel()} bytes too small for "
                         f"{n} payload + {scale.numel() * 4} scale bytes")
    if layout == "heads":
        q = q.permute(1, 0, 2)
        scale = scale.permute(1, 0, 2) if scale.ndim == 3             else scale.permute(1, 0)
    else:
        assert layout == "tokens", f"unknown layout {layout!r}"
    out.narrow(0, 0, n).copy_(q.reshape(-1).contiguous().view(torch.uint8))
    out.narrow(0, n, scale.numel() * 4).copy_(
        scale.reshape(-1).contiguous().float().view(torch.uint8))
    return out


def unpack_kv_block(row: torch.Tensor, block_tokens: int, n_kv_heads: int,
                    head_dim: int):
    """Inverse of :func:`pack_kv_block`: ``(q, scale)`` views into ``row``.

    Views, not copies — the caller may hand these straight to a kernel.
    """
    return unpack_kv_block_grouped(row, block_tokens, n_kv_heads, head_dim,
                                   1)


def unpack_kv_block_grouped(row: torch.Tensor, block_tokens: int,
                            n_kv_heads: int, head_dim: int, groups: int,
                            *, layout: str = "tokens"):
    """Grouped-scale variant: the scale tail holds ``groups`` fp32 values
    per (token, head); ``groups == 1`` is the per-row layout and returns
    the scale squeezed to ``[T, H]`` so both callers see the shape the
    reference dequant expects for their layout. Always returns
    ``[T, H, ...]`` regardless of the on-disk ``layout`` (heads-major
    rows come back permuted — a COPY for heads, views for tokens)."""
    n = block_tokens * n_kv_heads * head_dim
    if layout == "heads":
        q = row.narrow(0, 0, n).view(FP8_DTYPE).view(
            n_kv_heads, block_tokens, head_dim).permute(1, 0, 2)
        s = row.narrow(0, n, block_tokens * n_kv_heads * groups * 4).view(
            torch.float32).view(n_kv_heads, block_tokens, groups
                                ).permute(1, 0, 2)
    else:
        assert layout == "tokens", f"unknown layout {layout!r}"
        q = row.narrow(0, 0, n).view(FP8_DTYPE).view(
            block_tokens, n_kv_heads, head_dim)
        s = row.narrow(0, n, block_tokens * n_kv_heads * groups * 4).view(
            torch.float32).view(block_tokens, n_kv_heads, groups)
    return q, (s.squeeze(-1) if groups == 1 else s)


# --------------------------------------------------------------------------
# Fused T=1 append (e4b F1 Stage B, arm B2)
# --------------------------------------------------------------------------
# The eager decode append is ~25 launches per layer (two 5-kernel
# quantizes, ~10 address-math ops, 4 scatters), every one at the GPU's
# ~1.2 us launch quantum -- launch-bound, not arithmetic-bound (the F1
# Stage A census). This kernel does one SIDE's address-math + quantize +
# store in a single launch. The math is quantize_kv_fp8's exactly: fp32
# amax per group, scale = amax/E4M3_MAX with the all-zero group pinned to
# 1.0, x/scale cast to e4m3 (saturating RNE on both paths); fp32 max and
# same-operand divides carry no reduction-order rounding, so the fused
# path is BITWISE against the reference -- asserted by
# test_fp8_kv_append.py on randomized states, and re-asserted on-box
# before any timed arm.

try:  # triton is Linux-only (see pyproject); the torch surface above
    import triton  # must stay importable without it
    import triton.language as tl
    HAS_TRITON = True
except Exception:  # pragma: no cover - non-Linux
    triton = None
    HAS_TRITON = False

if HAS_TRITON:
    @triton.jit
    def _fp8_append_t1_side(
        x_ptr,            # [H, D] input (bf16/fp16/fp32), contiguous
        pool_u8,          # flat uint8 view of this layer's pool arena
        pool_i32,         # SAME memory viewed int32 (scale stores)
        tbl_ptr,          # [blocks_per_seq] int32 block table row
        lens_ptr,         # int32 scalar: tokens already in this sequence
        row_bytes,        # bytes per pool row
        pay_bytes,        # payload region size (scales start here)
        bt,               # tokens per block
        H: tl.constexpr,
        D: tl.constexpr,
        GROUPS: tl.constexpr,   # scale groups per (token, head) row
        GS: tl.constexpr,       # D // GROUPS
        E4M3_MAX: tl.constexpr,
    ):
        h = tl.program_id(0)
        pos = tl.load(lens_ptr).to(tl.int64)
        blk = pos // bt
        fill = pos - blk * bt
        # int64 BEFORE any byte product: row * row_bytes overflows int32
        # on multi-GiB arenas (the nf4_grouped lesson).
        row = tl.load(tbl_ptr + blk).to(tl.int64)
        pay = row * row_bytes + fill * (H * D) + h * D
        sc = (row * row_bytes + pay_bytes
              + (fill * (H * GROUPS) + h * GROUPS) * 4)
        for g in tl.static_range(GROUPS):
            offs = tl.arange(0, GS)
            x = tl.load(x_ptr + h * D + g * GS + offs).to(tl.float32)
            amax = tl.max(tl.abs(x), axis=0)
            scale = tl.where(amax > 0, amax / E4M3_MAX, 1.0)
            q = (x / scale).to(tl.float8e4nv).to(tl.uint8, bitcast=True)
            tl.store(pool_u8 + pay + g * GS + offs, q)
            tl.store(pool_i32 + sc // 4 + g,
                     scale.to(tl.float32).to(tl.int32, bitcast=True))


def fp8_kv_append_t1(x, pool_flat_u8, tbl_row_i32, lens_scalar_i32,
                     row_bytes: int, pay_bytes: int, block_tokens: int,
                     groups: int):
    """One-launch T=1 append of one side into a paged pool row.

    ``x``: [H, D] (or [1, H, D]) float tensor for the new token.
    ``pool_flat_u8``: flat uint8 view of the layer's pool arena.
    ``tbl_row_i32``: this sequence's int32 block-table row.
    ``lens_scalar_i32``: 1-element int32 view of tokens-seen; READ by the
    kernel, never written -- the caller publishes the increment after,
    preserving the exact in-stream ordering append_graph_t1 has today.

    Capture-safe by construction: every address comes from device state.
    """
    if x.dim() == 3:
        x = x.squeeze(0)
    H, D = x.shape
    if D % groups:
        raise ValueError(f"groups {groups} must divide head_dim {D}")
    if pool_flat_u8.dtype != torch.uint8 or not pool_flat_u8.is_contiguous():
        raise ValueError("pool view must be contiguous uint8")
    if int(pay_bytes) % 4 or int(row_bytes) % 4:
        # scale stores go through an int32 view of the same bytes; a
        # misaligned scale region would corrupt neighbours silently
        raise ValueError("row/payload sizes must be 4-byte aligned")
    pool_i32 = pool_flat_u8.view(torch.int32)
    _fp8_append_t1_side[(H,)](
        x.contiguous(), pool_flat_u8, pool_i32, tbl_row_i32,
        lens_scalar_i32, int(row_bytes), int(pay_bytes),
        int(block_tokens), H=H, D=D, GROUPS=groups, GS=D // groups,
        E4M3_MAX=E4M3_MAX, num_warps=1)
