# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Paged decode attention over FP8 KV blocks (hybrid Stage 2, Phase 7).

One Triton kernel: batch-decode attention reading E4M3 K/V straight from
the paged pool's packed rows, dequantizing **in registers** (invariant 2 —
no dequantized KV tensor exists in any memory tier), online-softmax
accumulating in fp32. The gate it answers to (G7): sustain ≥70% of the
box's measured ``B_vram`` while doing it.

Three design decisions, each anchored to a measured precedent:

* **Grid over (sequence, KV head), never (sequence, query head).** GQA
  shares one K/V head across G query heads; a per-query-head program
  re-reads K/V G times (8–16x on the target models). The in-tree record
  shows exactly this class of mistake producing a 19x-wrong baseline
  (``kv_cache.py``, corrected 2026-07-25: repeat_interleave vs
  ``enable_gqa=True``). Each program here loads a K/V tile once and
  serves all G query heads of its group from registers.
* **Manual E4M3 decode from uint8.** Six integer/float ops per element
  (sign, exponent, mantissa, subnormal fold) rather than Triton's fp8
  dtype path — portable to every sm_80+ box in the deployment class, and
  free at a memory-bound operating point. This is also why FP8 can win
  where the fused NF4 kernel lost 11.6x: byte→float bit math instead of
  a 16-entry LUT gather per element.
* **The quantizer cannot produce inf/NaN** (amax-derived scales, see
  ``fp8_kv``), so the decode skips the NaN branch (0x7F/0xFF) instead of
  spending ops on a case the writer cannot emit. The reference dequant
  agrees by construction.

Block layout is the Phase-6/7 contract: 16-token rows, tokens-major
``[16, H_kv, D]`` E4M3 payload followed by fp32 scales — per (token,
head) for V, optionally per (token, head, D//group) for K (the measured
quality fix: grouped key scales are what pass the ≤0.5% clause on both
probe models). K and V live in separate pool partitions; one block table
indexes both.

Positions beyond a sequence's length are masked by token index — garbage
bytes in a partially-filled tail block are loaded and discarded, never
scored.
"""
from __future__ import annotations

import torch

try:  # torch ships triton on CUDA installs; keep import-safe elsewhere
    import triton
    import triton.language as tl
    _TRITON = True
except Exception:  # pragma: no cover
    _TRITON = False


def paged_attn_available() -> bool:
    return _TRITON and torch.cuda.is_available()


if _TRITON:

    @triton.jit
    def _e4m3_to_f32(u):
        """E4M3FN byte -> fp32. bias 7; subnormals are man/8 * 2^-6; the
        NaN encodings (0x7F/0xFF) are unreachable from an amax-scaled
        writer and are deliberately not special-cased."""
        u = u.to(tl.uint32)
        sign = tl.where((u & 0x80) != 0, -1.0, 1.0)
        exp = (u >> 3) & 0xF
        man = (u & 0x7).to(tl.float32)
        norm = (1.0 + man / 8.0) * tl.exp2(exp.to(tl.float32) - 7.0)
        sub = man / 8.0 * 0.015625            # 2^-6
        return sign * tl.where(exp == 0, sub, norm)

    @triton.jit
    def _fp8_paged_decode(
        q_ptr, o_ptr, kpool_u8, vpool_u8, kpool_f32, vpool_f32,
        table_ptr, seqlen_ptr,
        stride_qb, stride_qh, stride_ob, stride_oh, stride_tb,
        k_row_bytes, v_row_bytes,
        sm_scale,
        H_KV: tl.constexpr, G: tl.constexpr, D: tl.constexpr,
        BT: tl.constexpr, KTILE: tl.constexpr,
        NG_K: tl.constexpr, NG_V: tl.constexpr,
        BLOCK_G: tl.constexpr,
    ):
        pid = tl.program_id(0)
        b = pid // H_KV
        h = pid % H_KV

        t_len = tl.load(seqlen_ptr + b)

        offs_g = tl.arange(0, BLOCK_G)
        offs_d = tl.arange(0, D)
        g_mask = offs_g < G

        # q rows for this KV group: heads h*G .. h*G+G-1
        q_ptrs = (q_ptr + b * stride_qb + (h * G + offs_g)[:, None]
                  * stride_qh + offs_d[None, :])
        q = tl.load(q_ptrs, mask=g_mask[:, None], other=0.0).to(tl.float32)

        m_i = tl.full([BLOCK_G], float("-inf"), tl.float32)
        l_i = tl.zeros([BLOCK_G], tl.float32)
        acc = tl.zeros([BLOCK_G, D], tl.float32)

        offs_t = tl.arange(0, KTILE)              # tokens within a tile
        # payload byte offset of (token, head, d) inside a block row
        pay_off = ((offs_t % BT)[:, None] * (H_KV * D) + h * D
                   + offs_d[None, :])
        offs_gk = tl.arange(0, NG_K)
        offs_gv = tl.arange(0, NG_V)

        for start in range(0, t_len, KTILE):
            tok = start + offs_t
            t_mask = tok < t_len
            # gather block base rows through the table
            blk = tl.load(table_ptr + b * stride_tb + tok // BT,
                          mask=t_mask, other=0)

            # ---- K tile: [KTILE, D] fp8 -> f32, scaled ----
            k_base = blk * k_row_bytes
            ku = tl.load(kpool_u8 + k_base[:, None] + pay_off,
                         mask=t_mask[:, None], other=0)
            k = _e4m3_to_f32(ku)
            # scale float32 index: (payload_bytes + ((t%BT)*H+h)*NG + d//gs)/4
            # Scales load at their NATURAL [KTILE, NG] shape and apply via
            # a register reshape — loading them expanded to [KTILE, D]
            # (a D-wide repeat of NG values) cost 32 KB of shared memory
            # per tile and blew the sm_86 budget at D=128. Every address
            # term is kept 2-D: a bare 1-D base broadcasts along the LAST
            # axis, which compiles whenever D == KTILE and silently reads
            # scrambled scales (43% of outputs wrong before the [:, None]).
            ks_off = (k_base[:, None] // 4 + BT * H_KV * D // 4
                      + ((offs_t % BT) * H_KV + h)[:, None] * NG_K
                      + offs_gk[None, :])
            ks = tl.load(kpool_f32 + ks_off, mask=t_mask[:, None], other=1.0)
            k = tl.reshape(k, (KTILE, NG_K, D // NG_K))
            k = k * ks[:, :, None]
            k = tl.reshape(k, (KTILE, D))

            # ---- scores [BLOCK_G, KTILE], masked beyond t_len ----
            s = tl.dot(q, tl.trans(k)) * sm_scale
            s = tl.where(t_mask[None, :], s, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            alpha = tl.exp2((m_i - m_new) * 1.4426950408889634)
            p = tl.exp2((s - m_new[:, None]) * 1.4426950408889634)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None]

            # ---- V tile, same addressing with V's scale layout ----
            v_base = blk * v_row_bytes
            vu = tl.load(vpool_u8 + v_base[:, None] + pay_off,
                         mask=t_mask[:, None], other=0)
            v = _e4m3_to_f32(vu)
            vs_off = (v_base[:, None] // 4 + BT * H_KV * D // 4
                      + ((offs_t % BT) * H_KV + h)[:, None] * NG_V
                      + offs_gv[None, :])
            vs = tl.load(vpool_f32 + vs_off, mask=t_mask[:, None], other=1.0)
            v = tl.reshape(v, (KTILE, NG_V, D // NG_V))
            v = v * vs[:, :, None]
            v = tl.reshape(v, (KTILE, D))

            acc += tl.dot(p.to(v.dtype), v)
            m_i = m_new

        out = acc / l_i[:, None]
        o_ptrs = (o_ptr + b * stride_ob + (h * G + offs_g)[:, None]
                  * stride_oh + offs_d[None, :])
        tl.store(o_ptrs, out.to(o_ptr.dtype.element_ty),
                 mask=g_mask[:, None])


def fp8_paged_decode_attention(q, k_pool, v_pool, block_table, seq_lens, *,
                               n_kv_heads: int, head_dim: int,
                               block_tokens: int = 16,
                               k_groups: int = 1, v_groups: int = 1,
                               ktile: int = 64, sm_scale: float | None = None):
    """Decode attention over packed FP8 KV pool rows.

    q            [B, H_q, D] bf16/fp16 — ONE decode token per sequence.
    k_pool/v_pool  flat uint8 pools of packed block rows (payload then
                 scales, ``fp8_kv.pack_kv_block`` layout).
    block_table  [B, MAX_BLOCKS] int32 — row index per 16-token block;
                 one table serves both pools (lockstep appends).
    seq_lens     [B] int32 tokens per sequence.

    Returns [B, H_q, D] in q's dtype. Reduction order is fixed per config
    (tile-sequential online softmax) — the serving-tolerance side of the
    D0 determinism split.
    """
    assert paged_attn_available(), "needs CUDA + triton"
    B, Hq, D = q.shape
    assert D == head_dim and Hq % n_kv_heads == 0
    G = Hq // n_kv_heads
    block_g = max(16, triton.next_power_of_2(G))
    assert D % k_groups == 0 and D % v_groups == 0
    assert ktile % block_tokens == 0
    if sm_scale is None:
        sm_scale = D ** -0.5

    from fp8_kv import kv_block_bytes
    k_row = kv_block_bytes(block_tokens, n_kv_heads, head_dim) \
        + block_tokens * n_kv_heads * 4 * (k_groups - 1)
    v_row = kv_block_bytes(block_tokens, n_kv_heads, head_dim) \
        + block_tokens * n_kv_heads * 4 * (v_groups - 1)
    assert k_pool.numel() % k_row == 0 and v_pool.numel() % v_row == 0

    o = torch.empty_like(q)
    grid = (B * n_kv_heads,)
    _fp8_paged_decode[grid](
        q, o, k_pool, v_pool,
        k_pool.view(torch.float32), v_pool.view(torch.float32),
        block_table, seq_lens.to(torch.int32),
        q.stride(0), q.stride(1), o.stride(0), o.stride(1),
        block_table.stride(0),
        k_row, v_row, sm_scale,
        H_KV=n_kv_heads, G=G, D=D, BT=block_tokens, KTILE=ktile,
        NG_K=k_groups, NG_V=v_groups, BLOCK_G=block_g,
        num_warps=4, num_stages=2,
    )
    return o


def paged_attn_ref(q, k_pool, v_pool, block_table, seq_lens, *,
                   n_kv_heads: int, head_dim: int, block_tokens: int = 16,
                   k_groups: int = 1, v_groups: int = 1):
    """Pure-torch oracle: unpack every block through the reference dequant,
    run fp32 SDPA per sequence. Slow — test sizes only."""
    from fp8_kv import dequant_kv_fp8_ref, unpack_kv_block_grouped

    B, Hq, D = q.shape
    G = Hq // n_kv_heads
    outs = []
    for b in range(B):
        t = int(seq_lens[b])
        n_blk = (t + block_tokens - 1) // block_tokens
        ks, vs = [], []
        for i in range(n_blk):
            row = int(block_table[b, i])
            for pool, groups, dst in ((k_pool, k_groups, ks),
                                      (v_pool, v_groups, vs)):
                row_bytes = (block_tokens * n_kv_heads * head_dim
                             + block_tokens * n_kv_heads * groups * 4)
                raw = pool[row * row_bytes:(row + 1) * row_bytes]
                qq, ss = unpack_kv_block_grouped(
                    raw, block_tokens, n_kv_heads, head_dim, groups)
                dst.append(dequant_kv_fp8_ref(qq, ss, dtype=torch.float32))
        k = torch.cat(ks)[:t].permute(1, 0, 2)    # [H_kv, t, D]
        v = torch.cat(vs)[:t].permute(1, 0, 2)
        qq = q[b].float().view(n_kv_heads, G, D)
        att = torch.einsum("hgd,htd->hgt", qq, k) * (D ** -0.5)
        w = torch.softmax(att, dim=-1)
        o = torch.einsum("hgt,htd->hgd", w, v)
        outs.append(o.reshape(Hq, D))
    return torch.stack(outs).to(q.dtype)
