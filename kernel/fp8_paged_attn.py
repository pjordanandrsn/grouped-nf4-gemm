# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Paged decode attention over FP8 KV blocks (hybrid Stage 2, Phase 7).

Flash-decode over the paged pool: E4M3 K/V read straight from packed rows,
dequantized **in registers** (invariant 2 — no dequantized KV tensor exists
in any memory tier), online softmax accumulated in fp32, sequences
partitioned across SPLITS with a tiny combine kernel. The gate it answers
to (G7): sustain ≥70% of the box's measured ``B_vram``.

Design decisions, each anchored to a measured fact:

* **Grid over (sequence, KV head, split), never (sequence, query head).**
  GQA shares one K/V head across G query heads; a per-query-head program
  re-reads K/V G times (8–16x on the target models) — the kernel form of
  the ``enable_gqa`` mistake behind the in-tree 19x-wrong baseline
  (``kv_cache.py``, corrected 2026-07-25). And without the split axis the
  grid is B × H_kv programs — four CTAs at batch 1 — which measured
  4.8 GB/s on a ~200 GB/s card: the first version of this kernel was
  occupancy-starved, not bandwidth-bound. Splits are sized from the
  device's SM count at launch.
* **E4M3 decode is bit assembly, not arithmetic.** A normal's f32 bits are
  ``sign | (exp+120)<<23 | man<<20``, one bitcast; subnormals are a
  constant multiply. The first version used ``tl.exp2`` per element —
  16K SFU ops per tile iteration in a kernel whose budget is memory.
  This is also why FP8 can win where the fused NF4 kernel lost 11.6x:
  bit math, not a 16-entry LUT gather per element. The amax-scaled
  writer cannot emit inf/NaN (see ``fp8_kv``), so the NaN encodings are
  deliberately not special-cased.
* **Scales load at their natural [KTILE, NG] shape** and apply via a
  register reshape — loading them expanded to [KTILE, D] (a D-wide
  repeat) cost 32 KB of shared memory per tile and blew the sm_86 budget
  at D=128.
* **Head-packed variant built, measured, default-off.** One CTA per
  (sequence, split) consuming ALL kv heads reads every 512 B token line
  whole — and still LOST to the split kernel on the class card, 574 vs
  806 GB/s at B=25/T=4K: with a 96 MB L2 over a ~106 MB pool the
  scattered sibling-quarter reads mostly hit L2 anyway, while packing
  pays 4x tensor-core work and one heavyweight CTA's register pressure.
  Kept behind ``pack_heads=`` (same format, launch-time choice) for
  cards where the L2-to-pool ratio inverts the trade.

Block layout is the Phase-6/7 contract: 16-token rows, tokens-major
``[16, H_kv, D]`` E4M3 payload followed by fp32 scales — per (token,
head) for V, optionally per (token, head, D//group) for K (grouped key
scales are the measured quality fix that passes the ≤0.5% clause on both
probe models). K and V live in separate pool partitions; one block table
indexes both. Positions beyond a sequence's length are masked by token
index — garbage bytes in a partial tail block are loaded and discarded,
never scored.

Reduction order is fixed per (config, split count) — the
serving-tolerance side of the D0 determinism split.
"""
from __future__ import annotations

import os

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
        u32 = u.to(tl.uint32)
        exp = (u32 >> 3) & 0xF
        man = u32 & 0x7
        bits = ((u32 & 0x80) << 24) | ((exp + 120) << 23) | (man << 20)
        norm = bits.to(tl.float32, bitcast=True)
        sub = tl.where((u32 & 0x80) != 0, -1.0, 1.0) \
            * (man.to(tl.float32) * 0.001953125)
        return tl.where(exp == 0, sub, norm)

    @triton.jit
    def _fp8_paged_decode_split(
        q_ptr, kpool_u8, vpool_u8, kpool_f32, vpool_f32,
        table_ptr, seqlen_ptr,
        m_ptr, l_ptr, acc_ptr,
        o_ptr, counter_ptr,
        stride_qb, stride_qh, stride_tb,
        stride_ob, stride_oh,
        k_row_bytes, v_row_bytes,
        sm_scale, n_split,
        H_KV: tl.constexpr, G: tl.constexpr, D: tl.constexpr,
        BT: tl.constexpr, KTILE: tl.constexpr,
        NG_K: tl.constexpr, NG_V: tl.constexpr,
        BLOCK_G: tl.constexpr,
        FUSE_COMBINE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        s_id = pid % n_split
        bh = pid // n_split
        b = bh // H_KV
        h = bh % H_KV

        t_len = tl.load(seqlen_ptr + b)
        # per-sequence, KTILE-aligned span for this split
        n_tiles = (t_len + KTILE - 1) // KTILE
        tiles_per = (n_tiles + n_split - 1) // n_split
        t_lo = s_id * tiles_per * KTILE
        t_hi = tl.minimum(t_len, (s_id + 1) * tiles_per * KTILE)

        offs_g = tl.arange(0, BLOCK_G)
        offs_d = tl.arange(0, D)
        g_mask = offs_g < G

        q_ptrs = (q_ptr + b * stride_qb + (h * G + offs_g)[:, None]
                  * stride_qh + offs_d[None, :])
        q = tl.load(q_ptrs, mask=g_mask[:, None], other=0.0).to(tl.float32)

        m_i = tl.full([BLOCK_G], float("-inf"), tl.float32)
        l_i = tl.zeros([BLOCK_G], tl.float32)
        acc = tl.zeros([BLOCK_G, D], tl.float32)

        offs_t = tl.arange(0, KTILE)
        # tokens-major payload [BT, H, D] — measured against a heads-major
        # variant and KEPT: the H_kv programs of one sequence walk the
        # same 512 B lines together, and that L2 cooperation beat
        # per-CTA contiguity 86.3 to 74.2 GB/s at the same config.
        # offs_d carries contiguity/alignment hints so the compiler can
        # prove each row's D bytes form one aligned run THROUGH the
        # gather and emit wide vector loads.
        pay_off = ((offs_t % BT)[:, None] * (H_KV * D) + h * D
                   + tl.max_contiguous(tl.multiple_of(offs_d, D), D)[None, :])
        offs_gk = tl.arange(0, NG_K)
        offs_gv = tl.arange(0, NG_V)

        for start in range(t_lo, t_hi, KTILE):
            tok = start + offs_t
            t_mask = tok < t_hi
            blk = tl.load(table_ptr + b * stride_tb + tok // BT,
                          mask=t_mask, other=0)

            k_base = blk * k_row_bytes
            ku = tl.load(kpool_u8 + k_base[:, None] + pay_off,
                         mask=t_mask[:, None], other=0)
            k = _e4m3_to_f32(ku)
            ks_off = (k_base[:, None] // 4 + BT * H_KV * D // 4
                      + ((offs_t % BT) * H_KV + h)[:, None] * NG_K
                      + offs_gk[None, :])
            ks = tl.load(kpool_f32 + ks_off, mask=t_mask[:, None], other=1.0)
            k = tl.reshape(k, (KTILE, NG_K, D // NG_K))
            k = k * ks[:, :, None]
            k = tl.reshape(k, (KTILE, D))

            s = tl.dot(q, tl.trans(k)) * sm_scale
            s = tl.where(t_mask[None, :], s, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            alpha = tl.exp2((m_i - m_new) * 1.4426950408889634)
            p = tl.exp2((s - m_new[:, None]) * 1.4426950408889634)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None]

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

        # partials: [B, H_KV, n_split, BLOCK_G(, D)], fp32
        part = (bh * n_split + s_id) * BLOCK_G
        tl.store(m_ptr + part + offs_g, m_i)
        tl.store(l_ptr + part + offs_g, l_i)
        tl.store(acc_ptr + part * D + offs_g[:, None] * D + offs_d[None, :],
                 acc)

        if FUSE_COMBINE:
            # PREREG-f2-tail T1: the same stream-k fixup the packed and
            # f8dot kernels already carry -- the LAST split CTA for this
            # (b, h) combines the partials in place, with the identical
            # fixed 0..n_split reduction order as `_fp8_combine`, so the
            # result is bitwise-equal and only the launch disappears.
            arrived = tl.atomic_add(counter_ptr + bh, 1, sem="acq_rel")
            if arrived == n_split - 1:
                m_glob = tl.full([BLOCK_G], float("-inf"), tl.float32)
                for s_i in range(0, n_split):
                    m_s = tl.load(m_ptr + (bh * n_split + s_i) * BLOCK_G
                                  + offs_g)
                    m_glob = tl.maximum(m_glob, m_s)
                m_glob = tl.where(m_glob == float("-inf"), 0.0, m_glob)
                l_tot = tl.zeros([BLOCK_G], tl.float32)
                out = tl.zeros([BLOCK_G, D], tl.float32)
                for s_i in range(0, n_split):
                    base = (bh * n_split + s_i) * BLOCK_G
                    m_s = tl.load(m_ptr + base + offs_g)
                    l_s = tl.load(l_ptr + base + offs_g)
                    a_s = tl.load(acc_ptr + base * D + offs_g[:, None] * D
                                  + offs_d[None, :])
                    w = tl.exp2((m_s - m_glob) * 1.4426950408889634)
                    l_tot += l_s * w
                    out += a_s * w[:, None]
                out = out / l_tot[:, None]
                o_ptrs = (o_ptr + b * stride_ob
                          + (h * G + offs_g)[:, None] * stride_oh
                          + offs_d[None, :])
                tl.store(o_ptrs, out.to(o_ptr.dtype.element_ty),
                         mask=g_mask[:, None])
                tl.store(counter_ptr + bh, 0)

    @triton.jit
    def _fp8_paged_decode_packed(
        q_ptr, kpool_u8, vpool_u8, kpool_f32, vpool_f32,
        table_ptr, seqlen_ptr,
        m_ptr, l_ptr, acc_ptr,
        stride_qb, stride_qh, stride_tb,
        k_row_bytes, v_row_bytes,
        sm_scale, n_split,
        H_KV: tl.constexpr, G: tl.constexpr, D: tl.constexpr,
        BT: tl.constexpr,
        NG_K: tl.constexpr, NG_V: tl.constexpr,
        BLOCK_Q: tl.constexpr,
    ):
        # Head-PACKED variant: one CTA consumes ALL kv heads of its token
        # span, so every 512 B token line (H_KV * D payload bytes) is read
        # whole by the program that fetched it. The tokens-major split
        # kernel reads a 1/H_KV slice per CTA and leans on L2 to serve
        # siblings; measured on the quiet 5090 that recovered only half of
        # B_vram (806 GB/s / 51%) with the split axis saturated — line
        # utilization, not occupancy, was the remaining structural loss.
        #
        # The tile IS one block (BT tokens): payload loads collapse to one
        # contiguous [BT, H_KV*D] read per iteration and the scale tail to
        # one [BT, H_KV*NG] read — no modulo arithmetic survives into the
        # gather. Scores are ONE tensor-core dot [BLOCK_Q, D] x
        # [D, BT*H_KV] under a block-diagonal validity mask
        # (q-head-group == token's kv head); the wasted (H_KV-1)/H_KV of
        # the MACs are bf16 tensor-core ops in a kernel whose budget is
        # DRAM bytes — fp32 CUDA-core dots here would cost more than the
        # loads they serve.
        pid = tl.program_id(0)
        s_id = pid % n_split
        b = pid // n_split

        t_len = tl.load(seqlen_ptr + b)
        n_blocks = (t_len + BT - 1) // BT
        blocks_per = (n_blocks + n_split - 1) // n_split
        blk_lo = s_id * blocks_per
        blk_hi = tl.minimum(n_blocks, blk_lo + blocks_per)

        offs_q = tl.arange(0, BLOCK_Q)
        offs_d = tl.arange(0, D)
        q_mask = offs_q < G * H_KV
        q_ptrs = (q_ptr + b * stride_qb + offs_q[:, None] * stride_qh
                  + offs_d[None, :])
        qt = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0) \
            .to(tl.bfloat16)

        # column c of a tile = (token t, kv head h) with c = t*H_KV + h
        offs_c = tl.arange(0, BT * H_KV)
        c_head = offs_c % H_KV
        c_tok = offs_c // H_KV
        diag = (offs_q[:, None] // G) == c_head[None, :]

        offs_hd = tl.max_contiguous(
            tl.multiple_of(tl.arange(0, H_KV * D), H_KV * D), H_KV * D)
        offs_s = tl.arange(0, H_KV * NG_K)
        offs_sv = tl.arange(0, H_KV * NG_V)
        offs_t = tl.arange(0, BT)

        m_i = tl.full([BLOCK_Q], float("-inf"), tl.float32)
        l_i = tl.zeros([BLOCK_Q], tl.float32)
        acc = tl.zeros([BLOCK_Q, D], tl.float32)

        for blk_i in range(blk_lo, blk_hi):
            blk = tl.load(table_ptr + b * stride_tb + blk_i)
            t0 = blk_i * BT
            t_mask_c = (t0 + c_tok) < t_len

            k_base = blk * k_row_bytes
            ku = tl.load(kpool_u8 + k_base + offs_t[:, None] * (H_KV * D)
                         + offs_hd[None, :])
            k = _e4m3_to_f32(ku)
            ks = tl.load(kpool_f32 + k_base // 4 + BT * H_KV * D // 4
                         + offs_t[:, None] * (H_KV * NG_K)
                         + offs_s[None, :])
            k = tl.reshape(k, (BT, H_KV * NG_K, D // NG_K))
            k = k * ks[:, :, None]
            # [BT, H_KV*D] row-major == [(t, h), D] — the reshape is free
            k = tl.reshape(k, (BT * H_KV, D)).to(tl.bfloat16)

            s = tl.dot(qt, tl.trans(k)).to(tl.float32) * sm_scale
            s = tl.where(diag & t_mask_c[None, :], s, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            alpha = tl.exp2((m_i - m_new) * 1.4426950408889634)
            p = tl.exp2((s - m_new[:, None]) * 1.4426950408889634)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None]

            v_base = blk * v_row_bytes
            vu = tl.load(vpool_u8 + v_base + offs_t[:, None] * (H_KV * D)
                         + offs_hd[None, :])
            v = _e4m3_to_f32(vu)
            vs = tl.load(vpool_f32 + v_base // 4 + BT * H_KV * D // 4
                         + offs_t[:, None] * (H_KV * NG_V)
                         + offs_sv[None, :])
            v = tl.reshape(v, (BT, H_KV * NG_V, D // NG_V))
            v = v * vs[:, :, None]
            v = tl.reshape(v, (BT * H_KV, D)).to(tl.bfloat16)

            # invalid columns carry p == 0 (exp of -inf), so the full dot
            # credits each q row with only its own head's tokens
            acc += tl.dot(p.to(tl.bfloat16), v).to(tl.float32)
            m_i = m_new

        # partial layout matches the combine kernel at H_KV=1, G=Hq:
        # [B, n_split, BLOCK_Q(, D)]
        part = (b * n_split + s_id) * BLOCK_Q
        tl.store(m_ptr + part + offs_q, m_i)
        tl.store(l_ptr + part + offs_q, l_i)
        tl.store(acc_ptr + part * D + offs_q[:, None] * D + offs_d[None, :],
                 acc)

    @triton.jit
    def _q_sub(q_ptrs, offs_dg, off, g_mask, q_s):
        qf = tl.load(q_ptrs + (off + offs_dg)[None, :],
                     mask=g_mask[:, None], other=0.0).to(tl.float32)
        return (qf * q_s).to(tl.float8e4nv)

    @triton.jit
    def _k_scored(q8, kpool_u8, kpool_f32, k_base, ks_base, pay_row,
                  offs_dg, off, j, t_mask):
        """One scale group's contribution to the score tile: fp8 dot of
        the group's raw payload columns, scaled per token AFTER the dot
        (the group scale is constant across the dot's reduction axis)."""
        # payload is read exactly once per call per CTA: stream it
        # marked first-out of cache — letting it claim L1
        # displaces the q/scale tiles that DO get reused
        ku = tl.load(kpool_u8 + k_base[:, None] + pay_row
                     + (off + offs_dg)[None, :],
                     mask=t_mask[:, None], other=0,
                     eviction_policy="evict_first")
        k8 = ku.to(tl.float8e4nv, bitcast=True)
        ks = tl.load(kpool_f32 + ks_base + j, mask=t_mask, other=1.0)
        return tl.dot(q8, tl.trans(k8)) * ks[None, :]

    @triton.jit
    def _fp8_paged_decode_split_f8dot(
        q_ptr, kpool_u8, vpool_u8, kpool_f32, vpool_f32,
        table_ptr, seqlen_ptr,
        m_ptr, l_ptr, acc_ptr,
        o_ptr, counter_ptr,
        stride_qb, stride_qh, stride_tb,
        stride_ob, stride_oh,
        k_row_bytes, v_row_bytes,
        sm_scale, n_split,
        H_KV: tl.constexpr, G: tl.constexpr, D: tl.constexpr,
        BT: tl.constexpr, KTILE: tl.constexpr,
        NG_K: tl.constexpr,
        BLOCK_G: tl.constexpr,
        FUSE_COMBINE: tl.constexpr,
        LAYOUT_HEADS: tl.constexpr,
    ):
        # FP8-tensor-core variant: the E4M3 payload is BITCAST and fed to
        # fp8 MMA units — no decode at all. The sweep surface demanded
        # this: stages flat (not latency-bound), w=2 > w=4, kt=32 > kt=64
        # by 65% — the wall was the ~40 integer ops per PAYLOAD ELEMENT of
        # bit-assembly decode + scale expansion, i.e. per-SM issue
        # throughput, and the fix is to stop touching payload elements
        # with ALU entirely. Scales fold AFTER the dots, onto [G, KTILE]
        # score tiles (D-fold fewer elements):
        #
        # * K, grouped scales: one sub-dot per scale group j over its
        #   D/NG_K columns, then s += dot_j * ks_j[None, :] — algebraically
        #   exact because a group's scale is constant across the dot's
        #   reduction axis.
        # * q: scaled to fill e4m3's range once per program, the factor
        #   divided back out of sm_scale (fa3's q-scaling trick).
        # * V, per-row scales: fold into P — p8 = p * (vs * C), with the
        #   tile-dependent range factor C = 448/max(vs) keeping products
        #   out of e4m3's subnormal floor (raw p*vs ~ 1e-3 would round to
        #   1-3 mantissa bits). C changes per tile; the correction rides
        #   the online-softmax rescale that already multiplies acc by
        #   alpha each iteration, and divides out once at the end — zero
        #   extra D-wide work.
        # * l stays the f32 sum of UNQUANTIZED p, m unchanged: only the
        #   numerator pays fp8 rounding, and it averages over the tile.
        #
        # Numerics: q and p each carry e4m3's ~6% per-element rounding
        # into the dots (the STORED bytes are identical to the decode
        # path's). This is the documented serving-tolerance side of
        # invariant 4' — measured in test_fp8_paged_attn against the f32
        # oracle — and quality certification of fp8 COMPUTE (as opposed
        # to fp8 STORAGE, which the G7 oracle measured) is explicitly
        # still owed if this mode becomes the default.
        P448: tl.constexpr = 448.0
        DG: tl.constexpr = D // NG_K
        pid = tl.program_id(0)
        s_id = pid % n_split
        bh = pid // n_split
        b = bh // H_KV
        h = bh % H_KV

        t_len = tl.load(seqlen_ptr + b)
        n_tiles = (t_len + KTILE - 1) // KTILE
        tiles_per = (n_tiles + n_split - 1) // n_split
        t_lo = s_id * tiles_per * KTILE
        t_hi = tl.minimum(t_len, (s_id + 1) * tiles_per * KTILE)

        offs_g = tl.arange(0, BLOCK_G)
        offs_d = tl.arange(0, D)
        offs_dg = tl.max_contiguous(
            tl.multiple_of(tl.arange(0, DG), DG), DG)
        g_mask = offs_g < G

        q_ptrs = (q_ptr + b * stride_qb + (h * G + offs_g)[:, None]
                  * stride_qh)
        qf = tl.load(q_ptrs + offs_d[None, :], mask=g_mask[:, None],
                     other=0.0).to(tl.float32)
        q_amax = tl.max(tl.abs(qf))
        q_s = tl.where(q_amax > 0, P448 / q_amax, 1.0)
        # static q sub-tiles, one per scale group, hoisted out of the
        # token loop (Triton's tracer has no lists — NG_K in {1, 2, 4}
        # is unrolled by constexpr guards, wrapper-asserted)
        q8_0 = _q_sub(q_ptrs, offs_dg, 0 * DG, g_mask, q_s)
        q8_1 = q8_0
        q8_2 = q8_0
        q8_3 = q8_0
        if NG_K >= 2:
            q8_1 = _q_sub(q_ptrs, offs_dg, 1 * DG, g_mask, q_s)
        if NG_K >= 4:
            q8_2 = _q_sub(q_ptrs, offs_dg, 2 * DG, g_mask, q_s)
            q8_3 = _q_sub(q_ptrs, offs_dg, 3 * DG, g_mask, q_s)
        scale_fold = sm_scale / q_s

        m_i = tl.full([BLOCK_G], float("-inf"), tl.float32)
        l_i = tl.zeros([BLOCK_G], tl.float32)
        acc = tl.zeros([BLOCK_G, D], tl.float32)
        c_prev = 1.0

        offs_t = tl.arange(0, KTILE)
        # heads-major gives this CTA one PRIVATE contiguous BT*D-byte run
        # per block (plus a contiguous scale run); tokens-major interleaves
        # heads within each 512 B token line and leans on sibling CTAs
        # hitting L2 — which layout wins depends on the kernel's
        # bottleneck, so both exist and both are measured.
        if LAYOUT_HEADS:
            pay_row = h * (BT * D) + (offs_t % BT)[:, None] * D
        else:
            pay_row = (offs_t % BT)[:, None] * (H_KV * D) + h * D
        pay_v = pay_row + tl.max_contiguous(
            tl.multiple_of(offs_d, D), D)[None, :]

        for start in range(t_lo, t_hi, KTILE):
            tok = start + offs_t
            t_mask = tok < t_hi
            blk = tl.load(table_ptr + b * stride_tb + tok // BT,
                          mask=t_mask, other=0)

            k_base = blk * k_row_bytes
            if LAYOUT_HEADS:
                ks_base = (k_base // 4 + BT * H_KV * D // 4
                           + (h * BT + (offs_t % BT)) * NG_K)
            else:
                ks_base = (k_base // 4 + BT * H_KV * D // 4
                           + ((offs_t % BT) * H_KV + h) * NG_K)
            s = _k_scored(q8_0, kpool_u8, kpool_f32, k_base, ks_base,
                          pay_row, offs_dg, 0 * DG, 0, t_mask)
            if NG_K >= 2:
                s += _k_scored(q8_1, kpool_u8, kpool_f32, k_base, ks_base,
                               pay_row, offs_dg, 1 * DG, 1, t_mask)
            if NG_K >= 4:
                s += _k_scored(q8_2, kpool_u8, kpool_f32, k_base, ks_base,
                               pay_row, offs_dg, 2 * DG, 2, t_mask)
                s += _k_scored(q8_3, kpool_u8, kpool_f32, k_base, ks_base,
                               pay_row, offs_dg, 3 * DG, 3, t_mask)
            s = s * scale_fold
            s = tl.where(t_mask[None, :], s, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            alpha = tl.exp2((m_i - m_new) * 1.4426950408889634)
            p = tl.exp2((s - m_new[:, None]) * 1.4426950408889634)
            l_i = l_i * alpha + tl.sum(p, axis=1)

            v_base = blk * v_row_bytes
            vu = tl.load(vpool_u8 + v_base[:, None] + pay_v,
                         mask=t_mask[:, None], other=0,
                         eviction_policy="evict_first")
            v8 = vu.to(tl.float8e4nv, bitcast=True)
            if LAYOUT_HEADS:
                vs_off = h * BT + (offs_t % BT)
            else:
                vs_off = (offs_t % BT) * H_KV + h
            vs = tl.load(vpool_f32 + v_base // 4 + BT * H_KV * D // 4
                         + vs_off,
                         mask=t_mask, other=1.0)
            # masked-out lanes carry p == 0, so their vs value is inert;
            # exclude them from the range factor anyway (other=1.0 above
            # would otherwise drag C toward 448)
            c_new = P448 / tl.max(tl.where(t_mask, vs, 0.0))
            p8 = (p * (vs * c_new)[None, :]).to(tl.float8e4nv)
            acc = acc * (alpha * (c_new / c_prev))[:, None] \
                + tl.dot(p8, v8)
            c_prev = c_new
            m_i = m_new

        acc = acc / c_prev
        part = (bh * n_split + s_id) * BLOCK_G
        tl.store(m_ptr + part + offs_g, m_i)
        tl.store(l_ptr + part + offs_g, l_i)
        tl.store(acc_ptr + part * D + offs_g[:, None] * D + offs_d[None, :],
                 acc)

        if FUSE_COMBINE:
            # stream-k-style fixup: the LAST split CTA to arrive for this
            # (b, h) combines all partials in place. The acq_rel atomic
            # orders each CTA's partial stores before its arrival and the
            # last arriver's loads after — partials are seconds-old and
            # combine reads come from L2, so the separate combine kernel's
            # launch + DRAM round-trip disappears. Reduction order is the
            # same fixed 0..n_split loop as the standalone kernel: which
            # CTA runs it changes, what it computes does not (the
            # serving-determinism contract is per config+split count).
            arrived = tl.atomic_add(counter_ptr + bh, 1, sem="acq_rel")
            if arrived == n_split - 1:
                m_glob = tl.full([BLOCK_G], float("-inf"), tl.float32)
                for s_i in range(0, n_split):
                    m_s = tl.load(m_ptr + (bh * n_split + s_i) * BLOCK_G
                                  + offs_g)
                    m_glob = tl.maximum(m_glob, m_s)
                m_glob = tl.where(m_glob == float("-inf"), 0.0, m_glob)
                l_tot = tl.zeros([BLOCK_G], tl.float32)
                out = tl.zeros([BLOCK_G, D], tl.float32)
                for s_i in range(0, n_split):
                    base = (bh * n_split + s_i) * BLOCK_G
                    m_s = tl.load(m_ptr + base + offs_g)
                    l_s = tl.load(l_ptr + base + offs_g)
                    a_s = tl.load(acc_ptr + base * D + offs_g[:, None] * D
                                  + offs_d[None, :])
                    w = tl.exp2((m_s - m_glob) * 1.4426950408889634)
                    l_tot += l_s * w
                    out += a_s * w[:, None]
                out = out / l_tot[:, None]
                o_ptrs = (o_ptr + b * stride_ob
                          + (h * G + offs_g)[:, None] * stride_oh
                          + offs_d[None, :])
                tl.store(o_ptrs, out.to(o_ptr.dtype.element_ty),
                         mask=g_mask[:, None])
                # self-cleaning: reset this (b, h) slot so the cached
                # counter buffer needs no per-call zeroing (a per-call
                # torch.zeros memset launch measured ~4% of the whole
                # kernel at serving shapes). Stream order makes this safe:
                # the next call's atomics on this stream come after.
                tl.store(counter_ptr + bh, 0)

    @triton.jit
    def _kc_scored(q8, kpool_u8, kpool_f32, k_base, ks_base, pay_c,
                   offs_dg, off, j, t_mask_c):
        """Packed-layout sub-dot: columns are (token, kv head) pairs; the
        four group loads of one iteration cover each 512 B payload line
        completely and adjacently in time."""
        ku = tl.load(kpool_u8 + k_base + pay_c[:, None]
                     + (off + offs_dg)[None, :],
                     mask=t_mask_c[:, None], other=0,
                     eviction_policy="evict_first")
        k8 = ku.to(tl.float8e4nv, bitcast=True)
        ks = tl.load(kpool_f32 + ks_base + j, mask=t_mask_c, other=1.0)
        return tl.dot(q8, tl.trans(k8)) * ks[None, :]

    @triton.jit
    def _fp8_paged_decode_packed_f8(
        q_ptr, kpool_u8, vpool_u8, kpool_f32, vpool_f32,
        table_ptr, seqlen_ptr,
        m_ptr, l_ptr, acc_ptr,
        o_ptr, counter_ptr,
        stride_qb, stride_qh, stride_tb,
        stride_ob, stride_oh,
        k_row_bytes, v_row_bytes,
        sm_scale, n_split,
        H_KV: tl.constexpr, G: tl.constexpr, D: tl.constexpr,
        BT: tl.constexpr,
        NG_K: tl.constexpr,
        BLOCK_Q: tl.constexpr,
        FUSE_COMBINE: tl.constexpr,
    ):
        # Packed structure + fp8 MMA: the combination neither parent could
        # run. The packed layout (one CTA owns ALL kv heads of its span,
        # whole 512 B token lines consumed where they are fetched) lost in
        # f32-decode mode to register pressure and 4x CUDA-core dot work;
        # the f8dot split kernel deleted the decode but still reads
        # quarter-lines per CTA and plateaus ~69% on DRAM page locality.
        # Here the tile is one block: payload loads are line-perfect
        # [BT, H*D] contiguous runs, scores are block-diagonal fp8
        # sub-dots at M=BLOCK_Q (full MMA tiles, not the split kernel's
        # M=16), and the e4m3 register tiles are a quarter the size the
        # f32 decode needed. Scale folds are the f8dot machinery: group
        # scales fold per COLUMN c=(t,h) after each sub-dot, V scales
        # fold into P under the per-tile range factor C riding the
        # online-softmax rescale.
        P448: tl.constexpr = 448.0
        DG: tl.constexpr = D // NG_K
        C: tl.constexpr = BT * H_KV
        pid = tl.program_id(0)
        s_id = pid % n_split
        b = pid // n_split

        t_len = tl.load(seqlen_ptr + b)
        n_blocks = (t_len + BT - 1) // BT
        blocks_per = (n_blocks + n_split - 1) // n_split
        blk_lo = s_id * blocks_per
        blk_hi = tl.minimum(n_blocks, blk_lo + blocks_per)

        offs_q = tl.arange(0, BLOCK_Q)
        offs_d = tl.arange(0, D)
        offs_dg = tl.max_contiguous(
            tl.multiple_of(tl.arange(0, DG), DG), DG)
        q_mask = offs_q < G * H_KV
        q_ptrs = q_ptr + b * stride_qb + offs_q[:, None] * stride_qh

        qf = tl.load(q_ptrs + offs_d[None, :], mask=q_mask[:, None],
                     other=0.0).to(tl.float32)
        q_amax = tl.max(tl.abs(qf))
        q_s = tl.where(q_amax > 0, P448 / q_amax, 1.0)
        q8_0 = _q_sub(q_ptrs, offs_dg, 0 * DG, q_mask, q_s)
        q8_1 = q8_0
        q8_2 = q8_0
        q8_3 = q8_0
        if NG_K >= 2:
            q8_1 = _q_sub(q_ptrs, offs_dg, 1 * DG, q_mask, q_s)
        if NG_K >= 4:
            q8_2 = _q_sub(q_ptrs, offs_dg, 2 * DG, q_mask, q_s)
            q8_3 = _q_sub(q_ptrs, offs_dg, 3 * DG, q_mask, q_s)
        scale_fold = sm_scale / q_s

        # column c of a score tile = (token t, kv head h), c = t*H_KV + h
        offs_c = tl.arange(0, C)
        c_head = offs_c % H_KV
        c_tok = offs_c // H_KV
        diag = (offs_q[:, None] // G) == c_head[None, :]
        # payload byte offset of column c's group-j slice inside a row
        pay_c = c_tok * (H_KV * D) + c_head * D
        offs_vhd = tl.max_contiguous(
            tl.multiple_of(tl.arange(0, H_KV * D), H_KV * D), H_KV * D)

        m_i = tl.full([BLOCK_Q], float("-inf"), tl.float32)
        l_i = tl.zeros([BLOCK_Q], tl.float32)
        acc = tl.zeros([BLOCK_Q, D], tl.float32)
        c_prev = 1.0

        for blk_i in range(blk_lo, blk_hi):
            blk = tl.load(table_ptr + b * stride_tb + blk_i)
            t_mask_c = (blk_i * BT + c_tok) < t_len

            k_base = blk * k_row_bytes
            ks_base = k_base // 4 + BT * H_KV * D // 4 + offs_c * NG_K
            s = _kc_scored(q8_0, kpool_u8, kpool_f32, k_base, ks_base,
                           pay_c, offs_dg, 0 * DG, 0, t_mask_c)
            if NG_K >= 2:
                s += _kc_scored(q8_1, kpool_u8, kpool_f32, k_base, ks_base,
                                pay_c, offs_dg, 1 * DG, 1, t_mask_c)
            if NG_K >= 4:
                s += _kc_scored(q8_2, kpool_u8, kpool_f32, k_base, ks_base,
                                pay_c, offs_dg, 2 * DG, 2, t_mask_c)
                s += _kc_scored(q8_3, kpool_u8, kpool_f32, k_base, ks_base,
                                pay_c, offs_dg, 3 * DG, 3, t_mask_c)
            s = s * scale_fold
            s = tl.where(diag & t_mask_c[None, :], s, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            alpha = tl.exp2((m_i - m_new) * 1.4426950408889634)
            pr = tl.exp2((s - m_new[:, None]) * 1.4426950408889634)
            l_i = l_i * alpha + tl.sum(pr, axis=1)

            v_base = blk * v_row_bytes
            vu = tl.load(vpool_u8 + v_base
                         + tl.arange(0, BT)[:, None] * (H_KV * D)
                         + offs_vhd[None, :],
                         mask=((blk_i * BT + tl.arange(0, BT))
                               < t_len)[:, None], other=0,
                         eviction_policy="evict_first")
            v8 = tl.reshape(vu, (C, D)).to(tl.float8e4nv, bitcast=True)
            vs = tl.load(vpool_f32 + v_base // 4 + BT * H_KV * D // 4
                         + offs_c, mask=t_mask_c, other=1.0)
            c_new = P448 / tl.max(tl.where(t_mask_c, vs, 0.0))
            p8 = (pr * (vs * c_new)[None, :]).to(tl.float8e4nv)
            acc = acc * (alpha * (c_new / c_prev))[:, None]                 + tl.dot(p8, v8)
            c_prev = c_new
            m_i = m_new

        acc = acc / c_prev
        part = (b * n_split + s_id) * BLOCK_Q
        tl.store(m_ptr + part + offs_q, m_i)
        tl.store(l_ptr + part + offs_q, l_i)
        tl.store(acc_ptr + part * D + offs_q[:, None] * D + offs_d[None, :],
                 acc)

        if FUSE_COMBINE:
            arrived = tl.atomic_add(counter_ptr + b, 1, sem="acq_rel")
            if arrived == n_split - 1:
                m_glob = tl.full([BLOCK_Q], float("-inf"), tl.float32)
                for s_i in range(0, n_split):
                    m_s = tl.load(m_ptr + (b * n_split + s_i) * BLOCK_Q
                                  + offs_q)
                    m_glob = tl.maximum(m_glob, m_s)
                m_glob = tl.where(m_glob == float("-inf"), 0.0, m_glob)
                l_tot = tl.zeros([BLOCK_Q], tl.float32)
                out = tl.zeros([BLOCK_Q, D], tl.float32)
                for s_i in range(0, n_split):
                    base = (b * n_split + s_i) * BLOCK_Q
                    m_s = tl.load(m_ptr + base + offs_q)
                    l_s = tl.load(l_ptr + base + offs_q)
                    a_s = tl.load(acc_ptr + base * D + offs_q[:, None] * D
                                  + offs_d[None, :])
                    w = tl.exp2((m_s - m_glob) * 1.4426950408889634)
                    l_tot += l_s * w
                    out += a_s * w[:, None]
                out = out / l_tot[:, None]
                o_ptrs = (o_ptr + b * stride_ob
                          + offs_q[:, None] * stride_oh + offs_d[None, :])
                tl.store(o_ptrs, out.to(o_ptr.dtype.element_ty),
                         mask=q_mask[:, None])
                tl.store(counter_ptr + b, 0)

    @triton.jit
    def _fp8_combine(
        m_ptr, l_ptr, acc_ptr, o_ptr,
        stride_ob, stride_oh,
        n_split,
        H_KV: tl.constexpr, G: tl.constexpr, D: tl.constexpr,
        BLOCK_G: tl.constexpr,
    ):
        bh = tl.program_id(0)
        b = bh // H_KV
        h = bh % H_KV
        offs_g = tl.arange(0, BLOCK_G)
        offs_d = tl.arange(0, D)
        g_mask = offs_g < G

        m_glob = tl.full([BLOCK_G], float("-inf"), tl.float32)
        for s in range(0, n_split):
            m_s = tl.load(m_ptr + (bh * n_split + s) * BLOCK_G + offs_g)
            m_glob = tl.maximum(m_glob, m_s)
        # an all-empty group (t_len 0) never occurs in decode; guard anyway
        m_glob = tl.where(m_glob == float("-inf"), 0.0, m_glob)

        l_tot = tl.zeros([BLOCK_G], tl.float32)
        out = tl.zeros([BLOCK_G, D], tl.float32)
        for s in range(0, n_split):
            base = (bh * n_split + s) * BLOCK_G
            m_s = tl.load(m_ptr + base + offs_g)
            l_s = tl.load(l_ptr + base + offs_g)
            a_s = tl.load(acc_ptr + base * D + offs_g[:, None] * D
                          + offs_d[None, :])
            w = tl.exp2((m_s - m_glob) * 1.4426950408889634)
            l_tot += l_s * w
            out += a_s * w[:, None]

        out = out / l_tot[:, None]
        o_ptrs = (o_ptr + b * stride_ob + (h * G + offs_g)[:, None]
                  * stride_oh + offs_d[None, :])
        tl.store(o_ptrs, out.to(o_ptr.dtype.element_ty),
                 mask=g_mask[:, None])


_FUSE_COUNTERS: dict = {}



def _f32_fuse_default() -> bool:
    """Effective fuse_combine default for the F32 SPLIT path when the
    caller passes None: ON since RESULTS-f2-tail (PARTIAL, ships --
    +0.041 ms under a 0.001 ms A/A, token-identical over 127 graph
    steps; receipts in kernel/receipts-f2/). GNF4_F32_FUSE_COMBINE=0
    is the rollback. The packed and fp8-compute paths certified their
    fused combine in their own cycle and keep an unconditional True
    default."""
    return os.environ.get("GNF4_F32_FUSE_COMBINE", "1") == "1"


#: PREREG-m3 mechanism receipt, the attention half. Same reasoning as
#: ``nf4_grouped._DISPATCH_COUNTS``: ``GNF4_ATTN_COMPUTE=fp8`` is a
#: request, and the caller may also pass ``compute=`` explicitly, so
#: reading the environment records an intention rather than an event.
#: Incremented where the mode is actually resolved, once per decode
#: call (once per capture under CUDA graphs).
_COMPUTE_COUNTS = {"f32": 0, "fp8": 0}


def compute_counts() -> dict:
    """Copy of the decode-attention compute-mode tally (PREREG-m3)."""
    return dict(_COMPUTE_COUNTS)


def reset_compute_counts() -> None:
    """Zero the tally, so a caller can scope it to one window."""
    for _k in _COMPUTE_COUNTS:
        _COMPUTE_COUNTS[_k] = 0


def fp8_compute_unsupported(q, head_dim: int, k_groups: int,
                            v_groups: int, *, pack_heads: bool = False,
                            block_tokens: int = 16, n_kv_heads: int = 1,
                            ktile: int | None = None) -> str | None:
    """Why fp8-COMPUTE cannot run on this call, or None if it can.

    SINGLE SOURCE OF TRUTH. The default selector below and the hard
    preconditions on the fp8 path both go through this, so the guard
    that decides "is fp8 safe here?" cannot drift away from the
    asserts that enforce it. Two copies of a predicate is how a
    default starts silently choosing a path its own asserts reject.

``ktile`` is checked only when the CALLER SUPPLIED it. An earlier
    version skipped it entirely, reasoning that ktile is derived from
    the resolved mode (64 for fp8) and so cannot gate that mode
    without circularity. That was wrong: ktile is a kwarg, and it is
    only derived when the caller passes None. A caller who passes
    ``ktile=16`` would have had the default select fp8 and then hit
    the leftover ``ktile >= 32`` assert on a call that previously ran
    f32 (review, gnf4#291). Supplied means it is an input like any
    other; None means the constraint is vacuous.

    ``layout`` genuinely cannot be a reason to refuse fp8: the f32
    path requires ``tokens`` while fp8 accepts ``tokens`` or
    ``heads``, so anything fp8 rejects, f32 rejects too.
    """
    if v_groups != 1:
        return ("fp8 compute folds V scales into P: per-row only "
                f"(v_groups={v_groups})")
    if k_groups not in (1, 2, 4):
        return f"fp8 compute unrolls k_groups in (1, 2, 4), got {k_groups}"
    if head_dim // k_groups < 32:
        return ("fp8 dot needs >=32-wide key scale groups "
                f"(head_dim {head_dim} // k_groups {k_groups})")
    if q.dtype not in (torch.bfloat16, torch.float16):
        return f"fp8 compute loads q as bf16/fp16, got {q.dtype}"
    if torch.cuda.get_device_capability(q.device) < (8, 9):
        cc = torch.cuda.get_device_capability(q.device)
        return f"fp8 tensor-core dots need sm_89+, this device is sm_{cc[0]}{cc[1]}"
    # The PACKED fp8 branch reduces over BT*H_kv instead of ktile and
    # so carries a precondition the split branch does not. A guard
    # that covered only the split path would still let the default
    # select fp8 into an assert -- there are TWO fp8 branches.
    if ktile is not None and ktile < 32:
        return f"fp8 P.V dot reduces over ktile: needs >= 32, got {ktile}"
    if pack_heads and block_tokens * n_kv_heads < 32:
        return ("packed fp8 P.V dot reduces over BT*H_kv: needs >= 32, "
                f"got {block_tokens} * {n_kv_heads}")
    return None


def _compute_default(q=None, head_dim: int | None = None,
                     k_groups: int = 1, v_groups: int = 1, *,
                     pack_heads: bool = False, block_tokens: int = 16,
                     n_kv_heads: int = 1,
                     ktile: int | None = None) -> str:
    """Which compute mode an unset ``GNF4_ATTN_COMPUTE`` selects.

    **fp8 is the default as of RESULTS-m3-default-on** (PASS: 8192
    scored tokens, dppl -0.0058 against a +-0.05 bar, 0.22 ms off the
    step). It was opt-in through K8 because the quality question was
    open; M3 closed it.

    But PASS licensed the default on QUALITY AND SPEED, not on
    APPLICABILITY -- the fp8 path asserts sm_89+, ``v_groups == 1``
    and more, none of which that cycle varied. Flipping
    unconditionally would turn a working f32 install into an
    AssertionError on every pre-Ada GPU. So the default is
    capability-conditional: fp8 where fp8 can run, the certified f32
    path otherwise.

    An EXPLICIT ``GNF4_ATTN_COMPUTE=fp8`` is never downgraded. A user
    who names the mode gets it, and the path's asserts tell them it is
    unavailable -- silently handing them f32 under the name they asked
    for is how a benchmark arm gets mislabelled
    ([[identical-output-is-not-identical-computation]]).

    An unrecognised value RAISES rather than falling back: a typo'd
    ``FP8`` that silently ran f32 would be recorded as an fp8 arm and
    mis-attribute the whole cycle.
    """
    v = os.environ.get("GNF4_ATTN_COMPUTE")
    if v is not None:
        if v not in ("f32", "fp8"):
            raise ValueError(
                f"GNF4_ATTN_COMPUTE={v!r} is not a compute mode; expected "
                "'f32' or 'fp8'. Refusing rather than silently running "
                "f32 and letting it be recorded as the fp8 arm.")
        return v
    if q is None or head_dim is None:
        # no call context to judge applicability against; the caller
        # gets the certified path rather than a guess it cannot check
        return "f32"
    return ("f32" if fp8_compute_unsupported(
        q, head_dim, k_groups, v_groups, pack_heads=pack_heads,
        block_tokens=block_tokens, n_kv_heads=n_kv_heads,
        ktile=ktile) else "fp8")

def _fuse_counters(n: int, device) -> torch.Tensor:
    """Zeroed-once arrival counters for the fused combine; slots are
    reset by the combining CTA itself, so reuse needs no memset (a
    per-call torch.zeros measured ~4% of the whole kernel).

    Keyed by (device, STREAM, n): launches on one stream are ordered, so
    sharing a buffer along a stream is race-free, while two concurrent
    streams of the same shape get distinct buffers — a shared slot across
    streams could combine early or not at all (review). The remaining
    theoretical staleness — a launch that increments but never resets
    because it ABORTED mid-flight — is unreachable in practice: a CUDA
    fault poisons the context and no later launch runs normally; for
    belt-and-braces (fresh process reusing a serialized context, exotic
    capture replays) `reset_fuse_counters()` clears the cache.
    """
    key = (device, torch.cuda.current_stream(device).cuda_stream, n)
    buf = _FUSE_COUNTERS.get(key)
    if buf is None:
        buf = torch.zeros(n, dtype=torch.int32, device=device)
        _FUSE_COUNTERS[key] = buf
    return buf


def reset_fuse_counters() -> None:
    """Drop all cached arrival counters (they re-create zeroed)."""
    _FUSE_COUNTERS.clear()


def fp8_paged_decode_attention(q, k_pool, v_pool, block_table, seq_lens, *,
                               n_kv_heads: int, head_dim: int,
                               block_tokens: int = 16,
                               k_groups: int = 1, v_groups: int = 1,
                               ktile: int | None = None,
                               n_split: int | None = None,
                               sm_scale: float | None = None,
                               num_warps: int | None = None,
                               num_stages: int = 3,
                               pack_heads: bool = False,
                               compute: str | None = None,
                               fuse_combine: bool | None = None,
                               layout: str = "tokens"):
    """Decode attention over packed FP8 KV pool rows.

    q            [B, H_q, D] bf16/fp16 — ONE decode token per sequence.
    k_pool/v_pool  flat uint8 pools of packed block rows (payload then
                 scales, ``fp8_kv.pack_kv_block`` layout).
    block_table  [B, MAX_BLOCKS] int32 — row index per 16-token block;
                 one table serves both pools (lockstep appends).
    seq_lens     [B] int32 tokens per sequence.
    n_split      sequence partitions per (seq, kv-head); default sizes the
                 grid to ~4 CTAs per SM (the first version's B×H_kv grid
                 was 4 CTAs total at batch 1 — occupancy, not bandwidth,
                 was its 40x problem).
    pack_heads   one CTA per (sequence, split) consuming ALL kv heads —
                 whole-line reads, tensor-core block-diagonal scores; the
                 tile is one BT-token block and ``ktile`` is ignored.
    compute      None (default): resolved from ``GNF4_ATTN_COMPUTE``
                 (PREREG-k8), which is "f32" unless exported — an
                 explicit argument always wins.
                 "f32": E4M3 decoded in registers, tf32 dots —
                 the bit-exact-est serving path. "fp8": payload bytes are
                 BITCAST into fp8 tensor-core dots, scales folded onto the
                 post-dot score tiles; requires sm_89+, ``v_groups == 1``,
                 ``head_dim // k_groups >= 32`` and ``ktile >= 32``.
                 Documented-tolerance path (q and p each pay one e4m3
                 rounding); fp8 COMPUTE quality is owed separately if it
                 becomes the default — the G7 oracle certified storage.

    Returns [B, H_q, D] in q's dtype.
    """
    assert paged_attn_available(), "needs CUDA + triton"
    if compute is None:
        # ktile here is still the RAW kwarg -- derivation happens
        # below, after the mode is known
        compute = _compute_default(q, head_dim, k_groups, v_groups,
                                   pack_heads=pack_heads,
                                   block_tokens=block_tokens,
                                   n_kv_heads=n_kv_heads, ktile=ktile)
    elif compute not in ("f32", "fp8"):
        raise ValueError(f"compute={compute!r}; expected 'f32' or 'fp8'")
    _COMPUTE_COUNTS[compute] += 1
    if fuse_combine is None:
        # per-path default: packed/fp8-compute fused combine is
        # certified; the f32 split port is under PREREG-f2-tail T1 and
        # stays opt-in until that verdict (explicit True/False from the
        # caller always wins)
        fuse_combine = True if compute == "fp8" else _f32_fuse_default()
    B, Hq, D = q.shape
    assert D == head_dim and Hq % n_kv_heads == 0
    G = Hq // n_kv_heads
    # per-mode measured defaults: the fp8 path wants bigger tiles and
    # more warps (its per-element decode is gone; the f32 path is
    # issue-bound and wants the opposite)
    if ktile is None:
        ktile = 64 if compute == "fp8" else 32
    if num_warps is None:
        num_warps = 4 if compute == "fp8" else 2
    block_g = max(16, triton.next_power_of_2(G))
    assert D % k_groups == 0 and D % v_groups == 0
    assert ktile % block_tokens == 0
    if sm_scale is None:
        sm_scale = D ** -0.5
    if n_split is None:
        # f32 decode: ~8 CTAs/SM (the sweep plateau — fewer starves the
        # serial tile loop, the B=1..8 latency pedestal; 2x more pays
        # q reload + combine for nothing). f8dot: ~2 CTAs/SM — with the
        # per-element decode deleted the kernel streams, longer spans
        # pipeline fine, and extra splits only buy partial overhead
        # (sweep winners sat at 2-4 splits at serving batches).
        sms = torch.cuda.get_device_properties(q.device).multi_processor_count
        ctas_per = B if pack_heads else B * n_kv_heads
        span = block_tokens if pack_heads else ktile
        # fp8: ~2.2 CTAs/SM — the qualified-box surface's winners land
        # exactly there (B=32 -> 2 splits, B=25 -> 3, the non-pow2 split
        # that fixed B=25's wave quantization, +6.9%)
        if compute == "fp8":
            want = max(1, (11 * sms) // (5 * max(1, ctas_per)))
        else:
            want = max(1, (8 * sms) // max(1, ctas_per))
        # capacity from the block table, NOT seq_lens.max(): the max()
        # is a device reduction + full sync PER CALL — measured ~12 us
        # on a ~123 us kernel (every "shipped defaults" row before this
        # fix paid ~10% harness-visible tax, and in a real decode loop
        # the sync also breaks pipelining). Capacity over-estimates
        # useful splits only for sequences far shorter than their table;
        # empty splits exit at t_lo >= t_hi and cost a combine slot.
        cap_tokens = int(block_table.shape[1]) * block_tokens
        max_useful = max(1, (cap_tokens + span - 1) // span)
        n_split = int(min(32, want, max_useful))

    from fp8_kv import kv_block_bytes
    k_row = kv_block_bytes(block_tokens, n_kv_heads, head_dim) \
        + block_tokens * n_kv_heads * 4 * (k_groups - 1)
    v_row = kv_block_bytes(block_tokens, n_kv_heads, head_dim) \
        + block_tokens * n_kv_heads * 4 * (v_groups - 1)
    assert k_pool.numel() % k_row == 0 and v_pool.numel() % v_row == 0

    if pack_heads:
        # combine expects the packed partial layout at H_KV=1, G=Hq
        assert layout == "tokens", \
            "packed variants read whole tokens-major lines by design"
        assert q.dtype == torch.bfloat16, "packed path loads q as bf16"
        block_q = max(16, triton.next_power_of_2(Hq))
        part = B * n_split * block_q
        m_buf = torch.empty(part, dtype=torch.float32, device=q.device)
        l_buf = torch.empty(part, dtype=torch.float32, device=q.device)
        acc_buf = torch.empty(part * D, dtype=torch.float32,
                              device=q.device)
        o = torch.empty_like(q)
        if compute == "fp8":
            # same predicate as the split branch and the default
            # selector -- one definition, three call sites
            _why = fp8_compute_unsupported(
                q, D, k_groups, v_groups, pack_heads=True,
                block_tokens=block_tokens, n_kv_heads=n_kv_heads)
            assert _why is None, _why
            counters = _fuse_counters(B, q.device) if fuse_combine \
                else m_buf
            _fp8_paged_decode_packed_f8[(B * n_split,)](
                q, k_pool, v_pool,
                k_pool.view(torch.float32), v_pool.view(torch.float32),
                block_table, seq_lens.to(torch.int32),
                m_buf, l_buf, acc_buf,
                o, counters,
                q.stride(0), q.stride(1), block_table.stride(0),
                o.stride(0), o.stride(1),
                k_row, v_row, sm_scale, n_split,
                H_KV=n_kv_heads, G=G, D=D, BT=block_tokens,
                NG_K=k_groups, BLOCK_Q=block_q,
                FUSE_COMBINE=fuse_combine,
                num_warps=num_warps, num_stages=num_stages,
            )
            if fuse_combine:
                return o
            _fp8_combine[(B,)](
                m_buf, l_buf, acc_buf, o,
                o.stride(0), o.stride(1), n_split,
                H_KV=1, G=Hq, D=D, BLOCK_G=block_q,
                num_warps=4,
            )
            return o
        _fp8_paged_decode_packed[(B * n_split,)](
            q, k_pool, v_pool,
            k_pool.view(torch.float32), v_pool.view(torch.float32),
            block_table, seq_lens.to(torch.int32),
            m_buf, l_buf, acc_buf,
            q.stride(0), q.stride(1), block_table.stride(0),
            k_row, v_row, sm_scale, n_split,
            H_KV=n_kv_heads, G=G, D=D, BT=block_tokens,
            NG_K=k_groups, NG_V=v_groups, BLOCK_Q=block_q,
            num_warps=num_warps, num_stages=num_stages,
        )
        _fp8_combine[(B,)](
            m_buf, l_buf, acc_buf, o,
            o.stride(0), o.stride(1), n_split,
            H_KV=1, G=Hq, D=D, BLOCK_G=block_q,
            num_warps=4,
        )
        return o

    o = torch.empty_like(q)
    part = B * n_kv_heads * n_split * block_g
    m_buf = torch.empty(part, dtype=torch.float32, device=q.device)
    l_buf = torch.empty(part, dtype=torch.float32, device=q.device)
    acc_buf = torch.empty(part * D, dtype=torch.float32, device=q.device)

    if compute == "fp8":
        # Same predicate the default selector uses, so a
        # capability-conditional default can never choose a path these
        # asserts reject. The kernels unroll one sub-dot per key scale
        # group via constexpr guards -- NG_K values outside the unroll
        # would silently score only the first groups and drop the rest
        # of the key (Bugbot, HIGH).
        _why = fp8_compute_unsupported(q, D, k_groups, v_groups,
                                       pack_heads=False)
        assert _why is None, _why
        assert ktile >= 32, "fp8 P.V dot reduces over ktile: needs >= 32"
        assert layout in ("tokens", "heads")
        counters = _fuse_counters(B * n_kv_heads, q.device) \
            if fuse_combine else m_buf
        _fp8_paged_decode_split_f8dot[(B * n_kv_heads * n_split,)](
            q, k_pool, v_pool,
            k_pool.view(torch.float32), v_pool.view(torch.float32),
            block_table, seq_lens.to(torch.int32),
            m_buf, l_buf, acc_buf,
            o, counters,
            q.stride(0), q.stride(1), block_table.stride(0),
            o.stride(0), o.stride(1),
            k_row, v_row, sm_scale, n_split,
            H_KV=n_kv_heads, G=G, D=D, BT=block_tokens, KTILE=ktile,
            NG_K=k_groups, BLOCK_G=block_g,
            FUSE_COMBINE=fuse_combine,
            LAYOUT_HEADS=(layout == "heads"),
            num_warps=num_warps, num_stages=num_stages,
        )
        if fuse_combine:
            return o
    else:
        assert compute == "f32", f"unknown compute mode {compute!r}"
        assert layout == "tokens", \
            "the decode path keeps tokens-major (heads-major exists for " \
            "the fp8 compute path, where the memory system is the wall)"
        counters = _fuse_counters(B * n_kv_heads, q.device) \
            if fuse_combine else m_buf  # inert placeholder when off
        _fp8_paged_decode_split[(B * n_kv_heads * n_split,)](
            q, k_pool, v_pool,
            k_pool.view(torch.float32), v_pool.view(torch.float32),
            block_table, seq_lens.to(torch.int32),
            m_buf, l_buf, acc_buf,
            o, counters,
            q.stride(0), q.stride(1), block_table.stride(0),
            o.stride(0), o.stride(1),
            k_row, v_row, sm_scale, n_split,
            H_KV=n_kv_heads, G=G, D=D, BT=block_tokens, KTILE=ktile,
            NG_K=k_groups, NG_V=v_groups, BLOCK_G=block_g,
            FUSE_COMBINE=fuse_combine,
            num_warps=num_warps, num_stages=num_stages,
        )
        if fuse_combine:
            return o
    _fp8_combine[(B * n_kv_heads,)](
        m_buf, l_buf, acc_buf, o,
        o.stride(0), o.stride(1), n_split,
        H_KV=n_kv_heads, G=G, D=D, BLOCK_G=block_g,
        num_warps=4,
    )
    return o


def paged_attn_ref(q, k_pool, v_pool, block_table, seq_lens, *,
                   n_kv_heads: int, head_dim: int, block_tokens: int = 16,
                   k_groups: int = 1, v_groups: int = 1,
                   layout: str = "tokens"):
    """Pure-torch oracle: unpack every block through the reference dequant,
    run fp32 attention per sequence. Slow — test sizes only."""
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
                    raw, block_tokens, n_kv_heads, head_dim, groups,
                    layout=layout)
                dst.append(dequant_kv_fp8_ref(qq, ss, dtype=torch.float32))
        k = torch.cat(ks)[:t].permute(1, 0, 2)    # [H_kv, t, D]
        v = torch.cat(vs)[:t].permute(1, 0, 2)
        qq = q[b].float().view(n_kv_heads, G, D)
        att = torch.einsum("hgd,htd->hgt", qq, k) * (D ** -0.5)
        w = torch.softmax(att, dim=-1)
        o = torch.einsum("hgt,htd->hgd", w, v)
        outs.append(o.reshape(Hq, D))
    return torch.stack(outs).to(q.dtype)
