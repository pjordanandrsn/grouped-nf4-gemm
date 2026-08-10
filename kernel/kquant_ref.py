# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Pure-torch GGUF k-quant dequant reference — the executable spec for the
Glimmer-era GGUF lane, mirroring mxfp4_pack_ref's role for the MoE formats.

Covers what real released files actually contain (adjudicated by parsing the
headers, not the filenames — 2026-08-10, gguf_reader.py):

  meta-models/Muse-Glimmer-30B-GGUF  kquant-dynamic : F32 Q6_K Q5_K Q4_K
  unsloth/Muse-Glimmer-30B-GGUF      UD-Q2_K_XL     : F32 Q3_K Q2_K Q4_K Q5_K Q6_K
  (+ Q8_0 for the Q8 tiers, F16/BF16 passthrough)

Dispatch is BY GGML TYPE PER TENSOR, never by provider or filename: a "Q4_K_M"
file is a MIX (attention in Q4_K, some ffn in Q6_K, norms in F32), and dynamic
quants (Unsloth UD-*) re-mix per tensor. Any provider's file decodes through
the same table by construction.

Source of the block layouts: gguf-py (the llama.cpp project's own numpy
implementation, `gguf.quants`), transcribed to torch operation-for-operation —
including the two easy-to-lose details its comments document:
  * Q4_K/Q5_K pack 8 six-bit (scale, min) pairs into 12 bytes with the high
    twos of bytes 0-7 spilling into bytes 8-11 (diagram in gguf-py source);
  * Q3_K's high-bit plane is INVERTED ("strangely, the offset is zero when the
    bitmask is 1") — q = lo2 - ((hbit ^ 1) << 2).
Bit-exactness against gguf-py is adjudicated in test_kquant_ref.py the same way
test_mxfp4_oracle adjudicated NIBBLE_LOW_FIRST: disagreement is STOP, not
tolerance. The i-quants (IQ2_*/IQ3_*, codebook formats) are a deliberate
non-goal here — only dedicated IQ files use them; raise, never guess.

Superblock geometry: QK_K = 256 elements for all *_K types; Q8_0 uses 32.
Rows quantize along GGUF ne[0] (the contiguous dim), so every decoded tensor
reshapes as [rows, ne0] with ne0 % block_elems == 0 (enforced by ggml).
"""
from __future__ import annotations

import torch

QK_K = 256

# ggml_type ids (ggml.h / gguf spec — stable since 2023).
GGML_F32, GGML_F16 = 0, 1
GGML_Q8_0 = 8
GGML_Q2_K, GGML_Q3_K, GGML_Q4_K, GGML_Q5_K, GGML_Q6_K = 10, 11, 12, 13, 14
GGML_BF16 = 30

GGML_TYPE_NAMES = {
    GGML_F32: "F32", GGML_F16: "F16", GGML_BF16: "BF16", GGML_Q8_0: "Q8_0",
    GGML_Q2_K: "Q2_K", GGML_Q3_K: "Q3_K", GGML_Q4_K: "Q4_K",
    GGML_Q5_K: "Q5_K", GGML_Q6_K: "Q6_K",
}


def _u8(blocks: torch.Tensor) -> torch.Tensor:
    assert blocks.dtype == torch.uint8 and blocks.dim() == 2, \
        (blocks.dtype, tuple(blocks.shape))
    return blocks.contiguous()


def _fp16_col(b: torch.Tensor) -> torch.Tensor:
    """[n, 2] uint8 -> [n, 1] fp32 via an fp16 view (byte order as stored)."""
    return b.contiguous().view(torch.float16).float()


def _get_scale_min_k4(scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[n, 12] uint8 -> ([n, 8], [n, 8]) six-bit scales and mins (Q4_K/Q5_K).

    Byte layout (gguf-py's diagram, capitals = scale bits, lowers = min bits):
      0-3  EEAAAAAA..HHDDDDDD   low 6 of sc[0..3], high 2 of sc[4..7]
      4-7  eeaaaaaa..hhdddddd   low 6 of  m[0..3], high 2 of  m[4..7]
      8-11 eeeeEEEE..hhhhHHHH   low 4 of sc[4..7] | low 4 of m[4..7]
    """
    s = scales.reshape(-1, 3, 4)
    d, m, m_d = s[:, 0], s[:, 1], s[:, 2]
    sc = torch.cat([d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)], dim=-1)
    mn = torch.cat([m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)], dim=-1)
    return sc, mn


def dequant_q8_0(blocks: torch.Tensor) -> torch.Tensor:
    """[n, 34] -> [n, 32].  Layout: fp16 d, int8 qs[32].  w = d * q."""
    b = _u8(blocks)
    d = _fp16_col(b[:, :2])
    q = b[:, 2:].view(torch.int8).float()
    return d * q


def dequant_q2_k(blocks: torch.Tensor) -> torch.Tensor:
    """[n, 84] -> [n, 256].  Layout: scales[16] (lo4=sc, hi4=min), qs[64]
    2-bit planes, fp16 d, fp16 dmin.  w = d*sc*q - dmin*min per 16-elem sub."""
    b = _u8(blocks)
    n = b.shape[0]
    scales, qs, d2, dm2 = b[:, :16], b[:, 16:80], b[:, 80:82], b[:, 82:84]
    d = _fp16_col(d2)
    dmin = _fp16_col(dm2)
    dl = (d * (scales & 0xF).float()).reshape(n, 16, 1)
    ml = (dmin * (scales >> 4).float()).reshape(n, 16, 1)
    shift = torch.tensor([0, 2, 4, 6], dtype=torch.uint8).reshape(1, 1, 4, 1)
    q = (qs.reshape(n, -1, 1, 32) >> shift) & 3
    q = q.reshape(n, 16, 16).float()
    return (dl * q - ml).reshape(n, QK_K)


def dequant_q3_k(blocks: torch.Tensor) -> torch.Tensor:
    """[n, 110] -> [n, 256].  Layout: hmask[32], qs[64], scales[12] (6-bit
    packed, see header diagram in gguf-py), fp16 d.  q = lo2 - ((h ^ 1) << 2)."""
    b = _u8(blocks)
    n = b.shape[0]
    hmask, qs, scales, d2 = b[:, :32], b[:, 32:96], b[:, 96:108], b[:, 108:110]
    d = _fp16_col(d2)
    lsc = scales[:, :8].reshape(n, 1, 8) >> torch.tensor([0, 4], dtype=torch.uint8).reshape(1, 2, 1)
    lsc = lsc.reshape(n, 16)
    hsc = scales[:, 8:].reshape(n, 1, 4) >> torch.tensor([0, 2, 4, 6], dtype=torch.uint8).reshape(1, 4, 1)
    hsc = hsc.reshape(n, 16)
    sc6 = (lsc & 0x0F) | ((hsc & 0x03) << 4)
    sc = (sc6.view(torch.int8).float() - 32.0)
    dl = (d * sc).reshape(n, 16, 1)
    ql = qs.reshape(n, -1, 1, 32) >> torch.tensor([0, 2, 4, 6], dtype=torch.uint8).reshape(1, 1, 4, 1)
    qh = hmask.reshape(n, -1, 1, 32) >> torch.arange(8, dtype=torch.uint8).reshape(1, 1, 8, 1)
    ql = ql.reshape(n, 16, 16) & 3
    qh = (qh.reshape(n, 16, 16) & 1) ^ 1   # inverted plane — see docstring
    q = (ql.to(torch.int8) - (qh << 2).to(torch.int8)).float()
    return (dl * q).reshape(n, QK_K)


def dequant_q4_k(blocks: torch.Tensor) -> torch.Tensor:
    """[n, 144] -> [n, 256].  Layout: fp16 d, fp16 dmin, scales[12] (6-bit
    packed pairs), qs[128] nibble planes.  w = d*sc*q - dmin*min per 32-sub."""
    b = _u8(blocks)
    n = b.shape[0]
    d = _fp16_col(b[:, 0:2])
    dmin = _fp16_col(b[:, 2:4])
    sc, mn = _get_scale_min_k4(b[:, 4:16])
    dl = (d * sc.float()).reshape(n, -1, 1)
    dm = (dmin * mn.float()).reshape(n, -1, 1)
    qs = b[:, 16:144].reshape(n, -1, 1, 32) >> torch.tensor([0, 4], dtype=torch.uint8).reshape(1, 1, 2, 1)
    q = (qs & 0x0F).reshape(n, -1, 32).float()
    return (dl * q - dm).reshape(n, QK_K)


def dequant_q5_k(blocks: torch.Tensor) -> torch.Tensor:
    """[n, 176] -> [n, 256].  Q4_K + qh[32] fifth-bit planes ahead of qs[128]."""
    b = _u8(blocks)
    n = b.shape[0]
    d = _fp16_col(b[:, 0:2])
    dmin = _fp16_col(b[:, 2:4])
    sc, mn = _get_scale_min_k4(b[:, 4:16])
    dl = (d * sc.float()).reshape(n, -1, 1)
    dm = (dmin * mn.float()).reshape(n, -1, 1)
    qh_b, qs_b = b[:, 16:48], b[:, 48:176]
    ql = qs_b.reshape(n, -1, 1, 32) >> torch.tensor([0, 4], dtype=torch.uint8).reshape(1, 1, 2, 1)
    qh = qh_b.reshape(n, -1, 1, 32) >> torch.arange(8, dtype=torch.uint8).reshape(1, 1, 8, 1)
    ql = (ql & 0x0F).reshape(n, -1, 32)
    qh = (qh & 0x01).reshape(n, -1, 32)
    q = (ql | (qh << 4)).float()
    return (dl * q - dm).reshape(n, QK_K)


def dequant_q6_k(blocks: torch.Tensor) -> torch.Tensor:
    """[n, 210] -> [n, 256].  Layout: ql[128], qh[64], int8 scales[16], fp16 d.
    q = ((lo4 | hi2<<4) as int8) - 32, per 16-elem sub-block scale."""
    b = _u8(blocks)
    n = b.shape[0]
    ql_b, qh_b, sc_b, d2 = b[:, :128], b[:, 128:192], b[:, 192:208], b[:, 208:210]
    sc = sc_b.view(torch.int8).float()
    d = _fp16_col(d2)
    dl = (d * sc).reshape(n, 16, 1)
    ql = ql_b.reshape(n, -1, 1, 64) >> torch.tensor([0, 4], dtype=torch.uint8).reshape(1, 1, 2, 1)
    ql = (ql & 0x0F).reshape(n, -1, 32)
    qh = qh_b.reshape(n, -1, 1, 32) >> torch.tensor([0, 2, 4, 6], dtype=torch.uint8).reshape(1, 1, 4, 1)
    qh = (qh & 0x03).reshape(n, -1, 32)
    q = ((ql | (qh << 4)).to(torch.int8) - 32).reshape(n, 16, 16).float()
    return (dl * q).reshape(n, QK_K)


# ggml_type -> (block_elems, block_bytes, dequant fn | None for passthrough)
GGML_DEQUANT = {
    GGML_F32: (1, 4, None),
    GGML_F16: (1, 2, None),
    GGML_BF16: (1, 2, None),
    GGML_Q8_0: (32, 34, dequant_q8_0),
    GGML_Q2_K: (QK_K, 84, dequant_q2_k),
    GGML_Q3_K: (QK_K, 110, dequant_q3_k),
    GGML_Q4_K: (QK_K, 144, dequant_q4_k),
    GGML_Q5_K: (QK_K, 176, dequant_q5_k),
    GGML_Q6_K: (QK_K, 210, dequant_q6_k),
}


def dequantize_ggml(ggml_type: int, data: bytes | torch.Tensor,
                    shape: tuple[int, ...]) -> torch.Tensor:
    """Decode one tensor's raw GGUF bytes to fp32 in torch orientation.

    `shape` is the LOGICAL torch shape (GGUF ne reversed — ne0 last). Blocks
    run along the last dim; ne0 % block_elems == 0 for every type ggml emits.
    Unknown/i-quant types raise (never guess a codebook format).
    """
    if ggml_type not in GGML_DEQUANT:
        raise ValueError(
            f"ggml type {ggml_type} ({GGML_TYPE_NAMES.get(ggml_type, '?')}) is "
            f"not in the k-quant lane — IQ/legacy formats are out of scope here")
    elems, bbytes, fn = GGML_DEQUANT[ggml_type]
    buf = (torch.frombuffer(bytearray(data), dtype=torch.uint8)
           if not isinstance(data, torch.Tensor) else data.to(torch.uint8))
    total = 1
    for s in shape:
        total *= s
    if fn is None:
        dt = {GGML_F32: torch.float32, GGML_F16: torch.float16,
              GGML_BF16: torch.bfloat16}[ggml_type]
        out = buf.view(dt).float()
        assert out.numel() == total, (out.numel(), shape)
        return out.reshape(shape)
    ne0 = shape[-1] if shape else total
    assert ne0 % elems == 0, f"ne0 {ne0} not a multiple of {elems} for {GGML_TYPE_NAMES[ggml_type]}"
    n_blocks = total // elems
    assert buf.numel() == n_blocks * bbytes, (buf.numel(), n_blocks, bbytes)
    out = fn(buf.reshape(n_blocks, bbytes))
    return out.reshape(shape)
