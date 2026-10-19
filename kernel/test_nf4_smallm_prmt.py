# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""K25's ``lut="prmt"`` (lane K39) on the CPU: the inline PTX it ships, EXECUTED by a small vectorized interpreter for
the six opcodes it uses, over every 16-bit chunk, against NF4_LUT's fp32 bits. Also its table words and the
wrapper's NVIDIA-only rules.

This checks the asm string itself, not a restatement of it. The interpreter (``triton.language`` without a GPU) cannot
run inline PTX, which is why the wrapper decodes ``"prmt"`` through the tree there. The compiled kernel's bit-identity
to ``"tree"`` is ``test_nf4_smallm_prmt_gpu.py``'s."""
import re

import numpy as np
import pytest
import torch

import nf4_grouped
import nf4_smallm

U32 = np.uint64(0xFFFFFFFF)


def _prmt(a, b, c):
    """PTX ``prmt.b32`` (default mode): output byte i takes selector field c[4i+3:4i] over the 8 bytes {a, b}. The
    field's low 3 bits pick the byte; bit 3 replicates that byte's sign bit."""
    out = np.zeros_like(a)
    for i in range(4):
        f = (c >> np.uint64(4 * i)) & np.uint64(0xF)
        sel = f & np.uint64(7)
        lo = (a >> (np.uint64(8) * (sel & np.uint64(3)))) & np.uint64(0xFF)
        hi = (b >> (np.uint64(8) * (sel & np.uint64(3)))) & np.uint64(0xFF)
        byte = np.where(sel < 4, lo, hi)
        byte = np.where(f & np.uint64(8), np.where(byte & np.uint64(0x80), np.uint64(0xFF), np.uint64(0)), byte)
        out |= byte << np.uint64(8 * i)
    return out


def _lop3(a, b, c, lut):
    out = np.zeros_like(a)
    for i in range(8):
        if (lut >> i) & 1:
            out |= (a if i & 4 else ~a & U32) & (b if i & 2 else ~b & U32) & (c if i & 1 else ~c & U32)
    return out


def run_ptx(text, inputs, n):
    """Execute the asm block ``text`` on ``n`` lanes at once. ``inputs`` maps ``$k`` -> uint64 arrays; returns every
    ``$k`` register after the block."""
    reg = {k: np.asarray(v, dtype=np.uint64) & U32 for k, v in inputs.items()}

    def val(tok):
        tok = tok.strip()
        if re.fullmatch(r"0x[0-9a-fA-F]+|\d+", tok):
            return np.full(n, int(tok, 0), dtype=np.uint64)
        return reg[tok]

    for raw in text.splitlines():
        ln = raw.strip().rstrip(";")
        if not ln or ln in "{}" or ln.startswith(".reg"):
            continue
        op, rest = ln.split(None, 1)
        args = [x.strip() for x in rest.split(",")]
        d, srcs = args[0], [val(x) for x in args[1:]]
        if op == "mov.b32":
            r = srcs[0]
        elif op == "and.b32":
            r = srcs[0] & srcs[1]
        elif op == "shr.b32":
            r = srcs[0] >> srcs[1]
        elif op == "mad.lo.u32":
            r = srcs[0] * srcs[1] + srcs[2]
        elif op == "prmt.b32":
            r = _prmt(*srcs)
        elif op == "lop3.b32":
            r = _lop3(srcs[0], srcs[1], srcs[2], int(args[4], 0))
        else:
            raise AssertionError(f"opcode {op!r} is not one the interpreter knows; extend it")
        reg[d] = r & U32
    return reg


def _codebook_bits():
    return torch.tensor(nf4_grouped.NF4_LUT, dtype=torch.float32).view(torch.int32).numpy().astype(np.uint64) & U32


@pytest.mark.parametrize("half", [0, 1])
def test_the_shipped_ptx_decodes_every_16_bit_chunk_to_the_codebook_bits(half):
    """Bytes 0-1 (half 0) and bytes 2-3 (half 1) of the input word, each over all 65,536 values. The other half is a
    fixed pattern, so the halves are shown independent. $0-$3 are the HIGH nibbles' values and $4-$7 the LOW
    nibbles', byte j at position j."""
    v = np.arange(1 << 16, dtype=np.uint64)
    word = (v << np.uint64(16 * half)) | (np.uint64(0x5AC3) << np.uint64(16 * (1 - half)))
    reg = run_ptx(nf4_smallm.PRMT_ASM_TEXT, {"$8": word}, v.size)
    cb = _codebook_bits()
    for j in range(4):
        byte = (word >> np.uint64(8 * j)) & np.uint64(0xFF)
        assert np.array_equal(reg[f"${j}"], cb[(byte >> np.uint64(4)).astype(np.int64)]), f"byte {j} HIGH nibble"
        assert np.array_equal(reg[f"${4 + j}"], cb[(byte & np.uint64(15)).astype(np.int64)]), f"byte {j} LOW nibble"


def test_the_table_words_are_the_gemv_decodes_own():
    """One source of truth: the 16 constants in the asm are ``nf4_grouped._bw_table_words()``, the byte planes
    ``_nf4_prmt32`` reads."""
    movs = re.findall(r"mov\.b32 t(\d+), (0x[0-9a-f]+);", nf4_smallm.PRMT_ASM_TEXT)
    assert [int(i) for i, _ in movs] == list(range(16))
    assert [int(w, 16) for _, w in movs] == [w & 0xFFFFFFFF for w in nf4_grouped._bw_table_words()]
    assert nf4_smallm._LUT_MODES["prmt"] == 4 and nf4_smallm.PRMT_CONSTRAINTS_TEXT.count("=r") == 8


def test_prmt_is_refused_on_rocm(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "6.2.0")
    x = torch.zeros(1, 64, dtype=torch.bfloat16)
    packed = torch.zeros(1, 32, 32, dtype=torch.uint8)
    absmax = torch.ones(1, 32, 1)
    t = torch.zeros(1, dtype=torch.int32)
    with pytest.raises(ValueError, match="inline-PTX decode for NVIDIA"):
        nf4_smallm.gemm_nf4_grouped_smallm(x, packed, absmax, t, t, t, lut="prmt")


def test_an_unknown_lut_names_prmt_among_the_choices():
    x = torch.zeros(1, 64, dtype=torch.bfloat16)
    t = torch.zeros(1, dtype=torch.int32)
    with pytest.raises(ValueError, match="'prmt'"):
        nf4_smallm.gemm_nf4_grouped_smallm(x, torch.zeros(1, 32, 32, dtype=torch.uint8), torch.ones(1, 32, 1), t, t, t,
                                           lut="reg")
