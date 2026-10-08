# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""K33 (kernel/PREREG-k33-nf4-decode-gemv-bw.md): ``_gemv_nf4_bw``, the bandwidth-targeted NF4 single-row decode GEMV
(``GNF4_GEMV_BW``), device-free (Triton interpreter mode, CPU).

Guards:
1. **The prmt32 decode's assembly, on any machine.** ``_nf4_prmt32``'s PTX is executed by a byte-exact model of
   ``prmt`` (default and sign-replicate modes) and ``lop3``, parsed from the kernel's own source string, and must decode
   every byte value in every byte position of a word (plus random words) to exactly the fp32 codebook bits, in k
   order. The selector constants are hand-derived; this is where a wrong one fails, before any GPU.
2. **The table words** are the fp32 codebook's byte planes.
3. **The kernel's arithmetic** (the select-tree decode, the only one the interpreter can run): every output within one
   bf16 step of the fp32 reference (``dequant_ref`` x fp32 activations), at N off the tile, K of one block, K off the
   K-step, unsorted and repeated expert ids, a row-strided view of a larger stack, and split-K; its error no worse than
   the scalar route's on the same call by more than the registered 5 %.
4. **One-hot readback** through split-K (whose final cast is torch's): every code of every nibble position reads
   ``dequant_ref`` rounded to bf16, exactly.
5. **The result never depends on the call's row count** (row 0 alone and inside a batch, bitwise).
6. **The switches:** off by default and the other routes untouched; ``1`` engages and the tally says which decode ran;
   ``auto`` engages nowhere until K33 reads; typos, an invalid plan, and ``prmt32`` under the interpreter are refused.

Set by conftest/CI: TRITON_INTERPRET=1. The interpreter's fp32 -> bf16 cast truncates where the compiled store rounds to
nearest, so single-pass outputs are checked within one bf16 step and the exact checks go through split-K. The compiled
checks -- prmt32 bitwise equal to the tree, the readback on the card, PTX and PDL -- are kernel/test_nf4_gemv_bw.py's.
"""
import os
os.environ.setdefault("TRITON_INTERPRET", "1")

import random  # noqa: E402
import re  # noqa: E402
import struct  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

pytest.importorskip("triton", reason="interpreter mode needs triton (Linux-only dependency)")

import nf4_grouped  # noqa: E402
from nf4_grouped import NF4_LUT, dequant_ref, gemm_4bit_grouped  # noqa: E402
from nf4_pack_ref import make_stack  # noqa: E402

if os.environ.get("TRITON_INTERPRET") != "1":
    pytest.skip("the interpreter contract; the compiled checks are test_nf4_gemv_bw.py's", allow_module_level=True)

M32 = 0xFFFFFFFF
BITS = [struct.unpack("<I", struct.pack("<f", v))[0] for v in NF4_LUT]


# ------------------------------------------------------- the PTX, modelled --

def _prmt(a, b, c):
    src = [(a >> (8 * i)) & 0xFF for i in range(4)] + [(b >> (8 * i)) & 0xFF for i in range(4)]
    out = 0
    for i in range(4):
        sel = (c >> (4 * i)) & 0xF
        byte = src[sel & 7]
        if sel & 8:                                       # sign mode: replicate the source byte's bit 7
            byte = 0xFF if byte & 0x80 else 0x00
        out |= byte << (8 * i)
    return out


def _lop3(a, b, c, imm):
    out = 0
    for bit in range(32):
        idx = (((a >> bit) & 1) << 2) | (((b >> bit) & 1) << 1) | ((c >> bit) & 1)
        out |= ((imm >> idx) & 1) << bit
    return out


def _kernel_asm():
    import inspect
    fn = nf4_grouped._nf4_prmt32
    src = inspect.getsource(getattr(fn, "fn", fn))
    m = re.search(r'inline_asm_elementwise\(\s*"""(.*?)"""', src, re.S)
    assert m, "no inline assembly in _nf4_prmt32"
    return m.group(1)


def _run_asm(asm, inputs):
    regs = dict(inputs)

    def val(tok):
        tok = tok.strip()
        if tok.startswith("$"):
            return regs[int(tok[1:])]
        return int(tok, 16) if tok.lower().startswith("0x") else (int(tok) if tok.isdigit() else regs[tok])
    for line in asm.splitlines():
        line = line.split("//")[0].strip().rstrip(";").strip()
        if not line or line in ("{", "}") or line.startswith(".reg"):
            continue
        op, _, rest = line.partition(" ")
        args = [x.strip() for x in rest.split(",")]
        dst = int(args[0][1:]) if args[0].startswith("$") else args[0]
        if op == "and.b32":
            v = val(args[1]) & val(args[2])
        elif op == "shl.b32":
            v = val(args[1]) << val(args[2])
        elif op == "shr.b32":
            v = val(args[1]) >> val(args[2])
        elif op == "prmt.b32":
            v = _prmt(val(args[1]), val(args[2]), val(args[3]))
        elif op == "lop3.b32":
            v = _lop3(val(args[1]), val(args[2]), val(args[3]), val(args[4]))
        else:
            raise AssertionError(f"instruction not modelled: {line!r}")
        regs[dst] = v & M32
    return regs


def _decode(word, asm, tables):
    regs = _run_asm(asm, {8: word & M32, **{9 + i: t & M32 for i, t in enumerate(tables)}})
    return [regs[i] for i in range(8)]


def _want(word):
    """Element e of a word is the nibble at shift (e//2)*8 + (4 if e even else 0): bitsandbytes' 2j-high order."""
    return [BITS[(word >> ((e // 2) * 8 + (4 if e % 2 == 0 else 0))) & 0xF] for e in range(8)]


def test_the_prmt32_assembly_decodes_every_byte_in_every_position_exactly():
    asm, tables = _kernel_asm(), nf4_grouped._bw_table_words()
    words = []
    for pos in range(4):
        for byte in range(256):
            rnd = random.Random(pos * 256 + byte).getrandbits(32)
            words.append((rnd & ~(0xFF << (8 * pos)) & M32) | (byte << (8 * pos)))
    words += [random.Random(10_000 + s).getrandbits(32) for s in range(500)] + [0, M32, 0x77777777, 0x88888888]
    bad = [hex(w) for w in words if _decode(w, asm, tables) != _want(w)]
    assert not bad, f"{len(bad)} of {len(words)} words decode wrong, e.g. {bad[:4]}"


def test_the_model_can_fail():
    """The check above must be able to catch a wrong selector: one flipped constant breaks it."""
    asm, tables = _kernel_asm(), nf4_grouped._bw_table_words()
    broken = asm.replace("0x9D8C", "0x9D8D", 1)
    assert broken != asm
    words = [random.Random(20_000 + s).getrandbits(32) for s in range(64)]
    assert sum(_decode(w, broken, tables) != _want(w) for w in words) > 16


def test_the_table_words_are_the_codebooks_byte_planes():
    t = [w & M32 for w in nf4_grouped._bw_table_words()]
    assert len(t) == 16
    for code in range(16):
        q, i = divmod(code, 4)
        got = sum(((t[4 * p + q] >> (8 * i)) & 0xFF) << (8 * p) for p in range(4))
        assert got == BITS[code], code


# ------------------------------------------------------------- the kernel --

def _call(N, K, ids, *, E=4, seed=0, bw="1", bw_config=None, B=None, A=None, acts=None, monkeypatch=None):
    if B is None:
        B, A = make_stack(E, N, K, seed=seed)
    if acts is None:
        acts = torch.randn(len(ids), K, dtype=torch.bfloat16, generator=torch.Generator().manual_seed(seed + 1))
    monkeypatch.setenv("GNF4_GEMV_BW", bw)
    out = gemm_4bit_grouped(acts, B, A, [1] * len(ids), torch.tensor(ids, dtype=torch.int32), bw_config=bw_config)
    return out, B, A, acts


def _ref(B, A, ids, acts, N, K):
    return torch.stack([dequant_ref(B[e], A[e], N, K).float() @ acts[r].float() for r, e in enumerate(ids)])


def _err(out, ref):
    return ((out.float() - ref).abs() / ref.abs().max().clamp_min(1e-6)).max().item()


SHAPES = [(48, 64), (32, 192), (40, 512), (16, 2880)]       # N off 16; one block; K off 256 (2880 = 11.25 K-steps)


@pytest.mark.parametrize("N,K", SHAPES)
def test_every_output_is_within_one_bf16_step_of_the_fp32_reference(monkeypatch, N, K):
    ids = [2, 0, 3, 0, 1]                                   # unsorted, a repeated expert
    nf4_grouped.reset_dispatch_counts()
    out, B, A, acts = _call(N, K, ids, monkeypatch=monkeypatch)
    ref = _ref(B, A, ids, acts, N, K)
    assert (out.float() - ref).abs().le(ref.abs() * 2**-7 + ref.abs().max() * 2**-12).all()
    tally = nf4_grouped.dispatch_counts()
    assert tally["bw_tree"] == 1 and tally["scalar"] == tally["dotpad"] == tally["bw_prmt32"] == 0, tally


@pytest.mark.parametrize("N,K", SHAPES[:3])
def test_no_worse_than_the_scalar_route(monkeypatch, N, K):
    ids = [1, 3, 0]
    bw_out, B, A, acts = _call(N, K, ids, monkeypatch=monkeypatch)
    sc_out, *_ = _call(N, K, ids, B=B, A=A, acts=acts, bw="0", monkeypatch=monkeypatch)
    ref = _ref(B, A, ids, acts, N, K)
    assert _err(bw_out, ref) <= max(1.05 * _err(sc_out, ref), 2**-8), (_err(bw_out, ref), _err(sc_out, ref))


def test_a_row_strided_view_of_a_larger_stack(monkeypatch):
    N, K = 32, 256
    B_big, A_big = make_stack(3, N + 24, K, seed=5)
    B, A = B_big[:, 8:8 + N], A_big[:, 8:8 + N]             # strides of the big stack, rows offset
    assert not B.is_contiguous()
    ids = [2, 1]
    out, *_ , acts = _call(N, K, ids, B=B, A=A, monkeypatch=monkeypatch)
    ref = _ref(B, A, ids, acts, N, K)
    assert (out.float() - ref).abs().le(ref.abs() * 2**-7 + ref.abs().max() * 2**-12).all()


@pytest.mark.parametrize("plan", [(16, 64, 4, 3), (16, 128, 2, 2), (32, 256, 4, 4)])
def test_split_k_matches_the_single_pass(monkeypatch, plan):
    N, K = 48, 1024
    ids = [0, 2, 1]
    nf4_grouped.reset_dispatch_counts()
    one, B, A, acts = _call(N, K, ids, bw_config=(16, 256, 4, 1), monkeypatch=monkeypatch)
    split, *_ = _call(N, K, ids, B=B, A=A, acts=acts, bw_config=plan, monkeypatch=monkeypatch)
    ref = _ref(B, A, ids, acts, N, K)
    for o in (one, split):
        assert (o.float() - ref).abs().le(ref.abs() * 2**-7 + ref.abs().max() * 2**-12).all()
    assert nf4_grouped.dispatch_counts()["bw_splitk"] == 1


def test_one_hot_reads_every_code_in_every_nibble_position_exactly(monkeypatch):
    """Through split-K (fp32 partials, torch's single cast), activation e_k reads column k of the dequantised weight,
    rounded to bf16 as torch rounds it -- for every k of the first two blocks, so every nibble position of every word,
    on weights that carry all 16 codes."""
    N, K = 16, 192
    B, A = make_stack(1, N, K, seed=11)
    codes = torch.stack([(B.to(torch.int32) >> 4) & 0xF, B.to(torch.int32) & 0xF], -1)
    assert torch.unique(codes).numel() == 16
    ks = list(range(128))
    acts = torch.zeros(len(ks), K, dtype=torch.bfloat16)
    for r, k in enumerate(ks):
        acts[r, k] = 1.0
    out, *_ = _call(N, K, [0] * len(ks), B=B, A=A, acts=acts, bw_config=(16, 64, 4, 3), monkeypatch=monkeypatch)
    w = dequant_ref(B[0], A[0], N, K)
    want = torch.stack([w[:, k] for k in ks]).to(torch.bfloat16)
    assert torch.equal(out, want)


def test_the_result_never_depends_on_the_row_count(monkeypatch):
    N, K = 32, 512
    B, A = make_stack(3, N, K, seed=3)
    acts = torch.randn(6, K, dtype=torch.bfloat16, generator=torch.Generator().manual_seed(4))
    many, *_ = _call(N, K, [1, 0, 2, 1, 1, 0], B=B, A=A, acts=acts, monkeypatch=monkeypatch)
    one, *_ = _call(N, K, [1], B=B, A=A, acts=acts[:1], monkeypatch=monkeypatch)
    assert torch.equal(many[:1], one)


# -------------------------------------------------------------- switches --

def test_off_leaves_the_other_routes(monkeypatch):
    monkeypatch.setenv("GNF4_GEMV_BW", "0")
    assert nf4_grouped._bw() == "0"
    nf4_grouped.reset_dispatch_counts()
    B, A = make_stack(2, 32, 128, seed=0)
    gemm_4bit_grouped(torch.randn(2, 128, dtype=torch.bfloat16), B, A, [1, 1], torch.tensor([0, 1], dtype=torch.int32))
    tally = nf4_grouped.dispatch_counts()
    assert tally["bw_tree"] == tally["bw_prmt32"] == 0 and tally["scalar"] + tally["scalar_splitk"] == 1, tally


def test_unset_reads_auto_and_engages_nowhere_off_a_large_part(monkeypatch):
    """The default is ``auto`` (P116 read DEFAULT_ON), but only at Qwen3's two shapes on >= 160-SM parts: the CPU's
    nominal count is 64, so the interpreter suites keep their routes."""
    monkeypatch.delenv("GNF4_GEMV_BW", raising=False)
    assert nf4_grouped._bw() == "auto"
    assert nf4_grouped._BW_SHAPES == frozenset({(1536, 2048), (2048, 768)})
    assert not nf4_grouped._bw_engages(1536, 2048, "cpu")
    nf4_grouped.reset_dispatch_counts()
    B, A = make_stack(2, 32, 128, seed=0)
    gemm_4bit_grouped(torch.randn(2, 128, dtype=torch.bfloat16), B, A, [1, 1], torch.tensor([0, 1], dtype=torch.int32))
    tally = nf4_grouped.dispatch_counts()
    assert tally["bw_tree"] == tally["bw_prmt32"] == 0 and tally["scalar"] + tally["scalar_splitk"] == 1, tally


@pytest.mark.parametrize("bad", ["true", "on", "2"])
def test_a_typo_is_refused(monkeypatch, bad):
    monkeypatch.setenv("GNF4_GEMV_BW", bad)
    with pytest.raises(ValueError, match="GNF4_GEMV_BW"):
        nf4_grouped._bw()


@pytest.mark.parametrize("plan", [(24, 256, 4, 1), (16, 192, 4, 1), (16, 96, 4, 1), (16, 256, 0, 1), (16, 256, 4, 0)])
def test_an_invalid_plan_is_refused(monkeypatch, plan):
    with pytest.raises(ValueError, match="plan"):
        _call(32, 256, [0], bw_config=plan, monkeypatch=monkeypatch)


def test_the_plan_override_is_read_per_shape(monkeypatch):
    monkeypatch.setenv("GNF4_GEMV_BW_PLAN", "32,256=32,128,2,2;1536,2048=16,512,8,1")
    assert nf4_grouped._bw_plan(32, 256) == (32, 128, 2, 2)
    assert nf4_grouped._bw_plan(1536, 2048) == (16, 512, 8, 1)              # the override beats the table
    assert nf4_grouped._bw_plan(2048, 768) == nf4_grouped._BW_PLANS[(2048, 768)]
    assert nf4_grouped._bw_plan(96, 128) == nf4_grouped._BW_PLAN_DEFAULT    # neither listed nor overridden


def test_the_plan_table_is_k33s_selection(monkeypatch):
    """K33's selected plan per family shape (kernel/receipts-k33/5090/k33.json), read with no override."""
    monkeypatch.delenv("GNF4_GEMV_BW_PLAN", raising=False)
    import json
    import pathlib
    want = {(1536, 2048): (16, 1024, 4, 1), (2048, 768): (16, 256, 4, 1), (1024, 1536): (16, 512, 8, 1),
            (1536, 512): (16, 256, 8, 1), (2048, 2048): (16, 1024, 4, 1), (2048, 1024): (16, 256, 4, 1)}
    assert nf4_grouped._BW_PLANS == want
    for (n, k), plan in want.items():
        assert nf4_grouped._bw_plan(n, k) == plan
    rec = pathlib.Path(__file__).resolve().parent / "receipts-k33" / "5090" / "k33.json"
    if rec.exists():
        sel = json.loads(rec.read_text())["plan"]
        fam = {"qwen3": ((1536, 2048), (2048, 768)), "granite": ((1024, 1536), (1536, 512)),
               "olmoe": ((2048, 2048), (2048, 1024))}
        for f, (gu, dn) in fam.items():
            assert want[gu] == tuple(sel[f"{f}/gate_up"]) and want[dn] == tuple(sel[f"{f}/down"]), f


def test_prmt32_under_the_interpreter_is_refused_not_downgraded(monkeypatch):
    monkeypatch.setenv("GNF4_GEMV_BW_DECODE", "prmt32")
    with pytest.raises(ValueError, match="compiled NVIDIA target"):
        _call(32, 128, [0], monkeypatch=monkeypatch)
    monkeypatch.setenv("GNF4_GEMV_BW_DECODE", "lut")
    with pytest.raises(ValueError, match="GNF4_GEMV_BW_DECODE"):
        nf4_grouped._bw_decode("cpu")
