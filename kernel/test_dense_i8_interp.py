# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""K37 (kernel/PREREG-k37-int8-attention-projections.md): ``dense_i8.linear_i8``, the int8-weight projection for 1 to 16
decode rows, over ``dense_gemv.quantize_rows_int8``'s grid.

Device-free (Triton interpreter mode, CPU; ``TRITON_INTERPRET=1``, set by conftest/CI). The compiled checks, the
registered tolerance contract against cuBLAS among them, are kernel/test_dense_i8.py's. Guards:
1. **P149's prototypes, verbatim:** the sha256s of ``_gemv_i8_kernel``'s and ``_gemm_i8_small_kernel``'s sources equal
   the ones registered from experts4bit-qlora ``bench/p149/p149_kernels.py`` at ``ffb1632c``. An edit to either fails here.
2. **The plans:** P149 box K's choices, frozen, and the defaults for any other shape; none splits K.
3. **The quantizer is K36's, re-exported:** the same function object, and P149's own prototype quantizer (its source
   pinned here by sha256) gives bitwise-equal ``(q, s)`` on random, all-zero, single-spike and extreme rows and on a
   non-contiguous (transposed) weight. K36's ``q`` keeps such an input's layout, and ``linear_i8`` refuses it rather than
   inheriting it silently: a caller passes ``q.contiguous()``, which reads the same.
4. **The arithmetic:** every output reads the fp64 reference of the dequantized weights (``q * s``) within bf16's output
   rounding, at Mixtral's two shapes under their plans and at ragged shapes (N off the block, K off the chunk), for M in
   1, 2, 7 and 16, for ``[K]`` and ``[M, K]`` rows and a preallocated ``out``.
5. **One-hot readback, exact:** a one-hot row reads ``s * q[:, j]`` bit for bit, as this mode stores it, on both paths.
6. **Bitwise run to run:** no atomics, so the same call repeats bit for bit.
7. **Refusals:** M > 16, M == 0, another dtype, a K mismatch, non-contiguous ``q`` rows, a wrong or non-contiguous ``s``,
   a wrong ``out``, and x of three dimensions.

The interpreter's fp32 -> bf16 cast truncates where the compiled store rounds to nearest, so the bound here is bf16's
step, not cuBLAS's error; the contract is test_dense_i8.py's, on the card.
"""
import ast
import hashlib
import os
import pathlib

os.environ.setdefault("TRITON_INTERPRET", "1")

import pytest  # noqa: E402
import torch  # noqa: E402

pytest.importorskip("triton", reason="the kernel needs triton (Linux-only dependency)")

import dense_gemv  # noqa: E402
import dense_i8 as D  # noqa: E402

if os.environ.get("TRITON_INTERPRET") != "1":
    pytest.skip("the interpreter contract; the compiled checks are test_dense_i8.py's", allow_module_level=True)

DEV = "cpu"
TOL = 2.0 ** -7                                  # one bf16 step, relative, per output: the interpreter TRUNCATES fp32 to bf16

#: sha256 of each source (``def`` to its last line, LF) in experts4bit-qlora bench/p149/p149_kernels.py at ffb1632c
#: (that file's sha256: 8aa099d8fd8add8f35354e1a1a07d870327af9f758a7512ca2d83e47375354f1).
P149_SHA256 = {"_gemv_i8_kernel": "b96c4fd3e5c89cac00028546d945a18e2e572bf0d0763b9d67cf5ca1376acc3b",
               "_gemm_i8_small_kernel": "77be4e84ee769d5f2ce5eb50e85f94cae450d719ec59766d01f234e51d40b52c"}

#: P149's prototype quantizer, ``def`` to its last line as it stands in that file (sha256 below), for the bitwise
#: comparison with the re-exported K36 grid.
P149_QUANTIZER = '''def quantize_rows_int8(w):
    """``(q int8 [N, K], s fp32 [N])``: the fake-quant's symmetric per-row absmax (``s = absmax / 127``, round to nearest,
    clamp to [-127, 127], 1.0 for an all-zero row)."""
    wf = w.detach().float()
    amax = wf.abs().amax(dim=1)
    s = torch.where(amax > 0, amax / QMAX, torch.ones_like(amax))
    q = torch.clamp(torch.round(wf / s[:, None]), -QMAX, QMAX).to(torch.int8)
    return q.contiguous(), s.contiguous()'''
P149_QUANTIZER_SHA256 = "288655c1bc14cb1959b38574fb2bb43cc272bbed538a4c0cdc9ad41d9244bc5d"

SHAPES = [(4096, 4096), (1024, 4096), (37, 2560), (5, 1000), (1031, 2050)]


def _w(n, k, seed):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, k, generator=g) / k ** 0.5).to(torch.bfloat16).to(DEV)


def _x(m, k, seed):
    g = torch.Generator().manual_seed(seed + 7)
    return torch.randn(m, k, generator=g).to(torch.bfloat16).to(DEV)


def _ref(x, q, s):
    return x.reshape(-1, q.shape[1]).double().cpu() @ (q.double().cpu() * s.double().cpu()[:, None]).T


def _within_bf16(got, ref):
    got = got.reshape(ref.shape).double().cpu()
    return bool(((got - ref).abs() <= TOL * ref.abs() + 1e-6).all())


def _p149_quantizer():
    assert hashlib.sha256(P149_QUANTIZER.encode("utf-8")).hexdigest() == P149_QUANTIZER_SHA256
    ns = {"torch": torch, "QMAX": 127}
    exec(P149_QUANTIZER, ns)  # noqa: S102 -- the pinned prototype text, compared bit for bit below
    return ns["quantize_rows_int8"]


@pytest.mark.parametrize("name", sorted(P149_SHA256))
def test_the_kernels_are_p149s_prototypes_verbatim(name):
    src = pathlib.Path(D.__file__).read_text(encoding="utf-8")
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == name)
    seg = ast.get_source_segment(src, fn).replace("\r\n", "\n")
    assert [ast.unparse(d) for d in fn.decorator_list] == ["triton.jit"]
    assert hashlib.sha256(seg.encode("utf-8")).hexdigest() == P149_SHA256[name]


def test_the_plans_are_box_ks_choices_frozen_and_none_splits_k():
    assert D.PLANS_GEMV == {(4096, 4096): (4, 512, 4), (1024, 4096): (4, 2048, 8)}
    assert D.PLANS_SMALL == {(4096, 4096): (32, 256, 4), (1024, 4096): (32, 256, 4)}
    assert D.PLAN_GEMV_DEFAULT == (4, 512, 4) and D.PLAN_SMALL_DEFAULT == (32, 256, 4) and D.BLOCK_M == 16
    assert D.plan(4096, 4096, 1) == (4, 512, 4) and D.plan(1024, 4096, 1) == (4, 2048, 8)
    assert D.plan(4096, 4096, 16) == (32, 256, 4) and D.plan(37, 2560, 1) == D.PLAN_GEMV_DEFAULT
    assert D.plan(37, 2560, 7) == D.PLAN_SMALL_DEFAULT
    src = pathlib.Path(D.__file__).read_text(encoding="utf-8")
    assert "SPLITK=True" not in src.split("def linear_i8", 1)[1]        # the wrapper never takes a split branch


def test_the_quantizer_is_k36s_reexported_and_p149s_to_the_bit():
    assert D.quantize_rows_int8 is dense_gemv.quantize_rows_int8
    p149 = _p149_quantizer()
    w = _w(64, 300, 1).clone()
    w[3] = 0.0                                                         # an all-zero row
    w[5] = 0.0
    w[5, 17] = 3.5                                                     # a single spike
    w[7, :] = torch.tensor(65280.0).to(torch.bfloat16)                 # extreme magnitudes
    w[9, ::2] = -w[9, ::2].abs() * 1e4
    x = _x(1, 300, 3)
    for case in (w, w.T.contiguous().T, _w(300, 64, 2).T):            # contiguous, and two transposed views
        q0, s0 = D.quantize_rows_int8(case)
        q1, s1 = p149(case)
        assert torch.equal(q0, q1) and torch.equal(s0, s1)             # the same codes and scales, bit for bit
        assert q0.dtype == torch.int8 and s0.dtype == torch.float32 and s0.is_contiguous() and q1.is_contiguous()
        assert int(q0.min()) >= -127 and int(q0.max()) <= 127
        if q0.stride(1) != 1:                                          # K36 keeps a strided input's layout:
            with pytest.raises(ValueError, match="contiguous rows"):   # K37 refuses it, never inherits it
                D.linear_i8(x, q0, s0)
        assert torch.equal(D.linear_i8(x, q0.contiguous(), s0), D.linear_i8(x, q1, s1))


@pytest.mark.parametrize("n,k", SHAPES, ids=lambda v: str(v))
@pytest.mark.parametrize("m", [1, 2, 7, 16])
def test_every_output_reads_the_dequantized_weights_within_bf16_rounding(n, k, m):
    if (n, k) == (4096, 4096) and m not in (1, 16):
        pytest.skip("the big shape at the two registered row counts only (interpreter time)")
    q, s = D.quantize_rows_int8(_w(n, k, n + k))
    x = _x(m, k, m)
    got = D.linear_i8(x, q, s)
    assert got.shape == (m, n) and got.dtype == torch.bfloat16
    assert _within_bf16(got, _ref(x, q, s))


def test_a_flat_row_a_one_row_matrix_and_a_preallocated_out_agree():
    q, s = D.quantize_rows_int8(_w(1031, 2050, 3))
    x = _x(1, 2050, 4)
    a = D.linear_i8(x[0], q, s)
    b = D.linear_i8(x, q, s)
    out = torch.empty(1031, dtype=torch.bfloat16)
    c = D.linear_i8(x[0], q, s, out=out)
    assert a.shape == (1031,) and b.shape == (1, 1031) and c is out
    assert torch.equal(a, b[0]) and torch.equal(a, c)
    out16 = torch.empty(16, 1031, dtype=torch.bfloat16)
    x16 = _x(16, 2050, 5)
    assert D.linear_i8(x16, q, s, out=out16) is out16 and _within_bf16(out16, _ref(x16, q, s))


@pytest.mark.parametrize("m", [1, 16])
def test_a_one_hot_row_reads_s_times_q_exactly(m):
    n, k, j = 37, 2560, 1234
    q, s = D.quantize_rows_int8(_w(n, k, 6))
    x = torch.zeros(m, k, dtype=torch.bfloat16)
    x[:, j] = 1.0
    got = D.linear_i8(x, q, s)
    want = (s * q[:, j].float())                                       # fp32 product, then this mode's bf16 store
    for r in range(m):
        assert torch.equal(got[r], want.to(torch.bfloat16)) or torch.equal(
            got[r].view(torch.int16), (want.view(torch.int32) >> 16).to(torch.int16))


@pytest.mark.parametrize("m", [1, 16])
def test_the_same_call_repeats_bit_for_bit(m):
    q, s = D.quantize_rows_int8(_w(1024, 4096, 8))
    x = _x(m, 4096, 9)
    assert torch.equal(D.linear_i8(x, q, s), D.linear_i8(x, q, s))


def test_the_refusals():
    q, s = D.quantize_rows_int8(_w(37, 256, 10))
    x = _x(1, 256, 11)
    bad = [
        (_x(17, 256, 1), q, s, None, "1 to 16 rows"),
        (_x(1, 256, 1)[:0], q, s, None, "1 to 16 rows"),
        (x.float(), q, s, None, "bf16 x"),
        (x, q.to(torch.int16), s, None, "int8 q"),
        (x, q, s.double(), None, "fp32 s"),
        (_x(1, 255, 1), q, s, None, "K = 256"),
        (x, q.T.contiguous().T, s, None, "contiguous rows"),
        (x, q, s[:-1], None, "contiguous fp32"),
        (x, q, torch.stack([s, s], 1)[:, 0], None, "contiguous fp32"),
        (x, q, s, torch.empty(37, dtype=torch.bfloat16), "contiguous bf16"),
        (x, q, s, torch.empty(1, 37, dtype=torch.float32), "contiguous bf16"),
        (x[None], q, s, None, "[K] or [M, K]"),
    ]
    for xx, qq, ss, out, msg in bad:
        with pytest.raises(ValueError, match=msg.replace("[", r"\[").replace("]", r"\]")):
            D.linear_i8(xx, qq, ss, out=out)
