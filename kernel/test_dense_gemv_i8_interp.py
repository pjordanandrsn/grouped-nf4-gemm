# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""K36 (kernel/PREREG-k36-int8-head-gemv.md): ``dense_gemv.gemv_i8_bw``, the single-row int8-weight decode GEMV, and
``dense_gemv.quantize_rows_int8``, its weights' grid.

Device-free (Triton interpreter mode, CPU; ``TRITON_INTERPRET=1``, set by conftest/CI). The compiled checks, the
registered tolerance contract against cuBLAS among them, are kernel/test_dense_gemv_i8.py's. Guards:
1. **P148's prototypes, verbatim:** the sha256s of ``_gemv_i8_kernel``'s and ``quantize_rows_int8``'s sources equal the
   ones registered from experts4bit-qlora ``bench/p148/p148_kernels.py`` at ``caf5a28c``. An edit to either fails here.
2. **The plan:** P148's winner at the head, frozen; the same config for any other shape.
3. **The quantizer:** int8 in [-127, 127] (never -128), ``s = absmax / 127`` in fp32 with each row's absmax at +-127,
   every weight within half a step of its dequantized value, and an all-zero row at ``s = 1``, ``q = 0``.
4. **The arithmetic:** every output reads the fp64 reference of the dequantized weights (``q * s``) within bf16's output
   rounding, at the head's K under the head's plan (N = 4096: the full 151936 rows are the compiled file's) and at
   ragged shapes (N off the 4-row block, K off the 1024 chunk, K below one chunk), for ``[K]`` and ``[1, K]`` rows and a
   preallocated ``out``.
5. **One-hot readback, exact:** a one-hot row reads ``s * q[:, j]`` bit for bit, as this mode stores it.
6. **Bitwise run to run:** no atomics, so the same call repeats bit for bit.
7. **Refusals:** M > 1, a dtype other than bf16 x / int8 q / fp32 s, a K mismatch, non-contiguous weight rows, a wrong
   or non-contiguous scale, and a wrong ``out``.

The interpreter's fp32 -> bf16 cast truncates where the compiled store rounds to nearest, so the bound here is bf16's
step, not cuBLAS's error; the contract is test_dense_gemv_i8.py's, on the card.
"""
import ast
import hashlib
import os
import pathlib

os.environ.setdefault("TRITON_INTERPRET", "1")

import pytest  # noqa: E402
import torch  # noqa: E402

pytest.importorskip("triton", reason="the kernel needs triton (Linux-only dependency)")

import dense_gemv as D  # noqa: E402

if os.environ.get("TRITON_INTERPRET") != "1":
    pytest.skip("the interpreter contract; the compiled checks are test_dense_gemv_i8.py's", allow_module_level=True)

DEV = "cpu"

#: sha256 of each source (``def`` to its last line, LF) in experts4bit-qlora bench/p148/p148_kernels.py at caf5a28c
#: (that file's sha256: 9a7133fd879b7e71d4e282ae7ca8a594f66fd30ee1bfcbfa7460525a1a06ac91).
P148_SHA256 = {"_gemv_i8_kernel": "729955a718f9d0887cfabc5dca8e8644726d093703579ea1eb86e703f7cb59f7",
               "quantize_rows_int8": "8ea6dd9ee1b52594419f40339272059e413956aaed2a2f02d382450f854e8c6c"}
SHAPES = [(4096, 2048), (37, 2560), (5, 1000), (1031, 2050)]


def _w(n, k, seed):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, k, generator=g) / k ** 0.5).to(torch.bfloat16).to(DEV)


def _x(k, seed):
    g = torch.Generator().manual_seed(seed + 7)
    return torch.randn(k, generator=g).to(torch.bfloat16).to(DEV)


def _ref(x, q, s):
    return (q.double().cpu() * s.double().cpu()[:, None]) @ x.reshape(-1).double().cpu()


def _rel_rms(got, ref):
    got = got.reshape(-1).double().cpu()
    return float(((got - ref) ** 2).mean().sqrt() / (ref ** 2).mean().sqrt())


@pytest.mark.parametrize("name,decorators", [("_gemv_i8_kernel", ["triton.jit"]), ("quantize_rows_int8", [])])
def test_the_kernel_and_quantizer_are_p148s_prototypes_verbatim(name, decorators):
    src = pathlib.Path(D.__file__).read_text(encoding="utf-8")
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == name)
    seg = ast.get_source_segment(src, fn).replace("\r\n", "\n")
    assert [ast.unparse(d) for d in fn.decorator_list] == decorators
    assert hashlib.sha256(seg.encode("utf-8")).hexdigest() == P148_SHA256[name]


def test_the_plan_is_p148s_winner_frozen():
    assert D.PLANS_I8 == {(151936, 2048): (4, 1024, 4)}
    assert D.PLAN_I8_DEFAULT == (4, 1024, 4)
    assert D.plan_i8(151936, 2048) == (4, 1024, 4) and D.plan_i8(1, 1) == D.PLAN_I8_DEFAULT


def test_the_quantizer_grid():
    w = _w(64, 300, 9)
    w[3] = 0                                         # an all-zero row
    w[7, 11] = -2 * w[7].abs().max()                 # a row whose absmax is negative
    q, s = D.quantize_rows_int8(w)
    assert q.dtype == torch.int8 and q.shape == w.shape and s.dtype == torch.float32 and s.shape == (64,)
    assert int(q.min()) >= -127 and int(q.max()) <= 127
    amax = w.float().abs().amax(dim=1)
    nz = amax > 0
    assert torch.equal(s[nz], amax[nz] / 127.0)
    assert float(s[3]) == 1.0 and not bool(q[3].any())
    assert bool((q[nz].to(torch.int16).abs().amax(dim=1) == 127).all()) and int(q[7, 11]) == -127
    err = (q.double() * s.double()[:, None] - w.double()).abs()
    assert bool((err <= s.double()[:, None] * (0.5 + 1e-5)).all())


@pytest.mark.parametrize("n,k", SHAPES, ids=lambda v: str(v))
def test_every_shape_reads_the_fp64_reference_of_its_dequantized_weights(n, k):
    q, s = D.quantize_rows_int8(_w(n, k, n + k))
    x = _x(k, n + k)
    ref = _ref(x, q, s)
    out = D.gemv_i8_bw(x, q, s)
    assert out.shape == (n,) and out.dtype == torch.bfloat16
    # the interpreter TRUNCATES fp32 -> bf16: a relative RMS near 2^-7 / sqrt(3), about 4.5e-3 (the compiled store
    # rounds to nearest, about half that; test_dense_gemv_i8.py holds the card to 1.05 x cuBLAS)
    assert _rel_rms(out, ref) < 2.0 ** -7
    step = ref.abs() * 2.0 ** -7 + 1e-30                    # one bf16 step (the interpreter truncates)
    assert bool(((out.double().cpu() - ref).abs() <= step + ref.abs().max() * 2.0 ** -20).all())


def test_a_one_row_matrix_and_a_preallocated_out():
    n, k = 64, 3072
    q, s = D.quantize_rows_int8(_w(n, k, 3))
    x = _x(k, 3)
    flat = D.gemv_i8_bw(x, q, s)
    out = torch.empty(1, n, dtype=torch.bfloat16, device=DEV)
    got = D.gemv_i8_bw(x.reshape(1, k), q, s, out=out)
    assert got is out and got.shape == (1, n) and torch.equal(got.reshape(-1), flat)


@pytest.mark.parametrize("n,k,j", [(19, 2048, 1531), (4096, 2048, 2047), (5, 1000, 0)])
def test_a_one_hot_row_reads_one_scaled_column_exactly(n, k, j):
    q, s = D.quantize_rows_int8(_w(n, k, 5))
    x = torch.zeros(k, dtype=torch.bfloat16, device=DEV)
    x[j] = 1.0
    got = D.gemv_i8_bw(x, q, s).cpu()
    f32 = q[:, j].float().cpu() * s.cpu()                   # exact up to the store: q * 1.0, then one fp32 multiply
    truncated = (f32.view(torch.int32) & ~0xFFFF).view(torch.float32).to(torch.bfloat16)
    # this mode's store truncates; a nearest-rounding interpreter would read the second, bit for bit all the same
    assert torch.equal(got, truncated) or torch.equal(got, f32.to(torch.bfloat16))


def test_the_same_call_repeats_bit_for_bit():
    q, s = D.quantize_rows_int8(_w(4096, 2048, 11))
    x = _x(2048, 11)
    assert torch.equal(D.gemv_i8_bw(x, q, s), D.gemv_i8_bw(x, q, s))


def test_the_refusals():
    q, s = D.quantize_rows_int8(_w(16, 64, 1))
    x = _x(64, 1)
    with pytest.raises(ValueError, match="M = 2"):
        D.gemv_i8_bw(torch.stack([x, x]), q, s)
    with pytest.raises(ValueError, match="bf16 x"):
        D.gemv_i8_bw(x.float(), q, s)
    with pytest.raises(ValueError, match="int8 q"):
        D.gemv_i8_bw(x, q.to(torch.int16), s)
    with pytest.raises(ValueError, match="fp32 s"):
        D.gemv_i8_bw(x, q, s.to(torch.bfloat16))
    with pytest.raises(ValueError, match="K = 64"):
        D.gemv_i8_bw(_x(63, 1), q, s)
    qt, _ = D.quantize_rows_int8(_w(64, 16, 2))
    with pytest.raises(ValueError, match="stride"):
        D.gemv_i8_bw(x, qt.t(), s)
    with pytest.raises(ValueError, match="s must be"):
        D.gemv_i8_bw(x, q, s[:15])
    with pytest.raises(ValueError, match="s must be"):
        D.gemv_i8_bw(x, q, torch.stack([s, s], dim=1)[:, 0])
    with pytest.raises(ValueError, match="out must be"):
        D.gemv_i8_bw(x, q, s, out=torch.empty(15, dtype=torch.bfloat16, device=DEV))
    with pytest.raises(ValueError, match=r"\[K\] or \[1, K\]"):
        D.gemv_i8_bw(x.reshape(1, 1, 64), q, s)
