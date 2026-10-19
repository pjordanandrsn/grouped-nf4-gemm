# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""K35 (kernel/PREREG-k35-dense-bf16-decode-gemv.md): ``dense_gemv.gemv_bf16_bw``, the single-row dense bf16 decode GEMV.

Device-free (Triton interpreter mode, CPU; ``TRITON_INTERPRET=1``, set by conftest/CI). The compiled checks, the
registered tolerance contract against cuBLAS among them, are kernel/test_dense_gemv.py's. Guards:
1. **The kernel is P148's prototype, verbatim:** the sha256 of ``_gemv_bf16_kernel``'s source equals the one
   registered from experts4bit-qlora ``bench/p148/p148_kernels.py`` at ``caf5a28c``. An edit to the body fails here.
2. **The plan:** P148's winners, frozen; the default for any other shape.
3. **The arithmetic:** every output reads the fp64 reference within bf16's output rounding, at both served shapes and
   at ragged ones (N off the 2-row block, K off the 1024 chunk, K below one chunk), for ``[K]`` and ``[1, K]`` rows and
   a preallocated ``out``.
4. **One-hot readback, exact:** a one-hot row reads one weight column bit for bit (no rounding can intervene).
5. **Bitwise run to run:** no atomics, so the same call repeats bit for bit.
6. **Refusals:** M > 1, a dtype other than bf16, a K mismatch, non-contiguous weight rows, and a wrong ``out``.

The interpreter's fp32 -> bf16 cast truncates where the compiled store rounds to nearest, so the bound here is bf16's
step, not cuBLAS's error; the contract is test_dense_gemv.py's, on the card.
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
    pytest.skip("the interpreter contract; the compiled checks are test_dense_gemv.py's", allow_module_level=True)

DEV = "cpu"

#: sha256 of ``_gemv_bf16_kernel``'s source (``def`` to its last line, LF) in experts4bit-qlora
#: bench/p148/p148_kernels.py at caf5a28c (that file's sha256: 9a7133fd879b7e71d4e282ae7ca8a594f66fd30ee1bfcbfa7460525a1a06ac91).
P148_KERNEL_SHA256 = "f104918f516bb7b3d8f922b3f7b4b26db1117e1c12606429306dbb838e432bf0"
SERVED = [(5120, 2048), (2048, 4096)]
RAGGED = [(37, 2560), (5, 1000), (1031, 2050)]


def _w(n, k, seed):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, k, generator=g) / k ** 0.5).to(torch.bfloat16).to(DEV)


def _x(k, seed):
    g = torch.Generator().manual_seed(seed + 7)
    return torch.randn(k, generator=g).to(torch.bfloat16).to(DEV)


def _ref(x, w):
    return w.double().cpu() @ x.reshape(-1).double().cpu()


def _rel_rms(got, ref):
    got = got.reshape(-1).double().cpu()
    return float(((got - ref) ** 2).mean().sqrt() / (ref ** 2).mean().sqrt())


def test_the_kernel_is_p148s_prototype_verbatim():
    src = pathlib.Path(D.__file__).read_text(encoding="utf-8")
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "_gemv_bf16_kernel")
    seg = ast.get_source_segment(src, fn).replace("\r\n", "\n")
    assert [ast.unparse(d) for d in fn.decorator_list] == ["triton.jit"]
    assert hashlib.sha256(seg.encode("utf-8")).hexdigest() == P148_KERNEL_SHA256


def test_the_plan_is_p148s_winners_frozen():
    assert D.PLANS == {(5120, 2048): (2, 1024, 4), (2048, 4096): (2, 1024, 4)}
    assert D.PLAN_DEFAULT == (2, 1024, 4) and D.plan(1, 1) == D.PLAN_DEFAULT and D.plan(5120, 2048) == (2, 1024, 4)


@pytest.mark.parametrize("n,k", SERVED + RAGGED, ids=lambda v: str(v))
def test_every_shape_reads_the_fp64_reference(n, k):
    w, x = _w(n, k, n + k), _x(k, n + k)
    ref = _ref(x, w)
    out = D.gemv_bf16_bw(x, w)
    assert out.shape == (n,) and out.dtype == torch.bfloat16
    # the interpreter TRUNCATES fp32 -> bf16: a relative RMS near 2^-7 / sqrt(3), about 4.5e-3 (the compiled store
    # rounds to nearest, about half that; test_dense_gemv.py holds the card to 1.05 x cuBLAS)
    assert _rel_rms(out, ref) < 2.0 ** -7
    step = ref.abs() * 2.0 ** -7 + 1e-30                    # one bf16 step (the interpreter truncates)
    assert bool(((out.double().cpu() - ref).abs() <= step + ref.abs().max() * 2.0 ** -20).all())


def test_a_one_row_matrix_and_a_preallocated_out():
    n, k = 64, 3072
    w, x = _w(n, k, 3), _x(k, 3)
    flat = D.gemv_bf16_bw(x, w)
    out = torch.empty(1, n, dtype=torch.bfloat16, device=DEV)
    got = D.gemv_bf16_bw(x.reshape(1, k), w, out=out)
    assert got is out and got.shape == (1, n) and torch.equal(got.reshape(-1), flat)


@pytest.mark.parametrize("n,k,j", [(19, 2048, 1531), (2048, 4096, 4095), (5, 1000, 0)])
def test_a_one_hot_row_reads_one_weight_column_exactly(n, k, j):
    w = _w(n, k, 5)
    x = torch.zeros(k, dtype=torch.bfloat16, device=DEV)
    x[j] = 1.0
    assert torch.equal(D.gemv_bf16_bw(x, w).cpu(), w[:, j].cpu())


def test_the_same_call_repeats_bit_for_bit():
    w, x = _w(2048, 4096, 11), _x(4096, 11)
    a, b = D.gemv_bf16_bw(x, w), D.gemv_bf16_bw(x, w)
    assert torch.equal(a, b)


def test_the_refusals():
    w, x = _w(16, 64, 1), _x(64, 1)
    with pytest.raises(ValueError, match="M = 2"):
        D.gemv_bf16_bw(torch.stack([x, x]), w)
    with pytest.raises(ValueError, match="bf16"):
        D.gemv_bf16_bw(x.float(), w)
    with pytest.raises(ValueError, match="K = 64"):
        D.gemv_bf16_bw(_x(63, 1), w)
    with pytest.raises(ValueError, match="stride"):
        D.gemv_bf16_bw(x, _w(64, 16, 2).t())
    with pytest.raises(ValueError, match="out must be"):
        D.gemv_bf16_bw(x, w, out=torch.empty(15, dtype=torch.bfloat16, device=DEV))
    with pytest.raises(ValueError, match=r"\[K\] or \[1, K\]"):
        D.gemv_bf16_bw(x.reshape(1, 1, 64), w)
