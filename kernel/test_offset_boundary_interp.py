# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The 2^31 expert-offset boundary (gnf4#87) WITHOUT a GPU, under the Triton interpreter.

`kernel/test_offsets_2gib.py` and `kernel/test_expert_offset_boundary.py` already
straddle this boundary, but both require CUDA and ~2.3 GiB of free device memory,
so on every CI runner this project has they skip in full. Nothing that CI can run
fails when the `eid.to(tl.int64)` promotion is dropped -- and it HAS been dropped
once, silently, by a port: the MXFP4 kernels shipped without it in 0.13.2 and the
gap was found by a live illegal access months later (#205). A regression that can
only be caught on hardware CI does not have is not guarded.

This file closes that gap on CPU. Two facts make it possible.

**The interpreter does wrap.** The 0.14.0 CHANGELOG states that
``TRITON_INTERPRET=1`` "evaluates offsets with int64 semantics and the overflow
cannot manifest there". That is not true on triton 3.4.0 / numpy 2.3.2: the
interpreter evaluates ``int32_tensor * python_int`` under NEP 50, where the
Python int is weak, so the product stays int32 and wraps. Measured, and pinned
below by ``test_interpreter_arithmetic_still_wraps`` -- if a future triton or
numpy widens this, that canary fails and says so, instead of leaving every test
here quietly unable to detect the bug it was written for.

**The bytes can stay small while the offsets do not.** Each case allocates ~4 GiB
of *address space* with ``torch.empty`` and writes three small tiles into it.
Linux commits pages on touch, so resident memory stays a few hundred MiB
(measured 546 MiB peak inside a 7 GiB container, all of it torch itself). The
expert stack is a strided view whose base sits 2 GiB into that buffer, so BOTH
the correct offset (+2^31) and the wrapped one (-2^31) land inside a valid
mapping. A kernel that wraps therefore reads the WRONG BYTES rather than
segfaulting, which is what makes this an assertion instead of a crash.

Every case carries its own instrument check: the result must match the reference
for the high expert AND be far from the reference for the decoy tile sitting at
the wrapped address. A geometry that stopped straddling the boundary would pass
the first and fail the second, so a green run here cannot mean "the test no
longer engages the bug".

**A stride must FIT in int32 for the bug to be reachable at all.** Triton
specialises integer kernel arguments; a stride >= 2^31 is passed as int64 and
the product never wraps, whatever the index dtype. The first draft of this file
used ``stride = 2**31`` and passed against a deliberately unfixed kernel. The
strides below are all int32-representable, and ``test_geometry_straddles``
asserts it.

Proof of power (2026-09-21, triton 3.4.0 / torch 2.8.0 / numpy 2.3.2, CPU): run
against a copy of ``nf4_grouped.py`` with the four ``eid = eid.to(tl.int64)``
lines removed, the NF4 decode case reads the decoy (rel 1.27 vs the true
reference, 3.5e-03 vs the decoy) and fails; against the shipped tree it passes.
The #386 cases at the bottom -- the int64-word slot gathers and the fp8 KV
appenders -- were calibrated the same way on 2026-09-23: with exactly the five
straddling promotions removed, all five fail at the wrapped address
(kernel/receipts-386/interp/).

    cd kernel && TRITON_INTERPRET=1 python -m pytest test_offset_boundary_interp.py -q
"""
from __future__ import annotations

import mmap
import os

os.environ.setdefault("TRITON_INTERPRET", "1")

import pytest
import torch

pytest.importorskip("triton", reason="interpreter mode needs triton (Linux-only dependency)")

import triton                                                    # noqa: E402
import triton.language as tl                                     # noqa: E402

BOUNDARY = 2 ** 31
SLACK = 1 << 16
MAP_NORESERVE = getattr(mmap, "MAP_NORESERVE", 0x4000)          # Linux's value; named from 3.13


def _buffer(n_bytes: int, *, reserve: bool = False) -> torch.Tensor:
    """Address space only. torch.empty does not touch the pages and Linux
    commits on write, so just the tiles written below are ever resident
    (measured: 485 MiB RSS for a 16 GiB mapping, all of it torch).

    ``reserve=True`` maps MAP_NORESERVE instead, for the ~32 GiB spans of the
    int64-word gathers (#386). The heuristic overcommit check refuses a single
    torch.empty larger than RAM + swap -- a 16 GB CI runner -- and does not
    charge a MAP_NORESERVE mapping at all; pages still commit only on touch.
    torch.frombuffer holds the mapping for the tensor's lifetime."""
    if not reserve:
        return torch.empty(n_bytes, dtype=torch.uint8)
    m = mmap.mmap(-1, n_bytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS | MAP_NORESERVE,
                  prot=mmap.PROT_READ | mmap.PROT_WRITE)
    return torch.frombuffer(m, dtype=torch.uint8)


def _tiles(n_bytes: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, 256, (n_bytes,), generator=g, dtype=torch.uint8)
            for _ in range(3)]                                   # decoy, low, high


def _straddling_stack(stride: int, per_expert: int, seed: int, *, unit: int = 1,
                      reserve: bool = False):
    """A stack whose expert ``EID`` has a base offset past 2^31 ELEMENTS.

    ``unit`` is bytes per addressing element **as the kernel sees it**. It is 1
    where the wrapper passes the uint8 tensor, and 4 on the routes that pass
    ``B.view(torch.int32)`` (wide loads, dot-pad): those index the stack in
    32-bit words, so their wrap sits at 2^31 WORDS = 8 GiB of packed bytes,
    four times further out than the byte-addressed routes. Getting that wrong
    is not a small error -- it makes the case vacuous, because the product
    never reaches the boundary and the kernel reads correctly with or without
    the int64 promotion.

    Returns ``(buf, eid, decoy_bytes, base)``.
    """
    assert stride % unit == 0, (stride, unit)
    elems = stride // unit                       # the stride the KERNEL multiplies
    assert elems < BOUNDARY, (
        f"a stride of {elems} elements is >= 2^31, so Triton passes it as an "
        f"i64 argument and the product cannot wrap whatever the index dtype")
    eid = -(-BOUNDARY // elems)                  # first expert at/above 2^31 elements
    assert eid * elems >= BOUNDARY > (eid - 1) * elems
    base = BOUNDARY * unit                       # bytes: 2 GiB, or 8 GiB for words
    buf = _buffer(2 * base + per_expert + SLACK, reserve=reserve)
    decoy, low, high = _tiles(per_expert, seed)
    wrapped = _wrapped_offset(eid, stride, unit, base)   # the int32 image, in bytes
    assert 0 <= wrapped < base, (wrapped, base)
    buf[wrapped:wrapped + per_expert] = decoy
    buf[base:base + per_expert] = low
    buf[base + eid * stride:base + eid * stride + per_expert] = high
    return buf, eid, decoy, base


def _wrapped_offset(eid: int, stride: int, unit: int, base: int) -> int:
    """Where an int32 ``eid * (stride // unit)`` lands, in bytes into the buffer:
    the address a kernel that dropped the promotion reads or WRITES instead."""
    return base + (eid * (stride // unit) - 2 ** 32) * unit


def _rel(got, want) -> float:
    return ((got.float() - want.float()).norm()
            / want.float().norm().clamp_min(1e-6)).item()


def _verdict(got, want_true, want_decoy, tol, tag):
    """Both halves. The first is the regression; the second is the instrument --
    it fails if the geometry stopped engaging the boundary, so a pass here can
    never mean 'this test no longer looks at the bug'."""
    r_true = _rel(got, want_true)
    r_decoy = _rel(got, want_decoy)
    assert r_true < tol, (
        f"{tag}: high expert misread (rel {r_true:.3e} vs reference; "
        f"rel {r_decoy:.3e} vs the tile at the WRAPPED address"
        + (" -- the offset wrapped)" if r_decoy < tol else ")"))
    assert r_decoy > 10 * tol, (
        f"{tag}: instrument check failed -- the decoy tile is indistinguishable "
        f"from the true one (rel {r_decoy:.3e}); this geometry no longer proves "
        f"the read came from past 2^31")


# --------------------------------------------------------------------------
# The canary: this whole file can only detect the bug while the interpreter
# evaluates an int32 index times an int32-representable stride in int32.
# --------------------------------------------------------------------------
@triton.jit
def _unpromoted_offset(idx_ptr, out_ptr, stride):
    g = tl.program_id(0)
    i = tl.load(idx_ptr + g)                 # int32, deliberately NOT promoted
    tl.store(out_ptr + g, (i * stride).to(tl.int64))


def test_interpreter_arithmetic_still_wraps():
    """If this fails, every other test in this file has silently stopped being
    able to detect #87 -- the promotion would still be right, but its absence
    would no longer be observable here. Fix the suite, do not delete this."""
    stride = 8 * 1024 * 1024                                     # 8 MiB, the down_proj stride
    idx = torch.tensor([0, 255, 256, 257], dtype=torch.int32)
    out = torch.zeros(4, dtype=torch.int64)
    _unpromoted_offset[(4,)](idx, out, stride)
    got = out.tolist()
    assert got[:2] == [0, 255 * stride], f"below the boundary the interpreter already disagrees: {got}"
    assert got[2] < 0 and got[3] < 0, (
        "the Triton interpreter no longer wraps int32 offset arithmetic "
        f"(got {got}, expected the last two negative). The CPU tests in this "
        "file can no longer fail when the int64 promotion is dropped.")


def test_geometry_straddles():
    """Every stride used here must be int32-representable IN THE KERNEL'S OWN
    UNITS -- otherwise triton passes it as int64, the product never wraps, and
    the case is vacuous whether or not the promotion is present."""
    for stride, unit in ((NF4_STRIDE, 1), (MX_STRIDE, 1), (MXB32_STRIDE, 1),
                         (I4_STRIDE, 1), (WIDE_STRIDE, 4)):
        elems = stride // unit
        eid = -(-BOUNDARY // elems)
        assert elems < BOUNDARY
        assert eid * elems >= BOUNDARY > (eid - 1) * elems


def test_word_addressed_routes_sit_four_times_further_out():
    """The wide-load and dot-pad decode routes are handed ``B.view(torch.int32)``
    (``nf4_grouped`` lines 1262 and 1292), so the kernel multiplies the expert id
    by a stride counted in 32-bit WORDS. Their wrap is therefore at 2^31 words =
    8 GiB of packed bytes, not 2 GiB.

    This is worth pinning rather than commenting, because it makes a whole class
    of test silently vacuous: a fixture built on the 2 GiB BYTE boundary hands
    these kernels a product of only 2^29, which does not wrap, so the case passes
    with the int64 promotion removed. The first draft of this file had exactly
    that bug, and so does the wide arm of the shipped GPU suite
    (``test_expert_offset_boundary.py``) -- see the PR and the contract note.
    """
    b = torch.zeros(4, 8, 64, dtype=torch.uint8)
    assert b.view(torch.int32).stride(0) * 4 == b.stride(0), (
        "the int32 view no longer quarters the expert stride; the 4x factor "
        "this file's WIDE_* geometry is built on has changed")


# ------------------------------------------------------------------- NF4 --
NF4_N, NF4_K = 64, 128
NF4_STRIDE = 2 ** 20                                             # padded 1 MiB rows
NF4_PER = NF4_N * (NF4_K // 2)


@pytest.fixture(scope="module")
def nf4_case():
    import nf4_grouped as NG
    buf, eid, decoy, base = _straddling_stack(NF4_STRIDE, NF4_PER, seed=11)
    E = eid + 1
    B = torch.as_strided(buf, (E, NF4_N, NF4_K // 2),
                         (NF4_STRIDE, NF4_K // 2, 1), storage_offset=base)
    g = torch.Generator().manual_seed(12)
    A = (torch.rand(E, NF4_N, NF4_K // NG.BLOCKSIZE, generator=g) * 0.02 + 1e-3).float()
    w_true = NG.dequant_ref(B[eid].contiguous(), A[eid], NF4_N, NF4_K).float()
    w_decoy = NG.dequant_ref(decoy.reshape(NF4_N, NF4_K // 2), A[eid], NF4_N, NF4_K).float()
    return NG, B, A, eid, w_true, w_decoy


@pytest.mark.parametrize("route,kw,env", [
    ("scalar", dict(decode_config=(64, 2), split_k=1), {}),
    ("scalar_splitk", dict(decode_config=(64, 2), split_k=4), {}),
    ("scalar/vec", dict(decode_config=(64, 2), split_k=1), {"GNF4_GEMV_VEC_LOADS": "1"}),
])
def test_nf4_decode(nf4_case, route, kw, env, monkeypatch):
    NG, B, A, eid, w_true, w_decoy = nf4_case
    for k in ("GNF4_GEMV_WIDE_LOADS", "GNF4_GEMV_VEC_LOADS", "GNF4_GEMV_DOTPAD"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    if route.endswith("vec") and not NG.HAS_TL_INTERLEAVE:
        pytest.skip("this triton has no tl.interleave, so the vec route does not exist")
    g = torch.Generator().manual_seed(3)
    a = (torch.randn(1, NF4_K, generator=g) * 0.5).bfloat16()
    got = NG.gemm_4bit_grouped(a, B, A, [1], [eid], **kw)
    _verdict(got, a.float() @ w_true.T, a.float() @ w_decoy.T, 5e-2, f"nf4 decode {route}")


def test_nf4_m_tile(nf4_case):
    """prefill_variant=0: the register-LUT mainloop uses tl.gather, which the
    interpreter cannot execute (it is a GPU-only path here)."""
    NG, B, A, eid, w_true, w_decoy = nf4_case
    g = torch.Generator().manual_seed(4)
    a = (torch.randn(4, NF4_K, generator=g) * 0.5).bfloat16()
    got = NG.gemm_4bit_grouped(a, B, A, [4], [eid], prefill_variant=0)
    _verdict(got, a.float() @ w_true.T, a.float() @ w_decoy.T, 5e-2, "nf4 m-tile")


def test_nf4_dgrad(nf4_case):
    NG, B, A, eid, w_true, w_decoy = nf4_case
    g = torch.Generator().manual_seed(5)
    go = (torch.randn(4, NF4_N, generator=g) * 0.1).bfloat16()
    got = NG.dgrad_4bit_grouped(go, B, A, [4], [eid])
    _verdict(got, go.float() @ w_true, go.float() @ w_decoy, 5e-2, "nf4 dgrad")


def test_nf4_grouped_launch_spans_the_boundary(nf4_case):
    """One launch reading an expert below the boundary and one above it."""
    NG, B, A, eid, w_true, w_decoy = nf4_case
    g = torch.Generator().manual_seed(6)
    a = (torch.randn(2, NF4_K, generator=g) * 0.5).bfloat16()
    got = NG.gemm_4bit_grouped(a, B, A, [1, 1], [0, eid],
                               decode_config=(64, 2), split_k=1)
    w_lo = NG.dequant_ref(B[0].contiguous(), A[0], NF4_N, NF4_K).float()
    assert _rel(got[0:1], a[0:1].float() @ w_lo.T) < 5e-2, "low expert misread"
    _verdict(got[1:2], a[1:2].float() @ w_true.T, a[1:2].float() @ w_decoy.T,
             5e-2, "nf4 grouped both-sides")


# The wide-load route, at ITS boundary: 2^31 int32 words = 8 GiB of packed
# bytes, so the mapping is 16 GiB of address space (still only three resident
# tiles). If the host refuses the mapping the test skips saying exactly that --
# it is the one case here whose geometry CI may not be able to afford.
WIDE_STRIDE = 2 ** 20                                            # bytes; 2^18 words


@pytest.fixture(scope="module")
def nf4_wide_case():
    import nf4_grouped as NG
    try:
        buf, eid, decoy, base = _straddling_stack(WIDE_STRIDE, NF4_PER, seed=23, unit=4)
    except (RuntimeError, MemoryError) as e:                     # noqa: PERF203
        pytest.skip(f"the wide route wraps at 2^31 int32 WORDS, so this case needs "
                    f"a 16 GiB mapping (lazily committed, ~0.5 GiB resident) and "
                    f"this host refused it: {e}")
    E = eid + 1
    B = torch.as_strided(buf, (E, NF4_N, NF4_K // 2),
                         (WIDE_STRIDE, NF4_K // 2, 1), storage_offset=base)
    g = torch.Generator().manual_seed(24)
    A = (torch.rand(E, NF4_N, NF4_K // NG.BLOCKSIZE, generator=g) * 0.02 + 1e-3).float()
    w_true = NG.dequant_ref(B[eid].contiguous(), A[eid], NF4_N, NF4_K).float()
    w_decoy = NG.dequant_ref(decoy.reshape(NF4_N, NF4_K // 2), A[eid], NF4_N, NF4_K).float()
    return NG, B, A, eid, w_true, w_decoy


def test_nf4_decode_wide_loads(nf4_wide_case, monkeypatch):
    NG, B, A, eid, w_true, w_decoy = nf4_wide_case
    monkeypatch.setenv("GNF4_GEMV_WIDE_LOADS", "1")
    monkeypatch.delenv("GNF4_GEMV_VEC_LOADS", raising=False)
    g = torch.Generator().manual_seed(25)
    a = (torch.randn(1, NF4_K, generator=g) * 0.5).bfloat16()
    got = NG.gemm_4bit_grouped(a, B, A, [1], [eid], decode_config=(64, 2), split_k=1)
    _verdict(got, a.float() @ w_true.T, a.float() @ w_decoy.T, 5e-2, "nf4 decode wide")


# ----------------------------------------------------------------- MXFP4 --
MX_N, MX_K = 64, 128
MX_STRIDE = 2 ** 20
MX_PER = MX_N * (MX_K // 2)


@pytest.fixture(scope="module")
def mx_case():
    import mxfp4_grouped as MX
    from mxfp4_pack_ref import dequant_mxfp4
    buf, eid, decoy, base = _straddling_stack(MX_STRIDE, MX_PER, seed=13)
    E = eid + 1
    B = torch.as_strided(buf, (E, MX_N, MX_K // 2),
                         (MX_STRIDE, MX_K // 2, 1), storage_offset=base)
    g = torch.Generator().manual_seed(14)
    S = torch.empty(E, MX_N, MX_K // 32, dtype=torch.uint8)      # lazy: 2 tiles written
    S[0] = torch.randint(100, 140, (MX_N, MX_K // 32), generator=g, dtype=torch.uint8)
    S[eid] = torch.randint(100, 140, (MX_N, MX_K // 32), generator=g, dtype=torch.uint8)
    nb = MX_K // 32
    w_true = dequant_mxfp4(B[eid].contiguous().reshape(MX_N, nb, 16), S[eid]).float()
    w_decoy = dequant_mxfp4(decoy.reshape(MX_N, nb, 16), S[eid]).float()
    return MX, B, S, eid, w_true, w_decoy


@pytest.mark.parametrize("rows,tag", [(1, "gemv"), (4, "m-tile")])
def test_mxfp4_grouped(mx_case, rows, tag):
    MX, B, S, eid, w_true, w_decoy = mx_case
    g = torch.Generator().manual_seed(15)
    a = (torch.randn(rows, MX_K, generator=g) * 0.5).bfloat16()
    got = MX.gemm_mxfp4_grouped(a, B, S, [rows], [eid])
    _verdict(got, a.float() @ w_true.T, a.float() @ w_decoy.T, 2e-2, f"mxfp4 {tag}")


# `gemv_mxfp4_b32` recomputes the expert base from the SHAPES
# (`eid * N * (K // 2)`) instead of reading `blocks.stride(0)`, so a padded
# layout never reaches it -- the first draft of this file paired it with the
# fixture above and it read untouched pages and returned zeros. It needs a
# genuinely contiguous stack, which the lazily-committed buffer still affords.
MXB32_N, MXB32_K = 64, 128
MXB32_STRIDE = MXB32_N * (MXB32_K // 2)                          # 4096, natural


@pytest.fixture(scope="module")
def mx_b32_case():
    import mxfp4_grouped as MX
    from mxfp4_pack_ref import dequant_mxfp4
    buf, eid, decoy, base = _straddling_stack(MXB32_STRIDE, MXB32_STRIDE, seed=17)
    E = eid + 1
    B = torch.as_strided(buf, (E, MXB32_N, MXB32_K // 2),
                         (MXB32_STRIDE, MXB32_K // 2, 1), storage_offset=base)
    g = torch.Generator().manual_seed(18)
    S = torch.empty(E, MXB32_N, MXB32_K // 32, dtype=torch.uint8)   # lazy
    for i in (0, eid):
        S[i] = torch.randint(100, 140, (MXB32_N, MXB32_K // 32),
                             generator=g, dtype=torch.uint8)
    nb = MXB32_K // 32
    w_true = dequant_mxfp4(B[eid].contiguous().reshape(MXB32_N, nb, 16), S[eid]).float()
    w_decoy = dequant_mxfp4(decoy.reshape(MXB32_N, nb, 16), S[eid]).float()
    return MX, B, S, eid, w_true, w_decoy


def test_mxfp4_gemv_b32(mx_b32_case):
    MX, B, S, eid, w_true, w_decoy = mx_b32_case
    from int4_b32 import quant_x_rows
    g = torch.Generator().manual_seed(16)
    a = (torch.randn(1, MXB32_K, generator=g) * 0.5).bfloat16()
    xq, xs = quant_x_rows(a)
    x = xq.float() * xs.repeat_interleave(32, dim=1)
    got = MX.gemv_mxfp4_b32(xq, xs, B, S,
                            torch.tensor([eid], dtype=torch.int32), MXB32_N, MXB32_K)
    _verdict(got, x @ w_true.T, x @ w_decoy.T, 2e-2, "mxfp4 gemv_b32")


# -------------------------------------------------------------- int4-b32 --
# These kernels do NOT take a stride argument -- they recompute the expert base
# as `eid * N * (K // 2)` from the shapes -- so the stack must be genuinely
# contiguous and the stride is the natural per-expert size. The buffer is still
# lazily committed, so only the two written tiles are resident.
I4_N, I4_K = 64, 128
I4_STRIDE = I4_N * (I4_K // 2)                                   # 4096, natural
I4_PER = I4_STRIDE


@pytest.fixture(scope="module")
def i4_case():
    buf, eid, decoy, base = _straddling_stack(I4_STRIDE, I4_PER, seed=19)
    E = eid + 1
    P = torch.as_strided(buf, (E, I4_N, I4_K // 2),
                         (I4_STRIDE, I4_K // 2, 1), storage_offset=base)
    g = torch.Generator().manual_seed(20)
    S = torch.empty(E, I4_N, I4_K // 32, dtype=torch.float16)    # lazy
    S[0] = (torch.rand(I4_N, I4_K // 32, generator=g) * 0.05 + 1e-3).half()
    S[eid] = (torch.rand(I4_N, I4_K // 32, generator=g) * 0.05 + 1e-3).half()
    from int4_pack_ref import dequant_int4_ref
    w_true = dequant_int4_ref(P[eid].contiguous(), S[eid], I4_N, I4_K).float()
    w_decoy = dequant_int4_ref(decoy.reshape(I4_N, I4_K // 2), S[eid], I4_N, I4_K).float()
    return P, S, eid, w_true, w_decoy


def test_int4_b32_gemv(i4_case):
    from int4_b32 import gemv_int4_b32, quant_x_rows
    P, S, eid, w_true, w_decoy = i4_case
    g = torch.Generator().manual_seed(21)
    x = (torch.randn(1, I4_K, generator=g) * 0.5).bfloat16()
    xq, xs = quant_x_rows(x)
    xd = xq.float() * xs.repeat_interleave(32, dim=1)
    got = gemv_int4_b32(xq, xs, P, S, torch.tensor([eid], dtype=torch.int32),
                        I4_N, I4_K)
    _verdict(got, xd @ w_true.T, xd @ w_decoy.T, 1e-2, "int4-b32 gemv")


def test_int4_b32_m_tile(i4_case):
    from int4_b32 import gemm_int4_b32_grouped_captured, quant_x_rows
    from nf4_grouped import build_group_tiles_device
    P, S, eid, w_true, w_decoy = i4_case
    g = torch.Generator().manual_seed(22)
    x = (torch.randn(4, I4_K, generator=g) * 0.5).bfloat16()
    xq, xs = quant_x_rows(x)
    flat = torch.full((4,), eid, dtype=torch.int32)
    t_row0, t_rows, t_group, order, _c = build_group_tiles_device(flat, P.shape[0], 16)
    got = gemm_int4_b32_grouped_captured(xq.index_select(0, order),
                                         xs.index_select(0, order), P, S,
                                         t_row0, t_rows, t_group, block_m=16)
    xd = (xq.float() * xs.repeat_interleave(32, dim=1)).index_select(0, order)
    _verdict(got, xd @ w_true.T, xd @ w_decoy.T, 1e-2, "int4-b32 m-tile")


# ------------------------------------------------ int64-word slot gathers --
# #386. host_gather._gather_rows and the mxfp4_pipelined / mxfp4_residency slot
# gathers move int64 WORDS: they multiply an int32 expert id or slot index by
# ``row_words``, so their wrap sits at 2^31 words = 16 GiB, and an int32 wrap
# moves an address by exactly 2^32 words = 32 GiB. Every straddling case here
# therefore spans ~32 GiB of address space, mapped MAP_NORESERVE (``_buffer``).
# The interpreter executes every program, so rows are large and grids small:
# one chunk per slot, and only the slot under test does any work. The gathers
# COPY bytes, so the verdicts are exact equality.
GW_TILE = 1024                                   # int64 words compared per case


def _copy_verdict(got, true, decoy, tag):
    assert torch.equal(got, true), (
        f"{tag}: the row past 2^31 words was not copied"
        + (" -- it copied the tile at the WRAPPED address" if torch.equal(got, decoy) else ""))
    assert not torch.equal(true, decoy), (
        f"{tag}: instrument check failed -- the decoy equals the true tile")


HG_ROW_WORDS = 2 ** 20                           # 8 MiB rows: expert 2048 sits at 2^31 words


@pytest.fixture(scope="module")
def host_gather_case():
    stride = HG_ROW_WORDS * 8                    # rows are contiguous: row_words IS the stride,
    buf, eid, decoy, base = _straddling_stack(stride, stride, seed=41, unit=8, reserve=True)
    host = torch.as_strided(buf, (eid + 1, stride), (stride, 1), storage_offset=base)
    return host, eid, decoy[:GW_TILE * 8]        # so whole rows are allocated; a prefix is compared


def test_host_gather_rows(host_gather_case):
    """``gather_expert_rows`` reads ``src + want * row_words``; the int32 operand
    is the expert id, loaded from ``ids``. Through the public wrapper."""
    from host_gather import gather_expert_rows
    host, eid, decoy = host_gather_case
    dst = torch.zeros(1, host.shape[1], dtype=torch.uint8)
    gather_expert_rows(dst, host, torch.tensor([eid], dtype=torch.int32), block=1 << 16)
    _copy_verdict(dst[0, :GW_TILE * 8], host[eid, :GW_TILE * 8], decoy, "host_gather")


SG_ROW_WORDS = 2 ** 21                           # 16 MiB rows: slot 1024 sits at 2^31 words


@pytest.fixture(scope="module")
def slot_gather_case():
    """The slot gathers WRITE past the boundary (``dst + slot * row_words``;
    ``slot`` is the int32 program id), so the stack is the destination: slot
    ``slot_hi``'s row starts 2^31 words in, and its int32 image holds a decoy
    that a correct store never touches."""
    stride = SG_ROW_WORDS * 8
    buf, slot_hi, decoy, base = _straddling_stack(stride, stride, seed=43, unit=8, reserve=True)
    dst = torch.as_strided(buf, (slot_hi + 1, stride), (stride, 1), storage_offset=base)
    g = torch.Generator().manual_seed(44)
    tile = torch.randint(0, 256, (GW_TILE * 8,), generator=g, dtype=torch.uint8)
    return (buf, dst, slot_hi, decoy[:GW_TILE * 8], _wrapped_offset(slot_hi, stride, 8, base),
            tile)


def _slot_launch_state(case):
    """Reset what an earlier case wrote, and build tables where ``want == have``
    for every slot but ``slot_hi``: those programs return at once."""
    buf, dst, slot_hi, decoy, wrapped, tile = case
    dst[slot_hi, :tile.numel()] = 0
    buf[wrapped:wrapped + tile.numel()] = decoy
    want = torch.zeros(dst.shape[0], dtype=torch.int64)
    have = torch.zeros(dst.shape[0], dtype=torch.int64)
    want[slot_hi] = tile.data_ptr()
    have[slot_hi] = -1
    return want, have


def _slot_verdict(case, tag):
    buf, dst, slot_hi, decoy, wrapped, tile = case
    got = dst[slot_hi, :tile.numel()]
    at_wrap = buf[wrapped:wrapped + tile.numel()]
    assert torch.equal(got, tile), (
        f"{tag}: slot {slot_hi} (2^31 words in) did not receive its row"
        + (" -- the store landed at the WRAPPED address" if torch.equal(at_wrap, tile) else ""))
    assert torch.equal(at_wrap, decoy), f"{tag}: the int32 image of the slot's address was written"
    assert not torch.equal(tile, decoy), f"{tag}: instrument check failed -- tile equals decoy"


def test_mxfp4_pipelined_slot_gather(slot_gather_case):
    from mxfp4_pipelined import _gather_kernel
    buf, dst, slot_hi, decoy, wrapped, tile = slot_gather_case
    want, have = _slot_launch_state(slot_gather_case)
    _gather_kernel()[(dst.shape[0], 1)](dst.view(torch.int64), want, have, SG_ROW_WORDS,
                                        BLOCK=GW_TILE, num_warps=4)
    _slot_verdict(slot_gather_case, "mxfp4_pipelined gather")


def test_mxfp4_residency_perm_gather(slot_gather_case):
    from mxfp4_residency import _perm_gather_kernel
    buf, dst, slot_hi, decoy, wrapped, tile = slot_gather_case
    want, have = _slot_launch_state(slot_gather_case)
    zero = torch.zeros(1, dtype=torch.int64)                     # one piece: src 0 -> dst 0
    n = torch.tensor([GW_TILE], dtype=torch.int64)
    _perm_gather_kernel()[(dst.shape[0], 1)](dst.view(torch.int64), want, have, zero, zero, n,
                                             SG_ROW_WORDS, BLOCK=GW_TILE, num_warps=4)
    _slot_verdict(slot_gather_case, "mxfp4_residency perm gather")


# ------------------------------------------------- fp8 KV cache appenders --
# #386. ``fp8_kv_append_t1`` / ``_bt1`` WRITE one token into block-table row
# ``row`` at ``row * row_bytes``: byte-addressed, so the wrap is at 2^31 bytes
# and the ordinary 4 GiB span serves. The public wrappers refuse CPU tensors
# (a triton-less or CPU call must fail as an availability error, not inside
# triton's driver), so these launch the kernels with the wrappers' arguments.
# What is checked is WHERE the bytes land, not their e4m3 rounding -- that is
# test_fp8_kv_append.py's bitwise GPU gate, which the interpreter must never
# stand in for: the token read back from the true row dequantizes to the input
# within e4m3's own error, and the row at the int32 image still holds its decoy.
FP8_H, FP8_D, FP8_BT, FP8_G = 2, 128, 4, 2
FP8_PAY = FP8_BT * FP8_H * FP8_D                                 # payload bytes per row
FP8_ROW = FP8_PAY + FP8_BT * FP8_H * FP8_G * 4                   # + the fp32 scales: 1088


@pytest.fixture(scope="module")
def fp8_pool_case():
    import fp8_kv
    buf, row_hi, decoy, base = _straddling_stack(FP8_ROW, FP8_ROW, seed=45)
    return fp8_kv, buf, buf[base:], row_hi, decoy, _wrapped_offset(row_hi, FP8_ROW, 1, base)


def _fp8_token(pool, row, fill):
    """The token at (row, fill), dequantized from the paged layout: payload at
    ``fill * H * D``, fp32 scales at ``pay + fill * H * groups * 4``."""
    r = pool[row * FP8_ROW:(row + 1) * FP8_ROW]
    q = r[fill * FP8_H * FP8_D:(fill + 1) * FP8_H * FP8_D].view(torch.float8_e4m3fn).float()
    s0 = FP8_PAY + fill * FP8_H * FP8_G * 4
    s = r[s0:s0 + FP8_H * FP8_G * 4].view(torch.float32)
    return (q.reshape(FP8_H, FP8_G, -1) * s.reshape(FP8_H, FP8_G, 1)).reshape(FP8_H, FP8_D)


def _fp8_reset(case, rows):
    fp8_kv, buf, pool, row_hi, decoy, wrapped = case
    for r in rows:
        pool[r * FP8_ROW:(r + 1) * FP8_ROW] = 0                  # a zero row decodes to 0: rel 1
    buf[wrapped:wrapped + FP8_ROW] = decoy


def _fp8_verdict(case, fill, x, tag):
    fp8_kv, buf, pool, row_hi, decoy, wrapped = case
    r = _rel(_fp8_token(pool, row_hi, fill), x)
    wrote_wrap = not torch.equal(buf[wrapped:wrapped + FP8_ROW], decoy)
    assert r < 0.1, (f"{tag}: row {row_hi} (past 2^31 bytes) does not hold the token (rel {r:.3e})"
                     + (" -- the bytes landed at the WRAPPED address" if wrote_wrap else ""))
    assert not wrote_wrap, f"{tag}: the row at the int32 image of the address was written"


def test_fp8_append_t1(fp8_pool_case):
    fp8_kv, buf, pool, row_hi, decoy, wrapped = fp8_pool_case
    _fp8_reset(fp8_pool_case, [row_hi])
    g = torch.Generator().manual_seed(46)
    x = torch.randn(FP8_H, FP8_D, generator=g)
    fill = 1
    fp8_kv._fp8_append_t1_side[(FP8_H,)](
        x, pool, pool.view(torch.int32), torch.tensor([row_hi], dtype=torch.int32),
        torch.tensor([fill], dtype=torch.int32), FP8_ROW, FP8_PAY, FP8_BT,
        H=FP8_H, D=FP8_D, GROUPS=FP8_G, GS=FP8_D // FP8_G, E4M3_MAX=fp8_kv.E4M3_MAX, num_warps=1)
    _fp8_verdict(fp8_pool_case, fill, x, "fp8 append t1")


def test_fp8_append_bt1_spans_the_boundary(fp8_pool_case):
    """One batched launch appending to a slot whose row is below the boundary
    and one whose row is above it."""
    fp8_kv, buf, pool, row_hi, decoy, wrapped = fp8_pool_case
    _fp8_reset(fp8_pool_case, [0, row_hi])
    g = torch.Generator().manual_seed(47)
    x = torch.randn(2, FP8_H, FP8_D, generator=g)
    fills = (2, 1)
    fp8_kv._fp8_append_bt1_side[(2, FP8_H)](
        x, pool, pool.view(torch.int32), torch.tensor([[0], [row_hi]], dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32), torch.tensor(fills, dtype=torch.int32),
        1, FP8_ROW, FP8_PAY, FP8_BT,
        H=FP8_H, D=FP8_D, GROUPS=FP8_G, GS=FP8_D // FP8_G, E4M3_MAX=fp8_kv.E4M3_MAX, num_warps=1)
    assert _rel(_fp8_token(pool, 0, fills[0]), x[0]) < 0.1, "fp8 append bt1: the low row misread"
    _fp8_verdict(fp8_pool_case, fills[1], x[1], "fp8 append bt1")
