# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The word-addressed NF4 decode routes -- wide loads and dot-pad -- at THEIR 2^31 boundary, on a GPU (#374).

Both routes are handed ``B.view(torch.int32)`` by the wrapper, so their kernels multiply the expert id by a
stride counted in 32-bit WORDS and wrap at 2^31 words = 8 GiB of packed bytes -- four times further out than
the byte-addressed routes. ``test_expert_offset_boundary.py`` builds its stacks on the 2^31 BYTE geometry, so
its wide and dot-pad arms exercise the routes without reaching their own boundary (they pass with the int64
promotion removed); ``test_offset_boundary_interp.py`` covers the wide route's word boundary on CPU through
lazily committed host memory. Dot-pad is the DEFAULT decode route on >= 160-SM parts at its census shapes since
M3, and until this file no test anywhere put it past its boundary.

Device memory is not lazily committed, so the geometry is real here: one uint8 buffer of a little over
16 GiB. Expert ``EID`` -- the first whose base is at or past 2^31 words -- sits past the 8 GiB mark, and a
DECOY tile sits at the address the int32 product wraps to, inside the same buffer. A kernel that wraps
therefore reads the decoy instead of faulting, which makes the regression an assertion, and every case also
asserts the decoy is distinguishable from the true tile, so a geometry that stopped straddling fails rather
than passing. About 17.5 GiB of free device memory per case: skipped below it, saying so (an RTX 5090 fits;
a 12 GiB card never will).

Proof of power: lane B374 (kernel/PREREG-b374-word-boundary-gpu.md) runs this file twice on an RTX 5090 --
against the installed kernels (every case must PASS, none skipped) and against a copy of nf4_grouped.py with
all SIX eid promotions removed (every case must FAIL reading the decoy). Six, not four: the dot-pad kernels
promote in the load itself (``eid = tl.load(eids_ptr + g).to(tl.int64)``), so a copy stripped of only the
four ``eid = eid.to(tl.int64)`` lines leaves dot-pad promoted and would "calibrate" nothing. Read 2026-09-23
(RESULTS-b374-word-boundary-gpu.md): 4/4 pass shipped, 4/4 read the decoy unpromoted.

    cd kernel && python -m pytest test_offset_boundary_words_gpu.py -q
"""
from __future__ import annotations

import gc
import os
import sys

import pytest

torch = pytest.importorskip("torch")

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="the word boundary needs a CUDA device")

BOUNDARY = 2 ** 31          # int32 elements; for these routes the element is a 4-byte word
UNIT = 4                    # bytes per addressing element on the word-addressed routes
SLACK = 1 << 16
MIN_FREE = int(17.5 * 2 ** 30)
DEV = "cuda"


def _geometry(stride: int, per_expert: int):
    """``(eid, base, wrapped, true_at, total)`` in bytes for a stack of per-expert ``stride`` bytes.

    ``eid`` is the first expert whose base is at or past 2^31 words; the stack starts at ``base`` = 8 GiB into
    the buffer so that the int32 image of ``eid * stride_words`` -- negative -- still lands inside it, at
    ``wrapped``; the true tile is at ``true_at``; ``total`` is the buffer size."""
    assert stride % UNIT == 0, stride
    elems = stride // UNIT
    assert elems < BOUNDARY, "a stride >= 2^31 elements is passed to Triton as i64 and cannot wrap"
    eid = -(-BOUNDARY // elems)
    assert eid * elems >= BOUNDARY > (eid - 1) * elems
    base = BOUNDARY * UNIT
    wrapped = base + (eid * elems - 2 ** 32) * UNIT
    assert 0 <= wrapped and wrapped + per_expert <= base, (wrapped, base)
    true_at = base + eid * stride
    return eid, base, wrapped, true_at, true_at + per_expert + SLACK


def _need(total_bytes: int):
    free, _ = torch.cuda.mem_get_info()
    if free < max(MIN_FREE, total_bytes + (1 << 30)):
        pytest.skip(f"the word-addressed routes wrap at 2^31 WORDS (8 GiB of packed bytes), so this case needs "
                    f"~{max(MIN_FREE, total_bytes) / 2 ** 30:.1f} GiB of free device memory; this device has "
                    f"{free / 2 ** 30:.1f} GiB free")


def _rel(got, want) -> float:
    return ((got.float() - want.float()).norm() / want.float().norm().clamp_min(1e-6)).item()


def _verdict(got, want_true, want_decoy, tol, tag):
    """Both halves: the regression (the true tile was read), and the instrument (the decoy is
    distinguishable, so a pass can never mean the geometry stopped reaching the boundary)."""
    r_true, r_decoy = _rel(got, want_true), _rel(got, want_decoy)
    assert r_true < tol, (f"{tag}: high expert misread (rel {r_true:.3e} vs reference; rel {r_decoy:.3e} vs the "
                          f"tile at the WRAPPED address" + (" -- the offset wrapped)" if r_decoy < tol else ")"))
    assert r_decoy > 10 * tol, (f"{tag}: instrument check failed -- the decoy is indistinguishable from the true "
                                f"tile (rel {r_decoy:.3e}); this geometry no longer proves the read came from past 2^31")


def _case(N: int, K: int, stride: int, seed: int):
    """The stack for one shape: a strided view whose expert ``eid`` lies past 2^31 words, a decoy at the
    wrapped address, and the references for both."""
    import nf4_grouped as NG
    per = N * (K // 2)
    eid, base, wrapped, true_at, total = _geometry(stride, per)
    _need(total)
    buf = torch.empty(total, dtype=torch.uint8, device=DEV)
    g = torch.Generator(device=DEV).manual_seed(seed)
    decoy, low, high = (torch.randint(0, 256, (per,), generator=g, dtype=torch.uint8, device=DEV) for _ in range(3))
    buf[wrapped:wrapped + per] = decoy
    buf[base:base + per] = low
    buf[true_at:true_at + per] = high
    E = eid + 1
    B = torch.as_strided(buf, (E, N, K // 2), (stride, K // 2, 1), storage_offset=base)
    A = (torch.rand(E, N, K // NG.BLOCKSIZE, generator=g, device=DEV) * 0.02 + 1e-3).float()
    w_true = NG.dequant_ref(high.reshape(N, K // 2), A[eid], N, K).float()
    w_decoy = NG.dequant_ref(decoy.reshape(N, K // 2), A[eid], N, K).float()
    a = (torch.randn(1, K, generator=g, device=DEV) * 0.5).bfloat16()
    return NG, B, A, eid, a, a.float() @ w_true.T, a.float() @ w_decoy.T, buf


def _route(NG, before) -> list[str]:
    after = NG.dispatch_counts()
    return [k for k in after if after[k] != before.get(k)]


@pytest.fixture(autouse=True)
def _release():
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@needs_cuda
@pytest.mark.parametrize("split_k, label", [(1, "scalar"), (4, "scalar_splitk")])
def test_wide_loads_past_the_word_boundary(monkeypatch, split_k, label):
    """NF4 decode with wide (uint32-word) loads: N=256, K=1024 (128 KiB per expert) on a 1 MiB stride, so
    expert 8192's base is exactly 2^31 words (8 GiB) and the wrapped product lands on the stack's base."""
    monkeypatch.setenv("GNF4_GEMV_WIDE_LOADS", "1")
    monkeypatch.delenv("GNF4_GEMV_VEC_LOADS", raising=False)
    monkeypatch.setenv("GNF4_GEMV_DOTPAD", "0")
    NG, B, A, eid, a, want, decoy, _buf = _case(256, 1024, 1 << 20, seed=31)
    before = NG.dispatch_counts()
    got = NG.gemm_4bit_grouped(a, B, A, [1], [eid], decode_config=(64, 2), split_k=split_k)
    torch.cuda.synchronize()
    assert _route(NG, before) == [label], f"wide decode routed {_route(NG, before)}, expected [{label!r}]"
    _verdict(got, want, decoy, 5e-2, f"nf4 decode wide ({label}), expert {eid}")


@needs_cuda
@pytest.mark.parametrize("split_k, label", [(None, "dotpad"), (4, "dotpad_splitk")])
def test_dotpad_past_the_word_boundary(monkeypatch, split_k, label):
    """NF4 dot-pad decode -- the default route on >= 160-SM parts -- at its gate_up census shape N=1536,
    K=2048 (1.5 MiB per expert, packed contiguously), so expert 5462's base is 1 MiB past 2^31 words."""
    import nf4_grouped as NG
    if (1536, 2048) not in NG._DOTPAD_CONFIGS or NG._sm_count(torch.device(DEV)) < 160:
        pytest.skip("dot-pad engages only at its census shapes on >= 160-SM parts; this device has "
                    f"{NG._sm_count(torch.device(DEV))} SMs")
    monkeypatch.delenv("GNF4_GEMV_DOTPAD", raising=False)                  # the shipped default: on
    monkeypatch.delenv("GNF4_GEMV_SPLITK", raising=False)
    NG, B, A, eid, a, want, decoy, _buf = _case(1536, 2048, 1536 * 1024, seed=37)
    before = NG.dispatch_counts()
    got = NG.gemm_4bit_grouped(a, B, A, [1], [eid], split_k=split_k)
    torch.cuda.synchronize()
    assert _route(NG, before) == [label], f"dot-pad decode routed {_route(NG, before)}, expected [{label!r}]"
    _verdict(got, want, decoy, 5e-2, f"nf4 decode dot-pad ({label}), expert {eid}")


def test_the_geometry_straddles_the_word_boundary():
    """Pure arithmetic, no device: each case's expert base is at or past 2^31 WORDS, its int32 image lands
    inside the buffer, and the stride itself fits int32 (else Triton passes it as i64 and nothing can wrap)."""
    for stride, per in ((1 << 20, 256 * 512), (1536 * 1024, 1536 * 1024)):
        eid, base, wrapped, true_at, total = _geometry(stride, per)
        assert eid * (stride // UNIT) >= BOUNDARY and stride // UNIT < BOUNDARY
        assert true_at - base >= BOUNDARY * UNIT and 0 <= wrapped < base
        assert total < 17 * 2 ** 30
