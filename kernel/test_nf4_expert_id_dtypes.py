# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""``gemm_4bit_grouped`` takes int64 expert ids as they are (``EXPERT_ID_DTYPES``; experts4bit-qlora#1313, lane P127).

Every grouped NF4 kernel loads its id and widens it to int64 before any stride product, so an int64 CUDA tensor needs
no cast launch. On a CUDA card, at every route: the singleton decode's bandwidth GEMV (one pass and split-K), its
scalar route, dot-pad (forced past its SM-count gate) and the M-tile; int64 ids give bitwise int32's output, and the
bandwidth route is handed the caller's tensor itself. Correctness only."""
import os

import pytest
import torch

if os.environ.get("TRITON_INTERPRET") == "1":
    pytest.skip("CPU tensors take the host-ids path; the pass-through is a CUDA-tensor path", allow_module_level=True)

pytest.importorskip("triton", reason="the compiled kernels need triton")
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="the compiled kernels need a CUDA device")

import nf4_grouped  # noqa: E402
from nf4_grouped import EXPERT_ID_DTYPES, gemm_4bit_grouped  # noqa: E402
from nf4_pack_ref import make_stack  # noqa: E402

DEV = "cuda"


def test_the_dtypes_read_as_they_are():
    assert EXPERT_ID_DTYPES == (torch.int32, torch.int64)


def _both(B, A, acts, sizes, ids, **kw):
    i32 = torch.tensor(ids, dtype=torch.int32, device=DEV)
    i64 = torch.tensor(ids, dtype=torch.int64, device=DEV)
    a = gemm_4bit_grouped(acts, B, A, sizes, i32, **kw)
    b = gemm_4bit_grouped(acts, B, A, sizes, i64, **kw)
    torch.cuda.synchronize()
    return a, b


@needs_cuda
@pytest.mark.parametrize("route", ["bw", "bw_splitk", "scalar", "dotpad"])
@pytest.mark.parametrize("N,K", [(1536, 2048), (2048, 768)])
def test_the_singleton_decode_reads_int64_ids_bitwise(monkeypatch, route, N, K):
    monkeypatch.setenv("GNF4_GEMV_BW", "1" if route.startswith("bw") else "0")
    monkeypatch.setenv("GNF4_GEMV_DOTPAD", "1" if route == "dotpad" else "0")
    if route == "dotpad":
        if (N, K) not in nf4_grouped._DOTPAD_CONFIGS:
            pytest.skip("dot-pad has no config at this shape")
        monkeypatch.setattr(nf4_grouped, "_sm_count", lambda dev: 170)        # past its >= 160-SM gate on any card
    B, A = make_stack(6, N, K, seed=3, device=DEV)
    ids = [5, 0, 3, 3, 1, 4, 2, 0]
    acts = torch.randn(len(ids), K, device=DEV, dtype=torch.bfloat16)
    kw = {"bw_config": (16, 256, 4, 2)} if route == "bw_splitk" else {}
    a, b = _both(B, A, acts, [1] * len(ids), ids, **kw)
    assert torch.equal(a, b), route


@needs_cuda
def test_the_bandwidth_route_is_handed_the_caller_s_int64_tensor(monkeypatch):
    monkeypatch.setenv("GNF4_GEMV_BW", "1")
    seen = []
    real = nf4_grouped._gemv_nf4_bw_launch

    def rec(a_cat, B, absmax, eids, *rest, **kw):
        seen.append(eids)
        return real(a_cat, B, absmax, eids, *rest, **kw)
    monkeypatch.setattr(nf4_grouped, "_gemv_nf4_bw_launch", rec)
    B, A = make_stack(4, 1536, 2048, seed=5, device=DEV)
    ids = torch.tensor([3, 1, 0, 2], dtype=torch.int64, device=DEV)
    gemm_4bit_grouped(torch.randn(4, 2048, device=DEV, dtype=torch.bfloat16), B, A, [1] * 4, ids)
    assert len(seen) == 1 and seen[0] is ids, "no cast: the kernel reads the caller's int64 ids"
    f32 = torch.tensor([3, 1, 0, 2], dtype=torch.float32, device=DEV)            # any other dtype is still cast
    gemm_4bit_grouped(torch.randn(4, 2048, device=DEV, dtype=torch.bfloat16), B, A, [1] * 4, f32)
    assert seen[1].dtype == torch.int32


@needs_cuda
@pytest.mark.parametrize("sizes", [[3, 1, 2], [16, 5], [1, 1, 7, 1]])
def test_the_m_tile_reads_int64_ids_bitwise(monkeypatch, sizes):
    monkeypatch.setenv("GNF4_GEMV_BW", "0")
    N, K = 1536, 2048
    B, A = make_stack(8, N, K, seed=7, device=DEV)
    ids = [6, 2, 7, 0][:len(sizes)]
    acts = torch.randn(sum(sizes), K, device=DEV, dtype=torch.bfloat16)
    a, b = _both(B, A, acts, sizes, ids)
    assert torch.equal(a, b), sizes
