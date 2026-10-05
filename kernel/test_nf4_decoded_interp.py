# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""GNF4_TRAIN_GEMM=decoded (``nf4_route.decoded_forward`` / ``decoded_dgrad``): the correctness contract, device-free (Triton
interpreter mode, CPU).

Guards:
1. experts4bit-qlora RD1's correctness gate, on every call: the route's relative error against an fp32 reference (``dequant_ref``
   in fp32 times the fp32 activations, per group) is at most ``GATE_X`` = 2 times dense's. Dense's arithmetic is reproduced
   here in torch -- the expert decoded to bf16, times the bf16 activations, fp32 accumulation, one bf16 rounding -- because the
   ``dense`` route itself is CUDA-only. Under the interpreter the route reads ~1.6x dense's error, not ~1x as compiled: the
   interpreter's fp32 -> bf16 cast truncates (it ignores ``fp_downcast_rounding``) where the compiled store rounds to nearest;
   so guard 1b is the sharper one here;
1b. every output is within one bf16 ulp of the exact fp32 product of the same bf16 operands;
2. both tilings (``DECODED_TILES_SMALL`` / ``_LARGE``), ragged and empty groups, N and K off the tile multiples;
3. the chunk plan moves no output bit: a call capped by ``GNF4_DECODED_MAX_BYTES`` to one or two groups per chunk is
   bit-identical to the uncapped call, an all-empty chunk included;
4. device-form (tensor) sizes and expert ids give the list form's output, bit for bit;
5. the route counts itself.

Set by conftest/CI: TRITON_INTERPRET=1 (the interpreter's dot is fp32 arithmetic on the bf16 operands). ``dequant_groups`` is
supplied by ``dequant_ref`` rounded to bf16 in torch here: its own kernel truncates under the interpreter for the same reason,
and its bit-equality to ``dequant_ref`` is test_nf4_route.py's, on a GPU. So this file reads the route's own GEMM kernel and
chunk plan. The GPU checks -- the compiled bf16 tensor-core arithmetic, against the real ``dense`` route and the fused kernels
-- are in test_nf4_route.py.
"""
import os
os.environ.setdefault("TRITON_INTERPRET", "1")

import pytest
import torch

pytest.importorskip("triton", reason="interpreter mode needs triton (Linux-only dependency)")

import nf4_route                                                                   # noqa: E402
from nf4_grouped import dequant_ref                                                # noqa: E402
from nf4_pack_ref import make_stack                                                # noqa: E402

if os.environ.get("TRITON_INTERPRET") != "1":
    pytest.skip("the interpreter contract; the compiled checks are test_nf4_route.py's", allow_module_level=True)

GATE_X = 2.0

# (N, K, sizes, expert ids): small enough for the interpreter, ragged, an empty group, N and K off the tile multiples, and one
# case at >= DECODED_ROWS_SPLIT mean rows so the large tiling runs.
CASES = [
    (64, 128, [5, 0, 33, 7], [7, 1, 2, 4]),
    (96, 192, [1, 1, 40, 2, 9], [0, 2, 3, 5, 6]),
    (130, 64, [17, 3], [6, 1]),
    (48, 256, [90, 100], [3, 5]),
]


@pytest.fixture(autouse=True)
def _dequant_from_ref(monkeypatch):
    def dq(B, A, eids, N, K):
        ids = [int(e) for e in eids.tolist()]
        if not ids:
            return torch.empty(0, N, K, dtype=torch.bfloat16)
        return torch.stack([dequant_ref(B[e], A[e], N, K).to(torch.bfloat16) for e in ids])
    monkeypatch.setattr(nf4_route, "dequant_groups", dq)


def _ref32(x, B, A, sizes, eids, N, K, mode):
    out = torch.zeros(x.shape[0], N if mode == "fwd" else K, dtype=torch.float32)
    r0 = 0
    for e, n in zip(eids, sizes):
        w = dequant_ref(B[e], A[e], N, K).float()
        out[r0:r0 + n] = x[r0:r0 + n].float() @ (w.t() if mode == "fwd" else w)
        r0 += n
    return out


def _dense(x, B, A, sizes, eids, N, K, mode):
    """``dense_forward`` / ``dense_dgrad``'s arithmetic in torch: bf16 operands, fp32 accumulation, one bf16 rounding."""
    out = torch.zeros(x.shape[0], N if mode == "fwd" else K, dtype=torch.bfloat16)
    r0 = 0
    for e, n in zip(eids, sizes):
        w = dequant_ref(B[e], A[e], N, K).to(torch.bfloat16).float()
        out[r0:r0 + n] = (x[r0:r0 + n].float() @ (w.t() if mode == "fwd" else w)).to(torch.bfloat16)
        r0 += n
    return out


def _rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12))


def _inputs(N, K, sizes, mode, seed):
    B, A = make_stack(8, N, K, seed=seed)
    g = torch.Generator().manual_seed(seed + 1)
    x = (torch.randn(sum(sizes), K if mode == "fwd" else N, generator=g) * 0.5).to(torch.bfloat16)
    return B, A, x


def _call(mode, *args):
    return (nf4_route.decoded_forward if mode == "fwd" else nf4_route.decoded_dgrad)(*args)


@pytest.mark.parametrize("mode", ["fwd", "dgrad"])
@pytest.mark.parametrize("N,K,sizes,eids", CASES)
def test_the_rd1_gate_holds(monkeypatch, mode, N, K, sizes, eids):
    monkeypatch.delenv("GNF4_DECODED_MAX_BYTES", raising=False)
    B, A, x = _inputs(N, K, sizes, mode, seed=N + K)
    y = _call(mode, x, B, A, sizes, eids)
    assert y.shape == (sum(sizes), N if mode == "fwd" else K) and y.dtype == torch.bfloat16
    ref = _ref32(x, B, A, sizes, eids, N, K, mode)
    err, dense_err = _rel(y, ref), _rel(_dense(x, B, A, sizes, eids, N, K, mode), ref)
    assert dense_err > 0
    assert err <= GATE_X * dense_err, (mode, N, K, err, dense_err)


@pytest.mark.parametrize("mode", ["fwd", "dgrad"])
@pytest.mark.parametrize("N,K,sizes,eids", CASES)
def test_every_output_is_within_one_bf16_ulp_of_its_exact_product(monkeypatch, mode, N, K, sizes, eids):
    monkeypatch.delenv("GNF4_DECODED_MAX_BYTES", raising=False)
    B, A, x = _inputs(N, K, sizes, mode, seed=N + K + 1)
    y = _call(mode, x, B, A, sizes, eids).float()
    exact = torch.zeros_like(y)
    r0 = 0
    for e, n in zip(eids, sizes):
        w = dequant_ref(B[e], A[e], N, K).to(torch.bfloat16).double()
        exact[r0:r0 + n] = (x[r0:r0 + n].double() @ (w.t() if mode == "fwd" else w)).float()
        r0 += n
    ulp = torch.exp2(torch.floor(torch.log2(exact.abs().clamp_min(2.0 ** -120))) - 7)
    worst = ((y - exact).abs() / ulp).max().item()
    assert worst <= 1.0, (mode, N, K, worst)


@pytest.mark.parametrize("mode", ["fwd", "dgrad"])
@pytest.mark.parametrize("N,K,sizes,eids", [(64, 128, [5, 0, 33, 7], [7, 1, 2, 4]), (96, 192, [0, 0, 40, 2, 9], [0, 2, 3, 5, 6])])
def test_the_chunk_plan_moves_no_bit(monkeypatch, mode, N, K, sizes, eids):
    B, A, x = _inputs(N, K, sizes, mode, seed=3)
    monkeypatch.delenv("GNF4_DECODED_MAX_BYTES", raising=False)
    whole = _call(mode, x, B, A, sizes, eids)
    for per_chunk in (1, 2):                                   # (0, 0) is an all-empty chunk in the second case at one per chunk
        monkeypatch.setenv("GNF4_DECODED_MAX_BYTES", str(per_chunk * N * K * 2))
        assert len(nf4_route.decoded_chunks(len(sizes), N, K, nf4_route.decoded_max_bytes())) == -(-len(sizes) // per_chunk)
        assert torch.equal(_call(mode, x, B, A, sizes, eids), whole), per_chunk


def test_tensor_sizes_and_ids_match_lists(monkeypatch):
    monkeypatch.delenv("GNF4_DECODED_MAX_BYTES", raising=False)
    N, K, sizes, eids = CASES[0]
    B, A, x = _inputs(N, K, sizes, "fwd", seed=5)
    lists = nf4_route.decoded_forward(x, B, A, sizes, eids)
    tensors = nf4_route.decoded_forward(x, B, A, torch.tensor(sizes), torch.tensor(eids, dtype=torch.int32))
    assert torch.equal(lists, tensors)


def test_the_route_counts_itself(monkeypatch):
    monkeypatch.delenv("GNF4_DECODED_MAX_BYTES", raising=False)
    nf4_route.ROUTE_STATS.update(decoded_fwd=0, decoded_dgrad=0, dense_fwd=0, dense_dgrad=0, fwd=0, dgrad=0)
    N, K, sizes, eids = CASES[2]
    B, A, x = _inputs(N, K, sizes, "fwd", seed=7)
    nf4_route.decoded_forward(x, B, A, sizes, eids)
    _, _, g = _inputs(N, K, sizes, "dgrad", seed=7)
    nf4_route.decoded_dgrad(g, B, A, sizes, eids)
    assert nf4_route.ROUTE_STATS == {"fwd": 0, "dgrad": 0, "dense_fwd": 0, "dense_dgrad": 0, "decoded_fwd": 1, "decoded_dgrad": 1}
