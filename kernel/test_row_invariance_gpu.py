# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Row-count invariance of the kernels lane P63 read EXACT, on a GPU (experts4bit-qlora#708).

A token decoded alone (one token's ``top_k`` rows in one call) and the same token inside a call over T tokens
must get the same bits from these kernels: none of them plans its arithmetic from the row count.
experts4bit-qlora's speculative verify (``FORCE_SINGLETON_GROUPS``) and its device-grouped decode GEMV (up to 256
routed rows) rely on it for their T > 1 rows to equal T = 1 decode. Lane P63 read it on an RTX 5090 at
Qwen3-30B-A3B's layer-0 gate_up with the model's own activations and routing (kernel/receipts-p63/5090/); these
tests assert it with ``torch.equal`` on random data at the served shapes, on any CUDA part:

- ``gemv_int4_b32``: one program per row. Its split-K count comes from ``_plan``, whose row term is gated to
  parts with <= ``SPLITK_R_TERM_MAX_SMS`` SMs; above that the plan is N-only, so every row count gets the same
  split count and the partials are reduced in the same order. The test pins the > 64-SM plan (the 5090's), so
  it asserts the route P63 read on any part. On a <= 64-SM part the box's own plan can change the split count
  with the rows; that is not claimed and not tested here.
- the NF4 dot-pad decode GEMV: one program per row, no split-K unless ``GNF4_GEMV_SPLITK`` is set. It is the
  default decode route at its census shapes on >= 160-SM parts; the test forces the dispatch there on any part
  and asserts from ``dispatch_counts()`` that the dot-pad kernel ran.
- ``combine_rows``: one program per token, slot order, a block size that does not depend on T.

``test_the_check_sees_a_split_k_change`` is the control: the scalar NF4 GEMV with its split count changed
between the single-token calls and the batched call (what the K1 plan does on a 5090 with
``GNF4_GEMV_DOTPAD=0``, which P63 read REORDER) must NOT come out bit-equal, so a comparison that could never
fail would fail it.

    cd kernel && python -m pytest test_row_invariance_gpu.py -q
"""
from __future__ import annotations

import os
import sys

import pytest

torch = pytest.importorskip("torch")

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="row-count invariance is a property of the compiled kernels: needs CUDA")

DEV = "cuda"
TOP_K = 8                        # Qwen3-30B-A3B routes 8 of 128 experts per token
EXPERTS = 16                     # enough distinct experts for the routing to repeat and to spread
TOKENS = (16, 17, 160)           # P63's verify windows and its prefill
QWEN3_SHAPES = [(1536, 2048), (2048, 768)]   # (N, K): gate_up and down, the _DOTPAD_CONFIGS entries
BIG_PART_SMS = 170               # an RTX 5090: above SPLITK_R_TERM_MAX_SMS, at or above dot-pad's 160


def _routing(T: int, seed: int):
    """``[T * TOP_K]`` int32 expert ids, token-major, TOP_K distinct experts per token (a router's top-k)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.stack([torch.randperm(EXPERTS, generator=g)[:TOP_K] for _ in range(T)])
    return ids.reshape(-1).to(torch.int32).to(DEV)


def _tokens(T: int, K: int, seed: int):
    """``[T, K]`` bf16 activations and their ``[T * TOP_K, K]`` routed rows (token t's row repeated per slot)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(T, K, generator=g).to(torch.bfloat16).to(DEV)
    return x, x.repeat_interleave(TOP_K, 0).contiguous()


def _assert_rows_are_single_token_calls(fn, rows, ids, T: int, what: str):
    """One call over all T tokens' rows vs T calls of one token's TOP_K rows each: every row bit for bit."""
    batched = fn(rows, ids)
    singles = torch.cat([fn(rows[t * TOP_K:(t + 1) * TOP_K].contiguous(), ids[t * TOP_K:(t + 1) * TOP_K])
                         for t in range(T)], 0)
    torch.cuda.synchronize()
    assert batched.shape == singles.shape, (what, batched.shape, singles.shape)
    same = (batched.view(torch.int16) == singles.view(torch.int16)).all(dim=1)
    assert torch.equal(batched, singles), (
        f"{what}: {int((~same).sum())} of {same.numel()} rows differ from the token's own single-token call")


# ------------------------------------------------------------------------------------------------ int4 GEMV --
@needs_cuda
@pytest.mark.parametrize("N,K", QWEN3_SHAPES)
@pytest.mark.parametrize("T", TOKENS)
def test_int4_gemv_rows_are_their_own_single_token_call(monkeypatch, N, K, T):
    import int4_b32 as I
    from int4_pack_ref import pack_int4_b32
    monkeypatch.delenv("GNF4_GEMV_FUSED_REDUCE", raising=False)     # the default two-launch reduce
    monkeypatch.setattr(I, "_sm_count", lambda device: BIG_PART_SMS)
    # the mechanism, asserted rather than assumed: the plan these rows get does not move with the row count
    assert I._plan(N, K, TOP_K, BIG_PART_SMS) == I._plan(N, K, T * TOP_K, BIG_PART_SMS)
    g = torch.Generator(device="cpu").manual_seed(N + K)
    packs = [pack_int4_b32(torch.randn(N, K, generator=g) * 0.02) for _ in range(EXPERTS)]
    P = torch.stack([p for p, _ in packs]).to(DEV)
    S = torch.stack([s for _, s in packs]).to(DEV)
    _x, rows = _tokens(T, K, seed=T)
    ids = _routing(T, seed=T + 1)

    def gemv(xr, eids):
        xq, xs = I.quant_x_rows(xr)
        return I.gemv_int4_b32(xq, xs, P, S, eids, N, K)
    _assert_rows_are_single_token_calls(gemv, rows, ids, T, f"gemv_int4_b32 ({N}, {K}) at T={T}")


# ---------------------------------------------------------------------------------------- NF4 dot-pad GEMV --
def _nf4_stack(N, K):
    from nf4_pack_ref import make_stack
    return make_stack(EXPERTS, N, K, seed=N + K, device=DEV)


@needs_cuda
@pytest.mark.parametrize("N,K", QWEN3_SHAPES)
@pytest.mark.parametrize("T", TOKENS)
def test_nf4_dotpad_rows_are_their_own_single_token_call(monkeypatch, N, K, T):
    import nf4_grouped as NG
    assert (N, K) in NG._DOTPAD_CONFIGS, "the dot-pad kernel no longer registers this census shape"
    monkeypatch.delenv("GNF4_GEMV_DOTPAD", raising=False)          # the shipped default: on
    monkeypatch.delenv("GNF4_GEMV_SPLITK", raising=False)          # dot-pad split-K is opt-in; not this route
    monkeypatch.setattr(NG, "_sm_count", lambda device: BIG_PART_SMS)
    B, A = _nf4_stack(N, K)
    _x, rows = _tokens(T, K, seed=T + 2)
    ids = _routing(T, seed=T + 3)

    def decode(xr, eids):                                          # the singleton contract: one row per group
        return NG.gemm_4bit_grouped(xr, B, A, [1] * xr.shape[0], eids)
    NG.reset_dispatch_counts()
    _assert_rows_are_single_token_calls(decode, rows, ids, T, f"NF4 dot-pad ({N}, {K}) at T={T}")
    counts = NG.dispatch_counts()
    assert counts["dotpad"] == T + 1 and sum(counts.values()) == T + 1, (
        f"expected the dot-pad kernel on every call, got {counts}")


@needs_cuda
def test_the_check_sees_a_split_k_change(monkeypatch):
    """The control. The scalar NF4 GEMV with one split for the single-token calls and four for the batched
    call groups the same fp32 products differently; the comparison above must report it."""
    import nf4_grouped as NG
    monkeypatch.setenv("GNF4_GEMV_DOTPAD", "0")
    monkeypatch.delenv("GNF4_GEMV_SPLITK", raising=False)
    N, K, T = 1536, 2048, 17
    B, A = _nf4_stack(N, K)
    _x, rows = _tokens(T, K, seed=5)
    ids = _routing(T, seed=6)

    def split_by_rows(xr, eids):
        sk = 1 if xr.shape[0] == TOP_K else 4
        return NG.gemm_4bit_grouped(xr, B, A, [1] * xr.shape[0], eids, split_k=sk)
    with pytest.raises(AssertionError, match="rows differ"):
        _assert_rows_are_single_token_calls(split_by_rows, rows, ids, T, "control: scalar GEMV, split-K 1 vs 4")


# ------------------------------------------------------------------------------------------------ combine_rows --
@needs_cuda
@pytest.mark.parametrize("k,H", [(8, 2048), (8, 1536), (2, 4096), (4, 2880), (8, 2816)])
def test_combine_rows_rows_are_their_own_single_token_call(k, H):
    """Token t's combined row from a T-token call is its row from a one-token call, at T in {2, 16, 17, 64, 160}
    (P63's census; (8, 2048) is Qwen3's (top_k, hidden), the others are B393's families)."""
    from int4_b32 import combine_rows
    g = torch.Generator(device="cpu").manual_seed(k * H)
    Tmax = 160
    dn = torch.randn(Tmax * k, H, generator=g).to(torch.bfloat16).to(DEV)
    w = torch.softmax(torch.randn(Tmax, k, generator=g), -1).reshape(-1).to(DEV)
    singles = torch.cat([combine_rows(dn[t * k:(t + 1) * k], w[t * k:(t + 1) * k], k) for t in range(Tmax)], 0)
    for T in (2, 16, 17, 64, 160):
        got = combine_rows(dn[:T * k], w[:T * k], k)
        torch.cuda.synchronize()
        assert torch.equal(got, singles[:T]), f"combine_rows (k={k}, H={H}) at T={T} differs from T=1"
