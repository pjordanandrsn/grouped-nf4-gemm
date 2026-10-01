# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""K25 grouped small-M NF4 GEMM: the correctness contract, device-free (Triton interpreter mode, CPU).

Guards:
1. every sorted output row is within one bf16 output ulp of ``x[src] @ dequant_ref(packed[e], absmax[e]).T`` -- with
   the weight rounded to bf16 when compiled (the dequant-then-GEMM path's operand), fp32 under the interpreter;
2. the masked K tail (KC not dividing K) matches the reference too;
3. the two codebook decodes (pair, load) are bit-identical: the same fp32 values by construction;
4. the in-kernel gather (``order``) is bit-identical to gathering first;
5. strided views of a fused stack (the expert and row strides free) are bit-identical to a contiguous copy;
6. K23's ``scatter`` is the unsort and ``gather_div`` reads token rows as their expansion, bit for bit;
7. deterministic;
8. a layout, dtype, KC or decode mismatch is refused before any launch;
9. compiled only: the plan (BLOCK_N, KC, warps, stages) moves no output bit, K19's K20 property;
10. compiled only: the weight operand is exactly ``dequant_ref(...).to(bf16)`` -- one expert, one row of ones per
    64-block reads each output as that block sum, compared bit for bit.

Set by conftest/CI: TRITON_INTERPRET=1 (fp32 dot operands there). The same file compiled on a GPU exercises the bf16
tensor-core arithmetic.
"""
import os
os.environ.setdefault("TRITON_INTERPRET", "1")

import pytest
import torch

pytest.importorskip("triton", reason="interpreter mode needs triton (Linux-only dependency)")

from int4_b32 import build_group_tiles_fused                                     # noqa: E402
from nf4_grouped import dequant_ref                                                # noqa: E402
from nf4_pack_ref import quantize_pack_nf4                                       # noqa: E402
from nf4_smallm import gemm_nf4_grouped_smallm, pair_lut                         # noqa: E402

INTERP = os.environ.get("TRITON_INTERPRET", "0") == "1"
DEV = "cpu" if INTERP else ("cuda" if torch.cuda.is_available() else None)
if DEV is None:
    pytest.skip("compiled mode needs a CUDA device; set TRITON_INTERPRET=1 for the CPU contract run", allow_module_level=True)

LUTS = ["pair", "load"]


def _stack(E, N, K, seed=0):
    g = torch.Generator().manual_seed(seed)
    packed, absmax, deq = [], [], []
    for _ in range(E):
        w = torch.randn(N, K, generator=g) * 0.05
        p, a = quantize_pack_nf4(w)                         # [N, K//2] u8, [N, K//64] fp32
        packed.append(p)
        absmax.append(a)
        deq.append(dequant_ref(p, a, N, K))
    return torch.stack(packed).contiguous(), torch.stack(absmax).contiguous(), deq


def _case(E, N, K, eids, seed=0):
    g = torch.Generator().manual_seed(seed + 1)
    packed, absmax, deq = _stack(E, N, K, seed)
    x = (torch.randn(len(eids), K, generator=g) * 0.5).to(torch.bfloat16)
    e = torch.tensor(eids, dtype=torch.int32)
    tiles = build_group_tiles_fused(e.to(DEV), E, 16)
    return x, packed, absmax, deq, e, tiles


def _ref_sorted(x, deq, e, order):
    o = order.cpu()
    wt = (lambda w: w) if INTERP else (lambda w: w.to(torch.bfloat16).float())
    return torch.stack([x[o[r]].float() @ wt(deq[int(e[o[r]])]).t() for r in range(len(o))])


CASES = [
    (4, 64, 256, [0, 1, 2, 3, 1, 1, 0, 2, 3, 3, 0, 1], "12 rows over 4 experts"),
    (4, 100, 128, [2] * 5 + [0] * 3 + [3] * 4, "N not a multiple of BLOCK_N; expert 1 unused"),
    (3, 48, 64, [0, 1, 2] * 4, "K=64 (KC 64)"),
    (4, 64, 128, [1] * 20 + [3] * 2, "an expert with 20 rows spans two tiles"),
    (2, 32, 2880, [0, 1, 1, 0, 1], "gpt-oss K=2880 (KC 64)"),
]
# the masked K tail: KC that does not divide K
TAIL_CASES = [(2, 32, 2880, 256, [0, 1, 1, 0, 1], "K=2880 at KC 256"), (2, 48, 2880, 128, [1, 0, 1], "K=2880 at KC 128"),
              (3, 40, 192, 128, [0, 1, 2, 2], "K=192 at KC 128"), (2, 24, 320, 256, [1, 1, 0], "K=320 at KC 256")]


def _kc_for(K):
    return next(c for c in (256, 128, 64) if K % c == 0)


def _args(x, packed, absmax, row0, rows, grp):
    return (x.to(DEV), packed.to(DEV), absmax.to(DEV), row0, rows, grp)


@pytest.mark.parametrize("E,N,K,eids,label", CASES, ids=[c[-1] for c in CASES])
def test_matches_dequant_reference(E, N, K, eids, label):
    x, packed, absmax, deq, e, (row0, rows, grp, order, _c) = _case(E, N, K, eids)
    y = gemm_nf4_grouped_smallm(*_args(x, packed, absmax, row0, rows, grp), order, kc=_kc_for(K))
    assert y.shape == (len(eids), N) and y.dtype == torch.bfloat16
    ref = _ref_sorted(x, deq, e, order)
    err = float((y.float().cpu() - ref).abs().max())
    assert err <= 2 ** -7 * float(ref.abs().max()), f"{label}: max |y - ref| {err} exceeds one bf16 ulp"


@pytest.mark.parametrize("E,N,K,kc,eids,label", TAIL_CASES, ids=[c[-1] for c in TAIL_CASES])
def test_masked_tail_matches_dequant_reference(E, N, K, kc, eids, label):
    x, packed, absmax, deq, e, (row0, rows, grp, order, _c) = _case(E, N, K, eids)
    y = gemm_nf4_grouped_smallm(*_args(x, packed, absmax, row0, rows, grp), order, kc=kc)
    ref = _ref_sorted(x, deq, e, order)
    err = float((y.float().cpu() - ref).abs().max())
    assert err <= 2 ** -7 * float(ref.abs().max()), f"{label}: max |y - ref| {err} exceeds one bf16 ulp"


def test_pair_lut_holds_both_codebook_values_of_each_byte():
    from nf4_grouped import NF4_LUT
    t = pair_lut("cpu")
    lut = torch.tensor(NF4_LUT, dtype=torch.float32)
    lo = (t & 0xFFFFFFFF).to(torch.int32).view(torch.float32)
    hi = (t >> 32).to(torch.int32).view(torch.float32)
    b = torch.arange(256)
    assert torch.equal(lo, lut[b >> 4]) and torch.equal(hi, lut[b & 0xF])


@pytest.mark.parametrize("E,N,K,eids,label", [CASES[0], CASES[3], CASES[4]], ids=[c[-1] for c in (CASES[0], CASES[3], CASES[4])])
def test_codebook_decodes_are_bit_identical(E, N, K, eids, label):
    x, packed, absmax, deq, e, (row0, rows, grp, order, _c) = _case(E, N, K, eids)
    args = _args(x, packed, absmax, row0, rows, grp)
    outs = {m: gemm_nf4_grouped_smallm(*args, order, kc=_kc_for(K), lut=m) for m in LUTS}
    for m, y in outs.items():
        assert torch.equal(y, outs["pair"]), f"{label}: lut={m} differs from lut=pair"


def test_gather_path_is_bitwise_the_presorted_path():
    x, packed, absmax, deq, e, (row0, rows, grp, order, _c) = _case(*CASES[0][:4])
    a = gemm_nf4_grouped_smallm(*_args(x, packed, absmax, row0, rows, grp), order)
    xs = x[order.cpu()].contiguous()
    b = gemm_nf4_grouped_smallm(*_args(xs, packed, absmax, row0, rows, grp), None)
    assert torch.equal(a, b)


def test_strided_views_of_a_fused_stack_match_a_contiguous_copy():
    """The served NF4 stacks are views (gate_up fused along N; experts sliced from a larger stack): only the last
    dimension must be contiguous."""
    E, N, K = 4, 64, 256
    eids = [0, 1, 2, 3, 1, 1, 0, 2, 3, 3, 0, 1]
    x, packed, absmax, deq, e, (row0, rows, grp, order, _c) = _case(E, 2 * N, K, eids)
    pv, av = packed[:, N:, :], absmax[:, N:, :]                     # the second half of a fused [E, 2N] stack
    assert not pv.is_contiguous() and not av.is_contiguous()
    got = gemm_nf4_grouped_smallm(*_args(x, pv, av, row0, rows, grp), order)
    want = gemm_nf4_grouped_smallm(*_args(x, pv.contiguous(), av.contiguous(), row0, rows, grp), order)
    assert torch.equal(got, want)


def test_deterministic():
    x, packed, absmax, deq, e, (row0, rows, grp, order, _c) = _case(*CASES[3][:4])
    args = _args(x, packed, absmax, row0, rows, grp)
    assert torch.equal(gemm_nf4_grouped_smallm(*args, order), gemm_nf4_grouped_smallm(*args, order))


@pytest.mark.parametrize("E,N,K,eids,label", CASES[:4], ids=[c[-1] for c in CASES[:4]])
def test_scatter_is_the_unsort(E, N, K, eids, label):
    x, packed, absmax, _deq, _e, (row0, rows, grp, order, _c) = _case(E, N, K, eids)
    w = (packed.to(DEV), absmax.to(DEV), row0, rows, grp)
    xs = x.to(DEV)
    kc = _kc_for(K)
    y = gemm_nf4_grouped_smallm(xs, *w, order, kc=kc)
    got = gemm_nf4_grouped_smallm(xs, *w, order, kc=kc, scatter=order)
    assert torch.equal(got, torch.empty_like(y).index_copy_(0, order, y)), label
    x_sorted = xs.index_select(0, order).contiguous()
    y2 = gemm_nf4_grouped_smallm(x_sorted, *w, kc=kc)
    got2 = gemm_nf4_grouped_smallm(x_sorted, *w, kc=kc, scatter=order)
    assert torch.equal(got2, torch.empty_like(y2).index_copy_(0, order, y2)), label


@pytest.mark.parametrize("k", [1, 2, 8])
def test_gather_div_reads_token_rows_as_their_expansion(k):
    E, N, K = 4, 64, 128
    T = 6
    eids = [(3 * t + s) % E for t in range(T) for s in range(k)]
    x, packed, absmax, _deq, _e, (row0, rows, grp, order, _c) = _case(E, N, K, eids, seed=k)
    xt = x[:T].to(DEV)
    w = (packed.to(DEV), absmax.to(DEV), row0, rows, grp)
    expanded = xt.repeat_interleave(k, 0)
    for kw in ({}, {"scatter": order}):
        want = gemm_nf4_grouped_smallm(expanded, *w, order, kc=128, **kw)
        got = gemm_nf4_grouped_smallm(xt, *w, order, kc=128, gather_div=k, **kw)
        assert torch.equal(got, want), (k, kw.keys())


def test_mismatches_are_refused_before_launch():
    x, packed, absmax, deq, e, (row0, rows, grp, order, _c) = _case(*CASES[0][:4])
    w = (packed.to(DEV), absmax.to(DEV), row0, rows, grp)
    xd = x.to(DEV)
    with pytest.raises(ValueError, match="layout mismatch"):
        gemm_nf4_grouped_smallm(xd[:, :128], *w, order)
    with pytest.raises(ValueError, match="must be uint8"):
        gemm_nf4_grouped_smallm(xd, packed.to(torch.int16).to(DEV), *w[1:], order)
    with pytest.raises(ValueError, match="absmax must be"):
        gemm_nf4_grouped_smallm(xd, w[0], absmax.to(torch.int32).to(DEV), *w[2:], order)
    with pytest.raises(ValueError, match="expected 64, 128 or 256"):
        gemm_nf4_grouped_smallm(xd, *w, order, kc=96)
    with pytest.raises(ValueError, match="expected 'pair' or 'load'"):
        gemm_nf4_grouped_smallm(xd, *w, order, lut="reg")
    with pytest.raises(ValueError, match="contiguous along their last"):
        gemm_nf4_grouped_smallm(xd, packed.transpose(1, 2).contiguous().transpose(1, 2).to(DEV), *w[1:], order)
    with pytest.raises(ValueError, match="scatter has"):
        gemm_nf4_grouped_smallm(xd, *w, order, scatter=order[:-1])
    with pytest.raises(ValueError, match="gather_div=2 needs order"):
        gemm_nf4_grouped_smallm(xd, *w, order, gather_div=2)
    with pytest.raises(ValueError, match="not a multiple of the NF4 block"):
        gemm_nf4_grouped_smallm(xd[:, :96], packed[:, :, :48].to(DEV), absmax[:, :, :1].to(DEV), *w[2:], order)


@pytest.mark.skipif(INTERP, reason="compiled only: under the interpreter the fp32 dot is numpy's and KC can move the last bit")
@pytest.mark.parametrize("K", [512, 1536, 2880], ids=["Granite down K512", "Granite gate_up K1536", "gpt-oss K2880"])
def test_plans_are_bit_identical_compiled(K):
    eids = [0, 1, 2, 3, 4, 5, 6, 7] * 6 + [3] * 17
    x, packed, absmax, deq, e, (row0, rows, grp, order, _c) = _case(8, 256, K, eids, seed=5)
    args = _args(x, packed, absmax, row0, rows, grp)
    ref = gemm_nf4_grouped_smallm(*args, order, block_n=64, kc=64, warps=4, stages=2)
    # plans that fit 99 KB of shared memory (the A2000's and the 5090's per-block limit): "load" overflows at
    # 32 x 256 and 64 x 128, "pair" at 128 x 256
    for bn, kc, wp, st, lut in [(32, 256, 4, 2, "pair"), (64, 128, 8, 3, "pair"), (128, 64, 4, 3, "pair"),
                                (32, 128, 4, 2, "load"), (16, 256, 4, 3, "pair"), (16, 256, 8, 2, "load"),
                                (64, 256, 4, 2, "pair")]:
        y = gemm_nf4_grouped_smallm(*args, order, block_n=bn, kc=kc, warps=wp, stages=st, lut=lut)
        assert torch.equal(y, ref), f"plan ({bn}, {kc}, {wp}, {st}, {lut}) moved an output bit at K={K}"


@pytest.mark.skipif(INTERP, reason="compiled only: the interpreter multiplies fp32 operands")
def test_weight_operand_is_the_bf16_dequant_compiled():
    """x = a one-hot row per output column block: with x = e_k (one 1 at column k), out[n] = bf16(W_bf16[n, k]) --
    every weight is read back through the MMA, exactly, and compared to ``dequant_ref(...).to(bf16)``."""
    E, N, K = 1, 64, 256
    x_all, packed, absmax, deq, e, _ = _case(E, N, K, [0])
    want = deq[0].to(torch.bfloat16)                                 # [N, K]
    for k in (0, 1, 63, 64, 130, 255):
        x = torch.zeros(1, K, dtype=torch.bfloat16)
        x[0, k] = 1.0
        row0, rows, grp, order, _c = build_group_tiles_fused(torch.zeros(1, dtype=torch.int32, device=DEV), E, 16)
        for lut in LUTS:
            y = gemm_nf4_grouped_smallm(*_args(x, packed, absmax, row0, rows, grp), order, lut=lut)
            assert torch.equal(y[0].cpu(), want[:, k]), f"column {k}, lut={lut}: the MMA operand is not the bf16 dequant"
