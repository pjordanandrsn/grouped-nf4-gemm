# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""K19 grouped small-M int4-b32 GEMM: the correctness contract, device-free (Triton interpreter mode, CPU).

Guards: (1) every sorted output row within one bf16 output ulp of ``x[src] @ dequant_int4_ref(packed[e]).T``;
(2) bit-identical to K16 (``gemm_int4_b32_smallm``, sk=1, the same block_n/kc) on each expert's rows -- grouping
changes no arithmetic, because a row's MMA output does not depend on the tile's other rows; (3) the in-kernel gather
(``order``) is bit-identical to gathering first and passing sorted rows; (4) an expert with more than 16 rows spans
several tiles and still matches; (5) deterministic; (6) a layout mismatch is refused before any launch; (7) lane K23's
``scatter`` store is bit-identical to the caller's ``index_copy_`` unsort, with and without the gather; (8) K23's
``gather_div`` reads token rows exactly as the gather reads their ``repeat_interleave`` expansion. Set by
conftest/CI: TRITON_INTERPRET=1 (the dot runs with fp32 operands there); the same file compiled on a GPU
(TRITON_INTERPRET=0) exercises the bf16 tensor-core arithmetic and owns the numerics claim.
"""
import os
os.environ.setdefault("TRITON_INTERPRET", "1")

import pytest
import torch

pytest.importorskip("triton", reason="interpreter mode needs triton (Linux-only dependency)")

from int4_b32 import build_group_tiles_fused                                              # noqa: E402
from int4_pack_ref import dequant_int4_ref, pack_int4_b32                                  # noqa: E402
from int4_smallm import gemm_int4_b32_grouped_smallm, gemm_int4_b32_smallm                  # noqa: E402

INTERP = os.environ.get("TRITON_INTERPRET", "0") == "1"
DEV = "cpu" if INTERP else ("cuda" if torch.cuda.is_available() else None)
if DEV is None:
    pytest.skip("compiled mode needs a CUDA device; set TRITON_INTERPRET=1 for the CPU contract run", allow_module_level=True)


def _stack(E, N, K, seed=0):
    g = torch.Generator().manual_seed(seed)
    packed, scales, deq = [], [], []
    for _ in range(E):
        w = torch.randn(N, K, generator=g) * 0.7
        p, s = pack_int4_b32(w)
        packed.append(p)
        scales.append(s)
        deq.append(dequant_int4_ref(p, s, N, K).float())
    return torch.stack(packed).contiguous(), torch.stack(scales).contiguous(), deq


def _case(E, N, K, eids, seed=0):
    g = torch.Generator().manual_seed(seed + 1)
    packed, scales, deq = _stack(E, N, K, seed)
    R = len(eids)
    x = (torch.randn(R, K, generator=g) * 0.5).to(torch.bfloat16)
    e = torch.tensor(eids, dtype=torch.int32)
    row0, rows, grp, order, counts = build_group_tiles_fused(e.to(DEV), E, 16)
    return x, packed, scales, deq, e, (row0, rows, grp, order, counts)


def _ref_sorted(x, deq, e, order):
    """The dequant-then-GEMM path's arithmetic per sorted row: bf16-rounded weights, fp32 accumulate."""
    o = order.cpu()
    return torch.stack([x[o[r]].float() @ deq[int(e[o[r]])].to(torch.bfloat16).float().t() for r in range(len(o))])


def _ulp_ok(y, ref):
    err = (y.float().cpu() - ref).abs().max()
    return bool(err <= 2 ** -7 * ref.abs().max()), float(err)


CASES = [
    (4, 64, 256, [0, 1, 2, 3, 1, 1, 0, 2, 3, 3, 0, 1], "12 rows over 4 experts"),
    (4, 100, 128, [2] * 5 + [0] * 3 + [3] * 4, "N not a multiple of BLOCK_N; expert 1 unused"),
    (3, 48, 96, [0, 1, 2] * 4, "checkout K=96 (KC 32)"),
    (4, 64, 128, [1] * 20 + [3] * 2, "an expert with 20 rows spans two tiles"),
]


@pytest.mark.parametrize("E,N,K,eids,label", CASES, ids=[c[-1] for c in CASES])
def test_matches_dequant_reference(E, N, K, eids, label):
    x, packed, scales, deq, e, (row0, rows, grp, order, _c) = _case(E, N, K, eids)
    y = gemm_int4_b32_grouped_smallm(x.to(DEV), packed.to(DEV), scales.to(DEV), row0, rows, grp, order)
    assert y.shape == (len(eids), N) and y.dtype == torch.bfloat16
    ok, err = _ulp_ok(y, _ref_sorted(x, deq, e, order))
    assert ok, f"{label}: max |y - ref| {err} exceeds one bf16 ulp"


@pytest.mark.parametrize("E,N,K,eids,label", CASES, ids=[c[-1] for c in CASES])
def test_bitwise_equals_k16_per_expert(E, N, K, eids, label):
    """Grouping changes no arithmetic: each expert's sorted rows equal K16 (sk=1, same tile config) on those rows."""
    x, packed, scales, deq, e, (row0, rows, grp, order, counts) = _case(E, N, K, eids)
    # the SAME tile config on both sides (K16's defaults): the claim is per plan. K19's own default differs
    # since K20 (BLOCK_N 32, KC 256); across plans the compiled kernel is bit-identical too (the test below), but under
    # the interpreter the fp32 dot is numpy's and KC can move the last bit.
    y = gemm_int4_b32_grouped_smallm(x.to(DEV), packed.to(DEV), scales.to(DEV), row0, rows, grp, order,
                                     block_n=64, kc=128).cpu()
    o = order.cpu()
    off = 0
    for ex, cnt in enumerate(counts.cpu().tolist()):
        for t0 in range(0, cnt, 16):                       # K16 serves <= 16 rows: compare tile by tile
            n = min(16, cnt - t0)
            src = o[off + t0: off + t0 + n]
            k16 = gemm_int4_b32_smallm(x[src].to(DEV), packed[ex].to(DEV), scales[ex].to(DEV), sk=1).cpu()
            assert torch.equal(y[off + t0: off + t0 + n], k16), f"{label}: expert {ex} rows {t0}..{t0 + n} differ from K16"
        off += cnt


def test_gather_path_is_bitwise_the_presorted_path():
    x, packed, scales, deq, e, (row0, rows, grp, order, _c) = _case(*CASES[0][:4])
    a = gemm_int4_b32_grouped_smallm(x.to(DEV), packed.to(DEV), scales.to(DEV), row0, rows, grp, order)
    xs = x[order.cpu()].contiguous()
    b = gemm_int4_b32_grouped_smallm(xs.to(DEV), packed.to(DEV), scales.to(DEV), row0, rows, grp, None)
    assert torch.equal(a, b)


def test_deterministic():
    x, packed, scales, deq, e, (row0, rows, grp, order, _c) = _case(*CASES[3][:4])
    a = gemm_int4_b32_grouped_smallm(x.to(DEV), packed.to(DEV), scales.to(DEV), row0, rows, grp, order)
    b = gemm_int4_b32_grouped_smallm(x.to(DEV), packed.to(DEV), scales.to(DEV), row0, rows, grp, order)
    assert torch.equal(a, b)


def test_layout_mismatch_is_refused_before_launch():
    x, packed, scales, deq, e, (row0, rows, grp, order, _c) = _case(*CASES[0][:4])
    with pytest.raises(ValueError, match="layout mismatch"):
        gemm_int4_b32_grouped_smallm(x[:, :128].to(DEV), packed.to(DEV), scales.to(DEV), row0, rows, grp, order)


@pytest.mark.skipif(INTERP, reason="compiled only: under the interpreter the fp32 dot is numpy's and KC can move the last bit")
@pytest.mark.parametrize("K", [768, 2048], ids=["down K768", "gate_up K2048"])
def test_plans_are_bit_identical_compiled(K):
    """K20: compiled, a plan changes speed and never an output bit. The MMA accumulates the same products in the
    same order whatever BLOCK_N and KC are (70 of 72 plans read equal on an RTX 5090), which is what let K20 move the
    default plan to BLOCK_N 32 / KC 256 with no quality read."""
    eids = [0, 1, 2, 3, 4, 5, 6, 7] * 6 + [3] * 17
    x, packed, scales, deq, e, (row0, rows, grp, order, _c) = _case(8, 256, K, eids, seed=5)
    args = (x.to(DEV), packed.to(DEV), scales.to(DEV), row0, rows, grp, order)
    ref = gemm_int4_b32_grouped_smallm(*args, block_n=64, kc=128, warps=4, stages=2)
    for bn, kc, wp, st in [(32, 256, 4, 2), (64, 256, 8, 3), (128, 128, 4, 3), (32, 64, 4, 2), (256, 128, 8, 2)]:
        y = gemm_int4_b32_grouped_smallm(*args, block_n=bn, kc=kc, warps=wp, stages=st)
        assert torch.equal(y, ref), f"plan ({bn}, {kc}, {wp}, {st}) moved an output bit at K={K}"


@pytest.mark.parametrize("E,N,K,eids,label", CASES, ids=[c[-1] for c in CASES])
def test_scatter_is_the_unsort(E, N, K, eids, label):
    """K23: ``scatter=order`` stores sorted row i at row order[i] -- the same bits as ``index_copy_`` of the sorted
    result, both for the gathered first projection and for the sorted-input second one."""
    x, packed, scales, _deq, _e, (row0, rows, grp, order, _c) = _case(E, N, K, eids)
    w = (packed.to(DEV), scales.to(DEV), row0, rows, grp)
    xs = x.to(DEV)
    y = gemm_int4_b32_grouped_smallm(xs, *w, order)
    got = gemm_int4_b32_grouped_smallm(xs, *w, order, scatter=order)
    assert torch.equal(got, torch.empty_like(y).index_copy_(0, order, y)), label
    x_sorted = xs.index_select(0, order).contiguous()
    y2 = gemm_int4_b32_grouped_smallm(x_sorted, *w)
    got2 = gemm_int4_b32_grouped_smallm(x_sorted, *w, scatter=order)
    assert torch.equal(got2, torch.empty_like(y2).index_copy_(0, order, y2)), label


def test_scatter_length_is_checked():
    x, packed, scales, _deq, _e, (row0, rows, grp, order, _c) = _case(*CASES[0][:4])
    with pytest.raises(ValueError, match="scatter has"):
        gemm_int4_b32_grouped_smallm(x.to(DEV), packed.to(DEV), scales.to(DEV), row0, rows, grp, order,
                                     scatter=order[:-1])


@pytest.mark.parametrize("k", [1, 2, 4])
def test_gather_div_reads_token_rows_as_their_expansion(k):
    """K23: x = T token rows, gather_div=k, order over T*k (token, slot) rows -- the same bits as handing K19 the
    expansion ``x.repeat_interleave(k, 0)``, with scatter too."""
    E, N, K = 4, 64, 128
    T = 6
    eids = [(3 * t + s) % E for t in range(T) for s in range(k)]
    x, packed, scales, _deq, _e, (row0, rows, grp, order, _c) = _case(E, N, K, eids, seed=k)
    xt = x[:T].to(DEV)
    w = (packed.to(DEV), scales.to(DEV), row0, rows, grp)
    expanded = xt.repeat_interleave(k, 0)
    for kw in ({}, {"scatter": order}):
        want = gemm_int4_b32_grouped_smallm(expanded, *w, order, **kw)
        got = gemm_int4_b32_grouped_smallm(xt, *w, order, gather_div=k, **kw)
        assert torch.equal(got, want), (k, kw.keys())


def test_gather_div_is_checked():
    x, packed, scales, _deq, _e, (row0, rows, grp, order, _c) = _case(*CASES[0][:4])
    w = (packed.to(DEV), scales.to(DEV), row0, rows, grp)
    with pytest.raises(ValueError, match="gather_div=2 needs order"):
        gemm_int4_b32_grouped_smallm(x.to(DEV), *w, order, gather_div=2)       # 12 token rows x 2 != 12
    with pytest.raises(ValueError, match="gather_div=2 needs order"):
        gemm_int4_b32_grouped_smallm(x[:6].to(DEV), *w, None, gather_div=2)    # no order
