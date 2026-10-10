# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""K19 over a split tile table (e4b#846, opt-in): full ``big_m``-row tiles at ``block_m=big_m``, then the 16-row
remainders, into one buffer (``int4_smallm.gemm_int4_b32_grouped_split`` over ``nf4_grouped.build_group_tiles_split``).

Guards, interpreted (TRITON_INTERPRET=1, CPU) and compiled:
(1) every sorted row within one bf16 output ulp of ``x[src] @ dequant_int4_ref(packed[e]).T``, for big_m 32 and 64,
    with a heavy expert spanning several tall tiles and a light tail; (2) ``scatter`` stores exactly the caller's
    ``index_copy_`` unsort of the sorted result, and ``gather_div`` reads token rows as their ``repeat_interleave``
    expansion, as on the 16-row table; (3) deterministic; (4) ``out=`` fills only the rows its tiles name;
    (5) ``block_m`` outside (16, 32, 64) and a wrong ``out`` are refused before any launch; (6) the default launch
    (``block_m=16``, no ``out``) is unchanged: the same bits as the call without the new arguments.
Compiled only: (7) the split result against the 16-row table, ``torch.equal`` or not, reported per K. Whether a row on
a tall tile is the same bits as on a 16-row tile is the card's codegen, read on each target and never assumed; the
interpreter's fp32 dot is numpy's, so it cannot answer it.
"""
import os
os.environ.setdefault("TRITON_INTERPRET", "1")

import pytest
import torch

pytest.importorskip("triton", reason="interpreter mode needs triton (Linux-only dependency)")

from int4_pack_ref import dequant_int4_ref, pack_int4_b32                                  # noqa: E402
from int4_smallm import gemm_int4_b32_grouped_smallm, gemm_int4_b32_grouped_split           # noqa: E402
from nf4_grouped import build_group_tiles_device, build_group_tiles_split                  # noqa: E402

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


def _eids(counts, seed=0):
    out = []
    for e, c in enumerate(counts):
        out.extend([e] * c)
    t = torch.tensor(out, dtype=torch.int32)
    return t[torch.randperm(t.numel(), generator=torch.Generator().manual_seed(seed))]


def _case(counts, N, K, big_m, seed=0):
    E = len(counts)
    packed, scales, deq = _stack(E, N, K, seed)
    e = _eids(counts, seed)
    x = (torch.randn(e.numel(), K, generator=torch.Generator().manual_seed(seed + 1)) * 0.5).to(torch.bfloat16)
    big, small, order, counts_t = build_group_tiles_split(e.to(DEV), E, big_m)
    return x, packed.to(DEV), scales.to(DEV), deq, e, big, small, order


def _ref_sorted(x, deq, e, order):
    o = order.cpu()
    return torch.stack([x[o[r]].float() @ deq[int(e[o[r]])].to(torch.bfloat16).float().t() for r in range(len(o))])


def _ulp_ok(y, ref):
    err = (y.float().cpu() - ref).abs().max()
    return bool(err <= 2 ** -7 * ref.abs().max()), float(err)


# a heavy expert over several tall tiles, an exact multiple, one just past it, a light tail, an unused expert
COUNTS = [150, 3, 64, 65, 0, 9, 17, 2]
CASES = [(32, 64, 128), (64, 64, 128), (64, 48, 96)]


@pytest.mark.parametrize("big_m,N,K", CASES, ids=[f"big{b}_N{n}_K{k}" for b, n, k in CASES])
def test_split_matches_dequant_reference(big_m, N, K):
    x, packed, scales, deq, e, big, small, order = _case(COUNTS, N, K, big_m)
    y = gemm_int4_b32_grouped_split(x.to(DEV), packed, scales, big, small, big_m, order)
    assert y.shape == (sum(COUNTS), N) and y.dtype == torch.bfloat16
    ok, err = _ulp_ok(y, _ref_sorted(x, deq, e, order))
    assert ok, f"big_m={big_m}: max |y - ref| {err} exceeds one bf16 ulp"


@pytest.mark.parametrize("big_m", [32, 64])
def test_split_scatter_is_the_unsort(big_m):
    x, packed, scales, _deq, _e, big, small, order = _case(COUNTS, 64, 128, big_m, seed=3)
    xs = x.to(DEV)
    y = gemm_int4_b32_grouped_split(xs, packed, scales, big, small, big_m, order)
    got = gemm_int4_b32_grouped_split(xs, packed, scales, big, small, big_m, order, scatter=order)
    assert torch.equal(got, torch.empty_like(y).index_copy_(0, order, y))
    x_sorted = xs.index_select(0, order).contiguous()
    y2 = gemm_int4_b32_grouped_split(x_sorted, packed, scales, big, small, big_m)
    got2 = gemm_int4_b32_grouped_split(x_sorted, packed, scales, big, small, big_m, scatter=order)
    assert torch.equal(got2, torch.empty_like(y2).index_copy_(0, order, y2))


@pytest.mark.parametrize("k", [2, 8])
def test_split_gather_div_reads_token_rows(k):
    T, E, N, K = 40, 4, 64, 128
    eids = [0 if (t + s) % 3 else (s % E) for t in range(T) for s in range(k)]    # expert 0 is heavy
    packed, scales, _deq = _stack(E, N, K, seed=k)
    e = torch.tensor(eids, dtype=torch.int32)
    xt = (torch.randn(T, K, generator=torch.Generator().manual_seed(k)) * 0.5).to(torch.bfloat16).to(DEV)
    big, small, order, _c = build_group_tiles_split(e.to(DEV), E, 32)
    w = (packed.to(DEV), scales.to(DEV), big, small, 32)
    for kw in ({}, {"scatter": order}):
        want = gemm_int4_b32_grouped_split(xt.repeat_interleave(k, 0), *w, order, **kw)
        got = gemm_int4_b32_grouped_split(xt, *w, order, gather_div=k, **kw)
        assert torch.equal(got, want), (k, kw.keys())


def test_split_is_deterministic():
    x, packed, scales, _deq, _e, big, small, order = _case(COUNTS, 64, 128, 64, seed=4)
    a = gemm_int4_b32_grouped_split(x.to(DEV), packed, scales, big, small, 64, order)
    b = gemm_int4_b32_grouped_split(x.to(DEV), packed, scales, big, small, 64, order)
    assert torch.equal(a, b)


def test_out_fills_only_the_rows_its_tiles_name():
    x, packed, scales, _deq, _e, big, small, order = _case(COUNTS, 64, 128, 64, seed=5)
    R = x.shape[0]
    sentinel = torch.full((R, 64), 7.0, dtype=torch.bfloat16, device=DEV)
    y = gemm_int4_b32_grouped_smallm(x.to(DEV), packed, scales, *big, order, block_m=64, out=sentinel)
    assert y.data_ptr() == sentinel.data_ptr()
    named = torch.zeros(R, dtype=torch.bool)
    for r0, rw in zip(big[0].tolist(), big[1].tolist()):
        named[r0:r0 + rw] = True
    assert bool((y.cpu()[~named] == 7.0).all()) and not bool((y.cpu()[named] == 7.0).all())


def test_refusals_before_launch():
    x, packed, scales, _deq, _e, big, small, order = _case(COUNTS, 64, 128, 64)
    xs = x.to(DEV)
    with pytest.raises(ValueError, match="block_m=48"):
        gemm_int4_b32_grouped_smallm(xs, packed, scales, *big, order, block_m=48)
    for bad in (torch.empty(xs.shape[0], 63, dtype=torch.bfloat16, device=DEV),
                torch.empty(xs.shape[0], 64, dtype=torch.float32, device=DEV)):
        with pytest.raises(ValueError, match="out must be"):
            gemm_int4_b32_grouped_smallm(xs, packed, scales, *big, order, block_m=64, out=bad)


def test_the_default_launch_is_unchanged():
    """``block_m=16`` and no ``out`` are the launch K19 always made: the same bits as the call without them."""
    x, packed, scales, _deq, e, _big, _small, _order = _case(COUNTS, 64, 128, 64, seed=6)
    row0, rows, grp, order, _c = build_group_tiles_device(e.to(DEV), len(COUNTS), 16)
    a = gemm_int4_b32_grouped_smallm(x.to(DEV), packed, scales, row0, rows, grp, order)
    b = gemm_int4_b32_grouped_smallm(x.to(DEV), packed, scales, row0, rows, grp, order, block_m=16, out=None)
    assert torch.equal(a, b)


@pytest.mark.skipif(INTERP, reason="compiled only: the interpreter's fp32 dot is numpy's and cannot read the card's codegen")
@pytest.mark.parametrize("big_m", [32, 64])
@pytest.mark.parametrize("K", [768, 2048], ids=["down K768", "gate_up K2048"])
def test_split_against_16_row_tiles_compiled(K, big_m):
    """The bit question, on this card: the split result against the 16-row table on a heavy-expert routing. Equal is
    the expectation (a row's reduction order does not depend on the tile's other rows); the test asserts it so a
    target whose codegen differs is caught here, before any speed read leans on it."""
    counts = [300] + [24] * 20 + [3] * 7
    x, packed, scales, _deq, e, big, small, order = _case(counts, 256, K, big_m, seed=9)
    row0, rows, grp, order16, _c = build_group_tiles_device(e.to(DEV), len(counts), 16)
    assert torch.equal(order, order16)
    ref = gemm_int4_b32_grouped_smallm(x.to(DEV), packed, scales, row0, rows, grp, order16)
    y = gemm_int4_b32_grouped_split(x.to(DEV), packed, scales, big, small, big_m, order)
    assert torch.equal(y, ref), (f"big_m={big_m} K={K}: the split moved output bits on "
                                 f"{torch.cuda.get_device_name()} (max |d| {(y.float() - ref.float()).abs().max().item()})")
