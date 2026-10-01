# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""K21 grouped small-M MXFP4 GEMM: the correctness contract, device-free (Triton interpreter mode, CPU).

Guards:
1. every sorted output row is within one bf16 output ulp of ``x[src] @ dequant_mxfp4(blocks[e], scales[e]).T``;
2. the in-kernel gather (``order``) is bit-identical to gathering first;
3. an expert with more than 16 rows spans several tiles and still matches;
4. deterministic;
5. a layout or dtype mismatch, or an unsupported KC, is refused before any launch;
5b. the masked K tail (KC not dividing K) matches the reference too;
6. compiled only: the plan (BLOCK_N, KC, warps, stages) moves no output bit, K19's K20 property.

The MXFP4 weights are exact in bf16, so the reference is the fp32 product of the bf16 activations and the exact
dequantised weights. Set by conftest/CI: TRITON_INTERPRET=1 (fp32 dot operands there). The same file compiled on a GPU
exercises the bf16 tensor-core arithmetic.
"""
import os
os.environ.setdefault("TRITON_INTERPRET", "1")

import pytest
import torch

pytest.importorskip("triton", reason="interpreter mode needs triton (Linux-only dependency)")

from int4_b32 import build_group_tiles_fused                                     # noqa: E402
from mxfp4_grouped import gemm_mxfp4_grouped_smallm                              # noqa: E402
from mxfp4_pack_ref import dequant_mxfp4, quantize_pack_mxfp4                    # noqa: E402

INTERP = os.environ.get("TRITON_INTERPRET", "0") == "1"
DEV = "cpu" if INTERP else ("cuda" if torch.cuda.is_available() else None)
if DEV is None:
    pytest.skip("compiled mode needs a CUDA device; set TRITON_INTERPRET=1 for the CPU contract run", allow_module_level=True)


def _stack(E, N, K, seed=0):
    g = torch.Generator().manual_seed(seed)
    blocks, scales, deq = [], [], []
    for _ in range(E):
        w = torch.randn(N, K, generator=g) * 0.05
        b, s = quantize_pack_mxfp4(w)                       # [N, K//32, 16], [N, K//32]
        blocks.append(b.reshape(N, K // 2))
        scales.append(s)
        deq.append(dequant_mxfp4(b, s))
    return torch.stack(blocks).contiguous(), torch.stack(scales).contiguous(), deq


def _case(E, N, K, eids, seed=0):
    g = torch.Generator().manual_seed(seed + 1)
    blocks, scales, deq = _stack(E, N, K, seed)
    x = (torch.randn(len(eids), K, generator=g) * 0.5).to(torch.bfloat16)
    e = torch.tensor(eids, dtype=torch.int32)
    tiles = build_group_tiles_fused(e.to(DEV), E, 16)
    return x, blocks, scales, deq, e, tiles


def _ref_sorted(x, deq, e, order):
    o = order.cpu()
    return torch.stack([x[o[r]].float() @ deq[int(e[o[r]])].float().t() for r in range(len(o))])


CASES = [
    (4, 64, 256, [0, 1, 2, 3, 1, 1, 0, 2, 3, 3, 0, 1], "12 rows over 4 experts"),
    (4, 100, 128, [2] * 5 + [0] * 3 + [3] * 4, "N not a multiple of BLOCK_N; expert 1 unused"),
    (3, 48, 96, [0, 1, 2] * 4, "K=96 (KC 32)"),
    (4, 64, 128, [1] * 20 + [3] * 2, "an expert with 20 rows spans two tiles"),
    (2, 32, 2880, [0, 1, 1, 0, 1], "gpt-oss K=2880 (KC 64)"),
]
# the masked K tail: KC that does not divide K (gpt-oss's K = 2880 at KC 256 = 11 full chunks + a 64-wide tail)
TAIL_CASES = [(2, 32, 2880, 256, [0, 1, 1, 0, 1], "K=2880 at KC 256"), (2, 48, 2880, 128, [1, 0, 1], "K=2880 at KC 128"),
              (3, 40, 96, 64, [0, 1, 2, 2], "K=96 at KC 64")]


@pytest.mark.parametrize("E,N,K,eids,label", CASES, ids=[c[-1] for c in CASES])
def test_matches_dequant_reference(E, N, K, eids, label):
    x, blocks, scales, deq, e, (row0, rows, grp, order, _c) = _case(E, N, K, eids)
    y = gemm_mxfp4_grouped_smallm(x.to(DEV), blocks.to(DEV), scales.to(DEV), row0, rows, grp, order)
    assert y.shape == (len(eids), N) and y.dtype == torch.bfloat16
    ref = _ref_sorted(x, deq, e, order)
    err = float((y.float().cpu() - ref).abs().max())
    assert err <= 2 ** -7 * float(ref.abs().max()), f"{label}: max |y - ref| {err} exceeds one bf16 ulp"


@pytest.mark.parametrize("E,N,K,kc,eids,label", TAIL_CASES, ids=[c[-1] for c in TAIL_CASES])
def test_masked_tail_matches_dequant_reference(E, N, K, kc, eids, label):
    x, blocks, scales, deq, e, (row0, rows, grp, order, _c) = _case(E, N, K, eids)
    y = gemm_mxfp4_grouped_smallm(x.to(DEV), blocks.to(DEV), scales.to(DEV), row0, rows, grp, order, kc=kc)
    ref = _ref_sorted(x, deq, e, order)
    err = float((y.float().cpu() - ref).abs().max())
    assert err <= 2 ** -7 * float(ref.abs().max()), f"{label}: max |y - ref| {err} exceeds one bf16 ulp"


def test_an_unsupported_kc_is_refused():
    x, blocks, scales, deq, e, (row0, rows, grp, order, _c) = _case(*CASES[0][:4])
    with pytest.raises(ValueError, match="expected 32, 64, 128 or 256"):
        gemm_mxfp4_grouped_smallm(x.to(DEV), blocks.to(DEV), scales.to(DEV), row0, rows, grp, order, kc=96)


def test_gather_path_is_bitwise_the_presorted_path():
    x, blocks, scales, deq, e, (row0, rows, grp, order, _c) = _case(*CASES[0][:4])
    a = gemm_mxfp4_grouped_smallm(x.to(DEV), blocks.to(DEV), scales.to(DEV), row0, rows, grp, order)
    xs = x[order.cpu()].contiguous()
    b = gemm_mxfp4_grouped_smallm(xs.to(DEV), blocks.to(DEV), scales.to(DEV), row0, rows, grp, None)
    assert torch.equal(a, b)


def test_deterministic():
    x, blocks, scales, deq, e, (row0, rows, grp, order, _c) = _case(*CASES[3][:4])
    args = (x.to(DEV), blocks.to(DEV), scales.to(DEV), row0, rows, grp, order)
    assert torch.equal(gemm_mxfp4_grouped_smallm(*args), gemm_mxfp4_grouped_smallm(*args))


def test_layout_and_dtype_mismatch_are_refused_before_launch():
    x, blocks, scales, deq, e, (row0, rows, grp, order, _c) = _case(*CASES[0][:4])
    with pytest.raises(ValueError, match="layout mismatch"):
        gemm_mxfp4_grouped_smallm(x[:, :128].to(DEV), blocks.to(DEV), scales.to(DEV), row0, rows, grp, order)
    with pytest.raises(ValueError, match="must be uint8"):
        gemm_mxfp4_grouped_smallm(x.to(DEV), blocks.to(DEV), scales.float().to(DEV), row0, rows, grp, order)


@pytest.mark.skipif(INTERP, reason="compiled only: under the interpreter the fp32 dot is numpy's and KC can move the last bit")
@pytest.mark.parametrize("K", [768, 2880], ids=["K768", "gpt-oss K2880"])
def test_plans_are_bit_identical_compiled(K):
    eids = [0, 1, 2, 3, 4, 5, 6, 7] * 6 + [3] * 17
    x, blocks, scales, deq, e, (row0, rows, grp, order, _c) = _case(8, 256, K, eids, seed=5)
    args = (x.to(DEV), blocks.to(DEV), scales.to(DEV), row0, rows, grp, order)
    ref = gemm_mxfp4_grouped_smallm(*args, block_n=64, kc=64, warps=4, stages=2)
    for bn, kc, wp, st in [(32, 256, 4, 2), (64, 128, 8, 3), (128, 64, 4, 3), (32, 32, 4, 2)]:   # 256/128 mask K=2880's tail
        y = gemm_mxfp4_grouped_smallm(*args, block_n=bn, kc=kc, warps=wp, stages=st)
        assert torch.equal(y, ref), f"plan ({bn}, {kc}, {wp}, {st}) moved an output bit at K={K}"
