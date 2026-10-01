# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""K19 grouped small-M int4-b32 GEMM: the correctness contract, device-free (Triton interpreter mode, CPU).

Guards: (1) every sorted output row within one bf16 output ulp of ``x[src] @ dequant_int4_ref(packed[e]).T``;
(2) bit-identical to K16 (``gemm_int4_b32_smallm``, sk=1, the same block_n/kc) on each expert's rows -- grouping
changes no arithmetic, because a row's MMA output does not depend on the tile's other rows; (3) the in-kernel gather
(``order``) is bit-identical to gathering first and passing sorted rows; (4) an expert with more than 16 rows spans
several tiles and still matches; (5) deterministic; (6) a layout mismatch is refused before any launch. Set by
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
    y = gemm_int4_b32_grouped_smallm(x.to(DEV), packed.to(DEV), scales.to(DEV), row0, rows, grp, order).cpu()
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
