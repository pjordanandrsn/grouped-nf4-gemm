"""Lane K18 (``PREREG-k18-grouped-expert-gemv.md``): the grouped split-K GEMV must be BITWISE the served GEMV.

``gemv_int4_b32_grouped`` shares each expert's weight loads across up to ``mt`` of its rows and computes those rows
with the served GEMV's arithmetic, storing every row's fp32 partial at its original index for the served reduce. So
for every input it returns ``torch.equal`` to ``gemv_int4_b32(..., fused_reduce=False)`` -- one differing element
refuses the kernel. Cases cover: one row, every row a distinct expert, every row the same expert (one expert split
across many tiles), skewed routing, R not a power of two, N not a multiple of BLOCK_N, SK > 1, and mt in {1, 2, 4, 8}.

Runs under ``TRITON_INTERPRET=1`` on CPU (named in ``.github/workflows/ci.yml``'s interpreter job and guarded in
``conftest._INTERP_FILES``); compiled on a CUDA device (``TRITON_INTERPRET=0``) the same cases plus Qwen3-30B-A3B's
expert shapes at R=128 run."""
import os
os.environ.setdefault("TRITON_INTERPRET", "1")

import pytest
import torch

from int4_b32 import gemv_int4_b32, gemv_int4_b32_grouped, quant_x_rows      # noqa: E402

INTERP = os.environ.get("TRITON_INTERPRET", "0") == "1"
if not INTERP and not torch.cuda.is_available():
    pytest.skip("compiled mode needs a CUDA device", allow_module_level=True)
DEV = "cpu" if INTERP else "cuda"


def _store(E, N, K, seed=0):
    g = torch.Generator().manual_seed(seed)
    packed = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, generator=g).to(DEV)
    scales = (torch.rand(E, N, K // 32, generator=g) * 0.02 + 1e-3).to(torch.float16).to(DEV)
    return packed, scales


def _check(E, N, K, eids, mt, seed=0):
    packed, scales = _store(E, N, K, seed)
    R = len(eids)
    x = (torch.randn(R, K, generator=torch.Generator().manual_seed(seed + 1)) / 4).to(torch.bfloat16).to(DEV)
    xq, xs = quant_x_rows(x)
    e = torch.tensor(eids, dtype=torch.int32, device=DEV)
    ref = gemv_int4_b32(xq, xs, packed, scales, e, N, K, fused_reduce=False)
    got = gemv_int4_b32_grouped(xq, xs, packed, scales, e, N, K, mt=mt)
    assert got.shape == ref.shape
    if not torch.equal(got, ref):
        d = (got.float() - ref.float()).abs()
        raise AssertionError(f"grouped != served: {int((d > 0).sum())} elements differ, max |delta| {float(d.max()):g}")


SMALL = [  # (E, N, K, eids, label)
    (4, 64, 64, [2], "one row"),
    (8, 96, 64, [5, 1, 7, 0, 3, 2, 6, 4], "all distinct"),
    (4, 64, 128, [3] * 9, "one expert, many tiles"),
    (8, 160, 64, [1, 1, 5, 1, 0, 5, 1, 7, 5, 1, 2], "skewed, R=11, N not a BLOCK_N multiple"),
    (6, 64, 256, [0, 5, 5, 2, 0, 0, 3, 5, 5, 1, 0, 4, 2], "R=13, K=256 (SK>1 at small N)"),
]


@pytest.mark.parametrize("mt", [1, 2, 4, 8])
@pytest.mark.parametrize("E,N,K,eids,label", SMALL, ids=[c[-1] for c in SMALL])
def test_grouped_is_bitwise_the_served_gemv(E, N, K, eids, label, mt):
    _check(E, N, K, eids, mt)


@pytest.mark.skipif(INTERP, reason="real shapes are compiled-only (the interpreter runs one program at a time)")
@pytest.mark.parametrize("N,K", [(1536, 2048), (2048, 768)], ids=["gate_up", "down"])
@pytest.mark.parametrize("pattern", ["uniform", "skewed", "one_hot"])
def test_grouped_real_shapes_r128(N, K, pattern):
    g = torch.Generator().manual_seed(7)
    if pattern == "uniform":
        eids = torch.randint(0, 128, (128,), generator=g).tolist()
    elif pattern == "skewed":                       # ~55 distinct, a few hot experts (P60's regime)
        hot = torch.randint(0, 128, (8,), generator=g).tolist()
        eids = [hot[i % 8] if i % 3 == 0 else int(torch.randint(0, 128, (1,), generator=g)) for i in range(128)]
    else:
        eids = [17] * 128
    _check(128, N, K, eids, mt=4, seed=3)


def test_capture_legal():
    """Under CUDA-graph capture the grouped call replays to the same bits (compiled only)."""
    if INTERP:
        pytest.skip("capture is compiled-only")
    packed, scales = _store(16, 256, 512)
    x = (torch.randn(32, 512) / 4).to(torch.bfloat16).to(DEV)
    xq, xs = quant_x_rows(x)
    e = torch.randint(0, 16, (32,), dtype=torch.int32, device=DEV)
    from int4_b32 import _plan, _sm_count
    sk = _plan(256, 512, 32, _sm_count(DEV))[2]
    part = torch.empty(sk * 32, 256, dtype=torch.float32, device=DEV)
    out = torch.empty(32, 256, dtype=torch.bfloat16, device=DEV)
    ref = gemv_int4_b32(xq, xs, packed, scales, e, 256, 512, fused_reduce=False).clone()
    gemv_int4_b32_grouped(xq, xs, packed, scales, e, 256, 512, part=part, out=out)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        gemv_int4_b32_grouped(xq, xs, packed, scales, e, 256, 512, part=part, out=out)
    out.zero_()
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, ref)
