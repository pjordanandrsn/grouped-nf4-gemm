"""Lane K17 (``PREREG-k17-fused-splitk-gemv.md``): the split-K reduce folded into the int4-b32 GEMV's own launch.

The contract, before any performance number:

1. **P1 -- bitwise identity.** ``gemv_int4_b32(..., fused_reduce=True)`` is ``torch.equal`` to ``fused_reduce=False``:
   the same fp32 partials are summed in the same split order and cast to bf16 once, so nothing can differ. A
   single differing element refuses the kernel.
2. The counter is re-armed after every launch (a second call on the same ``cnt`` returns the same bits) and a
   ``cnt`` of the wrong length is refused before the launch.
3. ``fused_reduce=False`` is byte-for-byte the shipped path (its output is what P1 compares against, and the
   existing ``test_int4_b32.py`` suite runs it unchanged).
4. The default is read from ``GNF4_GEMV_FUSED_REDUCE`` and is OFF until the lane reads.

Runs under ``TRITON_INTERPRET=1`` on CPU (the repository's interpreter job -- this file must be NAMED in
``.github/workflows/ci.yml``'s explicit test list, or it is inert). The interpreter executes one program at a
time in Python, so the registered six-shape x four-row-count matrix is reserved for the compiled run on a
CUDA device (``TRITON_INTERPRET=0``); under the interpreter the same tests run on small shapes that still
exercise SK > 1, N not a multiple of BLOCK_N, several experts, and R > 1."""
import os
os.environ.setdefault("TRITON_INTERPRET", "1")

import pytest
import torch

from int4_pack_ref import dequant_int4_ref, pack_int4_b32                                   # noqa: E402
from int4_b32 import (_plan, gemv_counter_len, gemv_fused_reduce_default, gemv_int4_b32,    # noqa: E402
                      quant_x_rows)

INTERP = os.environ.get("TRITON_INTERPRET", "0") == "1"
if not INTERP and not torch.cuda.is_available():
    pytest.skip("compiled mode needs a CUDA device; set TRITON_INTERPRET=1 for the CPU contract run", allow_module_level=True)
DEV = "cpu" if INTERP else "cuda"

# Interpreter: small, but SK > 1 on every case (the planner's SK follows N and K, so pick shapes it splits),
# one case whose N is not a multiple of BLOCK_N (the store masks), several experts, R > 1.
# Compiled (CUDA): the six registered shapes x four row counts of the pre-registration.
INTERP_CASES = [  # (E, N, K, R)
    (1, 256, 512, 1),
    (4, 256, 512, 8),
    (1, 192, 1024, 1),       # N not a multiple of 128
    (8, 128, 768, 16),
]
REGISTERED_SHAPES = [(1536, 2048), (2048, 768), (4096, 2048), (512, 2048), (2048, 4096), (5120, 2048)]
COMPILED_CASES = [(8, N, K, R) for (N, K) in REGISTERED_SHAPES for R in (1, 8, 16, 128)]
CASES = INTERP_CASES if INTERP else COMPILED_CASES


def _store(E, N, K, seed=0):
    torch.manual_seed(seed)
    w = torch.randn(E, N, K) / (K ** 0.5)
    packed, scales = zip(*(pack_int4_b32(w[e].float()) for e in range(E)))
    packed = torch.stack([p.reshape(N, K // 2) for p in packed]).contiguous().to(DEV)
    scales = torch.stack([s.reshape(N, K // 32) for s in scales]).contiguous().to(DEV)
    return w, packed, scales


def _rows(R, K, E, seed=1):
    torch.manual_seed(seed)
    x = (torch.randn(R, K) / 8).to(torch.bfloat16).to(DEV)
    eids = (torch.arange(R) % E).to(torch.int32).to(DEV)
    return x, eids


def _both(x, eids, packed, scales, N, K, **kw):
    xq, xs = quant_x_rows(x)
    a = gemv_int4_b32(xq, xs, packed, scales, eids, N, K, fused_reduce=False)
    b = gemv_int4_b32(xq, xs, packed, scales, eids, N, K, fused_reduce=True, **kw)
    return a, b


@pytest.mark.parametrize("E,N,K,R", CASES)
def test_p1_fused_reduce_is_bitwise_the_two_launch_result(E, N, K, R):
    """P1. Same partials, same split order, one bf16 cast -- torch.equal, not allclose."""
    w, packed, scales = _store(E, N, K)
    x, eids = _rows(R, K, E)
    bn, _wp, sk, _ku = _plan(N, K, R, 170)
    a, b = _both(x, eids, packed, scales, N, K)
    assert a.shape == b.shape == (R, N) and b.dtype == torch.bfloat16
    assert torch.equal(a, b), (
        f"fused reduce differs from the two-launch path on (E={E}, N={N}, K={K}, R={R}, SK={sk}): "
        f"max |delta| {(a.float() - b.float()).abs().max().item():g} -- the epilogue is not summing in split order")
    # and both are within one bf16 ulp of the dequantised reference (the property the shipped path already meets)
    ref = torch.stack([(x[r].float().cpu() @ dequant_int4_ref(packed[int(eids[r])].cpu(), scales[int(eids[r])].cpu(), N, K).T)
                       for r in range(R)])
    bound = 2.0 ** -7 * ref.abs().max()
    assert (b.float().cpu() - ref).abs().max() <= bound


def test_counter_is_rearmed_and_a_preallocated_workspace_is_reused():
    """A second launch on the same cnt/out/part returns the same bits, and cnt reads all-zero between launches."""
    E, N, K, R = INTERP_CASES[1] if INTERP else (8, 1536, 2048, 8)
    w, packed, scales = _store(E, N, K)
    x, eids = _rows(R, K, E)
    xq, xs = quant_x_rows(x)
    bn, _wp, sk, _ku = _plan(N, K, R, 170)
    cnt = torch.zeros(gemv_counter_len(N, R, bn), dtype=torch.int32, device=DEV)
    out = torch.empty(R, N, dtype=torch.bfloat16, device=DEV)
    part = torch.empty(sk * R, N, dtype=torch.float32, device=DEV)
    y1 = gemv_int4_b32(xq, xs, packed, scales, eids, N, K, part=part, cnt=cnt, out=out, fused_reduce=True).clone()
    assert int(cnt.abs().sum()) == 0, "the last arriver must re-arm its counter"
    y2 = gemv_int4_b32(xq, xs, packed, scales, eids, N, K, part=part, cnt=cnt, out=out, fused_reduce=True)
    assert torch.equal(y1, y2) and int(cnt.abs().sum()) == 0
    ref = gemv_int4_b32(xq, xs, packed, scales, eids, N, K, fused_reduce=False)
    assert torch.equal(y1, ref)


def test_a_wrong_sized_counter_is_refused_before_the_launch():
    E, N, K, R = INTERP_CASES[0] if INTERP else (1, 512, 2048, 1)
    w, packed, scales = _store(E, N, K)
    x, eids = _rows(R, K, E)
    xq, xs = quant_x_rows(x)
    bad = torch.zeros(3, dtype=torch.int32, device=DEV)
    with pytest.raises(ValueError, match="cnt must be int32 with"):
        gemv_int4_b32(xq, xs, packed, scales, eids, N, K, cnt=bad, fused_reduce=True)
    wrong_dtype = torch.zeros(gemv_counter_len(N, R), dtype=torch.int64, device=DEV)
    with pytest.raises(ValueError, match="cnt must be int32 with"):
        gemv_int4_b32(xq, xs, packed, scales, eids, N, K, cnt=wrong_dtype, fused_reduce=True)


def test_the_default_is_off_until_the_lane_reads(monkeypatch):
    """The flag is read at call time; unset means the shipped two-launch path (decision rule pending)."""
    monkeypatch.delenv("GNF4_GEMV_FUSED_REDUCE", raising=False)
    assert gemv_fused_reduce_default() is False
    monkeypatch.setenv("GNF4_GEMV_FUSED_REDUCE", "1")
    assert gemv_fused_reduce_default() is True
    monkeypatch.setenv("GNF4_GEMV_FUSED_REDUCE", "off")
    assert gemv_fused_reduce_default() is False
    # and the wrapper honours it when fused_reduce is not passed
    monkeypatch.setenv("GNF4_GEMV_FUSED_REDUCE", "1")
    E, N, K, R = INTERP_CASES[0] if INTERP else (1, 512, 2048, 1)
    w, packed, scales = _store(E, N, K)
    x, eids = _rows(R, K, E)
    xq, xs = quant_x_rows(x)
    y_env = gemv_int4_b32(xq, xs, packed, scales, eids, N, K)
    y_off = gemv_int4_b32(xq, xs, packed, scales, eids, N, K, fused_reduce=False)
    assert torch.equal(y_env, y_off)
