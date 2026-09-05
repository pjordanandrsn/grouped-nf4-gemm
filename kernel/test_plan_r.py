# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Row-count-aware GEMV plan: the R=128 overrides select only on the two
measured expert cells, every other row count / shape keeps the R=1 plan
(so partial buffers preallocated at R<=8 stay valid), a wrong-sized
``part`` is refused loudly, and the override configs compute the same
values as the reference (interp mode or GPU)."""
import os

import pytest

torch = pytest.importorskip("torch")

from int4_pack_ref import BLOCK, dequant_int4_ref, pack_int4_b32  # noqa: E402


def _gpu():
    if os.environ.get("TRITON_INTERPRET") == "1":
        return "cpu"
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA or TRITON_INTERPRET=1")
    return "cuda"


def test_overrides_select_only_on_measured_cells_at_batch_rows():
    pytest.importorskip("triton")
    from int4_b32 import _PLAN_R64, _PLAN_R_MIN, _plan
    for (N, K), cfg in _PLAN_R64.items():
        bn, wp, sk, ku = cfg
        assert (K // 32) % ku == 0, "KU must divide K//32"
        assert sk <= max(1, (K // 32) // ku), "never more splits than spans"
        assert _plan(N, K, _PLAN_R_MIN) == cfg
        assert _plan(N, K, 128) == cfg
        # R=1..8 (the B=1 stores preallocate at top-k rows) keep the R=1 plan
        for R in (1, 8, _PLAN_R_MIN - 1):
            assert _plan(N, K, R) == _plan(N, K)
        # a neighbouring shape at batch rows is NOT overridden
        assert _plan(N + 128, K, 128) == _plan(N + 128, K)
    assert _plan(1536, 2048, 128) == (64, 8, 2, 4)
    assert _plan(2048, 768, 128) == (256, 4, 4, 4)


def test_wrong_sized_part_is_refused():
    pytest.importorskip("triton")
    from int4_b32 import _plan, gemv_int4_b32, quant_x_rows
    dev = _gpu()
    N, K, E, R = 64, 128, 2, 3
    W = torch.randn(E, N, K) * 0.1
    pk, sc = zip(*[pack_int4_b32(W[e]) for e in range(E)])
    Wp, Sp = torch.stack(pk).to(dev), torch.stack(sc).to(dev)
    eids = torch.zeros(R, dtype=torch.int32, device=dev)
    xq, xs = quant_x_rows((torch.randn(R, K) * 0.2).to(dev, torch.bfloat16))
    _, _, sk, _ = _plan(N, K, R)
    bad = torch.empty(sk * (R + 1), N, dtype=torch.float32, device=dev)
    with pytest.raises(ValueError, match="does not match the plan"):
        gemv_int4_b32(xq, xs, Wp, Sp, eids, N, K, part=bad)
    good = torch.empty(sk * R, N, dtype=torch.float32, device=dev)
    gemv_int4_b32(xq, xs, Wp, Sp, eids, N, K, part=good)   # accepted


@pytest.mark.parametrize("N,K,cfg", [
    (64, 256, (64, 8, 2, 4)),      # the gate_up override config
    (256, 512, (256, 4, 4, 4)),    # the down override config
])
def test_override_configs_match_reference(monkeypatch, N, K, cfg):
    """Exercise each override config on a cell small enough for interp
    mode by mapping it onto that cell; the real cells are only reachable
    on a GPU and are certified by the box parity gate."""
    pytest.importorskip("triton")
    import int4_b32
    from int4_b32 import _PLAN_R_MIN, gemv_int4_b32, quant_x_rows
    monkeypatch.setitem(int4_b32._PLAN_R64, (N, K), cfg)
    dev = _gpu()
    E, R = 3, _PLAN_R_MIN
    torch.manual_seed(5)
    W = torch.randn(E, N, K) * 0.1
    pk, sc = zip(*[pack_int4_b32(W[e]) for e in range(E)])
    Wp = torch.stack(pk).to(dev).contiguous()
    Sp = torch.stack(sc).to(dev).contiguous()
    eids = (torch.arange(R) % E).to(dev).to(torch.int32)
    x = (torch.randn(R, K) * 0.2).to(dev, torch.bfloat16)
    xq, xs = quant_x_rows(x)
    assert int4_b32._plan(N, K, R) == cfg
    got = gemv_int4_b32(xq, xs, Wp, Sp, eids, N, K)
    ref = torch.stack([
        (dequant_int4_ref(Wp[int(e)].cpu(), Sp[int(e)].cpu(), N, K).to(dev)
         * (xq[i].float() * xs[i].repeat_interleave(BLOCK))[None, :]).sum(-1)
        for i, e in enumerate(eids)])
    assert (got.float() - ref).abs().max() <= ref.abs().max() * 2 ** -7
