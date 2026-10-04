"""GNF4_HOST_REUSE (on by default; =0 off): host-side reuse inside one MoE layer pass gives the SAME values, bit for bit.

Three pieces, all opt-in behind the one flag: ``to_device_i32`` returns the device tensor an identical earlier upload
produced (keyed on the integers, the device and the current stream; never under capture); ``lora_delta_grouped``'s
down call reuses the gate_up call's device plan; and host-known distinct expert ids gather the adapters through
``_GatherRows`` (a scatter backward) instead of advanced indexing (a sorted ``index_put_`` backward). The bar is
``torch.equal`` on the forward and every gradient against the flag off, for the gate_up -> down twin pattern the
memo exists for.
"""
import pytest
import torch

import nf4_grouped as NG
import nf4_qlora
from nf4_qlora import lora_delta_grouped

CUDA = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA, reason="the reuse paths are CUDA-only")


def _fresh():
    NG._UPLOAD_MEMO.clear()
    nf4_qlora._PLAN_MEMO.clear()
    for k in NG.HOST_REUSE_STATS:
        NG.HOST_REUSE_STATS[k] = 0


def _twin(sizes, eids, ad_dtype, scaling, seed=0):
    """gate_up then down off one grouping, both through lora_delta_grouped, then backward through both."""
    E, K1, N1, K2, N2, R = 8, 128, 96, 48, 128, 8
    g = torch.Generator().manual_seed(seed)
    total = sum(sizes)
    a = torch.randn(total, K1, generator=g).to(torch.bfloat16).cuda().requires_grad_(True)
    mk = lambda *s: (torch.randn(*s, generator=g) * 0.05).to(ad_dtype).cuda().requires_grad_(True)
    A1, B1, A2, B2 = mk(E, R, K1), mk(E, N1, R), mk(E, R, K2), mk(E, N2, R)
    d1 = lora_delta_grouped(a, A1, B1, sizes, eids, scaling)
    h = (d1[:, :K2].float().sigmoid() * 2).to(torch.bfloat16)
    d2 = lora_delta_grouped(h, A2, B2, sizes, eids, scaling)
    gout = torch.randn(d2.shape, generator=g).to(d2.dtype).cuda()
    (d2 * gout).float().sum().backward()
    return [t.detach().clone() for t in (d1, d2, a.grad, A1.grad, B1.grad, A2.grad, B2.grad)]


CASES = [
    ([3, 3, 3, 3], [0, 1, 2, 3]),
    ([5, 0, 1, 7], [2, 0, 3, 1]),
    ([9, 2, 0, 0, 4, 1], [5, 1, 4, 0, 2, 3]),
    ([4, 6, 2], [7, 7, 1]),                 # a repeated id: not unique -> advanced indexing stays
]


@pytest.mark.parametrize("sizes,eids", CASES)
@pytest.mark.parametrize("scaling", [1.0, 2.0])
@pytest.mark.parametrize("ad_dtype", [torch.bfloat16, torch.float32])
def test_reuse_is_bit_identical(monkeypatch, sizes, eids, scaling, ad_dtype):
    monkeypatch.setenv("NF4_QLORA_LORA_PATH", "padded")
    monkeypatch.setenv("GNF4_HOST_REUSE", "0")
    _fresh()
    off = _twin(sizes, eids, ad_dtype, scaling)
    assert NG.HOST_REUSE_STATS == {"upload_hits": 0, "upload_misses": 0, "plan_hits": 0, "plan_misses": 0}
    monkeypatch.setenv("GNF4_HOST_REUSE", "1")
    _fresh()
    on = _twin(sizes, eids, ad_dtype, scaling)
    assert NG.HOST_REUSE_STATS["plan_misses"] == 1 and NG.HOST_REUSE_STATS["plan_hits"] == 1
    for name, x, y in zip(("d1", "d2", "d_a", "d_A1", "d_B1", "d_A2", "d_B2"), off, on):
        assert x.dtype == y.dtype and x.shape == y.shape, name
        assert torch.equal(x, y), f"{name} differs (max |diff| {(x.float() - y.float()).abs().max().item()})"


def test_upload_memo_hits_only_identical_integers(monkeypatch):
    monkeypatch.setenv("GNF4_HOST_REUSE", "1")
    _fresh()
    (a,) = NG.to_device_i32(([4, 5, 6],), "cuda")
    (b,) = NG.to_device_i32(([4, 5, 6],), "cuda")
    (c,) = NG.to_device_i32(([4, 5, 7],), "cuda")
    x, y = NG.to_device_i32(([4, 5], [6]), "cuda")       # same ints, different split: a different key
    assert b.data_ptr() == a.data_ptr() and c.data_ptr() != a.data_ptr()
    assert x.data_ptr() != a.data_ptr()
    assert a.tolist() == [4, 5, 6] and c.tolist() == [4, 5, 7] and x.tolist() == [4, 5] and y.tolist() == [6]
    assert NG.HOST_REUSE_STATS["upload_hits"] == 1 and NG.HOST_REUSE_STATS["upload_misses"] == 3


def test_upload_memo_is_per_stream(monkeypatch):
    monkeypatch.setenv("GNF4_HOST_REUSE", "1")
    _fresh()
    (a,) = NG.to_device_i32(([1, 2, 3],), "cuda")
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        (b,) = NG.to_device_i32(([1, 2, 3],), "cuda")
    s.synchronize()
    assert b.data_ptr() != a.data_ptr() and b.tolist() == [1, 2, 3]


def test_upload_memo_is_bounded(monkeypatch):
    monkeypatch.setenv("GNF4_HOST_REUSE", "1")
    _fresh()
    for i in range(3 * NG._UPLOAD_MEMO_SIZE):
        NG.to_device_i32(([i, i + 1],), "cuda")
    assert len(NG._UPLOAD_MEMO) == NG._UPLOAD_MEMO_SIZE


def test_on_by_default(monkeypatch):
    monkeypatch.delenv("GNF4_HOST_REUSE", raising=False)
    _fresh()
    (a,) = NG.to_device_i32(([8, 9],), "cuda")
    (b,) = NG.to_device_i32(([8, 9],), "cuda")
    assert a.data_ptr() == b.data_ptr() and len(NG._UPLOAD_MEMO) == 1


def test_zero_turns_it_off(monkeypatch):
    monkeypatch.setenv("GNF4_HOST_REUSE", "0")
    _fresh()
    (a,) = NG.to_device_i32(([8, 9],), "cuda")
    (b,) = NG.to_device_i32(([8, 9],), "cuda")
    assert a.data_ptr() != b.data_ptr() and not NG._UPLOAD_MEMO


def test_capture_neither_reads_nor_fills_the_memo(monkeypatch):
    monkeypatch.setenv("GNF4_HOST_REUSE", "1")
    _fresh()
    NG._arena("cuda")                                   # the capture discipline: arena exists before capture
    (warm,) = NG.to_device_i32(([3, 1, 4],), "cuda")    # an uncaptured upload of the same ints is memoized
    assert len(NG._UPLOAD_MEMO) == 1
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s):
        with torch.cuda.graph(graph, stream=s):
            (t,) = NG.to_device_i32(([3, 1, 4],), "cuda")
            out = t * 2
    graph.replay(); torch.cuda.synchronize()
    assert t.data_ptr() != warm.data_ptr()             # the capture staged its own copy
    assert out.tolist() == [6, 2, 8]
    assert len(NG._UPLOAD_MEMO) == 1 and NG.HOST_REUSE_STATS["upload_hits"] == 0
