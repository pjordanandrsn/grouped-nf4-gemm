"""NF4_QLORA_COMPACT_DELTA=1: the lean padded LoRA delta as one autograd node that saves its input, not its padded block.

The bar is ``torch.equal`` on the forward and every gradient against the autograd path (flag off), for distinct and repeated
expert ids, bf16 and fp32 adapters, scaling 1 and not, with and without GNF4_HOST_REUSE (which is what supplies distinct-id
knowledge); plus the point of it: far fewer bytes saved for backward.
"""
import pytest
import torch

import nf4_grouped as NG
import nf4_qlora
from nf4_qlora import lora_delta_grouped

CUDA = torch.cuda.is_available()
DEVICES = ["cpu"] + (["cuda"] if CUDA else [])
CASES = [
    ([3, 3, 3, 3], [0, 1, 2, 3]),
    ([5, 0, 1, 7], [2, 0, 3, 1]),
    ([9, 2, 0, 0, 4, 1], [5, 1, 4, 0, 2, 3]),
    ([4, 6, 2], [7, 7, 1]),                 # a repeated id: the adapters' backward accumulates
    ([1], [3]),
]


def _run(dev, sizes, eids, ad_dtype, scaling, seed=0):
    E, K, N, R = 8, 128, 96, 8
    g = torch.Generator().manual_seed(seed)
    total = sum(sizes)
    a = torch.randn(total, K, generator=g).to(torch.bfloat16).to(dev).requires_grad_(True)
    A = (torch.randn(E, R, K, generator=g) * 0.05).to(ad_dtype).to(dev).requires_grad_(True)
    B = (torch.randn(E, N, R, generator=g) * 0.05).to(ad_dtype).to(dev).requires_grad_(True)
    out = lora_delta_grouped(a, A, B, sizes, eids, scaling)
    gout = torch.randn(out.shape, generator=g).to(out.dtype).to(dev)
    out.backward(gout)
    return out.detach(), a.grad, A.grad, B.grad


def _fresh():
    NG._UPLOAD_MEMO.clear()
    nf4_qlora._PLAN_MEMO.clear()


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("sizes,eids", CASES)
@pytest.mark.parametrize("ad_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("scaling", [1.0, 2.0])
@pytest.mark.parametrize("reuse", ["0", "1"])
def test_compact_delta_is_bit_identical(monkeypatch, dev, sizes, eids, ad_dtype, scaling, reuse):
    if reuse == "1" and dev != "cuda":
        pytest.skip("host reuse is CUDA-only")
    monkeypatch.setenv("NF4_QLORA_LORA_PATH", "padded")
    monkeypatch.setenv("GNF4_HOST_REUSE", reuse)
    monkeypatch.setenv("NF4_QLORA_COMPACT_DELTA", "0")
    _fresh()
    old = _run(dev, sizes, eids, ad_dtype, scaling)
    monkeypatch.setenv("NF4_QLORA_COMPACT_DELTA", "1")
    _fresh()
    new = _run(dev, sizes, eids, ad_dtype, scaling)
    for name, o, n in zip(("out", "d_a", "d_A", "d_B"), old, new):
        assert o.dtype == n.dtype and o.shape == n.shape, name
        assert torch.equal(o, n), f"{name} differs (max |diff| {(o.float() - n.float()).abs().max().item()})"


def _saved_bytes(dev, compact, monkeypatch):
    monkeypatch.setenv("NF4_QLORA_LORA_PATH", "padded")
    monkeypatch.setenv("NF4_QLORA_COMPACT_DELTA", "1" if compact else "0")
    E, K, N, R = 16, 512, 384, 16
    sizes = [60, 2, 2, 2, 2, 2, 2, 2]                 # one hot expert: the padded block is 8 x 60 rows for 74 real ones
    eids = list(range(8))
    a = torch.randn(sum(sizes), K, device=dev).requires_grad_(True)
    A = torch.randn(E, R, K, device=dev, requires_grad=True)
    B = torch.randn(E, N, R, device=dev, requires_grad=True)
    seen = {}

    def pack(t):
        if not isinstance(t, torch.nn.Parameter) and t.data_ptr() not in (a.data_ptr(), A.data_ptr(), B.data_ptr()):
            seen[t.untyped_storage().data_ptr()] = t.untyped_storage().nbytes()
        return t
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        lora_delta_grouped(a, A, B, sizes, eids, 1.0)
    return sum(seen.values())


@pytest.mark.parametrize("dev", DEVICES)
def test_compact_delta_saves_far_less(monkeypatch, dev):
    full, compact = _saved_bytes(dev, False, monkeypatch), _saved_bytes(dev, True, monkeypatch)
    assert compact * 8 < full, (compact, full)


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("NF4_QLORA_COMPACT_DELTA", raising=False)
    assert nf4_qlora._compact_delta_enabled() is False
