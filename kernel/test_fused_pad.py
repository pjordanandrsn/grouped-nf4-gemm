"""NF4_QLORA_FUSED_PAD=1: the lean single ladder's glue in single kernels (``lora_pad``), bit for bit the ladder's.

The ladder is called directly with a hand-built plan (``eid``, ``flat``, ``sz``, ``start``) and distinct ids, the route the
fused glue replaces (``_GatherRows`` + ``F.pad``; the plan memo that sets ``unique`` runs on CUDA only). The delta and every
gradient must be EQUAL, not close: the fused path changes only copies, exact casts and zeros. On a plain CPU run the
``lora_pad`` functions take their torch fallbacks (the autograd wiring); under ``TRITON_INTERPRET=1`` or on CUDA they run
the Triton kernels themselves.
"""
import pytest
import torch

import nf4_qlora
from nf4_qlora import _ladder_up, _lora_delta_padded_ladder

CUDA = torch.cuda.is_available()
DEVICES = ["cpu"] + (["cuda"] if CUDA else [])
#: (rows per group, distinct expert ids): uneven groups, a group count off the ladder's rungs (so G' > G pads groups), one
#: group, and a width that is already a rung.
CASES = [([3, 1, 5, 2, 7], [4, 0, 9, 2, 6]),
         ([1], [3]),
         ([4, 4, 4, 4, 4, 4], [0, 1, 2, 3, 4, 5]),
         ([9, 2, 2, 1, 6, 3, 1], [7, 1, 3, 0, 9, 5, 2])]


def _plan(sizes, eids, dev):
    G = len(sizes)
    width = _ladder_up(max(sizes))
    sz = torch.tensor(sizes, dtype=torch.int64, device=dev)
    start = torch.cumsum(sz, 0) - sz
    total = sum(sizes)
    shift = torch.repeat_interleave(torch.arange(G, device=dev) * width - start, sz, output_size=total)
    flat = torch.arange(total, device=dev) + shift
    eid = torch.tensor(eids, dtype=torch.int64, device=dev)
    return eid, flat, G, width, sz, start


def _run(monkeypatch, fused, dev, sizes, eids, act_dtype, ad_dtype, scaling, seed=0, K=96, N=80, R=8):
    monkeypatch.setenv("NF4_QLORA_FUSED_PAD", "1" if fused else "0")
    E = max(eids) + 1
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(sum(sizes), K, generator=g).to(act_dtype).to(dev).requires_grad_(True)
    A = (torch.randn(E, R, K, generator=g) * 0.05).to(ad_dtype).to(dev).requires_grad_(True)
    B = (torch.randn(E, N, R, generator=g) * 0.05).to(ad_dtype).to(dev).requires_grad_(True)
    gout = torch.randn(sum(sizes), N, generator=g).to(ad_dtype).to(dev)
    eid, flat, G, width, sz, start = _plan(sizes, eids, dev)
    before = nf4_qlora.FUSED_PAD_STATS["calls"]
    out = _lora_delta_padded_ladder(a, A, B, eid, flat, G, width, True, scaling, sz, start)
    out.backward(gout)
    return (out.detach(), a.grad, A.grad, B.grad), nf4_qlora.FUSED_PAD_STATS["calls"] - before


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("sizes,eids", CASES)
@pytest.mark.parametrize("act_dtype,ad_dtype", [(torch.bfloat16, torch.float32), (torch.bfloat16, torch.bfloat16),
                                                (torch.float32, torch.float32)])
@pytest.mark.parametrize("scaling", [1.0, 2.0])
def test_fused_pad_is_the_ladder_bit_for_bit(monkeypatch, dev, sizes, eids, act_dtype, ad_dtype, scaling):
    ref, n0 = _run(monkeypatch, False, dev, sizes, eids, act_dtype, ad_dtype, scaling)
    got, n1 = _run(monkeypatch, True, dev, sizes, eids, act_dtype, ad_dtype, scaling)
    assert (n0, n1) == (0, 1)
    for name, x, y in zip(("out", "d_a", "d_A", "d_B"), got, ref):
        assert x.dtype == y.dtype and x.shape == y.shape, name
        assert torch.equal(x, y), f"{name}: max |diff| {(x.float() - y.float()).abs().max().item():.3e}"


def test_fused_pad_needs_distinct_ids_and_a_plan(monkeypatch):
    """Repeated ids, or a caller without the plan's sz / start, take the ladder as before (the knob is never counted)."""
    monkeypatch.setenv("NF4_QLORA_FUSED_PAD", "1")
    sizes, eids = [3, 2], [1, 4]
    a = torch.randn(5, 16, dtype=torch.bfloat16)
    A, B = torch.randn(5, 4, 16), torch.randn(5, 8, 4)
    eid, flat, G, width, sz, start = _plan(sizes, eids, "cpu")
    before = nf4_qlora.FUSED_PAD_STATS["calls"]
    _lora_delta_padded_ladder(a, A, B, eid, flat, G, width, False, 1.0, sz, start)
    _lora_delta_padded_ladder(a, A, B, eid, flat, G, width, True, 1.0)
    assert nf4_qlora.FUSED_PAD_STATS["calls"] == before


def test_the_knob_is_off_unless_one(monkeypatch):
    for v, want in ((None, False), ("0", False), ("auto", False), ("1", True), (" 1 ", True)):
        if v is None:
            monkeypatch.delenv("NF4_QLORA_FUSED_PAD", raising=False)
        else:
            monkeypatch.setenv("NF4_QLORA_FUSED_PAD", v)
        assert nf4_qlora._fused_pad_enabled() is want, v
