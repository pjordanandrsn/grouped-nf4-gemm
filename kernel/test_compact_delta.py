"""NF4_QLORA_COMPACT_DELTA=1: the lean padded LoRA delta as one autograd node that saves its input, not its padded block.

The bar is ``torch.equal`` on the forward and every gradient against the autograd path (flag off), for distinct and repeated
expert ids, bf16 and fp32 adapters, scaling 1 and not, with and without GNF4_HOST_REUSE (which is what supplies distinct-id
knowledge); plus the point of it: far fewer bytes saved for backward, without a larger backward transient than the autograd
path's.
"""
import weakref

import pytest
import torch
import torch.utils._python_dispatch

import nf4_grouped as NG
import nf4_qlora
from nf4_qlora import lora_delta_grouped

CUDA = torch.cuda.is_available()
DEVICES = ["cpu"] + (["cuda"] if CUDA else [])


@pytest.fixture(autouse=True)
def _single_block_without_its_ladder(monkeypatch):
    """These tests hold other paths to the single block bit for bit. The single-block ladder (``NF4_QLORA_SINGLE_LADDER``,
    ``auto`` by default: engaged on fp32 adapters) changes the products' shapes, so it is off here; test_single_ladder.py
    holds it to the single block to rounding."""
    monkeypatch.setenv("NF4_QLORA_SINGLE_LADDER", "0")
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


def _peaks(dev, compact, ad_dtype, K, N, monkeypatch):
    """One forward + backward of a hot-expert delta: (out, d_a, d_A, d_B) and the forward and backward peaks of
    ``max_memory_allocated``, each above what was allocated before the forward (the inputs, adapters and output grad)."""
    monkeypatch.setenv("NF4_QLORA_LORA_PATH", "padded")
    monkeypatch.setenv("NF4_QLORA_COMPACT_DELTA", "1" if compact else "0")
    _fresh()
    E, R = 16, 16
    sizes = [60, 2, 2, 2, 2, 2, 2, 2]                 # the hot-expert case above: 8 x 60 padded rows for 74 real ones
    eids = list(range(8))
    g = torch.Generator().manual_seed(3)
    a = torch.randn(sum(sizes), K, generator=g).to(torch.bfloat16).to(dev).requires_grad_(True)
    A = (torch.randn(E, R, K, generator=g) * 0.05).to(ad_dtype).to(dev).requires_grad_(True)
    B = (torch.randn(E, N, R, generator=g) * 0.05).to(ad_dtype).to(dev).requires_grad_(True)
    gout = torch.randn(sum(sizes), N, generator=g).to(ad_dtype).to(dev)
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    out = lora_delta_grouped(a, A, B, sizes, eids, 1.0)
    torch.cuda.synchronize()
    fwd = torch.cuda.max_memory_allocated() - base
    torch.cuda.reset_peak_memory_stats()
    out.backward(gout)
    torch.cuda.synchronize()
    bwd = torch.cuda.max_memory_allocated() - base
    return (out.detach(), a.grad, A.grad, B.grad), fwd, bwd


@pytest.mark.skipif(not CUDA, reason="max_memory_allocated is a CUDA allocator statistic")
@pytest.mark.parametrize("ad_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("K,N", [(512, 384), (384, 512)])   # gate_up-like (K > N) and down-like (N > K)
@pytest.mark.parametrize("reuse", ["0", "1"])
def test_compact_backward_peak_at_most_autograd(monkeypatch, ad_dtype, K, N, reuse):
    """The compact node saves less, so its backward must not spend that on a larger transient: its peak over one forward +
    backward is at most the autograd path's. While its backward held every intermediate to the return (the padded output
    grad still live when the block and its grad were rebuilt), these cases failed on an RTX A2000 by 1.0-2.1 MB: 7 of 8
    under torch 2.8, all 8 under torch 2.11."""
    monkeypatch.setenv("GNF4_HOST_REUSE", reuse)
    old, fwd_old, bwd_old = _peaks("cuda", False, ad_dtype, K, N, monkeypatch)
    new, fwd_new, bwd_new = _peaks("cuda", True, ad_dtype, K, N, monkeypatch)
    for name, o, n in zip(("out", "d_a", "d_A", "d_B"), old, new):
        assert o.dtype == n.dtype and o.shape == n.shape, name
        assert torch.equal(o, n), f"{name} differs (max |diff| {(o.float() - n.float()).abs().max().item()})"
    assert fwd_new <= fwd_old, (fwd_new, fwd_old)
    assert bwd_new <= bwd_old, f"compact backward peak {bwd_new} B > autograd {bwd_old} B"
    assert max(fwd_new, bwd_new) <= max(fwd_old, bwd_old)


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("NF4_QLORA_COMPACT_DELTA", raising=False)
    assert nf4_qlora._compact_delta_enabled() is False


_NEW_ORDER = nf4_qlora._CompactPaddedDelta                       # captured before any test patches the module attribute


class _OldOrder(_NEW_ORDER):
    """The compact node's backward as it was before experts4bit-qlora TC1 amendment 75's lever (grouped-nf4-gemm a83f6bb3): the
    block's grad ``gx`` was built while the rebuilt block ``x`` was still live. A test-local reference only."""

    @staticmethod
    def backward(ctx, g):
        a_cat, lora_A, lora_B, eid, flat, h = ctx.saved_tensors
        G, W, unique = ctx.G, ctx.widest, ctx.unique
        if unique:
            A, B = lora_A.index_select(0, eid), lora_B.index_select(0, eid)
        else:
            A, B = lora_A[eid], lora_B[eid]
        gd = g.new_zeros((G * W,) + tuple(g.shape[1:]))
        gd.index_copy_(0, flat, g)
        gd = gd.view(G, W, -1)
        gh = gd.bmm(B)
        del B
        gBt = h.transpose(1, 2).bmm(gd)
        del gd, h
        x = torch.zeros(G * W, a_cat.shape[1], dtype=A.dtype, device=a_cat.device)
        x.index_copy_(0, flat, a_cat.to(A.dtype))
        gx = gh.bmm(A)
        del A
        gAt = x.view(G, W, -1).transpose(1, 2).bmm(gh)
        del x, gh
        grad_a = None
        if ctx.needs_input_grad[0]:
            grad_a = gx.view(G * W, -1).index_select(0, flat)
            del gx
            if grad_a.dtype != a_cat.dtype:
                grad_a = grad_a.to(a_cat.dtype)
        else:
            del gx
        gA = gB = None
        if ctx.needs_input_grad[1]:
            gA = torch.zeros_like(lora_A)
            if unique:
                gA.index_copy_(0, eid, gAt.transpose(1, 2))
            else:
                gA.index_put_((eid,), gAt.transpose(1, 2), accumulate=True)
        del gAt
        if ctx.needs_input_grad[2]:
            gB = torch.zeros_like(lora_B)
            if unique:
                gB.index_copy_(0, eid, gBt.transpose(1, 2))
            else:
                gB.index_put_((eid,), gBt.transpose(1, 2), accumulate=True)
        del gBt
        return grad_a, gA, gB, None, None, None, None, None


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("sizes,eids", CASES)
@pytest.mark.parametrize("ad_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("scaling", [1.0, 2.0])
@pytest.mark.parametrize("reuse", ["0", "1"])
def test_backward_order_is_bit_identical_to_the_old_order(monkeypatch, dev, sizes, eids, ad_dtype, scaling, reuse):
    """The reordered backward (the adapters' product before the block's grad, the block freed between them) gives the same
    bytes as the old order on every output and gradient: each product reads the same operands either way."""
    if reuse == "1" and dev != "cuda":
        pytest.skip("host reuse is CUDA-only")
    monkeypatch.setenv("NF4_QLORA_LORA_PATH", "padded")
    monkeypatch.setenv("GNF4_HOST_REUSE", reuse)
    monkeypatch.setenv("NF4_QLORA_COMPACT_DELTA", "1")
    _fresh()
    new = _run(dev, sizes, eids, ad_dtype, scaling)
    monkeypatch.setattr(nf4_qlora, "_CompactPaddedDelta", _OldOrder)
    _fresh()
    old = _run(dev, sizes, eids, ad_dtype, scaling)
    for name, o, n in zip(("out", "d_a", "d_A", "d_B"), old, new):
        assert o.dtype == n.dtype and o.shape == n.shape, name
        assert torch.equal(o, n), f"{name} differs (max |diff| {(o.float() - n.float()).abs().max().item()})"


class _BlockCensus(torch.utils._python_dispatch.TorchDispatchMode):
    """Inside the mode, every tensor an op returns with exactly ``numel`` elements is tracked by its storage until its Python
    object is freed (a weakref finalizer). ``most`` is the largest number of such storages ever live together. Views and
    in-place results share their base's storage and are not counted twice."""

    def __init__(self, numel):
        super().__init__()
        self.numel, self.live, self.most = numel, set(), 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for t in (out if isinstance(out, (tuple, list)) else (out,)):
            if isinstance(t, torch.Tensor) and t.numel() == self.numel:
                key = t.untyped_storage().data_ptr()
                if key not in self.live:
                    self.live.add(key)
                    weakref.finalize(t, self.live.discard, key)
                    self.most = max(self.most, len(self.live))
        return out


def _most_blocks_live(node, ad_dtype, monkeypatch):
    """One hot-expert delta through the compact node ``node``: the most [G * W, K] buffers live at once during its backward."""
    monkeypatch.setenv("NF4_QLORA_LORA_PATH", "padded")
    monkeypatch.setenv("NF4_QLORA_COMPACT_DELTA", "1")
    monkeypatch.setattr(nf4_qlora, "_CompactPaddedDelta", node)
    _fresh()
    E, K, N, R = 16, 512, 384, 16
    sizes = [60, 2, 2, 2, 2, 2, 2, 2]                 # G = 8 groups padded to W = 60 rows: the block is 8 x 60 x 512
    g = torch.Generator().manual_seed(5)
    a = torch.randn(sum(sizes), K, generator=g).to(torch.bfloat16).requires_grad_(True)
    A = (torch.randn(E, R, K, generator=g) * 0.05).to(ad_dtype).requires_grad_(True)
    B = (torch.randn(E, N, R, generator=g) * 0.05).to(ad_dtype).requires_grad_(True)
    out = lora_delta_grouped(a, A, B, sizes, list(range(8)), 1.0)
    gout = torch.randn(out.shape, generator=g).to(out.dtype)
    census = _BlockCensus(8 * 60 * K)                  # x is [G*W, K], gx is [G, W, K]: no other buffer has this many elements
    with census:
        out.backward(gout)
    return census.most


@pytest.mark.parametrize("ad_dtype", [torch.bfloat16, torch.float32])
def test_the_block_and_its_grad_are_never_live_together(monkeypatch, ad_dtype):
    """The point of the order: the rebuilt block x is freed before its grad gx is allocated, so the backward never holds two
    [G * W, K] buffers. The census proves it can see the difference: the old order holds two."""
    assert _most_blocks_live(_OldOrder, ad_dtype, monkeypatch) == 2
    assert _most_blocks_live(_NEW_ORDER, ad_dtype, monkeypatch) == 1


@pytest.mark.skipif(not CUDA, reason="max_memory_allocated is a CUDA allocator statistic")
@pytest.mark.parametrize("ad_dtype", [torch.bfloat16, torch.float32])
def test_reordered_backward_peak_drops_by_about_a_block(monkeypatch, ad_dtype):
    """On CUDA the compact backward's peak falls by about one [G * W, K] block against the old order, with the same bytes."""
    K = 512
    new, _, bwd_new = _peaks("cuda", True, ad_dtype, K, 384, monkeypatch)
    monkeypatch.setattr(nf4_qlora, "_CompactPaddedDelta", _OldOrder)
    old, _, bwd_old = _peaks("cuda", True, ad_dtype, K, 384, monkeypatch)
    for name, o, n in zip(("out", "d_a", "d_A", "d_B"), old, new):
        assert torch.equal(o, n), name
    block = 8 * 60 * K * torch.tensor([], dtype=ad_dtype).element_size()
    assert bwd_old - bwd_new >= 0.9 * block, (bwd_old, bwd_new, block)
