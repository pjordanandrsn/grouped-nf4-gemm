"""The padded LoRA delta after its launch trim is the SAME function, bit for bit.

``lora_delta_grouped``'s padded path now (a) indexes the padded block with one
flat row index instead of a (group, slot) pair, gathering back through a
function whose backward is a plain scatter (the rows are unique), (b) returns
the gathered rows instead of a zero fill plus a slice copy, and (c) applies
``scaling`` to the gathered rows rather than the padded block, and not at all
when it is 1. Every
one of those is exact, so the bar here is ``torch.equal`` -- forward AND every
gradient -- against the previous implementation, kept below verbatim as the
oracle. A tolerance would let a real change hide inside it.
"""
import pytest
import torch

import nf4_qlora
from nf4_qlora import lora_delta_grouped

CUDA = torch.cuda.is_available()
DEVICES = ["cpu"] + (["cuda"] if CUDA else [])


def _legacy_padded(a_cat, lora_A, lora_B, sizes, expert_ids, scaling):
    """The padded path as it stood before the trim (gnf4 133ad9d), host-list eids."""
    nz = [g for g in range(len(sizes)) if int(sizes[g]) > 0]
    rows = [int(sizes[g]) for g in nz]
    total, widest = sum(rows), max(rows)
    dev = a_cat.device
    sz = torch.tensor(rows, device=dev, dtype=torch.int64)
    eid = torch.tensor([int(expert_ids[g]) for g in nz], device=dev, dtype=torch.int64)
    grp = torch.repeat_interleave(torch.arange(len(rows), device=dev), sz, output_size=total)
    slot = torch.arange(total, device=dev) - (torch.cumsum(sz, 0) - sz)[grp]
    A, B = lora_A[eid], lora_B[eid]
    x = a_cat.new_zeros(len(rows), widest, a_cat.shape[1]).to(A.dtype)
    x[grp, slot] = a_cat.to(A.dtype)
    d = scaling * torch.bmm(torch.bmm(x, A.transpose(1, 2)), B.transpose(1, 2))
    out = torch.zeros(a_cat.shape[0], B.shape[1], dtype=d.dtype, device=dev)
    out[:total] = d[grp, slot]
    return out


CASES = [
    ([3, 3, 3, 3], [0, 1, 2, 3]),          # even groups
    ([5, 0, 1, 7], [2, 0, 3, 1]),          # an empty group, skew, permuted ids
    ([1], [3]),                            # a single row
    ([9, 2, 0, 0, 4, 1], [5, 1, 4, 0, 2, 3]),
]


def _run(fn, dev, act_dtype, ad_dtype, sizes, eids, scaling, seed=0):
    E, K, N, R = 6, 128, 96, 8
    g = torch.Generator().manual_seed(seed)
    total = sum(sizes)
    a = torch.randn(total, K, generator=g).to(act_dtype).to(dev).requires_grad_(True)
    A = (torch.randn(E, R, K, generator=g) * 0.05).to(ad_dtype).to(dev).requires_grad_(True)
    B = (torch.randn(E, N, R, generator=g) * 0.05).to(ad_dtype).to(dev).requires_grad_(True)
    out = fn(a, A, B, sizes, eids, scaling)
    gout = torch.randn(out.shape, generator=g).to(out.dtype).to(dev)
    out.backward(gout)
    return out.detach(), a.grad, A.grad, B.grad


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("sizes,eids", CASES)
@pytest.mark.parametrize("scaling", [1.0, 2.0, 0.5, 1.7])
@pytest.mark.parametrize("act_dtype,ad_dtype", [(torch.bfloat16, torch.bfloat16),
                                                (torch.bfloat16, torch.float32),
                                                (torch.float32, torch.float32)])
def test_padded_delta_is_bit_identical_to_the_legacy_path(monkeypatch, dev, sizes, eids,
                                                          scaling, act_dtype, ad_dtype):
    monkeypatch.setenv("NF4_QLORA_LORA_PATH", "padded")
    new = _run(lora_delta_grouped, dev, act_dtype, ad_dtype, sizes, eids, scaling)
    old = _run(_legacy_padded, dev, act_dtype, ad_dtype, sizes, eids, scaling)
    for name, n, o in zip(("out", "d_a", "d_A", "d_B"), new, old):
        assert n.dtype == o.dtype and n.shape == o.shape, name
        assert torch.equal(n, o), f"{name} differs (max |diff| {(n.float() - o.float()).abs().max().item()})"


@pytest.mark.skipif(not CUDA, reason="device-tensor expert ids need CUDA")
def test_device_expert_ids_take_the_same_values():
    """The device-eids branch selects groups on device; same bytes as host ids."""
    sizes, eids = [5, 0, 1, 7], [2, 0, 3, 1]
    host = _run(lora_delta_grouped, "cuda", torch.bfloat16, torch.float32, sizes, eids, 2.0)
    devd = _run(lambda a, A, B, s, e, sc: lora_delta_grouped(
                    a, A, B, s, torch.tensor(e, device="cuda", dtype=torch.int32), sc),
                "cuda", torch.bfloat16, torch.float32, sizes, eids, 2.0)
    for n, o in zip(host, devd):
        assert torch.equal(n, o)


def test_scaling_of_one_is_skipped_and_others_are_applied():
    d = torch.randn(4, 3)
    assert nf4_qlora._scaled(d, 1.0) is d
    assert nf4_qlora._scaled(d, 1) is d
    assert torch.equal(nf4_qlora._scaled(d, 2.0), 2.0 * d)
    t = torch.tensor(1.0)
    assert nf4_qlora._scaled(d, t) is not d        # a tensor is applied, never read


@pytest.mark.parametrize("path", ["loop", "padded"])
def test_every_path_keeps_the_scaling(monkeypatch, path):
    """The skip must only ever fire at exactly 1: alpha/r = 2 still doubles the delta."""
    monkeypatch.setenv("NF4_QLORA_LORA_PATH", path)
    sizes, eids = [3, 2], [1, 0]
    g = torch.Generator().manual_seed(1)
    a = torch.randn(5, 64, generator=g)
    A = torch.randn(2, 4, 64, generator=g)
    B = torch.randn(2, 32, 4, generator=g)
    one = lora_delta_grouped(a, A, B, sizes, eids, 1.0)
    two = lora_delta_grouped(a, A, B, sizes, eids, 2.0)
    assert torch.equal(two, 2.0 * one)


@pytest.mark.skipif(not CUDA, reason="launch counts are a CUDA property")
def test_padded_path_launches_fewer_kernels_than_the_legacy_path(monkeypatch):
    """The point of the change, measured: kernels per forward+backward at alpha == r."""
    from torch.profiler import profile, ProfilerActivity
    monkeypatch.setenv("NF4_QLORA_LORA_PATH", "padded")
    sizes, eids = [9, 2, 0, 0, 4, 1], [5, 1, 4, 0, 2, 3]

    def kernels(fn):
        _run(fn, "cuda", torch.bfloat16, torch.bfloat16, sizes, eids, 1.0)   # warm
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            _run(fn, "cuda", torch.bfloat16, torch.bfloat16, sizes, eids, 1.0)
            torch.cuda.synchronize()
        return sum(1 for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA)

    new, old = kernels(lora_delta_grouped), kernels(_legacy_padded)
    assert new < old, (new, old)
    monkeypatch.setenv("NF4_QLORA_LEAN_DELTA", "0")              # the switch reaches the old body
    assert kernels(lora_delta_grouped) > new


@pytest.mark.parametrize("dev", DEVICES)
def test_lean_delta_off_is_the_previous_body(monkeypatch, dev):
    """NF4_QLORA_LEAN_DELTA=0 (the A/B's legacy arm) gives the same bytes as the trimmed default."""
    monkeypatch.setenv("NF4_QLORA_LORA_PATH", "padded")
    sizes, eids = [5, 0, 1, 7], [2, 0, 3, 1]
    on = _run(lora_delta_grouped, dev, torch.bfloat16, torch.float32, sizes, eids, 2.0)
    monkeypatch.setenv("NF4_QLORA_LEAN_DELTA", "0")
    assert not nf4_qlora._lean_delta_enabled()
    off = _run(lora_delta_grouped, dev, torch.bfloat16, torch.float32, sizes, eids, 2.0)
    for n, o in zip(on, off):
        assert torch.equal(n, o)


def test_gather_rows_backward_is_the_scatter_autograd_would_have_done():
    """_GatherRows against index_select's own autograd, on a unique permutation."""
    g = torch.Generator().manual_seed(3)
    src = torch.randn(10, 6, generator=g, requires_grad=True)
    idx = torch.tensor([7, 0, 3, 9, 4])
    gout = torch.randn(5, 6, generator=g)
    nf4_qlora._GatherRows.apply(src, idx).backward(gout)
    mine = src.grad.clone(); src.grad = None
    src.index_select(0, idx).backward(gout)
    assert torch.equal(mine, src.grad)
