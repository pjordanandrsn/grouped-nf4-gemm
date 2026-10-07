"""NF4_QLORA_COMPACT_BUCKETS=1: the bucketed lean padded delta as one autograd node (``_CompactBucketedDelta``).

The bar is ``torch.equal`` against the bucketed autograd body (the flag off), for the output and every gradient: the node
issues the same forward ``bmm``s per bucket (into one preallocated output) and, in backward, the same calls ``BmmBackward0``,
the split / ``cat``, ``index_copy_`` and the adapters' gather would. Covered on every bucket case and dtype pair of
``test_lora_delta_pad_buckets.py``, unique and repeated expert ids, CPU and CUDA. The flag must leave the single block and
the ladder's plans alone, and on CUDA the memory a call holds between its forward and its backward must fall.
"""
import pytest
import torch

import nf4_qlora
from nf4_qlora import lora_delta_grouped
from test_lora_delta_pad_buckets import CASES, DEVICES, DTYPES, _fresh, _inputs, _run

CUDA = torch.cuda.is_available()


def _both(monkeypatch, inputs, sizes, eids, scaling, extra=None):
    """(the autograd body's results, the compact node's results, compact calls counted) with the given flags."""
    monkeypatch.setenv("NF4_QLORA_LORA_PATH", "padded")
    monkeypatch.setenv("NF4_QLORA_PAD_BUCKETS", "1")
    for k, v in (extra or {}).items():
        monkeypatch.setenv(k, v)
    res = {}
    for flag in ("0", "1"):
        monkeypatch.setenv("NF4_QLORA_COMPACT_BUCKETS", flag)
        _fresh()
        n0 = nf4_qlora.COMPACT_BUCKETS_STATS["calls"]
        res[flag] = _run(lora_delta_grouped, inputs, sizes, eids, scaling)
        res[flag + "n"] = nf4_qlora.COMPACT_BUCKETS_STATS["calls"] - n0
    return res["0"], res["1"], (res["0n"], res["1n"])


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("sizes,eids", CASES)
@pytest.mark.parametrize("act_dtype,ad_dtype", DTYPES)
@pytest.mark.parametrize("scaling", [1.0, 0.5])
def test_compact_buckets_are_the_same_bytes(monkeypatch, dev, sizes, eids, act_dtype, ad_dtype, scaling):
    inputs = _inputs(dev, sizes, eids, act_dtype, ad_dtype)
    ref, got, (n0, n1) = _both(monkeypatch, inputs, sizes, eids, scaling)
    assert n0 == 0 and n1 == 1, (n0, n1)
    for name, r, x in zip(("out", "grad a", "grad A", "grad B"), ref, got):
        assert torch.equal(r, x), name


@pytest.mark.parametrize("dev", DEVICES)
def test_the_flag_leaves_the_single_block_alone(monkeypatch, dev):
    """With buckets off the flag changes nothing: the single block runs, the node is never called."""
    sizes, eids = [60, 2, 2, 2, 2, 2, 2, 2], list(range(8))
    inputs = _inputs(dev, sizes, eids, torch.bfloat16, torch.float32)
    _, _, (n0, n1) = _both(monkeypatch, inputs, sizes, eids, 1.0, extra={"NF4_QLORA_PAD_BUCKETS": "0"})
    assert (n0, n1) == (0, 0)
    monkeypatch.setenv("NF4_QLORA_PAD_BUCKETS", "0")
    outs = []
    for flag in ("0", "1"):
        monkeypatch.setenv("NF4_QLORA_COMPACT_BUCKETS", flag)
        _fresh()
        outs.append(_run(lora_delta_grouped, inputs, sizes, eids, 1.0))
    for name, r, x in zip(("out", "grad a", "grad A", "grad B"), *outs):
        assert torch.equal(r, x), name


def test_the_flag_leaves_the_ladder_alone(monkeypatch):
    """A laddered plan keeps its own body (``_lora_delta_bucketed_ladder``): the node is never called and the values match."""
    sizes, eids = [2, 1, 0, 90, 3, 0, 1, 4, 75, 2, 5, 1], [9, 4, 0, 1, 7, 2, 11, 3, 6, 10, 8, 5]
    inputs = _inputs("cpu", sizes, eids, torch.bfloat16, torch.float32)
    ref, got, (n0, n1) = _both(monkeypatch, inputs, sizes, eids, 1.0, extra={"NF4_QLORA_PAD_BUCKETS_LADDER": "1"})
    assert (n0, n1) == (0, 0)
    for name, r, x in zip(("out", "grad a", "grad A", "grad B"), ref, got):
        assert torch.equal(r, x), name


@pytest.mark.skipif(not CUDA, reason="device memory is a CUDA statistic")
@pytest.mark.parametrize("ad_dtype", [torch.float32, torch.bfloat16])
def test_held_memory_falls_and_backward_peak_does_not_rise(monkeypatch, ad_dtype):
    """A Zipf-like routing over 64 experts (several buckets): the bytes a call holds from its forward to its backward fall,
    and its backward peak is not above the autograd body's."""
    monkeypatch.setenv("NF4_QLORA_LORA_PATH", "padded")
    monkeypatch.setenv("NF4_QLORA_PAD_BUCKETS", "1")
    g = torch.Generator().manual_seed(1)
    sizes = sorted((int(4000 / (i + 1) ** 0.8) % 900 + 1 for i in range(64)), reverse=True)
    eids = list(torch.randperm(64, generator=g).tolist())
    inputs = _inputs("cuda", sizes, eids, torch.bfloat16, ad_dtype, K=1024, N=768, R=16)
    held, peak = {}, {}
    for flag in ("0", "1"):
        monkeypatch.setenv("NF4_QLORA_COMPACT_BUCKETS", flag)
        _fresh()
        a, A, B, gout = (t.detach().clone() for t in inputs)
        a, A, B = (t.requires_grad_(True) for t in (a, A, B))
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        out = lora_delta_grouped(a, A, B, sizes, eids, 1.0)
        torch.cuda.synchronize()
        held[flag] = torch.cuda.memory_allocated() - base - out.numel() * out.element_size()
        torch.cuda.reset_peak_memory_stats()
        out.backward(gout)
        torch.cuda.synchronize()
        peak[flag] = torch.cuda.max_memory_allocated() - base
        del out, a, A, B
    print(f"\nheld after forward {held['0'] / 2**20:.1f} -> {held['1'] / 2**20:.1f} MiB; backward peak {peak['0'] / 2**20:.1f} -> "
          f"{peak['1'] / 2**20:.1f} MiB ({ad_dtype})")
    assert held["1"] < held["0"] / 2, held
    assert peak["1"] <= peak["0"], peak
