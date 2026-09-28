"""The MXFP4 prefill combine adds in a fixed order, so repeated calls give the same bits.

``Mxfp4PipelinedGptOss._forward_prefill`` sums each token's routed-expert outputs into
``out``. It used ``out.index_add_(0, rows, ...)``, and on CUDA ``index_add_`` adds with
float atomics, so a token that appears twice in one call has its terms summed in
whatever order the threads win. On the NAS RTX A2000, 50 identical calls at Kimi-K3
geometry gave 50 different fp32 outputs, and Kimi-K3's p(' Paris') moved from process
to process (experts4bit-qlora#761). ``_index_add_ordered_`` gives every ``index_add_``
call unique rows.

These pin that it is BITWISE the sequential loop, on CPU always and on CUDA when there
is one, and that repeated CUDA calls are bitwise identical. No triton is needed: the
helper is plain torch, so CPU CI runs the CPU half.
"""
import pytest
import torch

from mxfp4_pipelined import _index_add_ordered_

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _sequential(out, rows, src):
    """The reference: one fp32 add per term, in index order."""
    out = out.clone()
    for i, r in enumerate(rows.tolist()):
        out[r] += src[i]
    return out


def _case(n_rows, n, width, seed, device):
    g = torch.Generator().manual_seed(seed)
    rows = torch.randint(0, n_rows, (n,), generator=g)
    src = torch.randn(n, width, generator=g)
    # a NONZERO base, as out is after the first chunk: (c + a) + b != (c + b) + a
    base = torch.randn(n_rows, width, generator=g)
    return base.to(device), rows.to(device), src.to(device)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("n_rows,n", [
    (1, 7),         # every term on one row: seven passes
    (6, 96),        # Kimi-K3 prefill of 6 tokens, 16 experts each
    (90, 1440),     # a 90-token perplexity window
    (40, 30),       # mostly unique rows
    (5, 0),         # an empty chunk is a no-op
])
def test_equals_the_sequential_loop_bitwise(device, n_rows, n):
    base, rows, src = _case(n_rows, n, 257, seed=1000 * n_rows + n, device=device)
    got = _index_add_ordered_(base.clone(), rows, src)
    ref = _sequential(base.cpu(), rows.cpu(), src.cpu())
    assert got.device.type == device
    assert torch.equal(got.cpu(), ref)


@pytest.mark.parametrize("device", DEVICES)
def test_unique_rows_equal_plain_index_add(device):
    """With no repeated row there is no race to remove: one pass, plain index_add_'s bits."""
    base, _, src = _case(64, 64, 33, seed=5, device=device)
    rows = torch.randperm(64, generator=torch.Generator().manual_seed(6)).to(device)
    got = _index_add_ordered_(base.clone(), rows, src)
    assert torch.equal(got, base.clone().index_add_(0, rows, src))


def test_returns_out_and_writes_in_place():
    base, rows, src = _case(6, 96, 8, seed=7, device="cpu")
    out = base.clone()
    assert _index_add_ordered_(out, rows, src) is out
    assert not torch.equal(out, base)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the atomics race needs a CUDA device")
@pytest.mark.parametrize("n_rows,n", [(6, 96), (90, 1440)])
def test_repeated_cuda_calls_are_bitwise_identical(n_rows, n):
    """The property the prefill path lost: identical inputs, identical bits, over 50
    calls at Kimi-K3 width (7168)."""
    base, rows, src = _case(n_rows, n, 7168, seed=n, device="cuda")
    first = _index_add_ordered_(base.clone(), rows, src)
    for _ in range(49):
        assert torch.equal(_index_add_ordered_(base.clone(), rows, src), first)
