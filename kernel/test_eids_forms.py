"""Every `expert_ids` form a caller can plausibly hand over produces identical
results: Python list, CPU tensor, CUDA tensor.

The CPU-tensor column is the regression Bugbot caught on PR #85: the pre-change
code accepted a host `torch.tensor([...])` (its per-element `int()` never cared
where the tensor lived), and the first capturability rewrite regressed it —
`lora_delta_grouped` indexed it with CUDA indices (RuntimeError) and the GEMM
boundary passed it straight to a Triton launch. The rule since: a CUDA tensor
passes through; ANYTHING host — list or CPU tensor — converts once at the
boundary through the pinned path.

Bitwise equality, not tolerance: the form of an index carrier must not change a
single value.
"""
from __future__ import annotations

import pytest
import torch

import nf4_grouped as NG
import nf4_qlora as NQ

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="needs CUDA (form-equality of launches)")

E, N, K, RANK = 8, 128, 128, 4
SIZES = [3, 1, 0, 5, 2]              # jagged, with an empty group
EIDS = [0, 3, 7, 5, 2]


@pytest.fixture(scope="module")
def fx():
    from bitsandbytes import functional as BF
    g = torch.Generator(device="cpu").manual_seed(7)
    w = torch.randn(E, N, K, generator=g) * 0.02
    packed, states = [], []
    for e in range(E):
        q, st = BF.quantize_4bit(w[e].to("cuda", torch.bfloat16),
                                 blocksize=64, quant_type="nf4")
        packed.append(q)
        states.append(st)
    B, A = NG.repack_from_bnb(packed, states, N, K)
    a_cat = (torch.randn(sum(SIZES), K, generator=g) * 0.5).to("cuda", torch.bfloat16)
    go = (torch.randn(sum(SIZES), N, generator=g) * 0.1).to("cuda", torch.bfloat16)
    lA = (torch.randn(E, RANK, K, generator=g) * 0.01).to("cuda", torch.bfloat16)
    lB = (torch.randn(E, N, RANK, generator=g) * 0.01).to("cuda", torch.bfloat16)
    return B, A, a_cat, go, lA, lB


def forms():
    return {
        "list": EIDS,
        "cpu_tensor": torch.tensor(EIDS, dtype=torch.int32),          # no device=
        "cuda_tensor": torch.tensor(EIDS, dtype=torch.int32, device="cuda"),
    }


def _assert_all_equal(outs: dict):
    ref_name, ref = next(iter(outs.items()))
    for name, o in outs.items():
        assert torch.equal(o, ref), (
            f"expert_ids form {name!r} diverged from {ref_name!r}: "
            f"max|Δ|={(o.float() - ref.float()).abs().max().item():.3e}")


def test_gemm_forward_identical_across_eids_forms(fx):
    B, A, a_cat, _go, _lA, _lB = fx
    _assert_all_equal({n: NG.gemm_4bit_grouped(a_cat, B, A, SIZES, e)
                       for n, e in forms().items()})


def test_dgrad_identical_across_eids_forms(fx):
    B, A, _a, go, _lA, _lB = fx
    _assert_all_equal({n: NG.dgrad_4bit_grouped(go, B, A, SIZES, e)
                       for n, e in forms().items()})


def test_lora_delta_identical_across_eids_forms(fx):
    _B, _A, a_cat, _go, lA, lB = fx
    _assert_all_equal({n: NQ.lora_delta_grouped(a_cat, lA, lB, SIZES, e, 2.0)
                       for n, e in forms().items()})


def test_fused_train_backward_identical_across_eids_forms(fx):
    B, A, a_cat, _go, _lA, _lB = fx
    grads = {}
    for n, e in forms().items():
        x = a_cat.detach().clone().requires_grad_(True)
        out = NQ.gemm_4bit_grouped_train(x, B, A, SIZES, e, dgrad_kernel=True)
        out.float().pow(2).mean().backward()
        grads[n] = x.grad.detach()
    _assert_all_equal(grads)
