"""Gradient correctness for the differentiable grouped NF4 GEMM.

The fused kernel is forward-only, so training fell back to
dequantize-then-matmul. nf4_qlora wraps it so dL/dx flows. The bar: gradients
must match the dequantize-then-matmul path they replace, or the fused training
path is silently wrong in a way loss curves would hide for a long time.

The kernel itself is CUDA/Triton, so the fused arm is skipped without a GPU --
but the REFERENCE arm and the LoRA composition are pure torch and run on CPU,
which is what keeps this test meaningful in CI.
"""
import pytest
import torch

from nf4_grouped import dequant_ref
from nf4_pack_ref import quantize_pack_nf4
from nf4_qlora import (FusedGroupedNf4, fused_grouped_lora,  # noqa: F401
                       lora_delta_grouped)

E, N, K, R = 4, 64, 128, 8
CUDA = torch.cuda.is_available()


def _packed_stack(seed=0):
    torch.manual_seed(seed)
    packed, absmax = [], []
    for _ in range(E):
        w = torch.randn(N, K)
        p, a = quantize_pack_nf4(w)
        packed.append(p.reshape(N, K // 2))
        absmax.append(a.reshape(N, K // 64).float())
    return torch.stack(packed), torch.stack(absmax)


def _grouped_inputs(rows_per_expert=3):
    sizes = [rows_per_expert] * E
    expert_ids = list(range(E))
    a = torch.randn(rows_per_expert * E, K, dtype=torch.float32)
    return a, sizes, expert_ids


def _reference_forward(a_cat, packed, absmax, sizes, expert_ids):
    """What the fused path replaces: decode each expert, dense matmul."""
    outs, row = [], 0
    for g, e in enumerate(expert_ids):
        n = sizes[g]
        w = dequant_ref(packed[e], absmax[e], N, K).to(a_cat.dtype)
        outs.append(a_cat[row:row + n] @ w.T)
        row += n
    return torch.cat(outs, dim=0)


def test_lora_delta_matches_explicit_per_expert_math():
    """The grouped delta must equal B_e @ (A_e @ x) computed row by row."""
    packed, absmax = _packed_stack()
    a, sizes, eids = _grouped_inputs()
    A = torch.randn(E, R, K) * 0.05
    B = torch.randn(E, N, R) * 0.05
    got = lora_delta_grouped(a, A, B, sizes, eids)
    row = 0
    for g, e in enumerate(eids):
        n = sizes[g]
        want = (a[row:row + n] @ A[e].T) @ B[e].T
        torch.testing.assert_close(got[row:row + n], want, rtol=1e-5, atol=1e-5)
        row += n


def test_zero_B_delta_is_exactly_zero():
    """B is zero-initialised, so an untrained adapter must contribute NOTHING
    -- identically, not approximately. e4b's delegate-to-base decision rests on
    this being exact."""
    a, sizes, eids = _grouped_inputs()
    A = torch.randn(E, R, K)
    B = torch.zeros(E, N, R)
    d = lora_delta_grouped(a, A, B, sizes, eids)
    assert torch.count_nonzero(d) == 0


@pytest.mark.skipif(not CUDA, reason="fused kernel is CUDA/Triton only")
def test_fused_backward_matches_dequant_reference():
    """dL/dx through the fused kernel == dL/dx through decode-then-matmul."""
    packed, absmax = _packed_stack()
    a, sizes, eids = _grouped_inputs()
    packed_c, absmax_c = packed.cuda(), absmax.cuda()

    a_ref = a.clone().cuda().to(torch.bfloat16).requires_grad_(True)
    out_ref = _reference_forward(a_ref, packed_c, absmax_c, sizes, eids)
    out_ref.sum().backward()

    a_fus = a.clone().cuda().to(torch.bfloat16).requires_grad_(True)
    out_fus = FusedGroupedNf4.apply(a_fus, packed_c, absmax_c, sizes, eids)
    out_fus.sum().backward()

    # Tolerances from the DTYPE, not from what happens to pass. bf16 carries an
    # 8-bit mantissa (eps ~ 2**-8 ~ 3.9e-3); a K-term dot product accumulates
    # ~sqrt(K)*eps, so at K=128 elementwise agreement better than ~4% is not
    # available, and the kernel accumulates in a different order than a dense
    # matmul. Measured on sm_86 and sm_89: forward relative Frobenius error
    # 0.0027 -- an order of magnitude inside that bound.
    fwd_rel = ((out_fus.float() - out_ref.float()).norm()
               / out_ref.float().norm()).item()
    assert fwd_rel < 1e-2, f"forward relative error {fwd_rel} exceeds bf16 budget"
    torch.testing.assert_close(out_fus, out_ref, rtol=5e-2, atol=5e-2)
    # The gradient is grad_out @ W with W decoded by the same oracle the
    # reference uses, so this should be EXACT, not merely close. Measured 0.0
    # relative error on both architectures. Asserting exactness makes any future
    # drift in the backward loud.
    grad_rel = ((a_fus.grad.float() - a_ref.grad.float()).norm()
                / a_ref.grad.float().norm()).item()
    assert grad_rel == 0.0, f"backward is no longer exact: rel {grad_rel}"


@pytest.mark.skipif(not CUDA, reason="fused kernel is CUDA/Triton only")
def test_lora_params_receive_gradient_through_the_fused_path():
    """The whole point: A and B must train while the kernel does the base GEMM."""
    packed, absmax = _packed_stack()
    a, sizes, eids = _grouped_inputs()
    packed_c, absmax_c = packed.cuda(), absmax.cuda()
    a_c = a.cuda().to(torch.bfloat16)
    A = (torch.randn(E, R, K, device="cuda") * 0.05).requires_grad_(True)
    B = (torch.zeros(E, N, R, device="cuda")).requires_grad_(True)

    out = fused_grouped_lora(a_c, packed_c, absmax_c, sizes, eids, A, B)
    out.sum().backward()
    assert A.grad is not None and B.grad is not None
    # B starts at zero so dL/dA is zero, but dL/dB must not be
    assert torch.count_nonzero(B.grad) > 0, "B received no gradient"


@pytest.mark.skipif(not CUDA, reason="fused kernel is CUDA/Triton only")
def test_backward_holds_one_decoded_expert_at_a_time():
    """The memory guarantee: backward re-decodes per expert, so peak must scale
    with ONE dense expert, not all of them.

    Sized so weights actually dominate: at the toy size used elsewhere the whole
    stack is 64 KB and peak is swamped by allocator overhead (~17 MB), which is
    why an earlier version of this test asserted something it could not measure.
    """
    BIG_E, BIG_N, BIG_K = 32, 512, 1024
    torch.manual_seed(1)
    packed, absmax = [], []
    for _ in range(BIG_E):
        p, am = quantize_pack_nf4(torch.randn(BIG_N, BIG_K))
        packed.append(p.reshape(BIG_N, BIG_K // 2))
        absmax.append(am.reshape(BIG_N, BIG_K // 64).float())
    packed_c = torch.stack(packed).cuda()
    absmax_c = torch.stack(absmax).cuda()
    sizes = [2] * BIG_E
    eids = list(range(BIG_E))
    a_c = torch.randn(2 * BIG_E, BIG_K, device="cuda",
                      dtype=torch.bfloat16).requires_grad_(True)

    torch.cuda.synchronize(); torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    out = FusedGroupedNf4.apply(a_c, packed_c, absmax_c, sizes, eids)
    out.sum().backward()
    torch.cuda.synchronize()
    growth = torch.cuda.max_memory_allocated() - base

    one_dense = BIG_N * BIG_K * 4          # fp32 decode of a single expert
    all_dense = BIG_E * one_dense
    assert growth < all_dense / 4, (
        f"peak growth {growth} approaches all-experts-dense {all_dense}: "
        "backward looks like it is retaining decoded weights")
