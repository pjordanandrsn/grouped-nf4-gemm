# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Model-level gradient checks through frozen GGUF k-quant weights (bench/gguf-backward/CALIBRATION.md).

A tiny 2-layer model has an RMS norm, a LoRA-adapted square projection, a gated FFN and a LoRA-adapted down projection.
All four base weights are frozen k-quant bytes, decoded by `kquant_ref.KQuantLinearRef`.

- **fp64 gradcheck:** `compute_dtype=torch.float64` runs the SAME Function code in fp64, so
  `torch.autograd.gradcheck` checks its forward and backward inside a composed graph (norms, gating, residuals,
  adapters). Slow mode (the full Jacobian) is required: fast mode passed a backward that used W^T for the square
  projection.
- **Armed:** a backward off by 0.1%, a transposed weight in the backward, and a zeroed backward must each fail it.
- **The bf16 path:** the adapter and input gradients must equal those of the same model over the dense bf16 weight,
  bit for bit.
- **No statistical model-level bar against fp64.** Measured before writing these tests, the bf16 path sits 2e-3 to
  1.7e-2 from the fp64 model's gradients after two layers, and a 0.1% backward error falls inside that spread. So
  the exact instruments above are the checks.
"""
import pytest
import torch

pytest.importorskip("gguf")       # the fixture blocks come from test_kquant_backward, which needs gguf-py

import kquant_ref as kq  # noqa: E402
from kquant_ref import GGML_TYPE_NAMES, dequantize_ggml  # noqa: E402
from test_kquant_backward import TYPES, _blocks  # noqa: E402

H, F, T, R = 256, 512, 3, 2
SHAPES = {"q": (H, H), "g": (F, H), "u": (F, H), "d": (H, F)}


def _weights(gtype, seed=0):
    return {n: _blocks(gtype, N, K, seed + i).tobytes() for i, (n, (N, K)) in enumerate(SHAPES.items())}


def _rms(x):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)


def _model(x, aq, bq, ad, bd, lin, layers=2):
    for _ in range(layers):
        h = _rms(x)
        x = x + lin(h, "q") + (h @ aq.T) @ bq.T
        h = _rms(x)
        a = torch.nn.functional.silu(lin(h, "g")) * lin(h, "u")
        x = x + lin(a, "d") + (a @ ad.T) @ bd.T
    return x


def _gain(gtype, w):
    """A constant per weight that brings its output to unit scale: the synthetic blocks' W is 1-90x 1/sqrt(K) by type,
    and unscaled outputs reach ~1e6 through two gated layers, where finite differences lose their precision."""
    return {n: 1.0 / (float(dequantize_ggml(gtype, w[n], SHAPES[n]).std()) * SHAPES[n][1] ** 0.5) for n in SHAPES}


def _packed(gtype, w, compute_dtype):
    gain = _gain(gtype, w)
    return lambda h, n: kq.kquant_linear(h, w[n], gtype, SHAPES[n], compute_dtype).to(h.dtype) * gain[n]


def _inputs(dtype, seed=1):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(T, H, generator=g, dtype=torch.float64)
    ad = [torch.randn(*s, generator=g, dtype=torch.float64) * 0.05 for s in ((R, H), (H, R), (R, F), (H, R))]
    return [t.to(dtype).requires_grad_() for t in [x, *ad]]


@pytest.mark.parametrize("gtype", TYPES, ids=[GGML_TYPE_NAMES[t] for t in TYPES])
def test_fp64_gradcheck_through_the_frozen_kquant_weights(gtype):
    """Slow mode w.r.t. the input (one token, two layers), plus fast mode w.r.t. every input. Fast mode alone is NOT
    enough: it passes a backward that uses W^T for the square projection (test_gradcheck_is_armed[transpose]); the
    full Jacobian catches it."""
    w = _weights(gtype)
    lin = _packed(gtype, w, torch.float64)
    ins = _inputs(torch.float64)
    x1, adapters = ins[0][:1].detach().requires_grad_(), [t.detach() for t in ins[1:]]
    assert torch.autograd.gradcheck(lambda x: _model(x, *adapters, lin), [x1])
    assert torch.autograd.gradcheck(lambda *a: _model(*a, lin), ins, fast_mode=True)


def test_full_gradcheck_on_one_type():
    """The non-fast mode builds the whole Jacobian; one type, one layer and one token keep it cheap."""
    w = _weights(kq.GGML_Q4_K)
    lin = _packed(kq.GGML_Q4_K, w, torch.float64)
    ins = _inputs(torch.float64)
    x1 = ins[0][:1].detach().requires_grad_()
    assert torch.autograd.gradcheck(lambda x, *a: _model(x, *a, lin, layers=1), [x1, *ins[1:]])


def _broken_backward(kind):
    orig = kq.KQuantLinearRef.backward

    def bad(ctx, dy):
        out = orig(ctx, dy)
        dx = out[0]
        if kind == "scaled":
            dx = dx * 1.001
        elif kind == "zero":
            dx = torch.zeros_like(dx)
        elif kind == "transpose" and ctx.shape[0] == ctx.shape[1]:
            w = kq._decode_for(ctx.ggml_type, ctx.raw, ctx.shape, ctx.cd, dy.device)
            dx = kq._mm(dy, w.T, ctx.cd).to(ctx.x_dtype)
        return (dx,) + tuple(out[1:])
    return orig, bad


@pytest.mark.parametrize("kind", ["scaled", "zero", "transpose"])
def test_gradcheck_is_armed(kind):
    w = _weights(kq.GGML_Q4_K)
    lin = _packed(kq.GGML_Q4_K, w, torch.float64)
    orig, bad = _broken_backward(kind)
    ins = _inputs(torch.float64)
    x1, adapters = ins[0][:1].detach().requires_grad_(), [t.detach() for t in ins[1:]]
    kq.KQuantLinearRef.backward = staticmethod(bad)
    try:      # slow mode, w.r.t. one token: fast mode misses the transpose (see the test above)
        ok = torch.autograd.gradcheck(lambda x: _model(x, *adapters, lin), [x1], raise_exception=False)
    finally:
        kq.KQuantLinearRef.backward = orig
    assert ok is False, f"gradcheck passed a {kind} backward"


@pytest.mark.parametrize("gtype", TYPES, ids=[GGML_TYPE_NAMES[t] for t in TYPES])
def test_bf16_path_gradients_equal_the_dense_bf16_model_bitwise(gtype):
    w = _weights(gtype)
    dense = {n: dequantize_ggml(gtype, w[n], SHAPES[n]).to(torch.bfloat16) for n in SHAPES}
    grads = []
    gain = _gain(gtype, w)
    for lin in (_packed(gtype, w, torch.bfloat16),
                lambda h, n: (h.to(torch.bfloat16).float() @ dense[n].float().T).to(torch.bfloat16).to(h.dtype)
                * gain[n]):
        ins = _inputs(torch.float32)
        _model(*ins, lin).square().mean().backward()
        grads.append([t.grad for t in ins])
    for gp, gd in zip(*grads):
        assert gp.abs().sum() > 0 and torch.equal(gp, gd)
