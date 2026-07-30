"""Differentiable grouped NF4 GEMM — the training half of the fused lane.

``gemm_4bit_grouped`` is forward-only, so training could never reach it: any
graph that needs ``dL/dx`` through the expert projection had to fall back to
dequantize-then-matmul. This wraps the same kernel in an ``autograd.Function``
whose backward re-decodes one expert at a time, so the dequantized weight is
never stored across the forward-to-backward window.

This mirrors ``_FusedGroupedMxfp4`` in ``mxfp4_qlora.py`` exactly — same
recompute-in-backward guarantee, same "packed bytes are the only residency"
property — for the NF4/bitsandbytes layout instead of native MXFP4.

The base weight is frozen, so backward needs only ``grad_out @ W`` (the input
gradient). There is no ``dW``: nothing here trains the quantized weight.

Composing with LoRA: the delta must be added to the **pre-activation**
projection, because ``act(Wx + BAx) != act(Wx) + d`` for any cheap ``d``. So
callers add ``B(Ax)`` to this function's output *before* the SwiGLU, not after.
See ``fused_grouped_lora`` below, which does exactly that.
"""
from __future__ import annotations

import torch


class FusedGroupedNf4(torch.autograd.Function):
    """Grouped NF4 forward through the fused kernel; recompute-decode backward.

    ``a_cat`` is group-sorted ``[T_cat, K]``. ``packed`` is ``[E, N, K//2]``
    uint8 and ``absmax`` is ``[E, N, K//64]`` float — kernel-shaped views, both
    non-differentiable constants (stashed on ctx, not ``save_for_backward``:
    they are frozen storage, and saving them would imply a gradient).

    Backward: ``grad_a[rows_g] = grad_out[rows_g] @ decode(e_g)`` per group,
    one decoded expert live at a time.
    """

    @staticmethod
    def forward(ctx, a_cat, packed, absmax, sizes, expert_ids, weights_fn=None):
        from nf4_grouped import gemm_4bit_grouped

        out = gemm_4bit_grouped(a_cat, packed, absmax, sizes, expert_ids)
        # DO NOT stash the weight tensors themselves when a weights_fn is
        # supplied. e4b's expert offload keeps a SINGLE layer GPU-resident:
        # staging a layer evicts the previous one by reassigning ``.data``.
        # A tensor object held on ctx keeps that evicted storage alive by
        # refcount, so all 48 layers accumulate on device. Measured on a 24 GB
        # RTX 4090 / Qwen3-30B-A3B: the reference arm peaks at 9.13 GB while
        # holding tensors here OOMed at 22.41 GB asking for another 96 MiB.
        #
        # Holding a CALLABLE instead lets backward re-read whatever is staged
        # at the time it runs -- which under gradient checkpointing is exactly
        # this layer, because the recompute forward re-stages it first. Same
        # approach as ``_FrozenLinearRecomputeBackward``'s ``dequant_fn`` in
        # the MXFP4 lane.
        ctx.weights_fn = weights_fn
        if weights_fn is None:
            ctx.packed, ctx.absmax = packed, absmax
        else:
            ctx.packed = ctx.absmax = None
            ctx.wshape = (packed.shape[0], packed.shape[1], packed.shape[2])
        ctx.sizes = [int(s) for s in sizes]
        ctx.expert_ids = [int(e) for e in expert_ids]
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        from nf4_grouped import dequant_ref

        grad_a = None
        if ctx.needs_input_grad[0]:
            packed, absmax = ((ctx.packed, ctx.absmax) if ctx.weights_fn is None
                              else ctx.weights_fn())
            _E, N, half = packed.shape
            K = half * 2
            grad_out = grad_out.contiguous()
            grad_a = torch.empty(grad_out.shape[0], K, dtype=grad_out.dtype,
                                 device=grad_out.device)
            row = 0
            for g, e in enumerate(ctx.expert_ids):
                n = ctx.sizes[g]
                if n == 0:
                    continue
                # recomputed, one expert live at a time -- never stored.
                # Stage this expert's packed bytes only (a few MB), not the
                # whole stack, then let them fall out of scope.
                pe = packed[e]
                ae = absmax[e]
                if pe.device != grad_out.device:
                    pe = pe.to(grad_out.device, non_blocking=True)
                    ae = ae.to(grad_out.device, non_blocking=True)
                w = dequant_ref(pe, ae, N, K).to(grad_out.dtype)
                grad_a[row:row + n] = grad_out[row:row + n] @ w
                del pe, ae, w
                row += n
        return grad_a, None, None, None, None, None


def gemm_4bit_grouped_train(a_cat, packed, absmax, sizes, expert_ids,
                            weights_fn=None):
    """Differentiable ``gemm_4bit_grouped``. Same arguments, same output; the
    only difference is that ``a_cat`` may require grad.

    ``weights_fn``: optional zero-arg callable returning ``(packed, absmax)``
    at BACKWARD time. Pass it whenever the storage is offload-staged, so this
    function holds no reference that would defeat eviction. Omit it when the
    weights are permanently resident."""
    return FusedGroupedNf4.apply(a_cat, packed, absmax, sizes, expert_ids,
                                 weights_fn)


def lora_delta_grouped(a_cat, lora_A, lora_B, sizes, expert_ids, scaling=1.0):
    """Per-expert low-rank delta over a group-sorted activation block.

    ``a_cat`` is ``[T_cat, K]`` grouped by expert; ``lora_A`` is ``[E, r, K]``
    and ``lora_B`` is ``[E, N, r]``. Returns ``[T_cat, N]`` where each group's
    rows got ``scaling * (B_e @ (A_e @ x))``.

    ``scaling`` is LoRA's ``alpha / r`` and is NOT optional in practice: the
    reference adapter applies it (``ExpertsLoRA.scaling``), so omitting it makes
    every update the wrong size. Shipping it defaulted to 1.0 while the caller
    used alpha=16/r=8 made the fused delta exactly HALF the reference's, which
    trained visibly slower -- caught by the 48-layer parity gate at a median
    loss delta of 0.367 against a 0.05 band, after 16-layer parity had passed.

    Kept separate from the kernel call so the caller controls *where* the delta
    lands — for gate_up it must be added before the activation.
    """
    out = None
    row = 0
    for g, e in enumerate(expert_ids):
        n = int(sizes[g])
        if n == 0:
            continue
        x = a_cat[row:row + n]
        A, B = lora_A[e], lora_B[e]
        d = scaling * ((x.to(A.dtype) @ A.T) @ B.T)   # [n, r] -> [n, N]
        if out is None:
            out = torch.zeros(a_cat.shape[0], B.shape[0], dtype=d.dtype,
                              device=a_cat.device)
        out[row:row + n] = d
        row += n
    return out


def fused_grouped_lora(a_cat, packed, absmax, sizes, expert_ids,
                       lora_A=None, lora_B=None, weights_fn=None, scaling=1.0):
    """Frozen 4-bit projection through the fused kernel **plus** the trainable
    low-rank delta, returned pre-activation so callers can apply SwiGLU after.

    This is the composition the forward-only kernel could not express:
    ``W x`` fused and differentiable w.r.t. ``x``, ``B(Ax)`` differentiable
    w.r.t. ``A`` and ``B``, summed before any nonlinearity.
    """
    out = gemm_4bit_grouped_train(a_cat, packed, absmax, sizes, expert_ids,
                                  weights_fn=weights_fn)
    if lora_A is None or lora_B is None:
        return out
    delta = lora_delta_grouped(a_cat, lora_A, lora_B, sizes, expert_ids, scaling)
    return out + delta.to(out.dtype)
