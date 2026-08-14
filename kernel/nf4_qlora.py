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

# Padded-bmm cutoff for `lora_delta_grouped`: fall back to the per-expert loop once
# padding would inflate the row count past this multiple of the real rows. 4x is a
# guard against pathological router skew, not a tuned optimum — at uniform-ish
# routing the ratio sits near 1 and never approaches it.
_PAD_WASTE_LIMIT = 4.0


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
    def forward(ctx, a_cat, packed, absmax, sizes, expert_ids, weights_fn=None,
                dgrad_kernel=True):
        from nf4_grouped import gemm_4bit_grouped

        ctx.dgrad_kernel = dgrad_kernel
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
        # Kept AS GIVEN. This used to be `[int(e) for e in expert_ids]`, which on
        # a device tensor is one device-to-host sync PER GROUP -- eight of them
        # on an 8-group cell, measured -- and a sync is illegal inside a CUDA
        # graph capture. It was one of the five hazards that made the fused
        # training path uncapturable while the dequant-on-forward baseline
        # captured cleanly (bisected in bench/phase1/probe_capture_bisect.py).
        #
        # `sizes` stays a host sequence by contract: the kernel launch grid is
        # derived from it, so it has to be host-readable anyway. `expert_ids`
        # does not, and is passed through to the backward's dgrad kernel
        # untouched. The per-expert fallback loop in backward is the only reader
        # that needs host ints, and it materialises them ONCE, on its own branch,
        # rather than making every step pay for a path it usually does not take.
        ctx.sizes = sizes
        ctx.expert_ids = expert_ids
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        from nf4_grouped import dequant_ref, dgrad_4bit_grouped, dgrad_eligible

        grad_a = None
        if ctx.needs_input_grad[0]:
            packed, absmax = ((ctx.packed, ctx.absmax) if ctx.weights_fn is None
                              else ctx.weights_fn())
            _E, N, half = packed.shape
            K = half * 2
            grad_out = grad_out.contiguous()

            # Single-launch dgrad. ON by default since 2026-08-12.
            #
            # It was off on the argument that the loop below decodes with
            # `dequant_ref` -- the same oracle the reference path uses -- so its
            # gradient is EXACT, while the kernel accumulates in fp32 over a
            # different order and lands at ~2.9e-3 relative. Inside the bf16
            # budget, but not zero, so exactness was treated as something a
            # training run should not silently inherit.
            #
            # What that left out is the price, which 0.7.0 had already measured
            # and published: against this same per-expert decode oracle the
            # kernel runs 5.92 ms vs 61.78 ms on gate_up at E=256, and 3.28 ms
            # vs 85.12 ms on down (A2000, T_cat=4096) -- and the composed
            # training step 403.7 -> 26.5 ms. The loop materializes a decoded
            # expert per group, which is precisely the round trip the fused
            # forward exists to avoid, so the shipped default was paying the
            # forward's whole thesis back in the backward.
            #
            # 2.9e-3 sits an order of magnitude inside the bf16 mantissa budget
            # (eps ~3.9e-3, and a K-term dot accumulates ~sqrt(K) of it), so
            # this gradient is not distinguishable from the loop's at the dtype
            # training actually runs in.
            #
            # `dgrad_kernel=False` restores the exact loop, and is the right
            # choice for gradient-equivalence work: bit-exact A/B against a
            # reference trainer, or convergence forensics. The guards below are
            # unchanged -- an ineligible shape or offload-staged storage still
            # falls back to the loop -- so exactness is never merely a flag away
            # from being silently wrong.
            if ctx.dgrad_kernel and dgrad_eligible(grad_out, packed, absmax) is None:
                if packed.device == grad_out.device:
                    return ((dgrad_4bit_grouped(grad_out, packed, absmax,
                                                ctx.sizes, ctx.expert_ids),)
                            + (None,) * 6)
                # Offload-staged on another device: the kernel would need the
                # whole stack resident, which is the thing offload exists to
                # avoid. Per-expert staging below stays correct there.

            grad_a = torch.empty(grad_out.shape[0], K, dtype=grad_out.dtype,
                                 device=grad_out.device)
            # The fallback loop is the ONLY reader here that needs host ints, so
            # the device->host trip happens here and once (`.tolist()`), not per
            # element in every forward. This branch cannot be captured anyway --
            # it enqueues per-expert work from python, which is the cost the
            # fused forward exists to remove.
            sizes_h = (ctx.sizes if isinstance(ctx.sizes, (list, tuple))
                       else ctx.sizes.tolist())
            eids_h = (ctx.expert_ids if isinstance(ctx.expert_ids, (list, tuple))
                      else ctx.expert_ids.tolist())
            row = 0
            for g, e in enumerate(eids_h):
                n = int(sizes_h[g])
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
        return grad_a, None, None, None, None, None, None


def gemm_4bit_grouped_train(a_cat, packed, absmax, sizes, expert_ids,
                            weights_fn=None, dgrad_kernel=True):
    """Differentiable ``gemm_4bit_grouped``. Same arguments, same output; the
    only difference is that ``a_cat`` may require grad.

    ``weights_fn``: optional zero-arg callable returning ``(packed, absmax)``
    at BACKWARD time. Pass it whenever the storage is offload-staged, so this
    function holds no reference that would defeat eviction. Omit it when the
    weights are permanently resident."""
    return FusedGroupedNf4.apply(a_cat, packed, absmax, sizes, expert_ids,
                                 weights_fn, dgrad_kernel)


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

    Batched by default. This ran as a Python loop over experts, which put ``2E``
    matmul nodes per projection per layer on the autograd graph and paid for them
    again in backward. Padding the groups and running two ``bmm``s instead
    measured **3.0x on the end-to-end training step** at E=256 (401 -> 134 ms,
    A2000, 512 tokens, top_k 8, hidden 512) for +36% peak memory, with gradients
    agreeing to 1.6e-3 — inside the bf16 noise floor of ~6.5e-3. It was the
    cheapest win in the lane by a wide margin: batching the *backward's* decode
    loop bought only 1.24x and cost 4.4x peak memory, because that one has to
    materialize the weight stack and this one does not.

    Padding is the one hazard. Group sizes come from the router, so a hot expert
    makes ``max(sizes)`` large and the padded block ``G * max(sizes)`` rows wide
    regardless of how few rows are real. Past ``_PAD_WASTE_LIMIT`` the loop is
    faster and is used instead — pathological routing must not silently cost
    more than it did before this change.
    """
    # `sizes` is a host sequence by contract (the kernel launch grid comes off
    # it), so which groups are non-empty is a host-side fact and needs no device
    # read. `expert_ids` may be either form and is NEVER iterated in Python here:
    # on a device tensor that would be one sync per group.
    nz = [g for g in range(len(sizes)) if int(sizes[g]) > 0]
    if not nz:
        return None                      # unchanged: no rows, no delta tensor
    rows = [int(sizes[g]) for g in nz]
    total, widest = sum(rows), max(rows)
    if len(rows) * widest > _PAD_WASTE_LIMIT * total:
        return _lora_delta_grouped_loop(a_cat, lora_A, lora_B, sizes,
                                        expert_ids, scaling)

    dev = a_cat.device
    from nf4_grouped import to_device_i32
    if torch.is_tensor(expert_ids) and expert_ids.is_cuda:
        # Select the surviving groups ON DEVICE. One index_select, no round trip.
        sz_i32, nz_i = to_device_i32((rows, nz), dev)
        eid = expert_ids[nz_i.to(torch.int64)].to(torch.int64)
    else:
        # Host data: a list, or a CPU tensor (Bugbot, PR #85 — the old
        # per-element path accepted CPU tensors and indexing one with the CUDA
        # `nz_i` above raises). `int(expert_ids[g])` is host-only for both.
        sz_i32, eid_i32 = to_device_i32((rows, [int(expert_ids[g]) for g in nz]),
                                        dev)
        eid = eid_i32.to(torch.int64)
    sz = sz_i32.to(torch.int64)
    # Row -> (group, slot within group). Built on device: the whole point is to
    # stop enqueuing per-expert work from Python.
    #
    # `output_size=total` is load-bearing, not a micro-optimisation: without it
    # repeat_interleave has to READ `sz` to learn how long its output is, which
    # is a device-to-host sync and is illegal inside a CUDA graph capture. The
    # value is already known on the host (`sum(rows)`), so handing it over costs
    # nothing and removes the sync.
    grp = torch.repeat_interleave(torch.arange(len(rows), device=dev), sz,
                                  output_size=total)
    slot = torch.arange(total, device=dev) - (torch.cumsum(sz, 0) - sz)[grp]

    A, B = lora_A[eid], lora_B[eid]                    # [G, r, K], [G, N, r]
    x = a_cat.new_zeros(len(rows), widest, a_cat.shape[1]).to(A.dtype)
    x[grp, slot] = a_cat.to(A.dtype)
    d = scaling * torch.bmm(torch.bmm(x, A.transpose(1, 2)), B.transpose(1, 2))
    # Scatter back into the caller's row order. Zero-size groups were dropped
    # above, so `grp`/`slot` address exactly the real rows and nothing else.
    out = torch.zeros(a_cat.shape[0], B.shape[1], dtype=d.dtype, device=dev)
    out[:total] = d[grp, slot]
    return out


def _lora_delta_grouped_loop(a_cat, lora_A, lora_B, sizes, expert_ids, scaling=1.0):
    """The per-expert reference. Kept as the fallback for pathological group-size
    skew, and as the oracle the batched path is tested against."""
    out = None
    row = 0
    # One materialisation, not one per group: `enumerate` over a device tensor
    # syncs on every element. This path already enqueues per-expert work from
    # python and is not capturable either way, but it should not pay 2E syncs
    # to find that out.
    eids_h = (expert_ids if isinstance(expert_ids, (list, tuple))
              else expert_ids.tolist())
    for g, e in enumerate(eids_h):
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
                       lora_A=None, lora_B=None, weights_fn=None, scaling=1.0,
                       dgrad_kernel=True):
    """Frozen 4-bit projection through the fused kernel **plus** the trainable
    low-rank delta, returned pre-activation so callers can apply SwiGLU after.

    This is the composition the forward-only kernel could not express:
    ``W x`` fused and differentiable w.r.t. ``x``, ``B(Ax)`` differentiable
    w.r.t. ``A`` and ``B``, summed before any nonlinearity.
    """
    out = gemm_4bit_grouped_train(a_cat, packed, absmax, sizes, expert_ids,
                                  weights_fn=weights_fn, dgrad_kernel=dgrad_kernel)
    if lora_A is None or lora_B is None:
        return out
    delta = lora_delta_grouped(a_cat, lora_A, lora_B, sizes, expert_ids, scaling)
    return out + delta.to(out.dtype)
