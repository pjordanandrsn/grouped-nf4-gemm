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

import os

import torch

# Padded-bmm cutoff for `lora_delta_grouped`: fall back to the per-expert loop once
# padding would inflate the row count past this multiple of the real rows. 4x is a
# guard against pathological router skew, not a tuned optimum — at uniform-ish
# routing the ratio sits near 1 and never approaches it.
_PAD_WASTE_LIMIT = 4.0          # historical: the shipped `auto` rule until 0.32.1 (P46 read it as the defect); now an OPT-IN guard
_PAD_BYTES_LIMIT = 2 * 2 ** 30   # `auto` pads unless the padded block would exceed this many bytes (P46: structure, not flops)
# The adapter delta's path is a RECORDED choice (experts4bit-qlora P46, bench/p46/P46-PREREG.md / RESULTS-p46.md), never a
# silent branch. `NF4_QLORA_LORA_PATH` = auto | padded | loop | grouped_mm. The `auto` rule since 0.32.1: the padded bmm
# path unless the padded block `G * max(rows) * (K + N) * itemsize` would exceed `NF4_QLORA_PAD_BYTES_LIMIT` (default
# 2 GiB) -- what padding actually risks is MEMORY; the flops it wastes are r-rank matmuls that cost nothing next to the
# launches the loop pays (P46: at Qwen3-30B-A3B's field recipe the 4x waste guard sent >= 85 % of calls to the loop and the
# step took 24.5 s where the padded path took 4.2 s, same loss, same peak VRAM). `NF4_QLORA_PAD_WASTE_LIMIT`, when SET,
# re-arms the old flop-waste guard on top (loop above that ratio); unset means no waste guard. `grouped_mm` = two
# `torch._grouped_mm` calls over the jagged groups (no padding; CUDA, bf16, sm_90 in torch 2.8; refuses where torch has no
# kernel for the part). `LORA_PATH_STATS` counts calls per path (ints only -- consumers cast) and `LORA_PAD_WASTE` records
# the last / max padding-waste ratio seen, so a training census can say which path served a step and at what skew.
# `padded_bucketed` counts the calls `NF4_QLORA_PAD_BUCKETS=1` sent through the bucketed lean padded delta (instead of, not as
# well as, `padded`); each such call also writes `LORA_PAD_WASTE`'s `last_rows_single` (the one block's `G * widest`),
# `last_rows_bucketed` (the buckets' total padded rows) and `last_buckets`. Nothing else writes those three keys.
LORA_PATH_STATS = {"loop": 0, "padded": 0, "grouped_mm": 0, "padded_bucketed": 0}
#: Bucketed calls that took the compact node (the default; ``NF4_QLORA_COMPACT_BUCKETS=0`` off, :class:`_CompactBucketedDelta`), a subset of
#: ``LORA_PATH_STATS["padded_bucketed"]``.
COMPACT_BUCKETS_STATS = {"calls": 0}
# Which backward served each frozen-GEMM dgrad: the single-launch kernel, the grouped_mm route, the dense route, or the
# per-expert decode loop -- with the loop's reason (`dgrad_eligible`'s, offload-staged storage, or dgrad_kernel=False). The loop is
# exact and slow; it used to be taken silently, so a run asking for the kernel could not tell it had not had it.
DGRAD_STATS = {"kernel": 0, "grouped_mm": 0, "dense": 0, "decoded": 0, "loop": 0, "loop_reasons": {}}
LORA_PAD_WASTE = {"last": 0.0, "max": 0.0, "last_bytes": 0, "last_bytes_alloc": 0,
                  "last_rows_single": 0, "last_rows_bucketed": 0, "last_buckets": 0}


def _lora_path() -> str:
    import os
    v = os.environ.get("NF4_QLORA_LORA_PATH", "auto").strip().lower()
    if v not in ("auto", "padded", "loop", "grouped_mm"):
        raise ValueError(f"NF4_QLORA_LORA_PATH={v!r}: expected auto | padded | loop | grouped_mm")
    return v


def _pad_waste_limit():
    """The OPT-IN flop-waste guard: a ratio when `NF4_QLORA_PAD_WASTE_LIMIT` is set, else None (no waste guard)."""
    import os
    v = os.environ.get("NF4_QLORA_PAD_WASTE_LIMIT")
    return float(v) if v else None


def _pad_bytes_limit() -> int:
    import os
    v = os.environ.get("NF4_QLORA_PAD_BYTES_LIMIT")
    return int(float(v)) if v else _PAD_BYTES_LIMIT


def _lean_delta_enabled() -> bool:
    """The padded delta's trimmed body is the default; ``NF4_QLORA_LEAN_DELTA=0``
    restores the previous body -- bit-identical output and gradients -- so the
    two can be timed against each other on one box (experts4bit-qlora#945)."""
    import os
    return os.environ.get("NF4_QLORA_LEAN_DELTA", "1").strip() != "0"


# Within one bucket of the bucketed padded delta the widest group has at most this many times the narrowest group's rows.
_PAD_BUCKET_RATIO = 2


#: ``NF4_QLORA_PAD_BUCKETS=auto``'s gate: a call buckets when it carries at least this many routed rows (the sum of its group
#: sizes, tokens x top-k). Placed from a census of every delta call in experts4bit-qlora's TC1 (amendment 49's re-ask): at the
#: field recipe (Qwen3-30B-A3B, seq 2048, micro-batch 2) the calls carried 3,968 routed rows at the median and 9,040 at most; on
#: packed 4,096-token rows every call carries 32,768 (4,096 x top-8), where buckets took 4.29 GB off the fp32 arm's peak and
#: stepped it 0.893 (TC1 amendment 48). ``NF4_QLORA_PAD_BUCKETS_MIN_ROWS`` overrides it.
_PAD_BUCKETS_AUTO_MIN_ROWS = 16384


def _pad_buckets_mode() -> str:
    """``"auto"`` (the DEFAULT, also when unset: bucketed where the call carries at least ``_pad_buckets_min_rows()`` routed rows),
    ``"1"`` (every call bucketed) or ``"0"`` (the single block everywhere), from ``NF4_QLORA_PAD_BUCKETS``. Any other value is the
    single block. ``auto`` became the default by experts4bit-qlora TC1 amendment 50's registered rule: at the field recipe its gate never
    fired (49,152 calls per arm, all single block; held-out and peak unchanged), and on packed 4,096-token rows buckets stepped the fp32
    arm 0.893 of the single block's time with 4.29 GB off its peak, the bf16 arm 0.933 (amendment 48)."""
    v = os.environ.get("NF4_QLORA_PAD_BUCKETS", "").strip().lower()
    if v == "":
        return "auto"
    return v if v in ("1", "auto") else "0"


def _pad_buckets_min_rows() -> int:
    v = os.environ.get("NF4_QLORA_PAD_BUCKETS_MIN_ROWS")
    return int(v) if v else _PAD_BUCKETS_AUTO_MIN_ROWS


def _pad_buckets_enabled(total=None) -> bool:
    """Whether this call pads by buckets (``_lora_delta_bucketed``: each bucket of similar-sized groups padded to its own widest
    group instead of every group to the hottest one). ``NF4_QLORA_PAD_BUCKETS=1``: always; ``auto`` (the default, also when unset):
    when ``total`` (the call's routed rows) is at least ``_pad_buckets_min_rows()``; ``0``: never -- the single block, op for op."""
    mode = _pad_buckets_mode()
    if mode == "1":
        return True
    return mode == "auto" and total is not None and total >= _pad_buckets_min_rows()


def _pad_ladder_enabled() -> bool:
    """Off unless ``NF4_QLORA_PAD_BUCKETS_LADDER=1``: round every bucket's width AND group count up to ``_ladder_up``'s rungs, so
    the bucketed delta's batched products repeat their shapes from call to call. Opt-in: its first training A/B (experts4bit-qlora
    TC1 amendment 54, torch 2.8, fp32 adapters, on a host where the step was not host-bound) read 1.010 of the default buckets' time,
    with ``aten::bmm``'s CPU self time per call cut about tenfold; where the host is the bottleneck it is unread.

    Why: the router makes nearly every call's bucket shapes new, and a cuBLAS fp32 batched product costs host time per NEW shape
    (experts4bit-qlora TC1 amendment 24 on an RTX 5090: 119 us on a shape the process had not used against 38 us on a repeated one,
    the same under torch 2.8 and 2.12). Profiled in training (amendment 53), the bucketed fp32 arm's ``aten::bmm`` read about 268 us
    of CPU self time per call under torch 2.8 against 86 us under torch 2.12, while 59.7 % of torch 2.8's added step was not device time.
    Self time also counts waits on a full launch queue, so 268 us is an upper bound on its host work. A batched product's shape is
    ``(G_b, W_b)`` per projection, so both have to land on a fixed set for a shape to come back."""
    return os.environ.get("NF4_QLORA_PAD_BUCKETS_LADDER", "0").strip() == "1"


def _ladder_up(n: int) -> int:
    """The smallest rung >= ``n``: every integer up to 4, then four rungs per octave (``{4, 5, 6, 7} * 2**k``), so a rung is at
    most 25 % above ``n`` (``n <= _ladder_up(n) < 1.25 * n`` for ``n > 4``): ``n`` from 1 to 65,535 lands on 60 rungs."""
    n = int(n)
    if n <= 4:
        return max(n, 0)
    step = 1 << (n.bit_length() - 3)                   # 2**(floor(log2 n) - 2): a quarter of n's octave
    return -(-n // step) * step


def _bucket_groups(b) -> int:
    """A bucket's batch count: ``G_b``, or its laddered ``G_b'`` when the plan carries one (``(G_b, W_b, G_b')``)."""
    return b[2] if len(b) > 2 else b[0]


def _pad_buckets(rows):
    """The bucket plan for the non-empty groups' row counts ``rows`` (host ints, in the caller's group order).

    Sort the groups by rows (stable: equal counts keep the caller's order) and cut greedily from the narrowest: a
    bucket takes every next group whose rows are at most ``_PAD_BUCKET_RATIO`` times its first group's, so in every
    bucket ``W_b <= 2 * min_b`` and the bucketed block never holds more than twice the real rows (the single block holds
    ``G * max(rows)``, unbounded in the skew). Greedy from the narrowest is the fewest buckets under that rule.

    Returns ``(order, buckets, pstart)``: ``order`` the group positions in bucket order (the sort), ``buckets`` one
    ``(G_b, W_b)`` per consecutive run of ``order``, and ``pstart[j]`` the first padded row of group ``j`` (caller's order)
    in the concatenated blocks, whose total is ``sum(G_b * W_b)``. Pure host arithmetic: no device read.

    With ``NF4_QLORA_PAD_BUCKETS_LADDER=1`` each ``W_b`` is rounded up to ``_ladder_up``'s rung and each bucket carries a third
    entry, its batch count rounded the same way: ``(G_b, W_b, G_b')``. The ``G_b' - G_b`` padded groups follow the real ones in the
    bucket's block, hold no row and take zero adapters, so every real row's arithmetic is unchanged; the total is
    ``sum(G_b' * W_b)``, at most ``2 * 1.25 * 1.25`` times the real rows."""
    order = sorted(range(len(rows)), key=rows.__getitem__)
    buckets, pstart = [], [0] * len(rows)
    ladder = _pad_ladder_enabled()
    base = i = 0
    while i < len(order):
        lo, j = rows[order[i]], i
        while j < len(order) and rows[order[j]] <= _PAD_BUCKET_RATIO * lo:
            j += 1
        w = rows[order[j - 1]]
        if ladder:                                     # NF4_QLORA_PAD_BUCKETS_LADDER=1: the width and the batch count on rungs
            w = _ladder_up(w)
        for slot, g in enumerate(order[i:j]):
            pstart[g] = base + slot * w
        gp = _ladder_up(j - i) if ladder else j - i    # the padded groups (gp - (j - i) of them) sit after the real ones, all zero
        buckets.append((j - i, w, gp) if ladder else (j - i, w))
        base += gp * w
        i = j
    return order, buckets, pstart


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
        # GNF4_TRAIN_GEMM (nf4_route.py): `auto`, the default, takes the grouped_mm route -- dequantize + torch._grouped_mm,
        # forward and dgrad alike -- on compute capability 9.0; elsewhere the dense route (one expert at a time) for a call with
        # at most DENSE_AUTO_MAX_GROUPS present groups, the fused kernel above that; `fused` / `grouped_mm` / `dense` force one, and
        # `decoded` (opt-in only: auto never takes it) is dequant_groups + one grouped bf16 GEMM launch per chunk of groups.
        # The choice is made here, per call, and remembered, so a backward never mixes routes.
        from nf4_route import train_gemm_route
        ctx.route = train_gemm_route(a_cat.device, len(sizes))   # validates the value; raises on an unknown one
        if ctx.route == "grouped_mm":
            from nf4_route import grouped_mm_forward
            out = grouped_mm_forward(a_cat, packed, absmax, sizes, expert_ids)
        elif ctx.route == "dense":
            from nf4_route import dense_forward
            out = dense_forward(a_cat, packed, absmax, sizes, expert_ids)
        elif ctx.route == "decoded":
            from nf4_route import decoded_forward
            out = decoded_forward(a_cat, packed, absmax, sizes, expert_ids)
        else:
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
            # What that left out is the price. experts4bit-qlora's dgrad gate
            # (bench/dgrad-gate/, Qwen3-30B-A3B at 48 layers on a rented RTX
            # A6000, the published 0.7.0 wheel) read the fused training step at
            # 2.52x the reference loop with this kernel and 1.72x without it.
            # The loop materializes a decoded
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
            why = dgrad_eligible(grad_out, packed, absmax) if ctx.dgrad_kernel else "dgrad_kernel=False"
            if why is None:
                if packed.device == grad_out.device and getattr(ctx, "route", "fused") == "grouped_mm":
                    from nf4_route import grouped_mm_dgrad
                    DGRAD_STATS["grouped_mm"] += 1
                    return ((grouped_mm_dgrad(grad_out, packed, absmax, ctx.sizes, ctx.expert_ids),) + (None,) * 6)
                if packed.device == grad_out.device and getattr(ctx, "route", "fused") == "dense":
                    from nf4_route import dense_dgrad
                    DGRAD_STATS["dense"] += 1
                    return ((dense_dgrad(grad_out, packed, absmax, ctx.sizes, ctx.expert_ids),) + (None,) * 6)
                if packed.device == grad_out.device and getattr(ctx, "route", "fused") == "decoded":
                    from nf4_route import decoded_dgrad
                    DGRAD_STATS["decoded"] += 1
                    return ((decoded_dgrad(grad_out, packed, absmax, ctx.sizes, ctx.expert_ids),) + (None,) * 6)
                if packed.device == grad_out.device:
                    DGRAD_STATS["kernel"] += 1
                    return ((dgrad_4bit_grouped(grad_out, packed, absmax,
                                                ctx.sizes, ctx.expert_ids),)
                            + (None,) * 6)
                # Offload-staged on another device: the kernel would need the
                # whole stack resident, which is the thing offload exists to
                # avoid. Per-expert staging below stays correct there.
                why = "storage on another device (offload-staged)"
            DGRAD_STATS["loop"] += 1
            DGRAD_STATS["loop_reasons"][str(why)] = DGRAD_STATS["loop_reasons"].get(str(why), 0) + 1

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


class _GatherRows(torch.autograd.Function):
    """``src.index_select(0, idx)`` for UNIQUE ``idx``, whose backward is a plain
    scatter. Autograd's own backward for ``index_select`` is ``index_add_``, which
    must assume repeats and so adds atomically, where unique rows need only a
    plain scatter. With every row written at most once a copy is the same bytes:
    ``0 + g == g``."""

    @staticmethod
    def forward(ctx, src, idx):
        ctx.save_for_backward(idx)
        ctx.rows = src.shape[0]
        return src.index_select(0, idx)

    @staticmethod
    def backward(ctx, g):
        (idx,) = ctx.saved_tensors
        gs = g.new_zeros((ctx.rows,) + tuple(g.shape[1:]))
        gs.index_copy_(0, idx, g)
        return gs, None


def _scaled(d, scaling):
    """``scaling * d``, skipped when ``scaling`` is exactly 1: ``1.0 * x == x`` for
    every float, so the skip is bit-exact, and at alpha == r (the field recipe)
    the multiply was a full elementwise pass -- forward, recompute and backward --
    over every LoRA delta for nothing. A tensor ``scaling`` is always applied
    (comparing it would read the device)."""
    if isinstance(scaling, (int, float)) and scaling == 1:
        return d
    return scaling * d


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
    takes those nodes off the graph for +36% peak memory (RTX A2000, E=256,
    512 tokens, top_k 8, hidden 512), with gradients agreeing to 1.6e-3 —
    inside the bf16 noise floor of ~6.5e-3. Batching the *backward's* decode
    loop instead cost 4.4x peak memory, because that one has to materialize
    the weight stack and this one does not. (The step times read beside these
    were on the A2000, a correctness-only testbed, and are not quoted.)

    Padding is the one hazard. Group sizes come from the router, so a hot expert
    makes ``max(sizes)`` large and the padded block ``G * max(sizes)`` rows wide
    regardless of how few rows are real. Until 0.32.1 the rule sent any call past
    a 4x flop-waste ratio to the loop; experts4bit-qlora's P46 (RESULTS-p46.md)
    measured that guard choosing the loop for >= 85 % of calls at the field recipe
    and costing 20 s of a 24.5 s step -- the loop is launch-bound and the wasted
    flops are rank-r matmuls. The `auto` rule is now structural: pad unless the
    padded block would exceed ``_PAD_BYTES_LIMIT`` bytes (``NF4_QLORA_PAD_BYTES_LIMIT``),
    where the loop is what fits. ``NF4_QLORA_PAD_WASTE_LIMIT``, when set, re-arms the
    flop-waste guard on top. Either way the route changes, never the result.

    ``NF4_QLORA_PAD_BUCKETS=1`` (``0`` is the single block, op for op; the default
    is ``auto``, below) pads the lean path by buckets instead: the groups sorted by rows and cut
    greedily so that no bucket's widest group has more than twice its narrowest's
    rows, each bucket padded to its own widest (``_pad_buckets``,
    ``_lora_delta_bucketed``). The padded rows drop from ``G * max(rows)`` to at
    most twice the real rows: at 4,096 tokens, top-8 of 128 experts with a
    Zipf(1) router (the test's seeded case), from 13.5x the real rows to 1.35x,
    in six buckets. Values equal the single block's to rounding (the ``bmm``
    shapes differ). It applies where the single block would pad (the `auto` rule
    still sizes the single block), not to ``NF4_QLORA_LEAN_DELTA=0``'s body; with
    ``NF4_QLORA_COMPACT_DELTA=1`` also set, buckets win.
    ``LORA_PATH_STATS["padded_bucketed"]`` counts it. ``NF4_QLORA_PAD_BUCKETS=auto``
    buckets only a call carrying at least ``_PAD_BUCKETS_AUTO_MIN_ROWS`` (16,384)
    routed rows (``NF4_QLORA_PAD_BUCKETS_MIN_ROWS`` overrides it) and gives every
    smaller call the single block, op for op.
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
    path = _lora_path()
    waste = len(rows) * widest / total
    pad_bytes = len(rows) * widest * (a_cat.shape[1] + lora_B.shape[1]) * a_cat.element_size()
    LORA_PAD_WASTE["last"], LORA_PAD_WASTE["last_bytes"] = waste, pad_bytes
    # The padded block is allocated in the ADAPTER dtype (fp32 on a matched-init arm), so the bytes it really takes can be twice
    # `pad_bytes`, which sizes it at the activations' itemsize. Recorded, not acted on: the route rule is unchanged until a full
    # step reads what accounting it should use.
    LORA_PAD_WASTE["last_bytes_alloc"] = len(rows) * widest * (a_cat.shape[1] + lora_B.shape[1]) * max(
        a_cat.element_size(), lora_A.element_size())
    LORA_PAD_WASTE["max"] = max(LORA_PAD_WASTE["max"], waste)
    if path == "auto":
        wl = _pad_waste_limit()
        if pad_bytes > _pad_bytes_limit() or (wl is not None and waste > wl):
            path = "loop"
    if path == "loop":
        LORA_PATH_STATS["loop"] += 1
        return _lora_delta_grouped_loop(a_cat, lora_A, lora_B, sizes,
                                        expert_ids, scaling)
    if path == "grouped_mm":
        LORA_PATH_STATS["grouped_mm"] += 1
        return _lora_delta_grouped_mm(a_cat, lora_A, lora_B, rows, nz, expert_ids, scaling)
    if _pad_buckets_enabled(total) and _lean_delta_enabled():   # NF4_QLORA_PAD_BUCKETS=1, or auto past its row gate: padded per bucket
        LORA_PATH_STATS["padded_bucketed"] += 1
        return _lora_delta_grouped_bucketed(a_cat, lora_A, lora_B, rows, nz, expert_ids, widest, scaling)
    LORA_PATH_STATS["padded"] += 1

    dev = a_cat.device
    from nf4_grouped import to_device_i32, _host_reuse_enabled, _stream_key, HOST_REUSE_STATS
    host_ids = None if (torch.is_tensor(expert_ids) and expert_ids.is_cuda) else [int(expert_ids[g]) for g in nz]
    # GNF4_HOST_REUSE (on by default; =0 turns it off): the gate_up and down deltas of one MoE layer pass share their grouping, so the
    # down call reuses the gate_up call's device plan (`eid`, `flat`) instead of re-uploading the ids and
    # rebuilding the flat index -- about ten launches and one transfer per pass, values identical (the key is the
    # host grouping itself, plus device and stream). Host ids only (a device `expert_ids` would need a read to
    # key on), never under capture, lean path only.
    pkey = None
    if (host_ids is not None and _lean_delta_enabled() and dev.type == "cuda" and _host_reuse_enabled()
            and not torch.cuda.is_current_stream_capturing()):
        pkey = (tuple(rows), tuple(host_ids), _stream_key(dev))
        plan = _PLAN_MEMO.get(pkey)
        if plan is not None:
            HOST_REUSE_STATS["plan_hits"] += 1
            eid, flat, unique = plan
            return _lora_delta_padded(a_cat, lora_A, lora_B, eid, flat, len(rows), widest, unique, scaling)
    if host_ids is None:
        # Select the surviving groups ON DEVICE. One index_select, no round trip.
        sz_i32, nz_i = to_device_i32((rows, nz), dev)
        eid = expert_ids[nz_i.to(torch.int64)].to(torch.int64)
    else:
        # Host data: a list, or a CPU tensor (Bugbot, PR #85 — the old
        # per-element path accepted CPU tensors and indexing one with the CUDA
        # `nz_i` above raises). `int(expert_ids[g])` is host-only for both.
        sz_i32, eid_i32 = to_device_i32((rows, host_ids), dev)
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
    if not _lean_delta_enabled():           # the previous body, kept for the A/B
        grp = torch.repeat_interleave(torch.arange(len(rows), device=dev), sz,
                                      output_size=total)
        slot = torch.arange(total, device=dev) - (torch.cumsum(sz, 0) - sz)[grp]
        A, B = lora_A[eid], lora_B[eid]
        x = a_cat.new_zeros(len(rows), widest, a_cat.shape[1]).to(A.dtype)
        x[grp, slot] = a_cat.to(A.dtype)
        d = scaling * torch.bmm(torch.bmm(x, A.transpose(1, 2)), B.transpose(1, 2))
        out = torch.zeros(a_cat.shape[0], B.shape[1], dtype=d.dtype, device=dev)
        out[:total] = d[grp, slot]
        return out

    # One FLAT index (row -> padded row g * widest + slot) rather than the
    # (group, slot) pair: 1-D index_copy_ / index_select cost fewer launches than
    # two-index advanced indexing, whose backward sorted its indices first.
    G = len(rows)
    shift = torch.repeat_interleave(
        torch.arange(G, device=dev) * widest - (torch.cumsum(sz, 0) - sz), sz,
        output_size=total)
    flat = torch.arange(total, device=dev) + shift
    unique = False
    if pkey is not None:
        unique = len(set(host_ids)) == len(host_ids)
        _PLAN_MEMO.clear()                 # one entry: only the gate_up -> down twin repeats
        _PLAN_MEMO[pkey] = (eid, flat, unique)
        HOST_REUSE_STATS["plan_misses"] += 1
    return _lora_delta_padded(a_cat, lora_A, lora_B, eid, flat, G, widest, unique, scaling)


_PLAN_MEMO: dict = {}


def _lora_delta_padded(a_cat, lora_A, lora_B, eid, flat, G, widest, unique, scaling):
    """The lean padded delta given its device plan (``eid`` the [G] expert ids, ``flat`` row -> padded row).

    ``unique`` (host-known distinct ids, with GNF4_HOST_REUSE on) gathers the adapters through ``_GatherRows``
    instead of advanced indexing. Same forward values (both are row copies); the backward differs only in
    route: advanced indexing's backward is ``index_put_(accumulate=True)``, which SORTS its indices first (a
    radix sort plus index arithmetic, ~10 launches per adapter), while ``_GatherRows``' is a zero fill and
    one ``index_copy_`` -- no sort and no atomics (``index_select``'s own backward, ``index_add_``, adds
    atomically). With distinct ids every gradient row
    receives exactly one value on both routes, so the gradients are equal.
    """
    if _compact_delta_enabled():
        return _scaled(_CompactPaddedDelta.apply(a_cat, lora_A, lora_B, eid, flat, G, widest, unique), scaling)
    dev = a_cat.device
    if unique:
        A, B = _GatherRows.apply(lora_A, eid), _GatherRows.apply(lora_B, eid)
    else:
        A, B = lora_A[eid], lora_B[eid]                # [G, r, K], [G, N, r]
    x = torch.zeros(G * widest, a_cat.shape[1], dtype=A.dtype, device=dev)
    x.index_copy_(0, flat, a_cat.to(A.dtype))
    d = torch.bmm(torch.bmm(x.view(G, widest, -1), A.transpose(1, 2)),
                  B.transpose(1, 2))
    # Back into the caller's row order. Zero-size groups were dropped above, so
    # `flat` addresses exactly the real rows -- and `a_cat` has exactly `total`
    # of them (the copy above refuses anything else), so the gather IS the
    # output: no zero fill, no slice copy. `scaling` lands on the gathered
    # rows, not the padded block, and not at all when it is 1 -- both exact.
    return _scaled(_GatherRows.apply(d.view(G * widest, -1), flat), scaling)


def _lora_delta_grouped_bucketed(a_cat, lora_A, lora_B, rows, nz, expert_ids, widest, scaling):
    """``NF4_QLORA_PAD_BUCKETS=1``: the bucketed lean padded delta's device plan, then the delta.

    The single block's plan, built the same way -- the uploads batched in one ``to_device_i32`` and ``repeat_interleave``
    handed its ``output_size``, so nothing reads the device and the path is capture-safe exactly as the single block is --
    with two differences: ``eid`` is in bucket order (``_pad_buckets``' sort), and ``flat`` sends a caller row of group
    ``g`` to ``pstart[g] + slot`` in the concatenated bucket blocks rather than to ``g * widest + slot``.

    GNF4_HOST_REUSE: the plan memo is EXTENDED, not bypassed. A bucketed plan carries its bucket list and is stored under
    its own key (the single block's key plus a ``"buckets"`` tag, so neither path ever reads the other's plan), so the down
    call reuses the gate_up call's bucketed plan as the single block's does, under the same conditions (host ids, CUDA,
    not capturing) and in the same one-entry memo.
    """
    dev = a_cat.device
    from nf4_grouped import to_device_i32, _host_reuse_enabled, _stream_key, HOST_REUSE_STATS
    host_ids = None if (torch.is_tensor(expert_ids) and expert_ids.is_cuda) else [int(expert_ids[g]) for g in nz]
    single_rows = len(rows) * widest
    pkey = None
    if (host_ids is not None and dev.type == "cuda" and _host_reuse_enabled()
            and not torch.cuda.is_current_stream_capturing()):
        pkey = (tuple(rows), tuple(host_ids), _stream_key(dev), "buckets-ladder" if _pad_ladder_enabled() else "buckets")
        plan = _PLAN_MEMO.get(pkey)
        if plan is not None:
            HOST_REUSE_STATS["plan_hits"] += 1
            eid, flat, unique, buckets = plan
            _record_buckets(single_rows, buckets)
            return _lora_delta_bucketed(a_cat, lora_A, lora_B, eid, flat, buckets, unique, scaling)
    order, buckets, pstart = _pad_buckets(rows)
    _record_buckets(single_rows, buckets)
    if host_ids is None:
        # Device ids: select the surviving groups, in bucket order, ON DEVICE (one index_select, no round trip).
        sz_i32, nzb_i, ps_i32 = to_device_i32((rows, [nz[g] for g in order], pstart), dev)
        eid = expert_ids[nzb_i.to(torch.int64)].to(torch.int64)
    else:
        sz_i32, eid_i32, ps_i32 = to_device_i32((rows, [host_ids[g] for g in order], pstart), dev)
        eid = eid_i32.to(torch.int64)
    sz = sz_i32.to(torch.int64)
    total = sum(rows)
    shift = torch.repeat_interleave(ps_i32.to(torch.int64) - (torch.cumsum(sz, 0) - sz), sz, output_size=total)
    flat = torch.arange(total, device=dev) + shift
    unique = False
    if pkey is not None:
        unique = len(set(host_ids)) == len(host_ids)     # a permutation of the caller's ids: distinct iff they are
        _PLAN_MEMO.clear()
        _PLAN_MEMO[pkey] = (eid, flat, unique, buckets)
        HOST_REUSE_STATS["plan_misses"] += 1
    return _lora_delta_bucketed(a_cat, lora_A, lora_B, eid, flat, buckets, unique, scaling)


def _record_buckets(single_rows, buckets):
    """A bucketed call's padded rows beside the single block's, for a receipt (``LORA_PAD_WASTE``)."""
    LORA_PAD_WASTE["last_rows_single"] = single_rows
    LORA_PAD_WASTE["last_rows_bucketed"] = sum(_bucket_groups(b) * b[1] for b in buckets)
    LORA_PAD_WASTE["last_buckets"] = len(buckets)


def _lora_delta_bucketed(a_cat, lora_A, lora_B, eid, flat, buckets, unique, scaling):
    """The bucketed lean padded delta given its device plan (``eid`` the [G] expert ids in bucket order, ``flat`` caller
    row -> padded row, ``buckets`` the host ``(G_b, W_b)`` list).

    One zero-filled buffer of ``sum(G_b * W_b)`` rows in the adapters' dtype takes every real row in ONE ``index_copy_``,
    as the single block does, and is split into the buckets' ``[G_b * W_b, K]`` blocks: views, so the split copies
    nothing forward and its backward is one ``cat`` of the blocks' grads. The adapters are gathered once, in bucket order
    (``_GatherRows`` for host-known distinct ids, as on the single block), and split the same way. Each bucket runs the
    single block's two ``bmm``s at its own width ``W_b``; the buckets' outputs are concatenated (one bucket: no copy) and
    ONE gather through ``flat`` puts the real rows back in the caller's order (``_GatherRows``: the rows are unique, so its
    backward is a scatter). Only standard autograd ops, so the gradients take the single block's route, bucket by bucket.

    ``NF4_QLORA_COMPACT_DELTA`` is not consulted: with both flags set, buckets win (``_CompactPaddedDelta`` is a
    single-block body, and this path never calls it).

    Values: each real row gets the same arithmetic as on the single block (the same row times the same adapters, the
    padded rows zero either way), but the ``bmm``s run at other shapes, so a BLAS that picks its kernel, or splits a
    reduction, by shape can round differently. Equal to rounding, not promised bit for bit.
    """
    if _compact_buckets_enabled() and not any(len(b) > 2 for b in buckets):   # the compact node (default; =0 off; not the ladder's plans)
        COMPACT_BUCKETS_STATS["calls"] += 1
        return _scaled(_CompactBucketedDelta.apply(a_cat, lora_A, lora_B, eid, flat, tuple(buckets), unique), scaling)
    if unique:
        A, B = _GatherRows.apply(lora_A, eid), _GatherRows.apply(lora_B, eid)
    else:
        A, B = lora_A[eid], lora_B[eid]                # [G, r, K], [G, N, r], bucket order
    if any(len(b) > 2 for b in buckets):             # NF4_QLORA_PAD_BUCKETS_LADDER=1: laddered batch counts
        return _lora_delta_bucketed_ladder(a_cat, A, B, flat, buckets, scaling)
    x = torch.zeros(sum(g * w for g, w in buckets), a_cat.shape[1], dtype=A.dtype, device=a_cat.device)
    x.index_copy_(0, flat, a_cat.to(A.dtype))
    if len(buckets) == 1:
        ((g, w),) = buckets
        d = torch.bmm(torch.bmm(x.view(g, w, -1), A.transpose(1, 2)), B.transpose(1, 2)).view(g * w, -1)
    else:
        per = [g for g, _ in buckets]
        d = torch.cat([torch.bmm(torch.bmm(xb.view(g, w, -1), Ab.transpose(1, 2)), Bb.transpose(1, 2)).view(g * w, -1)
                       for xb, Ab, Bb, (g, w) in zip(x.split([g * w for g, w in buckets]), A.split(per), B.split(per),
                                                     buckets)])
    return _scaled(_GatherRows.apply(d, flat), scaling)


def _lora_delta_bucketed_ladder(a_cat, A, B, flat, buckets, scaling):
    """``_lora_delta_bucketed``'s body for a laddered plan (``(G_b, W_b, G_b')`` buckets): each bucket's gathered adapters are
    zero-padded from ``G_b`` to ``G_b'`` groups (``F.pad``: its backward drops the padded slots' gradient), so the two ``bmm``s run
    at ``[G_b', W_b, K]`` -- shapes on the ladder's rungs. The padded groups' rows are zero and ``flat`` never reads them."""
    import torch.nn.functional as F
    x = torch.zeros(sum(gp * w for _, w, gp in buckets), a_cat.shape[1], dtype=A.dtype, device=a_cat.device)
    x.index_copy_(0, flat, a_cat.to(A.dtype))
    per = [g for g, _, _ in buckets]
    outs = []
    for xb, Ab, Bb, (g, w, gp) in zip(x.split([gp * w for _, w, gp in buckets]), A.split(per), B.split(per), buckets):
        if gp > g:
            Ab, Bb = F.pad(Ab, (0, 0, 0, 0, 0, gp - g)), F.pad(Bb, (0, 0, 0, 0, 0, gp - g))
        outs.append(torch.bmm(torch.bmm(xb.view(gp, w, -1), Ab.transpose(1, 2)), Bb.transpose(1, 2)).view(gp * w, -1))
    d = outs[0] if len(outs) == 1 else torch.cat(outs)
    return _scaled(_GatherRows.apply(d, flat), scaling)


def _compact_buckets_enabled() -> bool:
    """On unless ``NF4_QLORA_COMPACT_BUCKETS=0``: the bucketed delta through :class:`_CompactBucketedDelta` (the same bytes,
    forward and gradients; a fraction of the memory held between a layer's forward and its backward). The default since
    experts4bit-qlora TC1 amendment 66 (Qwen3-30B-A3B, packed 4,096-token rows, one RTX 5090, torch 2.12): the matched arm's
    training-phase peak fell 0.654 GB and both arms stepped faster (0.972 matched, 0.977 shipped) with less device time per step
    (0.973 / 0.978), held-out unchanged. ``=0`` restores the autograd body exactly. Independent of ``NF4_QLORA_COMPACT_DELTA``,
    which governs the single block only; the single block and the ladder's plans never take this node."""
    return os.environ.get("NF4_QLORA_COMPACT_BUCKETS", "").strip() != "0"


class _CompactBucketedDelta(torch.autograd.Function):
    """``_lora_delta_bucketed``'s body as ONE autograd node: the same forward arithmetic and the same gradient bytes, with
    far less held between a layer's forward and its backward.

    The autograd body keeps, per projection, the zero-filled padded block ``x`` ``[sum(G_b * W_b), K]`` (each bucket's
    first ``bmm`` saves its slice) and the adapters'-dtype copy of ``a_cat`` (``index_copy_``'s backward keeps its
    source), and its forward builds every bucket's output before one ``cat``. On Qwen3-30B-A3B's packed 4,096-token rows
    with fp32 adapters, those sites (``nf4_qlora.py`` 672, 673 and 679) held about 1.70 GB at the training peak
    (experts4bit-qlora TC1 amendment 65). This node:
    - writes each bucket's two ``bmm``s into one preallocated output (``out=``, no list, no ``cat``) and returns the real
      rows by ``index_select``, as ``_GatherRows`` does;
    - saves only ``a_cat`` (an alias of the caller's tensor), the adapters, ``eid``, ``flat`` and the first ``bmm``s'
      ``[P, r]`` output;
    - in backward, scatters the incoming gradient as ``_GatherRows`` does, rebuilds ``x`` with the same fill and
      ``index_copy_``, and issues per bucket the calls ``BmmBackward0`` would on the same operand layouts, freeing each
      buffer at its last use.
    Every gradient is therefore the same bytes as the autograd body's (``kernel/test_compact_buckets.py``). The adapters'
    gradients are scattered as ``_GatherRows``'s backward (unique ids) or accumulated as advanced indexing's (repeated ids).
    """

    @staticmethod
    def forward(ctx, a_cat, lora_A, lora_B, eid, flat, buckets, unique):
        if unique:
            A, B = lora_A.index_select(0, eid), lora_B.index_select(0, eid)
        else:
            A, B = lora_A[eid], lora_B[eid]
        P = sum(g * w for g, w in buckets)
        R, N = A.shape[1], B.shape[1]
        x = torch.zeros(P, a_cat.shape[1], dtype=A.dtype, device=a_cat.device)
        x.index_copy_(0, flat, a_cat.to(A.dtype))
        h = torch.empty(P, R, dtype=A.dtype, device=a_cat.device)
        d = torch.empty(P, N, dtype=A.dtype, device=a_cat.device)
        so = sg = 0
        for g, w in buckets:
            hb = h[so:so + g * w].view(g, w, R)
            torch.bmm(x[so:so + g * w].view(g, w, -1), A[sg:sg + g].transpose(1, 2), out=hb)
            torch.bmm(hb, B[sg:sg + g].transpose(1, 2), out=d[so:so + g * w].view(g, w, N))
            so, sg = so + g * w, sg + g
        del x, A, B
        ctx.save_for_backward(a_cat, lora_A, lora_B, eid, flat, h)
        ctx.buckets, ctx.unique = buckets, unique
        return d.index_select(0, flat)

    @staticmethod
    def backward(ctx, g):
        a_cat, lora_A, lora_B, eid, flat, h = ctx.saved_tensors
        buckets, unique = ctx.buckets, ctx.unique
        if unique:
            A, B = lora_A.index_select(0, eid), lora_B.index_select(0, eid)
        else:
            A, B = lora_A[eid], lora_B[eid]
        P, R, N = h.shape[0], h.shape[1], B.shape[1]
        gd = g.new_zeros((P,) + tuple(g.shape[1:]))                       # _GatherRows' backward: a scatter
        gd.index_copy_(0, flat, g)
        gh = torch.empty_like(h)
        gBt = []
        so = sg = 0
        for gb, w in buckets:                                              # BmmBackward0 of d_b = h_b @ B_b^T
            gdb = gd[so:so + gb * w].view(gb, w, N)
            torch.bmm(gdb, B[sg:sg + gb], out=gh[so:so + gb * w].view(gb, w, R))
            gBt.append(h[so:so + gb * w].view(gb, w, R).transpose(1, 2).bmm(gdb))
            so, sg = so + gb * w, sg + gb
        del gd, B
        x = torch.zeros(P, a_cat.shape[1], dtype=A.dtype, device=a_cat.device)
        x.index_copy_(0, flat, a_cat.to(A.dtype))
        gx = torch.empty_like(x)
        gAt = []
        so = sg = 0
        for gb, w in buckets:                                              # BmmBackward0 of h_b = x_b @ A_b^T
            ghb = gh[so:so + gb * w].view(gb, w, R)
            torch.bmm(ghb, A[sg:sg + gb], out=gx[so:so + gb * w].view(gb, w, -1))
            gAt.append(x[so:so + gb * w].view(gb, w, -1).transpose(1, 2).bmm(ghb))
            so, sg = so + gb * w, sg + gb
        del x, gh, A
        grad_a = None
        if ctx.needs_input_grad[0]:
            grad_a = gx.index_select(0, flat)                              # index_copy_'s backward for its source
            del gx
            if grad_a.dtype != a_cat.dtype:
                grad_a = grad_a.to(a_cat.dtype)                            # the .to(A.dtype)'s backward
        else:
            del gx
        gA = gB = None
        if ctx.needs_input_grad[1]:
            gAt_all = gAt[0] if len(gAt) == 1 else torch.cat(gAt)          # the split's backward
            gA = torch.zeros_like(lora_A)
            if unique:
                gA.index_copy_(0, eid, gAt_all.transpose(1, 2))
            else:
                gA.index_put_((eid,), gAt_all.transpose(1, 2), accumulate=True)
            del gAt_all
        del gAt
        if ctx.needs_input_grad[2]:
            gBt_all = gBt[0] if len(gBt) == 1 else torch.cat(gBt)
            gB = torch.zeros_like(lora_B)
            if unique:
                gB.index_copy_(0, eid, gBt_all.transpose(1, 2))
            else:
                gB.index_put_((eid,), gBt_all.transpose(1, 2), accumulate=True)
        del gBt
        return grad_a, gA, gB, None, None, None, None


def _compact_delta_enabled() -> bool:
    """Off unless ``NF4_QLORA_COMPACT_DELTA=1``: the lean padded delta through ``_CompactPaddedDelta`` (same values, a
    fraction of the saved memory); opt-in until a within-box A/B decides the default."""
    return os.environ.get("NF4_QLORA_COMPACT_DELTA", "0").strip() == "1"


class _CompactPaddedDelta(torch.autograd.Function):
    """The lean padded LoRA delta as ONE autograd node that saves its INPUT, not its padded block.

    Under autograd the padded path saves, per projection, the zero-padded input block ``[G, widest, K]`` (the
    first bmm's operand) and the gathered adapters ``[G, r, K]`` / ``[G, N, r]``. With a hot expert the block is
    ``G * widest`` rows wide whatever the real row count: at Qwen3-30B-A3B's shape (380 tokens, top-8, fp32
    adapters) one MoE layer saved 229 MB, 85 MB of it the gate_up block alone. This node runs the same forward ops
    and saves ``a_cat`` (an alias of the caller's tensor), ``eid``, ``flat`` and the first bmm's ``[G, widest, r]``
    output; its backward re-gathers the adapters, rebuilds the block with the same zero fill and ``index_copy_``,
    and issues the same calls autograd's own backward would (``BmmBackward0``'s two products on the same operand
    layouts, ``index_copy_``'s ``index_select``, the gather's scatter, the adapters' scatter or sorted
    ``index_put_``), so every gradient is the same bytes. The rebuild costs one fill and one copy of the block in
    backward; the node replaces about ten autograd nodes per projection.
    """

    @staticmethod
    def forward(ctx, a_cat, lora_A, lora_B, eid, flat, G, widest, unique):
        if unique:
            A, B = lora_A.index_select(0, eid), lora_B.index_select(0, eid)
        else:
            A, B = lora_A[eid], lora_B[eid]
        x = torch.zeros(G * widest, a_cat.shape[1], dtype=A.dtype, device=a_cat.device)
        x.index_copy_(0, flat, a_cat.to(A.dtype))
        h = torch.bmm(x.view(G, widest, -1), A.transpose(1, 2))           # [G, widest, r]
        d = torch.bmm(h, B.transpose(1, 2))                                # [G, widest, N]
        ctx.save_for_backward(a_cat, lora_A, lora_B, eid, flat, h)
        ctx.G, ctx.widest, ctx.unique = G, widest, unique
        return d.view(G * widest, -1).index_select(0, flat)

    @staticmethod
    def backward(ctx, g):
        # Every intermediate is dropped at its last use, as autograd frees each node's buffers once that node has run. Held to
        # the return, the padded output grad `gd` [G, W, N] was still live while the block `x` and its grad `gx` (both
        # [G, W, K]) were built -- a larger transient than the autograd path's, which has freed `gd` before `gx` exists
        # (experts4bit-qlora TC1 amendment 36: training peak +0.23 GB with fp32 adapters). Only references move: every op,
        # operand layout, dtype and the order of the calls are unchanged, so every gradient is the same bytes.
        a_cat, lora_A, lora_B, eid, flat, h = ctx.saved_tensors
        G, W, unique = ctx.G, ctx.widest, ctx.unique
        if unique:
            A, B = lora_A.index_select(0, eid), lora_B.index_select(0, eid)
        else:
            A, B = lora_A[eid], lora_B[eid]
        gd = g.new_zeros((G * W,) + tuple(g.shape[1:]))                   # the gather's backward: a scatter
        gd.index_copy_(0, flat, g)                                         # (`g` itself is held by autograd until this returns)
        gd = gd.view(G, W, -1)
        gh = gd.bmm(B)                                                     # BmmBackward0 of d = h @ B^T
        del B
        gBt = h.transpose(1, 2).bmm(gd)
        del gd, h                                                          # before the block is rebuilt (`h` stays saved on ctx)
        x = torch.zeros(G * W, a_cat.shape[1], dtype=A.dtype, device=a_cat.device)
        x.index_copy_(0, flat, a_cat.to(A.dtype))
        gx = gh.bmm(A)                                                     # BmmBackward0 of h = x @ A^T
        del A
        gAt = x.view(G, W, -1).transpose(1, 2).bmm(gh)
        del x, gh
        grad_a = None
        if ctx.needs_input_grad[0]:
            grad_a = gx.view(G * W, -1).index_select(0, flat)              # index_copy_'s backward for its source
            del gx                                                         # before the cast's copy and the adapters' grads
            if grad_a.dtype != a_cat.dtype:
                grad_a = grad_a.to(a_cat.dtype)                            # the .to(A.dtype)'s backward
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


def _lora_delta_grouped_mm(a_cat, lora_A, lora_B, rows, nz, expert_ids, scaling=1.0):
    """The jagged path: two ``torch._grouped_mm`` calls over the expert-sorted rows with cumulative offsets -- no
    padding, no per-expert Python, no ``[G, widest, K]`` scratch. bf16 operands (the op's contract); the result is
    cast back to the adapter dtype so the caller's arithmetic is unchanged. Refuses, with the reason, where torch has
    no grouped-GEMM kernel for this device (the op raises), rather than falling back silently -- the fallback is a
    choice the caller makes by setting NF4_QLORA_LORA_PATH."""
    dev = a_cat.device
    if not a_cat.is_cuda:
        raise RuntimeError("NF4_QLORA_LORA_PATH=grouped_mm needs CUDA tensors (torch._grouped_mm has no CPU kernel for these shapes)")
    offs = torch.tensor(rows, device=dev, dtype=torch.int32).cumsum(0, dtype=torch.int32)
    total = int(sum(rows))
    if torch.is_tensor(expert_ids):
        eid = expert_ids[torch.tensor(nz, device=expert_ids.device, dtype=torch.int64)].to(torch.int64)
    else:
        eid = torch.tensor([int(expert_ids[g]) for g in nz], device=dev, dtype=torch.int64)
    A = lora_A[eid].to(torch.bfloat16)                     # [G, r, K]
    B = lora_B[eid].to(torch.bfloat16)                     # [G, N, r]
    x = a_cat[:total].to(torch.bfloat16).contiguous()
    try:
        h = torch._grouped_mm(x, A.transpose(1, 2).contiguous(), offs=offs)          # [total, r]
        d = torch._grouped_mm(h.to(torch.bfloat16).contiguous(), B.transpose(1, 2).contiguous(), offs=offs)   # [total, N]
    except (RuntimeError, NotImplementedError) as e:
        raise RuntimeError(f"NF4_QLORA_LORA_PATH=grouped_mm: torch._grouped_mm refused on this device/shape: {str(e)[:200]}") from e
    d = _scaled(d.to(lora_A.dtype), scaling)
    out = torch.zeros(a_cat.shape[0], B.shape[1], dtype=d.dtype, device=dev)
    out[:total] = d
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
        d = _scaled((x.to(A.dtype) @ A.T) @ B.T, scaling)   # [n, r] -> [n, N]
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
