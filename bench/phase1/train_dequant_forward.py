#!/usr/bin/env python3
"""Training axis vs the DEQUANT-ON-FORWARD pattern — the baseline practitioners
actually run.

The training axis has so far been measured against Unsloth's grouped GEMM. That
is a real comparator but it is not what people with this problem are running.
The pattern in the field — published by GenON at 743B (`mncai/nf4moe`,
`github.com/genonai/nf4moe`, `QuantizedNaiveMoe`), and re-derived by every
hand-rolled fused-MoE QLoRA path — is: **per routed expert, call
`bitsandbytes.functional.dequantize_4bit` on the packed weight inside the
forward, then `F.linear` on the result.** This leg measures that.

Arms, all fwd+bwd, CUDA-event timed, quantized base frozen, adapters trainable:

  G  `gnf4`            `fused_grouped_lora(..., dgrad_kernel=True)` — the
                       single-launch grouped NF4 GEMM over packed bytes.
  D  `dequant_forward` per-routed-expert `dequantize_4bit` + `F.linear`.
  U  `unsloth_4bit`    OPTIONAL, import-guarded, recorded skipped-with-reason
                       when absent. Present only so the two comparators can be
                       read off ONE instance; it does not supersede and is not
                       divided into the dequant_forward result.

Plus `D_routed`, a PROBE and not an arm: D with GenON's published routing
plumbing (one-hot hit mask, one `.tolist()` host sync, per-expert
`torch.where` + gather, `index_add_` accumulation) wrapped around the identical
compute, so what that plumbing costs is measured rather than assumed. It is
never the headline comparator — see `d_routed_arm`.

MEASUREMENT RULES (registered in PREREG-dequant-forward.md before any data):
  * every comparator is preceded by an ADJACENT re-time of G, and the cell
    opens with an adjacent G/G self-pair. Run 4's self-pair BRACKETED the cell
    (base first, twin last) and read ~4% on sub-0.2 ms cells purely from
    placement; that is the defect this ordering exists to remove.
  * any ratio inside the self-pair spread is NOT a measurement and the reducer
    marks it VOID.
  * iteration count is chosen per cell so each timed block runs >= --block-ms,
    so tiny cells are not read off a handful of launches.

WHAT NO ARM DOES: gradient checkpointing. A cell is one projection's forward
and backward, so checkpointing it would only insert a recompute of that same
projection; it is OFF for every arm, which is the "configured identically"
condition. Routing weights are not applied by any arm either — the cell ends at
the projection boundary, as it does on the decode axis.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as Fn

# Repo root from H2H_REPO/DQF_REPO or cwd, NEVER from __file__: this script is
# staged at /root/ while the repo lives at /root/gnf4, and a __file__-relative
# path resolves to a bench/phase1 that does not exist there.
_ROOT = Path(
    os.environ.get("DQF_REPO", os.environ.get("H2H_REPO", Path.cwd()))
).resolve()
if not (_ROOT / "bench" / "phase1" / "harness.py").exists():
    raise SystemExit(
        f"DQF_REPO/cwd={_ROOT} is not the repo root "
        f"(no bench/phase1/harness.py). Set DQF_REPO."
    )
sys.path.insert(0, str(_ROOT / "bench" / "phase1"))
sys.path.insert(0, str(_ROOT / "kernel"))
import harness as H  # noqa: E402
from nf4_qlora import fused_grouped_lora, lora_delta_grouped  # noqa: E402

RANK = 16
SCALING = 1.0  # identical in every arm; the arms differ in the base projection


# ------------------------------------------------------------------ timing
def _timed(fn, iters):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    ev0, ev1 = (torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True))
    ts = []
    for _ in range(iters):
        ev0.record()
        fn()
        ev1.record()
        torch.cuda.synchronize()
        ts.append(ev0.elapsed_time(ev1))
    return statistics.median(ts)


def _warm(fn, min_s: float):
    """Hold the GPU busy for min_s BEFORE any pilot or timed block.

    AMENDMENT 1. Run 1's consumer-card leg went VOID: 8 of 8 `decode_m8` cells
    failed the self-pair, and the timings inside them stepped DOWN across the
    cell (gemma down: 3.138, 3.135, 2.528, 1.579 ms). Every position-2 and
    position-3 cell on the same box was flat to ~0.3%. The cause is not the
    kernel and not the sample count: a stack build is a long CPU-bound stretch
    with the GPU idle, so the cell that runs immediately after it — always the
    smallest one — measures the card CLOCKING BACK UP. `_timed`'s 10 warm-up
    iterations are ~14 ms on a 1.4 ms cell, nowhere near enough to boost.

    The fix is wall-clock, not iteration-count, and it is applied identically to
    every arm, so it cannot favour one. No criterion, band or arm changes."""
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < min_s:
        fn()
    torch.cuda.synchronize()


def _pilot_iters(fn, block_ms: float, lo: int = 20, hi: int = 200) -> int:
    """Iterations so one timed block spans >= block_ms. Registered rule: the
    h2h self-pair failed on sub-0.2 ms cells read off too few launches."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    per_ms = (time.perf_counter() - t0) * 1000.0 / 5
    return int(min(hi, max(lo, round(block_ms / max(per_ms, 1e-3)))))


def _energy(fn, min_s: float = 1.2):
    """J per fwd+bwd step from NVML mean power over a >= min_s window. Returns
    (watts, joules_per_step, method, n_samples) or Nones when no sampler."""
    sampler = H.PowerSampler()
    torch.cuda.synchronize()
    sampler.start()
    t0 = time.monotonic()
    calls = 0
    while time.monotonic() - t0 < min_s:
        fn()
        calls += 1
    torch.cuda.synchronize()
    wall = time.monotonic() - t0
    watts, n = sampler.stop()
    if watts is None or calls == 0:
        return None, None, sampler.method, n
    return watts, watts * wall / calls, sampler.method, n


# ------------------------------------------------------------------- arms
class _DeqCount:
    """Positive-controllable call counter around the ONE dequant entry point.

    A patched-in counter that never fires looks exactly like a fast arm. The
    wiring gate asserts this count is >= the number of non-empty groups per
    forward, so an arm that silently reused a materialized weight cannot pass.
    Wrapped only during the gate, never during timing."""

    def __init__(self, stack):
        self.stack, self.n, self._orig = stack, 0, stack.dequant_bf16

    def __enter__(self):
        def counted(e):
            self.n += 1
            return self._orig(e)

        self.stack.dequant_bf16 = counted
        return self

    def __exit__(self, *a):
        # drop the instance attribute so the class method is live again
        self.stack.__dict__.pop("dequant_bf16", None)
        return False


def dequant_forward_arm(stack, a_cat, sizes, eids, lora_A, lora_B):
    """THE BASELINE ARM — the dequant-on-forward pattern as published.

    WHAT THIS IS. For each routed expert: `bitsandbytes.functional.
    dequantize_4bit` on that expert's packed NF4 weight, inside the forward,
    then `F.linear` on the dequantized weight. The dequant is constant w.r.t.
    the activations, so the projection stays differentiable in x and the frozen
    4-bit base is trainable-through. This is `QuantizedNaiveMoe._deq` +
    `F.linear(current_state, gate_up_w)` from `genonai/nf4moe`, at the same
    blocksize (64) and quant_type (nf4) the harness quantizes with.

    Structure reproduced from the published module rather than imagined:
      * dequant is per HIT expert, not over the whole stack;
      * one dequant call per expert per forward — the fixed per-step tax their
        writeup measures at ~2.5 s/step at 743B, and the reason they batch to a
        token budget (which is why this leg sweeps a token-budget axis);
      * the weight is materialized and dropped inside the loop body, so peak
        activation memory does not grow with expert count.

    WHAT THIS IS NOT. It is not Unsloth's kernel. It is not
    `torch._grouped_mm`. It is not a tuned grouped GEMM of any kind, and no
    result from this arm licenses a claim about one. It is also not the whole
    of `QuantizedNaiveMoe`: the routing plumbing that module carries — one-hot
    hit mask, `.tolist()` host sync, per-expert gather, `index_add_` scatter —
    is hoisted OUT of this arm and measured separately in `d_routed_arm`.

    That hoist is deliberate and it makes this baseline FASTER than what GenON
    published, which is the conservative direction. It is the same op boundary
    every other arm in this repo is measured at: the fused kernel takes
    pre-assembled rows per KERNEL_CONTRACT, and charging only this arm for
    routing plumbing would repeat exactly the asymmetry amendment 2 of the h2h
    leg existed to remove. Rows arrive group-sorted, so the per-expert gather
    degenerates to a contiguous slice.

    LoRA is the identical call the other arms make — `lora_delta_grouped` on
    the same `lora_A`/`lora_B`, same scaling, applied pre-activation. Both arms
    therefore pay that function's cost identically; it cancels in the
    difference and compresses the ratio toward 1."""

    def run():
        outs = []
        row = 0
        for g, e in enumerate(eids.tolist()):
            n = int(sizes[g])
            if n == 0:
                continue
            x = a_cat[row:row + n]
            w = stack.dequant_bf16(e)          # bnb dequantize_4bit, in-forward
            outs.append(Fn.linear(x, w))
            row += n
        out = torch.cat(outs)
        delta = lora_delta_grouped(a_cat, lora_A, lora_B, sizes, eids, SCALING)
        if delta is not None:
            out = out + delta.to(out.dtype)
        return out

    return run


def d_routed_arm(stack, a_tok, order, inv, sizes, eids, lora_A, lora_B, a_cat,
                 num_experts):
    """PROBE, not an arm: `dequant_forward` wearing GenON's published routing
    plumbing, so that plumbing's cost is measured instead of assumed.

    Reproduces `QuantizedNaiveMoe.forward` verbatim in shape: `zeros_like`
    accumulator, the one-hot expert mask built under `no_grad`, ONE `.tolist()`
    host sync for the hit list (their audit-#18 fix, which replaced thousands
    of per-expert GPU->CPU stalls), `torch.where` per hit expert, the gather
    `hidden_states[token_idx]`, and `index_add_` accumulation.

    Rows are placed in a fixed pseudo-random TOKEN order, because that is the
    order a real router produces: on the harness's group-sorted rows a verbatim
    `torch.where` would return a contiguous range and the gather would cost
    nothing, which would understate the pattern. The un-permute back to group
    order is harness bookkeeping and is charged to the probe — it inflates the
    probe, and the probe is not the headline.

    It carries the identical batched LoRA the arms carry, so `D_routed - D` is
    plumbing ALONE. GenON's module would additionally pay a per-expert loop
    LoRA if adapters were attached to the experts (their "per-expert A/B on the
    3D stacks"); this probe is not charged for that."""

    def run():
        final = torch.zeros(a_tok.shape[0], stack.spec.N,
                            device=a_tok.device, dtype=a_tok.dtype)
        with torch.no_grad():
            expert_mask = Fn.one_hot(order, num_classes=num_experts).t()  # [E,T]
            expert_hit = expert_mask.sum(dim=-1).gt(0).nonzero().flatten().tolist()
        for e in expert_hit:
            token_idx = torch.where(expert_mask[e])[0]
            x = a_tok[token_idx]
            w = stack.dequant_bf16(e)
            h = Fn.linear(x, w)
            final.index_add_(0, token_idx, h.to(final.dtype))
        out = final[inv]
        delta = lora_delta_grouped(a_cat, lora_A, lora_B, sizes, eids, SCALING)
        if delta is not None:
            out = out + delta.to(out.dtype)
        return out

    return run


def fused_arm(stack, a_cat, sizes, eids, lora_A, lora_B, B_pack, A_scale,
              dgrad_kernel=True):
    """gnf4: one launch over the packed bytes, recompute-decode backward.
    `dgrad_kernel=True` is the performance path and, since PR #49, the shipped
    default; `False` is the deliberately EXACT per-expert reference and is not
    a performance path, so timing it here would repeat run 3's error."""

    def run():
        return fused_grouped_lora(a_cat, B_pack, A_scale, sizes, eids,
                                  lora_A=lora_A, lora_B=lora_B,
                                  scaling=SCALING, dgrad_kernel=dgrad_kernel)

    return run


def unsloth_arm(stack, a_cat, sizes, eids, lora_A, lora_B, groups):
    """Unsloth's own kernel in the 4-bit-storage regime, dequant timed in —
    the SAME comparator the h2h training leg measured, carried here only so
    both comparators can be read off one instance. Reported in its own column;
    never divided into or conflated with the dequant_forward result."""

    def run():
        w_stack = torch.stack([stack.dequant_bf16(e) for e, _ in groups])
        out = H._unsloth_native_call(a_cat, w_stack, [int(s) for s in sizes])
        delta = lora_delta_grouped(a_cat, lora_A, lora_B, sizes, eids, SCALING)
        if delta is not None:
            out = out + delta.to(out.dtype)
        return out

    return run


# ------------------------------------------------------------------- cell
def _fidelity_sub(stack, groups, out, sizes, max_rows, max_groups):
    """`H.fidelity` restricted to a recorded subsample.

    Identical arithmetic — relative Frobenius error against `a_fp64 @
    dequant_ref_fp64(W).T`, the TOLERANCE_CONTRACT reference — but over at most
    `max_groups` groups and `max_rows` rows each. The full form is an fp64 GEMM
    per group and costs ~1.5 TFLOP fp64 on the largest token-budget cell, which
    is tens of minutes of rented card on an Ada part. Every expert is drawn
    from the same synthetic distribution, so the leading groups are
    representative; the caps are recorded per cell rather than assumed."""
    num = den = 0.0
    row = 0
    used = 0
    for g, (e, a) in enumerate(groups):
        n = int(sizes[g])
        if n == 0:
            continue
        if used < max_groups:
            m = min(n, max_rows)
            r = a[:m].to(torch.float64) @ stack.ref64(e).t()
            num += (out[row:row + m].to(torch.float64) - r).norm().item() ** 2
            den += r.norm().item() ** 2
            used += 1
        row += n
    return (num**0.5) / (den**0.5) if den else float("nan")


def cell(spec, regime, device, args, stack):
    groups = H.make_activations(spec, regime, device)
    sizes = [a.shape[0] for _, a in groups]
    eids = torch.tensor([e for e, _ in groups], dtype=torch.int32, device=device)
    a_cat = torch.cat([a for _, a in groups]).detach().requires_grad_(True)
    B_pack, A_scale = stack.fusedpack()
    T = a_cat.shape[0]

    row = {"model": spec.model, "proj": spec.proj, "regime": regime,
           "N": spec.N, "K": spec.K, "E": spec.E, "top_k": spec.top_k,
           "groups": len(groups), "rows": T, "rank": RANK,
           # C5: the fixture's routing behaviour, in EVERY receipt. These
           # regimes are constructed, not measured — `decode_m*` builds exactly
           # top_k groups where real OLMoE routing at T=32 hits ~58 of 64 — so
           # the occupancy and cv here describe a fiction and say which one.
           "routing": H.routing_summary(sizes, spec.E, fixture=regime)}

    # LoRA sized by the FULL expert count (lora_delta_grouped indexes by expert
    # id, not by group index) — [E,r,K] / [E,N,r], lora_B zero-init as standard.
    lora_A = (torch.randn(spec.E, RANK, spec.K, device=device,
                          dtype=torch.bfloat16) * 0.01).requires_grad_(True)
    lora_B = torch.zeros(spec.E, spec.N, RANK, device=device,
                         dtype=torch.bfloat16).requires_grad_(True)

    # Token-order fixture for the routed probe. Built once, outside timing —
    # fixture assembly is untimed for every arm in this repo.
    grp_of_row = torch.repeat_interleave(
        torch.arange(len(groups), device=device),
        torch.tensor(sizes, device=device))
    perm = torch.randperm(T, generator=torch.Generator().manual_seed(11)).to(device)
    order = eids.to(torch.int64)[grp_of_row][perm]   # token j -> its EXPERT id
    inv = torch.argsort(perm)
    a_tok = a_cat.detach()[perm].requires_grad_(True)

    def _zero():
        for t in (a_cat, a_tok, lora_A, lora_B):
            t.grad = None

    fwd = {
        "G": fused_arm(stack, a_cat, sizes, eids, lora_A, lora_B, B_pack, A_scale),
        "D": dequant_forward_arm(stack, a_cat, sizes, eids, lora_A, lora_B),
        "D_routed": d_routed_arm(stack, a_tok, order, inv, sizes, eids,
                                 lora_A, lora_B, a_cat, spec.E),
    }
    if args.unsloth:
        fwd["U"] = unsloth_arm(stack, a_cat, sizes, eids, lora_A, lora_B, groups)

    step = {}
    for k, f in fwd.items():
        def mk(f=f):
            def run():
                _zero()
                f().float().pow(2).mean().backward()
            return run
        step[k] = mk()

    try:
        # ---- wiring gate, before any timing --------------------------------
        gate = {}
        with torch.no_grad():
            og = fwd["G"]().detach()
            od = fwd["D"]().detach()
            orr = fwd["D_routed"]().detach()
        nz = sum(1 for s in sizes if s > 0)
        with _DeqCount(stack) as c:
            with torch.no_grad():
                fwd["D"]()
        gate["deq_calls_D"] = c.n
        gate["nonempty_groups"] = nz
        gate["deq_calls_ok"] = bool(c.n >= nz)
        # D_routed must reproduce D's values: same per-expert F.linear, disjoint
        # index_add_. A mismatch means the plumbing dropped or double-counted an
        # expert, which is the failure mode a timing-only probe would hide.
        # NOT bitwise: the gather hands the GEMM the same rows in a different
        # order, so blocking differs and fp32 lands ~1e-8 apart (measured). The
        # gate is per-ROW and scale-relative, because a dropped expert leaves
        # its rows wrong by O(1), not by a rounding step.
        gate["routed_max_rel"] = float(
            (orr - od).abs().max() / od.abs().max().clamp_min(1e-30))
        gate["routed_rows_differing"] = int(
            ((orr - od).abs().max(dim=1).values
             > 1e-2 * od.abs().max()).sum())
        gate["routed_matches_D"] = bool(gate["routed_rows_differing"] == 0)
        # Gradients present and finite in every arm (a dead arm is a fast arm).
        #
        # lora_A's gradient is EXACTLY ZERO here and that is correct, not a
        # wiring failure: the delta is B(Ax) with B zero-initialised, so
        # dL/dA = B^T(...) = 0 at LoRA init. Requiring it non-zero would void
        # every cell for a mathematical fact. Zero-init is kept because it is
        # standard LoRA, because it matches the arm run 4 was measured with,
        # and because it makes the delta exactly zero so b_rel measures the
        # base projection and nothing else. The A path is instead positive-
        # controlled below, at a non-zero B, and lora_B is restored after.
        gate["lora_A_grad_expected_zero_at_init"] = True
        for k in step:
            _zero()
            step[k]()
            gs = {"a": a_tok.grad if k == "D_routed" else a_cat.grad,
                  "lora_A": lora_A.grad, "lora_B": lora_B.grad}
            gate[f"grad_{k}"] = {
                n: (None if g is None else
                    {"finite": bool(torch.isfinite(g).all()),
                     "nonzero": bool(g.abs().sum() > 0)})
                for n, g in gs.items()}
        _zero()

        # Positive control on the A path: with B non-zero, dL/dA must be
        # non-zero in every arm. Without this the lora_A column above is
        # unfalsifiable and an arm that never wired A in would look identical.
        with torch.no_grad():
            lora_B.copy_(torch.randn_like(lora_B) * 0.01)
        for k in step:
            _zero()
            step[k]()
            g = lora_A.grad
            gate[f"gradA_at_nonzero_B_{k}"] = {
                "present": g is not None,
                "finite": bool(g is not None and torch.isfinite(g).all()),
                "nonzero": bool(g is not None and g.abs().sum() > 0)}
        with torch.no_grad():
            lora_B.zero_()
        _zero()
        row["gate"] = gate

        # ---- fidelity (B4): b_rel vs the fp64 exact GEMM --------------------
        # lora_B is zero-init so the delta is exactly zero here; the fidelity
        # number is the base projection's, which is what differs between arms.
        row["fid_rows_per_group"] = args.fid_rows
        row["fid_max_groups"] = args.fid_groups
        row["b_rel_G"] = _fidelity_sub(stack, groups, og, sizes,
                                       args.fid_rows, args.fid_groups)
        row["b_rel_D"] = _fidelity_sub(stack, groups, od, sizes,
                                       args.fid_rows, args.fid_groups)
        # Reported as FUSED-over-BASELINE, the orientation every fidelity claim
        # in this repo already uses ("fused <= 0.755x the dequant path's
        # error"). Below 1.0 means the fused kernel is the more accurate one.
        row["b_rel_G_over_D"] = (row["b_rel_G"] / row["b_rel_D"]
                                 if row["b_rel_D"] else None)
        del og, od, orr
        torch.cuda.empty_cache()

        if args.gate_only:
            # Correctness-only mode. Produces the wiring gate and b_rel and
            # NOTHING ELSE — no timing, no memory, no energy — so it can run on
            # the correctness-only home testbed without consuming the
            # registration window for any criterion that is about cost.
            row["status"] = "gate_only"
            stack.__dict__.pop("_unsloth_resident", None)
            torch.cuda.empty_cache()
            return row

        # ---- peak memory per step (M-axis of the real trade) ---------------
        # This is where the two arms genuinely differ and a speed-only table
        # would hide it. `F.linear` saves its WEIGHT for backward, so the
        # dequant-on-forward arm holds every hit expert's materialized bf16
        # weight across the forward->backward window; gnf4's autograd Function
        # re-decodes one expert at a time in backward and stores none. So the
        # dequant arm can buy speed with memory, and at a token budget where
        # the materialized set no longer fits it cannot buy it at all.
        mem = {}
        for k in step:
            _zero()
            torch.cuda.empty_cache()
            base = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            step[k]()
            mem[k] = {"peak_bytes": int(torch.cuda.max_memory_allocated()),
                      "resident_before_bytes": int(base)}
            mem[k]["transient_bytes"] = mem[k]["peak_bytes"] - base
        _zero()
        torch.cuda.empty_cache()
        if mem["G"]["transient_bytes"] > 0:
            row["mem_transient_d_over_g"] = (
                mem["D"]["transient_bytes"] / mem["G"]["transient_bytes"])
        row["mem"] = mem

        # ---- timing --------------------------------------------------------
        # AMENDMENT 1: clocks up before the pilot, not just before the block.
        _warm(step["G"], args.warm_s)
        row["warm_s"] = args.warm_s
        iters = _pilot_iters(step["G"], args.block_ms)
        row["iters"] = iters

        t = {}
        t["g_a"] = _timed(step["G"], iters)
        t["g_b"] = _timed(step["G"], iters)          # ADJACENT self-pair
        row["g_selfpair"] = t["g_b"] / t["g_a"]

        t["d_a"] = _timed(step["D"], iters)          # adjacent to g_b
        row["d_over_g"] = t["d_a"] / t["g_b"]
        t["d_b"] = _timed(step["D"], iters)          # ADJACENT self-pair
        row["d_selfpair"] = t["d_b"] / t["d_a"]

        t["g_c"] = _timed(step["G"], iters)
        t["dr_a"] = _timed(step["D_routed"], iters)
        row["dr_over_g"] = t["dr_a"] / t["g_c"]
        row["dr_over_d"] = t["dr_a"] / t["d_b"]

        if "U" in step:
            t["g_d"] = _timed(step["G"], iters)
            t["u_a"] = _timed(step["U"], iters)
            row["u_over_g"] = t["u_a"] / t["g_d"]

        t["g_end"] = _timed(step["G"], iters)
        row["g_drift"] = t["g_end"] / t["g_a"]        # whole-cell drift, reported

        # SHARED FLOOR, report-only. Both arms call the identical
        # `lora_delta_grouped` with identical arguments, and that function
        # iterates `expert_ids` in python — on a CUDA tensor that is one
        # device sync per group. Whatever it costs is added to BOTH arms and
        # therefore COMPRESSES every ratio toward 1.0. On the smallest cells it
        # can be most of the measured time, so it is measured rather than left
        # for a reader to wonder about. It is not subtracted from anything:
        # arithmetic on ratios after the fact is how instruments get talked
        # into saying what you wanted.
        def lora_only():
            _zero()
            d = lora_delta_grouped(a_cat, lora_A, lora_B, sizes, eids, SCALING)
            d.float().pow(2).mean().backward()
        t["lora_floor"] = _timed(lora_only, iters)
        row["lora_floor_frac_of_g"] = t["lora_floor"] / t["g_a"]
        row["ms"] = t

        # ---- GPU-busy fraction, beside the self-pair (C4) -------------------
        # Runs in EVERY leg now, not as an after-the-fact probe. The self-pair
        # says whether the box drifted; this says whether the ratio above is a
        # statement about kernels at all. Legs 2 and 3 were graded before this
        # existed and had their primary criterion demoted afterwards; that is
        # the mistake this instrument is here to stop repeating.
        if not args.no_busy:
            busy = {k: H.gpu_busy_fraction(step[k], args.busy_steps)
                    for k in ("G", "D")}
            row["gpu_busy"] = busy
            cls, lo, _fr = H.measurement_class(busy)
            row["measurement_class"] = cls        # "kernel" | "step_ratio"
            row["min_busy_fraction"] = lo
            row["measurement_class_note"] = H.MEASUREMENT_CLASS_NOTE

        # ---- energy per step (B2) ------------------------------------------
        if args.energy:
            en = {}
            for k in ("G", "D"):
                w, j, meth, n = _energy(step[k], args.energy_s)
                en[k] = {"watts": w, "j_per_step": j, "method": meth, "samples": n}
            if en["G"]["j_per_step"] and en["D"]["j_per_step"]:
                row["j_ratio_d_over_g"] = (
                    en["D"]["j_per_step"] / en["G"]["j_per_step"])
            row["energy"] = en

        row["status"] = "ok"
    except Exception as e:  # pragma: no cover - pod-side robustness
        row.update({"status": "skipped",
                    "reason": f"{type(e).__name__}: {str(e)[:220]}"})
    stack.__dict__.pop("_unsloth_resident", None)
    torch.cuda.empty_cache()
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--regimes", nargs="*",
                    default=["decode_m8", "tokbudget_2048", "tokbudget_11800"])
    ap.add_argument("--block-ms", type=float, default=250.0,
                    help="per-timed-block wall target; sets iters per cell")
    ap.add_argument("--warm-s", type=float, default=1.5,
                    help="AMENDMENT 1: seconds of GPU-busy warm-up before the "
                         "pilot and first timed block of every cell, so the "
                         "cell after a stack build does not measure clock "
                         "recovery. 0 reproduces run 1.")
    ap.add_argument("--busy-steps", type=int, default=50,
                    help="steps per arm for the GPU-busy fraction (C4). Runs by "
                         "default in every leg, beside the self-pair.")
    ap.add_argument("--no-busy", action="store_true",
                    help="skip the GPU-busy instrument. A cell without it is "
                         "labelled measurement_class=unknown, which is NOT the "
                         "same as 'kernel' and must not be printed as one.")
    ap.add_argument("--energy", action="store_true")
    ap.add_argument("--energy-s", type=float, default=1.5)
    ap.add_argument("--fid-rows", type=int, default=16,
                    help="rows per group used for the fp64 fidelity reference")
    ap.add_argument("--fid-groups", type=int, default=32,
                    help="groups used for the fp64 fidelity reference")
    ap.add_argument("--unsloth", action="store_true",
                    help="also time unsloth's kernel (import-guarded)")
    ap.add_argument("--gate-only", action="store_true",
                    help="wiring gate + fidelity only; no timing, memory or "
                         "energy. For the correctness-only home testbed.")
    ap.add_argument("--out", default=os.environ.get("DQF_OUT", "/root/dqf-out"))
    ap.add_argument("--tag", default="run1")
    args = ap.parse_args()

    if args.unsloth:
        try:
            H.unsloth_native_fingerprint()
        except Exception as e:
            print(f"unsloth unavailable, arm dropped: {type(e).__name__}: {e}")
            args.unsloth = False

    out = {
        "leg": "training axis vs dequant-on-forward",
        "TIER": "see PREREG-dequant-forward.md",
        "baseline_source": {
            "module": "QuantizedNaiveMoe",
            "repo": "https://github.com/genonai/nf4moe",
            "writeup": "https://huggingface.co/mncai/nf4moe",
            "quant": {"quant_type": "nf4", "blocksize": H.BLOCKSIZE},
        },
        "gpu": torch.cuda.get_device_name(0),
        "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
        "torch": torch.__version__,
        "rank": RANK, "scaling": SCALING,
        "grad_checkpointing": "OFF in every arm (single-projection cell)",
        "routing_weights": "not applied by any arm (projection boundary)",
        "regimes": args.regimes,
        "block_ms": args.block_ms,
        "rows": [],
    }
    try:
        import bitsandbytes as bnb
        out["bitsandbytes"] = bnb.__version__
    except Exception:
        pass
    if args.unsloth:
        out["unsloth"] = H.unsloth_native_fingerprint()

    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    artifact = dest / f"dequant_forward_{args.tag}.json"

    specs = H.census_specs(H.REPO / "census" / "shape_census.json", args.models)
    for spec in specs:
        # One stack per SPEC, shared by that spec's regimes: building it is a
        # per-expert CPU randn + quantize_4bit and dominates wall clock on the
        # E=128 shapes. Nothing in a cell mutates the packed bytes.
        stack = H.QuantStack(spec, "cuda")
        for regime in args.regimes:
            r = cell(spec, regime, "cuda", args, stack)
            out["rows"].append(r)
            print(json.dumps(r), flush=True)
            # Written after EVERY cell: a pod that dies mid-matrix must not
            # take the completed cells with it (teardown law 11).
            artifact.write_text(json.dumps(out, indent=1, default=str))
        del stack
        torch.cuda.empty_cache()

    ok = [r for r in out["rows"] if r.get("status") == "ok"]
    if args.gate_only:
        gated = [r for r in out["rows"] if r.get("status") == "gate_only"]
        bad = [r for r in gated if not (r["gate"]["deq_calls_ok"]
                                        and r["gate"]["routed_matches_D"])]
        print("GATE_ONLY cells=%d failures=%d  median b_rel G/D=%s" % (
            len(gated), len(bad),
            statistics.median([r["b_rel_G_over_D"] for r in gated
                               if r.get("b_rel_G_over_D")]) if gated else None))
        for r in bad:
            print("GATE_FAIL", r["model"], r["proj"], r["regime"], r["gate"])
    if ok:
        print("MEDIAN d/g=%.3f  dr/g=%.3f  g_self=%.3f  d_self=%.3f  cells=%d" % (
            statistics.median(r["d_over_g"] for r in ok),
            statistics.median(r["dr_over_g"] for r in ok),
            statistics.median(r["g_selfpair"] for r in ok),
            statistics.median(r["d_selfpair"] for r in ok),
            len(ok)))
        # The measurement class is printed with the medians, not buried in the
        # receipt, because it changes what the medians above MEAN.
        by = {}
        for r in ok:
            by[r.get("measurement_class", "unknown")] = \
                by.get(r.get("measurement_class", "unknown"), 0) + 1
        print("MEASUREMENT CLASS " + "  ".join(f"{k}={v}" for k, v in
                                               sorted(by.items())))
        sr = [r for r in ok if r.get("measurement_class") == "step_ratio"]
        if sr:
            print("  %d cell(s) are STEP RATIOS, not kernel measurements "
                  "(min busy fraction %.1f%%). Label them as such in any table:"
                  % (len(sr), 100 * min(r["min_busy_fraction"] for r in sr)))
            for r in sr:
                print("    %-28s %-8s %-16s d/g=%.3f  busy G=%.0f%% D=%.0f%%" % (
                    r["model"].split("/")[-1][:28], r["proj"], r["regime"],
                    r["d_over_g"],
                    100 * r["gpu_busy"]["G"]["busy_fraction"],
                    100 * r["gpu_busy"]["D"]["busy_fraction"]))
    print("DQF_DONE")


if __name__ == "__main__":
    main()
