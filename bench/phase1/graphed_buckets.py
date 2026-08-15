#!/usr/bin/env python3
"""DENSE-GROUPS bucketing: the fused training step as a replayable CUDA graph.

Grades `kernel/prereg_graphed_buckets.json` (OTS-stamped pre-data). The scheme,
verbatim from the registration:

  G_pad = E groups, every group padded to the same row count R (bucket set
  {1,2,4,8,16,32}, R = next_pow2(max real group size)), eids = arange(E) device
  tensor, sizes = [R]*E. Tile tables and launch grids are then step-invariant.

  Per step, OUTSIDE the graph: the host computes routing and writes the token
  activations and the token->slot index into persistent device buffers. The
  GRAPH: zero the padded activation buffer, scatter real rows to their slots,
  forward, loss over the real rows (their count T*top_k is constant), backward.

  The zero-fill is load-bearing: parameter gradients accumulate over rows, and
  a zeroed padding row contributes exactly zero to both LoRA grads (x=0 kills
  the A-term; Ax=0 kills the B-term). Real rows occupy each group's slot
  PREFIX, so per-group reductions see the same nonzero terms in the same order
  as the unbucketed path, followed by exact +0.0 terms — which is why the
  fidelity gate can demand bitwise equality rather than a tolerance.

Fidelity (F2, bitwise, no tolerance): bucketed-eager == shipped-unbucketed on
real-row outputs, a-grads, and (with LoRA) lora_A/lora_B grads; replayed graph
== bucketed-eager. If F2 fails, the race does not run.

Race (F3/F4): BOTH arms bucketed under the same scheme, both replayed. The
baseline arm is the same per-expert dequant+F.linear loop the graphed-race
probe captured, over the E static R-row slices.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as Fn

_ROOT = Path(os.environ.get("DQF_REPO", Path.cwd())).resolve()
if not (_ROOT / "bench" / "phase1" / "harness.py").exists():
    raise SystemExit(f"DQF_REPO/cwd={_ROOT} is not the repo root. Set DQF_REPO.")
sys.path.insert(0, str(_ROOT / "bench" / "phase1"))
sys.path.insert(0, str(_ROOT / "kernel"))
import harness as H  # noqa: E402
import routing_fixture as rf  # noqa: E402
from nf4_qlora import fused_grouped_lora  # noqa: E402

RESULTS = _ROOT / "bench" / "phase1" / "results"
BUCKETS = (1, 2, 4, 8, 16, 32)
RANK, SCALING = 16, 2.0


def pick_bucket(max_size: int) -> int:
    for r in BUCKETS:
        if max_size <= r:
            return r
    raise ValueError(f"group size {max_size} exceeds the registered bucket set "
                     f"{BUCKETS}; re-register before running this model/regime")


def slot_indices(groups, E: int, R: int, device):
    """Token(arrival order = group-major) -> padded slot. Group for expert e
    occupies slots [e*R, e*R + n_e); real rows fill the PREFIX of each group."""
    idx = []
    for e, a in groups:
        n = a.shape[0]
        base = int(e) * R
        idx.extend(range(base, base + n))
    return torch.tensor(idx, dtype=torch.int64, device=device)


class DenseGroupsStep:
    """One bucketed step (fused or baseline arm) with persistent buffers, an
    eager callable, and a capture/replay pair."""

    def __init__(self, arm, stack, spec, T_real: int, R: int, lora: bool,
                 device="cuda"):
        self.arm, self.spec, self.R, self.lora = arm, spec, R, lora
        E, K, N = spec.E, spec.K, spec.N
        self.stack = stack
        self.B_pack, self.A_scale = stack.fusedpack()
        self.sizes = [R] * E                      # static host list
        self.eids = torch.arange(E, dtype=torch.int32, device=device)
        self.tok = torch.zeros(T_real, K, dtype=torch.bfloat16, device=device)
        self.idx = torch.zeros(T_real, dtype=torch.int64, device=device)
        self.a_pad = torch.zeros(E * R, K, dtype=torch.bfloat16, device=device,
                                 requires_grad=True)
        if lora:
            g = torch.Generator(device="cpu").manual_seed(3)
            self.lora_A = (torch.randn(E, RANK, K, generator=g) * 0.01).to(
                device, torch.bfloat16).requires_grad_(True)
            self.lora_B = (torch.randn(E, N, RANK, generator=g) * 0.01).to(
                device, torch.bfloat16).requires_grad_(True)
        self.graph = None

    # -- the captured region ------------------------------------------------
    def _body(self):
        for t in (self.a_pad,) + ((self.lora_A, self.lora_B) if self.lora else ()):
            if t.grad is not None:
                t.grad.zero_()
        with torch.no_grad():
            self.a_pad.zero_()                       # load-bearing (prereg)
            self.a_pad.index_copy_(0, self.idx, self.tok)
        if self.arm == "fused":
            out = fused_grouped_lora(
                self.a_pad, self.B_pack, self.A_scale, self.sizes, self.eids,
                lora_A=self.lora_A if self.lora else None,
                lora_B=self.lora_B if self.lora else None,
                scaling=SCALING, dgrad_kernel=True)
        else:                                        # baseline: E static slices
            outs = []
            for e in range(self.spec.E):
                sl = self.a_pad[e * self.R:(e + 1) * self.R]
                o = Fn.linear(sl, self.stack.dequant_bf16(e))
                if self.lora:
                    o = o + SCALING * ((sl.to(self.lora_A.dtype) @ self.lora_A[e].T)
                                       @ self.lora_B[e].T)
                outs.append(o)
            out = torch.cat(outs)
        real = out.index_select(0, self.idx)
        real.float().pow(2).mean().backward()
        return real

    def eager(self):
        return self._body()

    def capture(self):
        self._body()
        torch.cuda.synchronize()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._body()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self._real_out = self._body()
        torch.cuda.synchronize()
        self.graph = g
        return g

    # -- per-step update (OUTSIDE the graph), then replay -------------------
    def load(self, groups):
        a_cat = torch.cat([a for _, a in groups])
        with torch.no_grad():
            self.tok.copy_(a_cat)
            self.idx.copy_(slot_indices(groups, self.spec.E, self.R, self.idx.device))

    def replay(self):
        self.graph.replay()


def _forward_with_padding_fill(step, junk_fill: bool):
    """Forward only, padding slots filled with junk instead of zeros — the
    amendment's leak test. Real rows are written AFTER the fill, exactly as the
    graph body orders zero-then-scatter."""
    with torch.no_grad():
        if junk_fill:
            step.a_pad.copy_(torch.randn_like(step.a_pad))
        else:
            step.a_pad.zero_()
        step.a_pad.index_copy_(0, step.idx, step.tok)
    out = fused_grouped_lora(step.a_pad, step.B_pack, step.A_scale, step.sizes,
                             step.eids, lora_A=step.lora_A, lora_B=step.lora_B,
                             scaling=SCALING, dgrad_kernel=True)
    return out.index_select(0, step.idx)


def _grads_with_garbage_padding(step, groups, fill="junk"):
    """Positive control (amendment 2): NaN in the padding must POISON the LoRA
    grads (0 * Inf = NaN through the row reductions) — the channel zero-fill
    actually closes. Ordinary garbage provably cannot contaminate them: the
    loss covers real rows only, so grad_out is exactly zero on padded rows."""
    for t in (step.a_pad, step.lora_A, step.lora_B):
        if t.grad is not None:
            t.grad.zero_()
    with torch.no_grad():
        if fill == "nan":
            step.a_pad.fill_(float("nan"))
        else:
            step.a_pad.copy_(torch.randn_like(step.a_pad))
        step.a_pad.index_copy_(0, step.idx, step.tok)
    out = fused_grouped_lora(step.a_pad, step.B_pack, step.A_scale, step.sizes,
                             step.eids, lora_A=step.lora_A, lora_B=step.lora_B,
                             scaling=SCALING, dgrad_kernel=True)
    out.index_select(0, step.idx).float().pow(2).mean().backward()
    return step.lora_A.grad.detach().clone()


def unbucketed_reference(stack, groups, lora, device="cuda"):
    """The shipped path on the same inputs, group-major order. Returns
    (real-row outputs, a_cat grad, loraA grad, loraB grad)."""
    spec = stack.spec
    sizes = [a.shape[0] for _, a in groups]
    eids = torch.tensor([e for e, _ in groups], dtype=torch.int32, device=device)
    a_cat = torch.cat([a for _, a in groups]).detach().requires_grad_(True)
    B_pack, A_scale = stack.fusedpack()
    if lora:
        g = torch.Generator(device="cpu").manual_seed(3)
        lA = (torch.randn(spec.E, RANK, spec.K, generator=g) * 0.01).to(
            device, torch.bfloat16).requires_grad_(True)
        lB = (torch.randn(spec.E, spec.N, RANK, generator=g) * 0.01).to(
            device, torch.bfloat16).requires_grad_(True)
    else:
        lA = lB = None
    out = fused_grouped_lora(a_cat, B_pack, A_scale, sizes, eids,
                             lora_A=lA, lora_B=lB, scaling=SCALING,
                             dgrad_kernel=True)
    out.float().pow(2).mean().backward()
    return (out.detach(), a_cat.grad.detach(),
            lA.grad.detach() if lora else None,
            lB.grad.detach() if lora else None)


def fidelity_cell(stack, spec, groups, lora, device="cuda"):
    """F2, bitwise. Token order is group-major in BOTH paths by construction."""
    T_real = sum(a.shape[0] for _, a in groups)
    R = pick_bucket(max(a.shape[0] for _, a in groups))
    step = DenseGroupsStep("fused", stack, spec, T_real, R, lora, device)
    step.load(groups)
    ref_out, ref_ga, ref_gA, ref_gB = unbucketed_reference(stack, groups, lora, device)

    got = step.eager().detach()
    ga = step.a_pad.grad.index_select(0, step.idx).detach()
    if not lora:
        # BASE path: bitwise as originally registered (and measured passing).
        checks = {"out_bitwise": torch.equal(got, ref_out),
                  "agrad_bitwise": torch.equal(ga, ref_ga)}
    else:
        # LoRA path per AMENDMENT 1: cuBLAS bmm kernel choice varies with batch
        # shape, so bitwise-vs-unbucketed was a broken oracle (it does not even
        # equal itself across routing draws). Three checks replace it:
        rel = lambda a, b: float(((a.float() - b.float()).norm()
                                  / b.float().norm().clamp_min(1e-30)))
        checks = {}
        # (a) LEAK, bitwise: garbage padding must not touch real rows (fwd).
        gA0 = step.lora_A.grad.detach().clone()
        gB0 = step.lora_B.grad.detach().clone()
        with torch.no_grad():
            junk = torch.randn_like(step.tok)
        got2 = _forward_with_padding_fill(step, junk_fill=True).detach()
        checks["leak_fwd_bitwise"] = torch.equal(got2, got)
        # (b) zero-fill determinism bitwise + garbage-contaminates positive
        # control (the leak test must be able to SEE contamination).
        step2 = DenseGroupsStep("fused", stack, spec, sum(a.shape[0] for _, a in groups),
                                step.R, True, device)
        step2.load(groups)
        step2.eager()
        checks["grad_zerofill_deterministic_bitwise"] = (
            torch.equal(step2.lora_A.grad, gA0)
            and torch.equal(step2.lora_B.grad, gB0))
        # AMENDMENT 2: garbage cannot contaminate (grad_out is exactly zero
        # on padded rows — measured, and it falsified the original rationale).
        # The REAL channel is NaN poisoning: 0 * Inf = NaN. The control proves
        # that channel exists and that zero-fill closes it.
        gjA = _grads_with_garbage_padding(step, groups, fill="nan")
        checks["nan_padding_poisons_grads_positive_control"] = (
            not bool(torch.isfinite(gjA).all()))
        checks["zerofill_grads_finite"] = bool(
            torch.isfinite(gA0).all() and torch.isfinite(gB0).all())
        # (c) value agreement vs the unbucketed path, bf16 floor 6.5e-3.
        checks["out_rel"] = rel(got, ref_out)
        checks["agrad_rel"] = rel(ga, ref_ga)
        checks["loraA_rel"] = rel(gA0, ref_gA)
        checks["loraB_rel"] = rel(gB0, ref_gB)
        checks["value_agreement_pass"] = all(
            checks[k] <= 6.5e-3 for k in
            ("out_rel", "agrad_rel", "loraA_rel", "loraB_rel"))

    step.capture()
    step.load(groups)                                  # same inputs, via update
    step.replay()
    torch.cuda.synchronize()
    checks["replay_eq_eager_bitwise"] = torch.equal(
        step._real_out.detach(), got)
    checks["R"] = R
    checks["rows_padded_over_real"] = spec.E * R / T_real
    import math
    block_m = R if R <= 64 else 128
    real_tiles = sum(math.ceil(a.shape[0] / block_m) for _, a in groups)
    checks["tiles_padded_over_real"] = (spec.E * math.ceil(R / block_m)) / real_tiles
    return checks


def race_cell(stack, spec, tokens, lora, args, device="cuda"):
    """F3 + F4 on one (model, proj, T). Fresh routing draw per step so the
    replay pattern is exercised as it would be used, not on one frozen step."""
    draws = [rf.routed_groups(spec, tokens, RESULTS, device, seed=s)
             for s in range(args.steps_pool)]
    T_real = sum(a.shape[0] for _, a in draws[0])
    R = pick_bucket(max(max(a.shape[0] for _, a in g) for g in draws))
    row = {"model": spec.model, "proj": spec.proj, "tokens": tokens, "R": R,
           "lora": lora, "rows_padded_over_real": spec.E * R / T_real}

    arms = {}
    for name in ("fused", "base"):
        s_ = DenseGroupsStep(name, stack, spec, T_real, R, lora, device)
        s_.load(draws[0])
        s_.capture()
        arms[name] = s_

    def timed_replay(s_, n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(n):
            s_.load(draws[i % len(draws)])
            s_.replay()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000.0 / n

    def timed_eager_fused(n):
        sizes0 = [a.shape[0] for _, a in draws[0]]
        eids0 = torch.tensor([e for e, _ in draws[0]], dtype=torch.int32,
                             device=device)
        a0 = torch.cat([a for _, a in draws[0]]).detach().requires_grad_(True)
        B_pack, A_scale = stack.fusedpack()

        def one():
            if a0.grad is not None:
                a0.grad.zero_()
            out = fused_grouped_lora(a0, B_pack, A_scale, sizes0, eids0,
                                     scaling=SCALING, dgrad_kernel=True)
            out.float().pow(2).mean().backward()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            one()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000.0 / n

    H_warm = getattr(__import__("importlib").import_module("e2e_train_arms"),
                     "warm_gpu", None)
    if H_warm:
        H_warm(1.5)
    n = args.steps
    t = {}
    t["g_a"] = timed_replay(arms["fused"], n)
    t["g_b"] = timed_replay(arms["fused"], n)          # replay self-pair
    t["d_a"] = timed_replay(arms["base"], n)
    t["d_b"] = timed_replay(arms["base"], n)
    t["fused_eager"] = timed_eager_fused(min(n, 60))
    row.update({
        "ms": t,
        "g_selfpair": t["g_b"] / t["g_a"],
        "d_selfpair": t["d_b"] / t["d_a"],
        "F3_d_over_g_graphed": t["d_a"] / t["g_b"],
        "F4_graphed_speedup_vs_shipped_eager": t["fused_eager"] / t["g_b"],
    })
    row["arena_owned_ints"] = getattr(
        __import__("nf4_grouped")._ARENAS.get(str(torch.device(device))), "off", 0)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["OLMoE"])
    ap.add_argument("--tokens", type=int, nargs="*", default=[32, 128])
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--steps-pool", type=int, default=8,
                    help="distinct routing draws cycled through replays")
    ap.add_argument("--fidelity-only", action="store_true")
    ap.add_argument("--out", default=os.environ.get("DQF_OUT", "/root/dqf-out"))
    ap.add_argument("--tag", default="gb1")
    args = ap.parse_args()

    out = {"probe": "dense-groups bucketing: fidelity + graphed-vs-graphed race",
           "prereg": "kernel/prereg_graphed_buckets.json",
           "gpu": torch.cuda.get_device_name(0),
           "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
           "torch": torch.__version__, "fidelity": [], "race": []}
    dest = Path(args.out); dest.mkdir(parents=True, exist_ok=True)
    art = dest / f"graphed_buckets_{args.tag}.json"

    for spec in H.census_specs(H.REPO / "census" / "shape_census.json", args.models):
        stack = H.QuantStack(spec, "cuda")
        for tokens in args.tokens:
            try:
                groups = rf.routed_groups(spec, tokens, RESULTS, "cuda", seed=0)
            except LookupError as e:
                out["fidelity"].append({"model": spec.model, "proj": spec.proj,
                                        "tokens": tokens, "status": "not_run",
                                        "reason": str(e)})
                continue
            for lora in (False, True):
                fc = {"model": spec.model, "proj": spec.proj, "tokens": tokens,
                      "lora": lora}
                fc.update(fidelity_cell(stack, spec, groups, lora))
                gates = [v for k, v in fc.items() if k.endswith("_bitwise")]
                gates += [fc["value_agreement_pass"],
                          fc["nan_padding_poisons_grads_positive_control"],
                          fc["zerofill_grads_finite"]] \
                    if lora else []
                fc["F2_pass"] = all(gates)
                out["fidelity"].append(fc)
                print("F2 %-14s %-8s T=%-4d lora=%-5s R=%-2d %s" % (
                    spec.model.split("/")[-1][:14], spec.proj, tokens,
                    lora, fc["R"], "PASS" if fc["F2_pass"] else
                    "FAIL " + str({k: v for k, v in fc.items()
                                   if k.endswith("_bitwise") and not v})),
                    flush=True)
                art.write_text(json.dumps(out, indent=1, default=str))
            if not args.fidelity_only:
                if not all(f.get("F2_pass") for f in out["fidelity"]
                           if f.get("tokens") == tokens):
                    print("F2 failed — the race does not run (stop rule)")
                    continue
                r = race_cell(stack, spec, tokens, False, args)
                out["race"].append(r)
                print("F3 %-8s T=%-4d d/g_graphed %.3f (self %.3f/%.3f)  "
                      "F4 eager/graphed %.2fx  waste rows %.2f" % (
                          spec.proj, tokens, r["F3_d_over_g_graphed"],
                          r["g_selfpair"], r["d_selfpair"],
                          r["F4_graphed_speedup_vs_shipped_eager"],
                          r["rows_padded_over_real"]), flush=True)
                art.write_text(json.dumps(out, indent=1, default=str))
        del stack
        torch.cuda.empty_cache()
    print("GRAPHED_BUCKETS_DONE ->", art)


if __name__ == "__main__":
    main()
