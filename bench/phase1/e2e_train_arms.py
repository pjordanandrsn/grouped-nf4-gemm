#!/usr/bin/env python3
"""END-TO-END: the two arms inside a real QLoRA finetune, not a fixture.

Every number in legs 1-4 is a microbenchmark: synthetic activations, measured
routing *histograms*, one op at a time. This runs the same comparison inside
`experts4bit_qlora`'s real training pipeline -- streaming NF4 load, frozen 4-bit
experts, per-expert LoRA, gradient checkpointing, AdamW -- and reports what a
user actually pays: seconds per step and peak VRAM.

ADAPTED FROM `experts4bit-qlora/bench/dgrad-gate/dgrad_gate.py` (2026-08-06,
e4b 0.11.0 / gnf4 0.7.0, A5000+A6000). That driver's design is kept because it
is right: ONE model load, adapter snapshot/restore between arms, so every arm
starts bit-identical on identical data. Not a fork of e4b -- this imports the
published package and patches nothing in it.

WHAT THE BASELINE IS, PRECISELY. `ExpertsLoRA.forward`'s per-expert Python loop:
dequantize-and-project one hit expert at a time, plus the low-rank delta. That
is the same FAMILY as GenON's `QuantizedNaiveMoE` and the arm legs 1-4 measured,
but it is e4b's code, not GenON's. Nothing here is a measurement of GenON's
implementation and it must not be reported as one.

FIVE THINGS THIS ADDS to the driver it came from, each because its absence would
have let a wrong number through:

1. SELF-PAIR / DRIFT GATE. `reference` runs twice, first and last. The ratio of
   the two is the instrument's own noise floor plus any drift across the sweep.
   Legs 1-3 of this program died three times on instruments that were never
   self-paired; a speedup smaller than the self-pair spread is not a speedup.

2. STEADY STATE, NOT MEAN-OVER-ALL. The source averages wall time over every
   step including the first, which carries allocator warmup, autotune and lazy
   init. Here each step is timed individually and the first `--warmup` are
   dropped, with the median of what remains reported alongside the mean the
   source would have produced, so the difference is visible rather than assumed.

3. REAL TEXT, AND RANDOM TOKENS, AS SEPARATE CELLS. The source feeds random
   token ids, arguing correctly that arm-vs-arm on identical inputs makes the
   text irrelevant. That holds for a dense model. It does NOT obviously hold for
   MoE: routing is a function of token content, and leg 4 established that
   per-expert SKEW is what sets this comparison. Random ids plausibly route
   flatter than prose. Both are run; if they disagree, the source's cost table
   was measuring a routing distribution no user will ever have.

4. MEASURED ROUTING, from the model itself. Occupancy and cv are captured off
   the live `top_k_index` during the run. Legs 1-4 drew from histograms captured
   once, offline; this reports what the router actually did during training, and
   closes the open item asking for occupancy on the real model rather than
   derived.

5. PEAK DECOMPOSED. `max_memory_allocated` is absolute and includes the resident
   model, so a 5% difference between arms can be a large difference in the part
   that actually varies. Resident baseline is recorded before each arm's first
   step and subtracted, so the transient -- the part legs 1-4 measured at
   18.7-48.6x -- is reported separately from the total.

Report-only against `kernel/prereg_dequant_forward_e2e.json`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

import torch

# The GPU-busy label rule (C4) lives in the harness so every leg applies the
# same bar. This driver sits beside it.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402


def _sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().to("cpu").contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def frozen_tensors(model):
    """Quantized expert storage, wherever it lives.

    Parameters ALONE is not enough: under `offload=1` e4b keeps the 4-bit expert
    bytes in pinned CPU RAM and streams a layer at a time, so a params-only scan
    hashed 32 tensors totalling 0.00 GiB on the first attempt -- an integrity
    check that would have reported every arm 'unchanged' having compared
    essentially nothing. Buffers are included and the byte total is asserted by
    the caller, because a check that cannot see is worse than no check: it reads
    as a pass.
    """
    seen = set()
    for n, p in model.named_parameters():
        if not p.requires_grad and ("expert" in n) and p.dtype == torch.uint8:
            seen.add(id(p))
            yield n, p
    for n, b in model.named_buffers():
        if b is not None and ("expert" in n) and b.dtype == torch.uint8 and id(b) not in seen:
            yield n, b


def frozen_hashes(model):
    h, nb = {}, 0
    for n, t in frozen_tensors(model):
        h[n] = _sha(t)
        nb += t.numel() * t.element_size()
    return h, nb


def trainable_snapshot(model):
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


def restore(model, snap):
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in snap:
                p.copy_(snap[n])


def rel(a, b):
    d = (a - b).norm().item()
    n = b.norm().item()
    return d / n if n else (0.0 if d == 0 else float("inf"))


def grads_now(model):
    return {n: p.grad.detach().float().clone()
            for n, p in model.named_parameters() if p.requires_grad and p.grad is not None}


# ---------------------------------------------------------------- data -----
def random_batches(tok, n, seq, device, seed=0):
    g = torch.Generator().manual_seed(seed)
    vocab = int(getattr(tok, "vocab_size", 32000))
    return [torch.randint(0, vocab, (1, seq), generator=g).to(device) for _ in range(n)]


def text_batches(tok, n, seq, device):
    """Real prose. Routing is a function of token content, so this is not
    cosmetic: it is the only cell whose routing distribution a user would see."""
    from datasets import load_dataset
    # Bare "wikitext" stopped resolving: current huggingface_hub requires
    # namespace/name and raises HfUriError on the legacy id.
    ds, err = None, None
    for repo in ("Salesforce/wikitext", "wikitext"):
        try:
            ds = load_dataset(repo, "wikitext-2-raw-v1", split="train")
            break
        except Exception as e:
            err = f"{repo}: {type(e).__name__}: {str(e)[:120]}"
    if ds is None:
        # Never silently fall back to random ids -- text-vs-random IS the cell.
        raise SystemExit(f"could not load wikitext ({err})")
    buf, out = [], []
    for row in ds:
        t = row["text"].strip()
        if not t:
            continue
        buf.extend(tok(t, add_special_tokens=False)["input_ids"])
        while len(buf) >= seq and len(out) < n:
            out.append(torch.tensor(buf[:seq], device=device).unsqueeze(0))
            buf = buf[seq:]
        if len(out) >= n:
            break
    if len(out) < n:
        raise SystemExit(f"wikitext gave {len(out)} of {n} batches at seq={seq}")
    return out


# ------------------------------------------------------------- routing -----
class RoutingTap:
    """Occupancy and skew off the LIVE router, from the same forwards being
    timed. `ExpertsLoRA.forward(hidden, top_k_index, top_k_weights)` -- the tap
    reads arg 1 and never touches the value."""

    def __init__(self, model, num_experts):
        self.E = num_experts
        self.counts = []
        self.handles = []
        for m in model.modules():
            if type(m).__name__ == "ExpertsLoRA":
                self.handles.append(m.register_forward_pre_hook(self._hook))

    def _hook(self, _mod, args):
        if len(args) < 2 or not torch.is_tensor(args[1]):
            return
        idx = args[1].detach().reshape(-1)
        c = torch.bincount(idx, minlength=self.E).float()
        self.counts.append(c.cpu())

    def summarise(self):
        if not self.counts:
            return None
        occ, cvs = [], []
        for c in self.counts:
            hit = int((c > 0).sum())
            occ.append(hit / self.E)
            nz = c[c > 0]
            if len(nz) > 1 and float(nz.mean()) > 0:
                cvs.append(float(nz.std(unbiased=False) / nz.mean()))
        return {"forwards_observed": len(self.counts),
                "occupancy_median": st.median(occ), "occupancy_min": min(occ),
                "occupancy_max": max(occ),
                "cv_median": st.median(cvs) if cvs else None,
                "E": self.E}

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []


# ----------------------------------------------------------------- arm -----
def plan_modes(data: str, discard_first: bool):
    """The run's data-mode schedule as (mode, is_burnin) pairs.

    TEN consecutive runs across THREE hosts (two shared L40S pods, a
    whole-machine 3060 Ti, a whole-machine A4000) drifted the FIRST data mode's
    reference self-pair below G1 (0.859–0.924) while the mode that ran second
    stayed clean — with zero CPU neighbours, so it is a property of this driver,
    not contention. A wall-clock GPU warm-up made it WORSE, and the drift spans
    the whole first mode (~120 steps), so a short burn-in is falsified too:
    settling takes about a mode.

    Hence the registered remedy (scope amendment 1): run the first mode TWICE
    and DISCARD the first pass. The burn-in pass runs the full arm set — the
    settling comes from doing the work — and its receipts land under a separate
    `burnin` key so no reducer can mistake them for cells.
    """
    modes = ["text", "random"] if data == "both" else [data]
    plan = [(m, False) for m in modes]
    if discard_first and plan:
        plan.insert(0, (plan[0][0], True))
    return plan


def warm_gpu(seconds: float, device="cuda"):
    """Wall-clock GPU-busy work before the FIRST timed arm of a mode.

    THE CLOCK-RECOVERY LAW, applied at leg level. `--warmup N` drops the first N
    STEPS of each arm. It does not cover the gap between a model load -- a long
    CPU-bound stretch with the GPU idle -- and the first arm timed after it,
    which reads the card boosting back up.

    Measured, 2026-08-14 L40S capturability gate: the FIRST data mode of every
    sweep blew the e2e prereg's G1 band (text 0.9564 / 0.9047) and worsened
    monotonically to 0.8926, while the mode that ran SECOND was clean throughout
    (1.0010-1.0453). Those cells were excluded and the gate could not close.

    The fix is wall-clock, not more iterations -- the same one that took the 4090
    from 9 void cells to 0 in the leg-1 re-run. More warm-up ITERATIONS do not
    work: a per-arm warmup of 4 steps is over in a fraction of the time the card
    needs to come back up.
    """
    if not torch.cuda.is_available() or seconds <= 0:
        return 0.0
    a = torch.randn(2048, 2048, device=device, dtype=torch.bfloat16)
    b = torch.randn(2048, 2048, device=device, dtype=torch.bfloat16)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        for _ in range(8):
            c = a @ b          # noqa: F841 -- discarded; the point is the work
    torch.cuda.synchronize()
    el = time.perf_counter() - t0
    del a, b
    torch.cuda.empty_cache()
    return el


def busy_fraction(model, opt, ids, m=3):
    """GPU-busy fraction of a real training step (C4), measured on this arm.

    Same definition as everywhere else in the harness -- summed CUDA kernel
    self-time per step over wall per step, back to back, no syncs between steps.

    Kept OUT of the timed loop: it runs after the timed steps, on one batch, so
    the profiler never touches the number the prereg grades.

    BUDGET THIS. It costs `2 + 2m` steps per arm, and an e2e step is seconds,
    not the milliseconds a microbench cell takes. At m=8 it added ~75% to the
    step count of every arm -- 1080 extra steps across a three-sweep gate run --
    and that overrun is what cost the 2026-08-14 capturability gate its final
    sweep and left the run UNUSABLE. The fractions it produces are stable to the
    percent (53/53, 34/34 across arms on the L40S), so m=3 buys the same label
    at a third of the cost. Raise it with --busy-steps if a cell is borderline
    against the 50% bar; do not raise it by default.
    """
    from torch.profiler import ProfilerActivity, profile

    def one():
        opt.zero_grad(set_to_none=False)
        model(input_ids=ids, labels=ids).loss.backward()

    for _ in range(2):
        one()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(m):
        one()
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - t0) * 1000.0 / m
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as pr:
        for _ in range(m):
            one()
        torch.cuda.synchronize()
    us = 0.0
    for e in pr.key_averages():
        v = getattr(e, "self_device_time_total", None)
        if v is None:
            v = getattr(e, "self_cuda_time_total", 0.0)
        us += float(v or 0.0)
    busy_ms = us / 1000.0 / m
    opt.zero_grad(set_to_none=True)
    return {"wall_per_step_ms": wall_ms, "gpu_busy_per_step_ms": busy_ms,
            "busy_fraction": (busy_ms / wall_ms) if wall_ms else None}


def run_arm(model, arm, batches, lr, enable, disable, warmup, tap_E,
            measure_busy=True, busy_steps=3):
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    resident = torch.cuda.memory_allocated()          # decomposition baseline
    torch.cuda.reset_peak_memory_stats()
    n_patched = enable(model) if enable else 0
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    tap = RoutingTap(model, tap_E) if tap_E else None

    losses, per_step, first_grads, routing = [], [], None, None
    t_all = time.perf_counter()
    for i, ids in enumerate(batches):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        out = model(input_ids=ids, labels=ids)
        out.loss.backward()
        if i == 0:
            first_grads = grads_now(model)
        opt.step()
        torch.cuda.synchronize()
        per_step.append(time.perf_counter() - t0)
        losses.append(float(out.loss.detach()))
        # One step's worth of routing is enough and keeps the tap off the timed
        # steps: every later step pays no hook at all.
        if tap is not None and i == 0:
            routing = tap.summarise()
            tap.close()
            tap = None
    mean_all = (time.perf_counter() - t_all) / max(1, len(batches))
    if tap is not None:
        routing = tap.summarise()
        tap.close()

    steady = per_step[warmup:] or per_step
    peak = torch.cuda.max_memory_allocated()
    # AFTER the timed steps and after peak is read, so neither is perturbed.
    busy = (busy_fraction(model, opt, batches[0], busy_steps)
            if measure_busy and busy_steps else None)
    if disable:
        disable(model)
    return dict(
        arm=arm, n_patched=n_patched, losses=losses,
        s_per_step=round(st.median(steady), 4),
        s_per_step_mean_all=round(mean_all, 4),          # what the source reported
        s_per_step_first=round(per_step[0], 4),
        s_per_step_spread=round((max(steady) - min(steady)) / st.median(steady), 4),
        steps_timed=len(per_step), warmup_dropped=warmup,
        train_peak_gb=round(peak / 2**30, 3),
        resident_gb=round(resident / 2**30, 3),
        transient_gb=round((peak - resident) / 2**30, 3),
        routing=routing,
        gpu_busy=busy,
    ), first_grads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--offload", type=int, default=1)
    # REAL TEXT IS THE DEFAULT (C5). Random ids are a fixture that flatters the
    # dequant baseline by 1.6-1.7x on every matched pair, because MoE routing is
    # content-dependent: random ids hit FEWER experts (occupancy 0.875 vs 0.984)
    # and far more unevenly (cv 1.463 vs 0.687), so the baseline's python loop
    # runs fewer iterations -- which is precisely the cost the fused path
    # removes. They stay available for work that genuinely needs
    # content-independence, and `both` still runs the matched pair, but a run
    # that says nothing about its fixture now gets prose.
    ap.add_argument("--data", default="text", choices=["text", "random", "both"])
    ap.add_argument("--warm-s", type=float, default=1.5,
                    help="wall-clock GPU-busy seconds before the first arm of "
                         "each data mode (clock recovery). 0 disables it.")
    ap.add_argument("--discard-first-mode", type=int, default=1,
                    help="run the first data mode twice and DISCARD the first "
                         "pass (default 1). Ten consecutive first modes drifted "
                         "below G1 across three hosts; settling takes ~a full "
                         "mode, so this is the registered remedy. 0 restores "
                         "the old single-pass behaviour.")
    ap.add_argument("--busy-steps", type=int, default=3,
                    help="profiled steps per arm for the GPU-busy label (C4). "
                         "Costs 2+2N steps per arm; 0 disables it and the cell "
                         "is then measurement_class=unknown, which is NOT "
                         "'kernel'.")
    ap.add_argument("--out", default=os.environ.get("DQF_OUT", "/root/dqf-out"))
    ap.add_argument("--tag", default="e2e")
    a = ap.parse_args()

    import experts4bit_qlora as e4b
    from experts4bit_qlora import (disable_batched_train, disable_fast_train,
                                   enable_batched_train, enable_fast_train,
                                   load_moe_4bit_streaming)
    from transformers import AutoTokenizer
    import nf4_grouped

    env = dict(gpu=torch.cuda.get_device_name(0),
               cap=list(torch.cuda.get_device_capability(0)),
               vram_total_gb=round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
               torch=torch.__version__, e4b=e4b.__version__,
               gnf4=getattr(__import__("nf4_grouped"), "__version__", "?"),
               gnf4_has_dgrad=hasattr(nf4_grouped, "dgrad_4bit_grouped"))
    print("env:", json.dumps(env), flush=True)

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    model, cfg = load_moe_4bit_streaming(a.model, "cuda", torch.bfloat16, r=a.r,
                                         alpha=a.alpha, quant_type="nf4",
                                         offload=bool(a.offload))
    if not a.offload:
        model.to("cuda")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    tcfg = getattr(cfg, "text_config", cfg)
    layers = getattr(tcfg, "num_hidden_layers", -1)
    n_exp = getattr(tcfg, "num_experts", getattr(tcfg, "n_routed_experts", 0))

    for n, p in model.named_parameters():
        p.requires_grad_("lora" in n and "experts" in n)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"layers={layers} experts={n_exp} trainable={n_train:,}", flush=True)

    base_snap = trainable_snapshot(model)
    frozen_before, nbytes = frozen_hashes(model)
    if not frozen_before:
        raise SystemExit("FATAL: hashed 0 frozen expert tensors — the integrity "
                         "check would be vacuous.")
    # A check that hashes a trivial number of bytes still PASSES, which is worse
    # than no check. Record it so a green integrity result can be read against
    # how much it actually covered.
    vacuous = nbytes < (0.25 * 2**30)
    # Under offload the only uint8 expert tensors resident are e4b's per-layer
    # STAGING buffers, reused as layers stream in and out -- their contents
    # change every step BY DESIGN. Hashing them reports "frozen bytes CHANGED"
    # for every arm including `reference` against itself, which is a false
    # positive about the instrument, not a finding about the arms. Measured:
    # offload=1 hashes 32 tensors / 0.00 GiB and flags all 5 arms; offload=0
    # hashes 3.00 GiB of the same model and flags none. So the integrity claim
    # is carried by the resident cell only, and under offload it is recorded as
    # NOT APPLICABLE rather than as a pass or a failure.
    integrity_applicable = bool(a.offload) is False and not vacuous
    if not integrity_applicable:
        print("frozen-storage check NOT APPLICABLE under offload=%d (%d tensors, "
              "%.2f GiB — these are streaming staging buffers, not frozen "
              "storage); the resident cell carries this claim"
              % (a.offload, len(frozen_before), nbytes / 2**30), flush=True)
    else:
        print("frozen tensors under check: %d (%.2f GiB)"
              % (len(frozen_before), nbytes / 2**30), flush=True)

    # `reference` runs FIRST and LAST. The pair brackets the whole sweep, so its
    # ratio is drift + noise; every other arm's margin is read against it.
    ARMS = [
        ("reference", None, None),
        ("fast_train", lambda m: enable_fast_train(m), disable_fast_train),
        ("fast_train_dgrad", lambda m: enable_fast_train(m, dgrad=True), disable_fast_train),
        ("batched", enable_batched_train, disable_batched_train),
        ("reference_selfpair", None, None),
    ]

    plan = plan_modes(a.data, bool(a.discard_first_mode))
    payload = dict(probe="end-to-end QLoRA training, fused vs per-expert dequant loop",
                   prereg="kernel/prereg_dequant_forward_e2e.json",
                   adapted_from="experts4bit-qlora/bench/dgrad-gate/dgrad_gate.py",
                   model=a.model, steps=a.steps, warmup=a.warmup, seq=a.seq,
                   layers=layers, num_experts=n_exp, trainable_params=n_train,
                   offload=bool(a.offload), env=env,
                   frozen_tensors_hashed=len(frozen_before),
                   frozen_bytes_hashed=nbytes,
                   frozen_check_vacuous=bool(vacuous),
                   integrity_applicable=integrity_applicable,
                   data_default="text (C5); random ids are opt-in",
                   clock_recovery_warm_s=a.warm_s,
                   random_id_understatement=(
                       "1.6-1.7x on every matched pair — any table citing a "
                       "random-id cell states this factor beside it"),
                   cells={})
    dest = Path(a.out)
    dest.mkdir(parents=True, exist_ok=True)
    art = dest / f"e2e_{a.tag}.json"

    payload["mode_plan"] = [f"{m}{'(burnin,discarded)' if b else ''}"
                            for m, b in plan]
    payload.setdefault("burnin", {})
    _batches_cache = {}
    for mode, is_burnin in plan:
        print(f"\n########## data={mode}{' BURN-IN (discarded)' if is_burnin else ''} ##########",
              flush=True)
        if mode not in _batches_cache:
            _batches_cache[mode] = (
                text_batches(tok, a.steps, a.seq, "cuda") if mode == "text"
                else random_batches(tok, a.steps, a.seq, "cuda"))
        batches = _batches_cache[mode]
        # Clock recovery, before the first arm of EVERY mode. Building batches
        # is itself a CPU-bound stretch with the GPU idle -- `text_batches`
        # tokenises wikitext -- so the mode that runs first is not the only one
        # exposed, it is just the worst. Cheap enough to do unconditionally.
        el = warm_gpu(a.warm_s)
        print(f"clock-recovery warm-up: {el:.2f}s GPU-busy before arm 1",
              flush=True)
        results, grads = {}, {}
        for name, en, dis in ARMS:
            restore(model, base_snap)
            print(f"--- arm {name} ({mode}) ---", flush=True)
            try:
                res, g = run_arm(model, name, batches, a.lr, en, dis, a.warmup, n_exp,
                                 busy_steps=a.busy_steps)
            except Exception as exc:
                print(f"arm {name} FAILED: {type(exc).__name__}: {exc}", flush=True)
                results[name] = dict(arm=name, failed=f"{type(exc).__name__}: {str(exc)[:300]}")
                continue
            after, _ = frozen_hashes(model)
            res["frozen_changed"] = len([k for k, v in frozen_before.items() if after.get(k) != v])
            # gnf4 is arch-gated and has to BUILD. When it does not, enable_*
            # returns 0, nothing is patched, and the arm silently runs the
            # reference loop -- reporting a ~1.00x that looks like a real
            # measurement of a fused path that never executed. An arm that
            # patched nothing is not a datapoint.
            if en is not None and not res["n_patched"]:
                res["INVALID_no_modules_patched"] = True
                print("  !! INVALID: %s patched 0 modules — this arm ran the "
                      "reference path, not its own" % name, flush=True)
            results[name], grads[name] = res, g
            print("  %.4f s/step (mean-all %.4f, first %.4f) | peak %.3f GB "
                  "(resident %.3f transient %.3f) | frozen_changed %d"
                  % (res["s_per_step"], res["s_per_step_mean_all"], res["s_per_step_first"],
                     res["train_peak_gb"], res["resident_gb"], res["transient_gb"],
                     res["frozen_changed"]), flush=True)

        ref = results.get("reference", {})
        ref_g = grads.get("reference", {})
        for name in list(results):
            r = results[name]
            if "failed" in r or not ref.get("s_per_step"):
                continue
            r["speedup_vs_reference"] = round(ref["s_per_step"] / r["s_per_step"], 4)
            r["peak_vs_reference"] = round(r["train_peak_gb"] / ref["train_peak_gb"], 4)
            if r["transient_gb"] > 0 and ref.get("transient_gb", 0) > 0:
                r["transient_vs_reference"] = round(r["transient_gb"] / ref["transient_gb"], 4)
            g = grads.get(name, {})
            per = {k: rel(g[k], ref_g[k]) for k in ref_g if k in g}
            r["grad_rel_mean"] = (sum(per.values()) / len(per)) if per else None
            r["grad_rel_worst"] = max(per.values()) if per else None
            if ref.get("losses") and r.get("losses"):
                dl = sorted(abs(x - y) for x, y in zip(r["losses"], ref["losses"]))
                r["loss_median_abs_delta"] = dl[len(dl) // 2]
            # REGISTERED LABEL (C4): the speedup above is a kernel result only
            # if BOTH arms in the pair are GPU-bound. Applied per arm, because
            # the reference and the arm can sit on opposite sides of the bar.
            (r["measurement_class"], r["min_busy_fraction"],
             _fr) = H.measurement_class({"reference": ref.get("gpu_busy"),
                                         name: r.get("gpu_busy")})
        payload["measurement_class_note"] = H.MEASUREMENT_CLASS_NOTE
        if is_burnin:
            # Discarded by REGISTRATION, not by judgement: these receipts exist
            # so the burn-in is auditable, under a key no reducer reads.
            payload["burnin"][mode] = results
        else:
            payload["cells"][mode] = results
        art.write_text(json.dumps(payload, indent=1, default=str))

    print("\n=== SUMMARY ===")
    for mode, results in payload["cells"].items():
        sp = results.get("reference_selfpair", {}).get("speedup_vs_reference")
        rt = (results.get("reference", {}) or {}).get("routing") or {}
        print("data=%-7s SELF-PAIR %s   routing occ %s cv %s%s" % (
            mode, ("%.4f" % sp) if sp else "n/a",
            ("%.3f" % rt["occupancy_median"]) if rt.get("occupancy_median") else "?",
            ("%.3f" % rt["cv_median"]) if rt.get("cv_median") else "?",
            "   [RANDOM-ID FIXTURE: understates the fused advantage by "
            "1.6-1.7x — state this factor beside any number below]"
            if mode == "random" else ""))
        for name, r in results.items():
            if "failed" in r:
                print("  %-20s FAILED %s" % (name, r["failed"][:60]))
                continue
            bf = (r.get("gpu_busy") or {}).get("busy_fraction")
            print("  %-20s %8.4f s/step  %6sx  peak %6.3f GB (%sx)  transient %6.3f GB  gradΔ %s  busy %s  %s"
                  % (name, r["s_per_step"],
                     ("%.3f" % r["speedup_vs_reference"]) if r.get("speedup_vs_reference") else "-",
                     r["train_peak_gb"],
                     ("%.3f" % r["peak_vs_reference"]) if r.get("peak_vs_reference") else "-",
                     r["transient_gb"],
                     ("%.1e" % r["grad_rel_mean"]) if r.get("grad_rel_mean") else "-",
                     ("%3.0f%%" % (100 * bf)) if bf else "  ?",
                     r.get("measurement_class", "unknown").upper()))
    print("E2E_DONE")


if __name__ == "__main__":
    main()
