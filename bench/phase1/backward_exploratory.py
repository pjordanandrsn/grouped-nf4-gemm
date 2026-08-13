#!/usr/bin/env python3
"""EXPLORATORY training-axis leg. Registered in the base prereg as unable to
change any verdict, and it does not.

AMENDMENT 2 corrects run 3, which was unfair in BOTH directions at once:
  * it timed gnf4 at `dgrad_kernel=False`, the DEFAULT — a deliberately EXACT
    reference path (per-expert loop decoding through `dequant_ref`), not a
    performance path. That pitted unsloth's TUNED backward against gnf4's
    REFERENCE backward.
  * it gave the gnf4 arm LoRA and the unsloth arm none, charging gnf4 for work
    unsloth was not doing.
Both are fixed here, together, so neither correction can be cherry-picked.

Arms per cell, all fwd+bwd, CUDA-event timed, base weights frozen (the LoRA
regime both stacks actually run, so unsloth computes dX and not dW):
  G_ref   : gnf4, dgrad_kernel=False — run 3's arm, kept so run 4 is comparable
            to run 3 WITHIN one run rather than across runs
  G_kernel: gnf4, dgrad_kernel=True — the single-launch dgrad over packed bytes
  U_4bit  : unsloth + an equivalent low-rank delta, dequant timed in
  U_bf16  : same, bf16-resident — their ceiling, reported not barred

G_kernel is re-timed immediately before each U arm (amendment 1's pairing rule);
its self-pair must hold [0.97,1.03] or the leg is VOID.

Residual asymmetry, still disclosed: gnf4 computes the delta through
`lora_delta_grouped` off packed bytes while the unsloth arm applies the same
delta outside its grouped GEMM, which is where unsloth's LoRA actually lives.
That is a difference in WHERE the delta is applied, not in whether it is paid.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
from pathlib import Path

import torch

# Repo root comes from H2H_REPO (or cwd), NEVER from __file__: this script is
# staged at /root/ while the repo lives at /root/h2h, so a __file__-relative
# path resolved to /root/bench/phase1 and died on ModuleNotFoundError AFTER the
# matrix had finished and the pod was minutes from teardown.
_ROOT = Path(os.environ.get("H2H_REPO", Path.cwd())).resolve()
if not (_ROOT / "bench" / "phase1" / "harness.py").exists():
    raise SystemExit(
        f"H2H_REPO/cwd={_ROOT} is not the repo root "
        f"(no bench/phase1/harness.py). Set H2H_REPO."
    )
sys.path.insert(0, str(_ROOT / "bench" / "phase1"))
sys.path.insert(0, str(_ROOT / "kernel"))
import harness as H  # noqa: E402
from nf4_qlora import fused_grouped_lora, lora_delta_grouped  # noqa: E402

RANK = 16
ITERS = 50


def _timed(fn, iters=ITERS):
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


def cell(spec, regime, device):
    stack = H.QuantStack(spec, device)
    groups = H.make_activations(spec, regime, device)
    sizes = [a.shape[0] for _, a in groups]
    eids = torch.tensor([e for e, _ in groups], dtype=torch.int32, device=device)
    a_cat = torch.cat([a for _, a in groups]).detach().requires_grad_(True)
    B_pack, A_scale = stack.fusedpack()

    # Shapes per lora_delta_grouped's contract: lora_A is [E, r, K] and lora_B
    # is [E, N, r] — NOT the [K, r] / [r, N] a reader might assume from the
    # usual LoRA writeup. lora_B starts at zero, which is standard LoRA init;
    # the delta is zero but the full forward and backward still execute, so the
    # timing is honest.
    lora_A = torch.randn(len(groups), RANK, spec.K, device=device,
                         dtype=torch.bfloat16) * 0.01
    lora_B = torch.zeros(len(groups), spec.N, RANK, device=device,
                         dtype=torch.bfloat16)
    lora_A.requires_grad_(True)
    lora_B.requires_grad_(True)

    W_resident = torch.stack([stack.dequant_bf16(e) for e, _ in groups]).detach()

    def _zero():
        for t in (a_cat, lora_A, lora_B):
            t.grad = None

    def g_arm(dgrad_kernel):
        def run():
            _zero()
            out = fused_grouped_lora(a_cat, B_pack, A_scale, sizes, eids,
                                     lora_A=lora_A, lora_B=lora_B,
                                     dgrad_kernel=dgrad_kernel)
            out.float().pow(2).mean().backward()
        return run

    def u_arm(resident):
        """AMENDMENT 2: the unsloth arm now carries an EQUIVALENT low-rank delta
        on the SAME lora_A/lora_B, so both sides compute dX *and* LoRA adapter
        gradients. Run 3 gave gnf4's arm LoRA and unsloth's arm none, which
        charged gnf4 for work unsloth was not doing. Applied outside the grouped
        GEMM, which is where unsloth's LoRA actually lives."""
        def run():
            _zero()
            W = W_resident if resident else torch.stack(
                [stack.dequant_bf16(e) for e, _ in groups])
            out = H._unsloth_native_call(a_cat, W, sizes)
            delta = lora_delta_grouped(a_cat, lora_A, lora_B, sizes, eids)
            if delta is not None:
                out = out + delta.to(out.dtype)
            out.float().pow(2).mean().backward()
        return run

    row = {"model": spec.model, "proj": spec.proj, "regime": regime,
           "N": spec.N, "K": spec.K, "rank": RANK}
    try:
        g_ref, g_ker = g_arm(False), g_arm(True)
        # T1: the reference default vs the single-launch dgrad, paired.
        row["g_ref_ms"] = _timed(g_ref)
        row["g_kernel_ms_for_ref"] = _timed(g_ker)
        row["T1_kernel_over_ref"] = row["g_ref_ms"] / row["g_kernel_ms_for_ref"]

        # G_kernel re-timed immediately before each comparator (amendment 1).
        row["g_ms_for_bf16"] = _timed(g_ker)
        row["u_bf16_ms"] = _timed(u_arm(True))
        row["g_ms_for_4bit"] = _timed(g_ker)
        row["u_4bit_ms"] = _timed(u_arm(False))
        row["g_selfpair"] = row["g_ms_for_4bit"] / row["g_ms_for_bf16"]
        row["u_bf16_over_g"] = row["u_bf16_ms"] / row["g_ms_for_bf16"]
        row["u_4bit_over_g"] = row["u_4bit_ms"] / row["g_ms_for_4bit"]
        row["status"] = "ok"
    except Exception as e:
        row.update({"status": "skipped", "reason": f"{type(e).__name__}: {str(e)[:180]}"})
    del stack
    torch.cuda.empty_cache()
    return row


def main():
    dev = "cuda"
    specs = H.census_specs(H.REPO / "census" / "shape_census.json", None)
    out = {"TIER": "EXPLORATORY — cannot change any registered verdict",
           "gpu": torch.cuda.get_device_name(0),
           "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
           "amendment": "2 — both arms now carry LoRA, and gnf4 runs the "
                        "single-launch dgrad (dgrad_kernel=True). G_ref retained "
                        "for the within-run T1 comparison against run 3's arm.",
           "asymmetry": "Frozen base, so unsloth computes dX not dW. Both arms "
                        "compute dX + LoRA dA/dB. Residual difference is WHERE "
                        "the delta is applied (gnf4 off packed bytes; unsloth "
                        "outside its GEMM), not whether it is paid.",
           "rows": []}
    for spec in specs:
        for regime in ("decode_m8", "prefill_s2048"):
            r = cell(spec, regime, dev)
            out["rows"].append(r)
            print(json.dumps(r), flush=True)
    dest = Path(os.environ.get("H2H_OUT", "/root/h2h-out"))
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "backward_exploratory.json").write_text(
        json.dumps(out, indent=1, default=str))
    ok = [r for r in out["rows"] if r.get("status") == "ok"]
    if ok:
        print("MEDIAN T1=%.3f  u_4bit/g=%.3f  u_bf16/g=%.3f  self=%.3f" % (
            statistics.median(r["T1_kernel_over_ref"] for r in ok),
            statistics.median(r["u_4bit_over_g"] for r in ok),
            statistics.median(r["u_bf16_over_g"] for r in ok),
            statistics.median(r["g_selfpair"] for r in ok)))
    print("BACKWARD_DONE")


if __name__ == "__main__":
    main()
