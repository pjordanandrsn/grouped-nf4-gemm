#!/usr/bin/env python3
"""Phase-1 baseline harness: census shapes x regimes x backends, with J/token receipts.

Measures the baselines the fused kernel must beat (gemm_predictions.json):
the dequant->bf16 grouped path (the e4b product path), bnb gemv_4bit at bs1
(the existing NF4-aware reference), and — import-guarded, recorded as skipped
when absent — Unsloth MoE backends and Marlin. The Phase-2 kernel drops into
the same registry, so its receipts land in the same JSON schema the thresholds
were registered against.

Fidelity per TOLERANCE_CONTRACT.md: the fp64 reference is exact math on the
SAME dequantized values (A_fp64 @ dequant_fp64(W).T), so per-cell error
measures GEMM reduction/rounding — not quantization loss, which every path
shares. The dequant path's B-rel per cell is the comparator the fused kernel's
2x bound is registered against.

Energy: NVML power sampling (pynvml if present, nvidia-smi polling otherwise)
over a >=1 s timed window; J/token = mean watts x window / tokens. Receipts
carry the sampling method and rate — a 50 Hz poll cannot resolve per-launch
spikes, only sustained draw, which is what the J/token claim is about.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
BLOCKSIZE = 64  # quantize_moe_experts default; KERNEL_CONTRACT convention pin


# ---------------------------------------------------------------- fixtures
@dataclass
class GemmSpec:
    model: str
    proj: str  # gate_up | down
    N: int
    K: int
    E: int
    top_k: int


def census_specs(census_path: Path, models: list[str] | None) -> list[GemmSpec]:
    d = json.loads(census_path.read_text())
    specs = []
    for m in d["models"]:
        if models and not any(s in m["model"] for s in models):
            continue
        for proj, nk in m["per_expert_gemms"].items():
            specs.append(
                GemmSpec(m["model"], proj, nk["N"], nk["K"], m["experts"], m["top_k"])
            )
    return specs


class QuantStack:
    """One fused expert stack, quantized per expert exactly as quantize_moe_experts
    does (per-expert quantize_4bit over the [N,K] slice, canonical #1949 layout).
    Keeps the fp64 dequantized copy for the reference GEMM."""

    def __init__(self, spec: GemmSpec, device: str, seed: int = 42):
        self.spec = spec
        g = torch.Generator(device="cpu").manual_seed(seed)
        w = torch.randn(spec.E, spec.N, spec.K, generator=g, dtype=torch.float32)
        w = (w * 0.02).to(device=device, dtype=torch.bfloat16)
        from bitsandbytes import functional as F

        self.packed, self.states = [], []
        for e in range(spec.E):
            q, st = F.quantize_4bit(w[e], blocksize=BLOCKSIZE, quant_type="nf4")
            self.packed.append(q)
            self.states.append(st)
        # ground truth = the values every path actually computes with
        self.w_ref64 = torch.stack(
            [F.dequantize_4bit(self.packed[e], self.states[e]) for e in range(spec.E)]
        ).to(torch.float64)
        del w

    def dequant_bf16(self, e: int) -> torch.Tensor:
        from bitsandbytes import functional as F

        return F.dequantize_4bit(self.packed[e], self.states[e])


def make_activations(spec: GemmSpec, regime: str, device: str, seed: int = 7):
    """Per-regime grouped problem: list of (expert_id, A[M,K] bf16)."""
    g = torch.Generator(device="cpu").manual_seed(seed)

    def act(m):
        return (torch.randn(m, spec.K, generator=g, dtype=torch.float32) * 0.5).to(
            device=device, dtype=torch.bfloat16
        )

    if regime == "decode_bs1":
        experts = list(range(spec.top_k))  # k experts, one token each
        return [(e, act(1)) for e in experts]
    if regime == "prefill_s2048":
        m = max(1, round(2048 * spec.top_k / spec.E))  # uniform routing, census note
        return [(e, act(m)) for e in range(spec.E)]
    raise ValueError(regime)


# ---------------------------------------------------------------- backends
def bk_dequant_grouped(stack: QuantStack, groups):
    """The e4b product path: dequantize the active experts to bf16 in global
    memory, then per-expert bf16 mm (the sparse loop the integration runs)."""
    outs = []
    for e, a in groups:
        w = stack.dequant_bf16(e)
        outs.append(a @ w.t())
    return outs


def bk_gemv4bit(stack: QuantStack, groups):
    """bnb's NF4-aware gemv at M=1 — dequantizes inside the kernel; the closest
    existing point to the fused claim. bs1 only (gemv semantics)."""
    from bitsandbytes import functional as F

    outs = []
    for e, a in groups:
        if a.shape[0] != 1:
            raise RuntimeError("gemv_4bit is M=1 only")
        outs.append(
            F.gemv_4bit(a, stack.packed[e].t(), state=stack.states[e])
        )
    return outs


def bk_unsloth(stack, groups):  # pragma: no cover - optional dependency
    import importlib

    if importlib.util.find_spec("unsloth") is None:
        raise ImportError("unsloth not installed")
    raise ImportError("unsloth present but MoE backend wiring not implemented in Phase 1 v0")


def bk_marlin(stack, groups):  # pragma: no cover - optional dependency
    import importlib

    if importlib.util.find_spec("vllm") is None:
        raise ImportError("vllm (marlin) not installed")
    raise ImportError("marlin present but repack wiring not implemented in Phase 1 v0")


BACKENDS = {
    "dequant_grouped": bk_dequant_grouped,
    "gemv_4bit": bk_gemv4bit,
    "unsloth": bk_unsloth,
    "marlin": bk_marlin,
}


# ---------------------------------------------------------------- measurement
class PowerSampler:
    """Mean GPU watts over start()..stop(). pynvml preferred; nvidia-smi poll
    fallback. Records its own method + achieved rate into the receipt."""

    def __init__(self, device_index: int = 0):
        self.samples: list[float] = []
        self._stop = threading.Event()
        self.method = "none"
        try:
            import pynvml

            pynvml.nvmlInit()
            self._h = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            self._read = lambda: pynvml.nvmlDeviceGetPowerUsage(self._h) / 1000.0
            self.method = "pynvml"
        except Exception:
            self._read = lambda: float(
                subprocess.run(
                    ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=2,
                ).stdout.strip().splitlines()[0]
            )
            self.method = "nvidia-smi"

    def _run(self):
        while not self._stop.is_set():
            try:
                self.samples.append(self._read())
            except Exception:
                pass
            time.sleep(0.02)

    def start(self):
        self.samples = []
        self._stop.clear()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def stop(self) -> tuple[float | None, int]:
        self._stop.set()
        self._t.join(timeout=2)
        return (statistics.mean(self.samples) if self.samples else None, len(self.samples))


def time_backend(fn, stack, groups, iters: int, device: str):
    for _ in range(min(10, iters)):  # warmup
        fn(stack, groups)
    torch.cuda.synchronize()
    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        ev0.record()
        fn(stack, groups)
        ev1.record()
        torch.cuda.synchronize()
        times.append(ev0.elapsed_time(ev1))
    return statistics.median(times)


def energy_window(fn, stack, groups, device: str, min_s: float = 1.2):
    """Repeat the call for >= min_s under the power sampler; J/call from mean W."""
    sampler = PowerSampler()
    torch.cuda.synchronize()
    sampler.start()
    t0 = time.monotonic()
    calls = 0
    while time.monotonic() - t0 < min_s:
        fn(stack, groups)
        calls += 1
    torch.cuda.synchronize()
    wall = time.monotonic() - t0
    watts, n = sampler.stop()
    if watts is None or calls == 0:
        return None, None, sampler.method, n
    return watts, watts * wall / calls, sampler.method, n


def fidelity(stack: QuantStack, groups, outs) -> float:
    """Relative Frobenius error vs the fp64 exact GEMM on identical dequantized values."""
    num = den = 0.0
    for (e, a), out in zip(groups, outs):
        ref = a.to(torch.float64) @ stack.w_ref64[e].t()
        num += (out.to(torch.float64) - ref).norm().item() ** 2
        den += ref.norm().item() ** 2
    return (num ** 0.5) / (den ** 0.5)


# ---------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None, help="substring filters over census models")
    ap.add_argument("--regimes", nargs="*", default=["decode_bs1", "prefill_s2048"])
    ap.add_argument("--backends", nargs="*", default=["dequant_grouped", "gemv_4bit", "unsloth", "marlin"])
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--no-energy", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true", help="tiny E/N/K, iters=3 (still needs CUDA)")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "Phase-1 baselines are GPU measurements"
    device = "cuda"
    specs = census_specs(REPO / "census" / "shape_census.json", args.models)
    if args.smoke:
        specs = [GemmSpec("smoke", "gate_up", 256, 128, 8, 2)]
        args.iters = 3

    env = {
        "gpu": torch.cuda.get_device_name(0),
        "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
        "torch": torch.__version__,
        "driver": subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True,
        ).stdout.strip(),
    }
    try:
        import bitsandbytes

        env["bitsandbytes"] = bitsandbytes.__version__
    except Exception as e:  # pragma: no cover
        env["bitsandbytes"] = f"unavailable: {e}"

    cells = []
    for spec in specs:
        print(f"== {spec.model} {spec.proj} N={spec.N} K={spec.K} E={spec.E} k={spec.top_k}")
        stack = QuantStack(spec, device)
        for regime in args.regimes:
            groups = make_activations(spec, regime, device)
            tokens = sum(a.shape[0] for _, a in groups)
            for name in args.backends:
                fn = BACKENDS[name]
                cell = {
                    "model": spec.model, "proj": spec.proj, "regime": regime,
                    "backend": name, **{k: getattr(spec, k) for k in ("N", "K", "E", "top_k")},
                    "tokens_per_call": tokens,
                }
                try:
                    if name == "gemv_4bit" and regime != "decode_bs1":
                        raise RuntimeError("gemv_4bit is bs1-only by definition")
                    outs = fn(stack, groups)
                    cell["b_rel_vs_fp64"] = fidelity(stack, groups, outs)
                    cell["ms_median"] = time_backend(fn, stack, groups, args.iters, device)
                    cell["tok_per_s"] = tokens / (cell["ms_median"] / 1e3)
                    if not args.no_energy:
                        watts, j_call, method, n = energy_window(fn, stack, groups, device)
                        cell.update({
                            "watts_mean": watts,
                            "j_per_token": (j_call / tokens) if j_call else None,
                            "power_method": method, "power_samples": n,
                        })
                    cell["status"] = "ok"
                    print(f"   {regime:>14} {name:<16} {cell['ms_median']:8.3f} ms "
                          f"{cell['tok_per_s']:10.1f} tok/s  err {cell['b_rel_vs_fp64']:.2e}")
                except Exception as e:
                    cell.update({"status": "skipped", "reason": str(e)[:200]})
                    print(f"   {regime:>14} {name:<16} skipped: {str(e)[:80]}")
                cells.append(cell)
        del stack
        torch.cuda.empty_cache()

    out = {"phase": 1, "spec": "gemm_predictions.json", "env": env,
           "blocksize": BLOCKSIZE, "cells": cells}
    path = Path(args.out or f"phase1_{env['gpu'].replace(' ', '_')}.json")
    path.write_text(json.dumps(out, indent=1))
    print(f"receipts -> {path} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
