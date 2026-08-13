#!/usr/bin/env python3
"""Wiring + correctness gate for the unsloth head-to-head arms. NO TIMING.

Runs free on the QNAP A2000, which is a CORRECTNESS-ONLY testbed (operator
instruction 2026-07-27): a shared production box whose wall times are worthless
(same kernel/shape has measured 0.687 ms and 1.519 ms across runs). Everything
here is a numeric-agreement or contract check, so contention cannot corrupt it.

Purpose: reject a broken arm for $0 instead of discovering it on a rented H100.

Three gates, each printing its RAW numbers rather than a bare PASS, because a
gate whose control is uninstrumented is not a gate:

  G1  the OLD `unsloth` probe is provably dead (positive control for the
      finding that motivated this work) AND the new import path is live.
  G2  unsloth's kernel agrees with the fp64 reference on the SAME dequantized
      weights, to the same norm-relative metric the harness already uses.
  G3  the 4-bit-storage and bf16-resident arms compute the SAME answer as each
      other (they differ only in whether the dequant is inside the timed
      region), so any later timing gap between them is attributable to that
      boundary and nothing else.

Exit code is 0 iff all three pass. Usage:

    python bench/phase1/validate_unsloth_native.py [--models OLMoE ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import harness as H  # noqa: E402


def gate_1_probe_is_dead(stack=None, groups=None) -> dict:
    """Prove EMPIRICALLY that the old `unsloth` backend does not execute
    unsloth's kernel — by running it and reading back which impl it used.

    An earlier draft of this gate asserted the wrong mechanism. It claimed the
    empty `unsloth/kernels/moe/__init__.py` makes
    `getattr(unsloth.kernels.moe, "grouped_gemm")` return None, so the probe
    skips silently. The files really are 0 bytes, but the inference was wrong:
    Python binds a submodule onto its parent package as an attribute when the
    submodule is imported, empty __init__ or not. The attribute therefore
    exists — as a MODULE.

    The real mechanism has two independent parts, either one sufficient:
      (a) `bk_unsloth` tries tgale96's `grouped_gemm.ops.gmm` FIRST and returns
          on success, so with that package installed the unsloth probes are
          never reached at all; and
      (b) if it did reach them, `getattr(...)` yields a module, and calling a
          module raises TypeError — so it could never have invoked their kernel
          in any environment.

    Asserting the OUTCOME (which impl ran) instead of the mechanism is what
    this gate now does, because the outcome is what the claim rests on."""
    import importlib

    out = {}
    moe = importlib.import_module("unsloth.kernels.moe")
    attr = getattr(moe, "grouped_gemm", None)
    out["unsloth.kernels.moe.__init__ bytes"] = Path(moe.__file__).stat().st_size
    out["getattr(moe,'grouped_gemm')"] = repr(attr)
    out["attr_is_callable"] = callable(attr)
    # (b): the old probe's expectation — a callable at that name — is
    # unsatisfiable. Not None, but not callable either.
    out["old_probe_expectation_unsatisfiable"] = not callable(attr)

    # (a) + the decisive check: actually RUN the old backend and see what it
    # used. This is the assertion that matters; everything above is colour.
    old_impl, old_err = None, None
    if stack is not None:
        H.IMPL_NOTE.pop("unsloth", None)
        try:
            H.bk_unsloth(stack, groups)
            old_impl = H.IMPL_NOTE.get("unsloth")
        except Exception as e:
            old_err = f"{type(e).__name__}: {e}"
    out["old_backend_impl_actually_used"] = old_impl
    out["old_backend_error"] = old_err
    out["old_backend_ran_unsloths_kernel"] = bool(
        old_impl and "unsloth.kernels.moe" in old_impl
    )

    fn, _ = H._unsloth_native_fn()
    out["new_entry_resolves"] = callable(fn)
    out["new_entry"] = H._UNSLOTH_NATIVE_ENTRY
    out["fingerprint"] = H.unsloth_native_fingerprint()

    out["PASS"] = bool(
        out["old_probe_expectation_unsatisfiable"]
        and not out["old_backend_ran_unsloths_kernel"]
        and out["new_entry_resolves"]
    )
    return out


def gate_2_and_3(spec, device: str, regime: str) -> dict:
    """G2 numerics vs fp64; G3 the two arms agree with each other.

    Run per REGIME, not once: unsloth's kernel and gnf4's both dispatch
    differently at M=1 (decode) than at the wide M of a prefill tile, so a
    single-regime gate would leave half the compared surface unexercised."""
    stack = H.QuantStack(spec, device)
    out = {
        "model": spec.model,
        "proj": spec.proj,
        "N": spec.N,
        "K": spec.K,
        "regime": regime,
    }

    groups = H.make_activations(spec, regime, device)

    outs_4bit = H.bk_unsloth_native(stack, groups)
    outs_bf16 = H.bk_unsloth_native_bf16(stack, groups)
    outs_fused = H.bk_fused_nf4(stack, groups)
    outs_dq = H.bk_dequant_grouped(stack, groups)

    # G2 — norm-relative error vs the fp64 reference, the harness's own metric.
    # Per-element relative error explodes near zero on outputs that sum K
    # random-signed terms; a known-good kernel failing is what surfaced that.
    out["b_rel_unsloth_4bit"] = H.fidelity(stack, groups, outs_4bit)
    out["b_rel_unsloth_bf16"] = H.fidelity(stack, groups, outs_bf16)
    out["b_rel_fused"] = H.fidelity(stack, groups, outs_fused)
    out["b_rel_dequant"] = H.fidelity(stack, groups, outs_dq)

    # The bar is comparative, not a fixed epsilon: unsloth's kernel must land in
    # the same class as the dequant path it replaces. A fixed epsilon would
    # encode this box's noise into a portable contract.
    out["G2_PASS"] = bool(
        out["b_rel_unsloth_4bit"] <= 2.0 * out["b_rel_dequant"]
        and out["b_rel_unsloth_4bit"] < 1e-1
    )

    # G3 — the two arms must be bit-identical to each other. They call one
    # kernel on identical values; only the dequant's position relative to the
    # timed region differs. If these ever diverge, the caching in the resident
    # arm is aliasing something it should not, and the ceiling number would be
    # measuring a different computation than the 4-bit cell.
    deltas = [
        (a.to(torch.float64) - b.to(torch.float64)).abs().max().item()
        for a, b in zip(outs_4bit, outs_bf16)
    ]
    out["max_abs_delta_4bit_vs_bf16"] = max(deltas) if deltas else None
    out["G3_PASS"] = bool(deltas) and max(deltas) == 0.0

    del stack
    torch.cuda.empty_cache()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["OLMoE"])
    ap.add_argument("--regimes", nargs="*", default=["decode_bs1", "prefill_s2048"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("FATAL: no CUDA device; this gate is a real-silicon check")
        return 2
    device = "cuda"

    report = {
        "gpu": torch.cuda.get_device_name(0),
        "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
        "torch": torch.__version__,
        "NOTE": "correctness/wiring only — this box is not a timing testbed",
    }

    specs = H.census_specs(H.REPO / "census" / "shape_census.json", args.models)
    if not specs:
        print(f"FATAL: no census spec matched {args.models}")
        return 2

    print("=== G1: the old backend does not run unsloth's kernel ===")
    try:
        # Needs a real fixture: the decisive check RUNS the old backend.
        g1_stack = H.QuantStack(specs[0], device)
        g1_groups = H.make_activations(specs[0], "decode_bs1", device)
        g1 = gate_1_probe_is_dead(g1_stack, g1_groups)
        del g1_stack
        torch.cuda.empty_cache()
    except Exception as e:
        g1 = {"PASS": False, "error": f"{type(e).__name__}: {e}"}
    report["G1"] = g1
    print(json.dumps(g1, indent=2, default=str))

    print("\n=== G2/G3: numerics ===")
    cells = []
    for spec in specs:
        for regime in args.regimes:
            try:
                c = gate_2_and_3(spec, device, regime)
            except Exception as e:
                c = {
                    "model": spec.model,
                    "proj": spec.proj,
                    "regime": regime,
                    "G2_PASS": False,
                    "G3_PASS": False,
                    "error": f"{type(e).__name__}: {e}",
                }
            cells.append(c)
            print(json.dumps(c, indent=2, default=str))
    report["cells"] = cells

    ok = (
        g1.get("PASS", False)
        and cells
        and all(c.get("G2_PASS") and c.get("G3_PASS") for c in cells)
    )
    report["VERDICT"] = "PASS" if ok else "FAIL"
    print(f"\nVERDICT: {report['VERDICT']}")

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, default=str))
        print(f"wrote {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
