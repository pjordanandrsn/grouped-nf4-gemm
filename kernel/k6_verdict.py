# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Verdict calculator for PREREG-k6-bespoke-gemv."""

import argparse
import json
import sys

PASS_US = 36.0
PARTIAL_US = 58.0


def verdict(rep):
    s = rep.get("summary") or {}
    if not s.get("noise_gate_pass"):
        return ("REFUSE", "noise gate failed")
    base = s.get("baseline_pair_us")
    if not base or base <= 0:
        return ("REFUSE", "no baseline pair")
    for name, c in (rep.get("cells") or {}).items():
        g = c.get("gate")
        if c.get("dot_pad_best") is None:
            return ("REFUSE", f"{name}: no config passed its "
                    "correctness gate -- the census pair is INCOMPLETE "
                    "and a single cell must not be judged as the pair")
        if not g or not g.get("pass"):
            return ("REFUSE", f"{name}: best dot-pad config failed its "
                    "correctness gate -- timing an incorrect kernel "
                    "certifies nothing")
    p = s.get("dot_pad_pair_us")
    if not p:
        return ("REFUSE", "no gated dot-pad pair time")
    if p <= PASS_US:
        return ("PASS", f"pair {p:.1f} us <= {PASS_US} (baseline "
                f"{base:.1f}) -- register K6-B productization")
    if p <= PARTIAL_US:
        # A/A condition rides the per-cell drift gates already enforced
        return ("PARTIAL", f"pair {p:.1f} us in ({PASS_US}, {PARTIAL_US}]"
                f" (baseline {base:.1f}) -- K6-B only if gain is real "
                "under A/A")
    return ("REFUTED", f"pair {p:.1f} us > {PARTIAL_US}: the compute "
            "wall stands against tensor cores with floor loads; the "
            "kernel lane closes")


def _fab(pair, base=72.4, noise=True, gate=True, missing_cell=False):
    cell = lambda: {"noise_gate_pass": noise,                # noqa: E731
                    "dot_pad_best": {"us": pair / 2},
                    "gate": {"pass": gate}}
    cells = {"gate_up": cell(), "down": cell()}
    if missing_cell:
        cells["down"] = {"noise_gate_pass": noise,
                         "dot_pad_best": None, "gate": None}
    return {"cells": cells,
            "summary": {"baseline_pair_us": base,
                        "dot_pad_pair_us": (None if (not gate
                                                     or missing_cell)
                                            else pair),
                        "noise_gate_pass": noise}}


def self_test():
    cases = [
        (_fab(30.0), "PASS"),
        (_fab(36.0), "PASS"),          # boundary
        (_fab(45.0), "PARTIAL"),
        (_fab(58.0), "PARTIAL"),       # boundary
        (_fab(65.0), "REFUTED"),
        (_fab(30.0, noise=False), "REFUSE"),
        (_fab(30.0, gate=False), "REFUSE"),
        # one cell with NO passing config: the pair is incomplete and
        # the other cell's time must not be judged as the pair
        (_fab(30.0, missing_cell=True), "REFUSE"),
        (dict(_fab(30.0), summary={"baseline_pair_us": None,
                                   "dot_pad_pair_us": 30.0,
                                   "noise_gate_pass": True}), "REFUSE"),
    ]
    for rep, want in cases:
        got, why = verdict(rep)
        assert got == want, (got, want, why)
    print(f"self-test PASS ({len(cases)} cases: all three outcomes, "
          "both boundaries, every refusal)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report", nargs="?")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return
    if not a.report:
        sys.exit("need a report (or --self-test)")
    v, why = verdict(json.loads(open(a.report).read()))
    print(f"K6 VERDICT: {v}\n  {why}")
    if v == "REFUSE":
        sys.exit(2)


if __name__ == "__main__":
    main()
