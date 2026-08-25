# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Verdict calculator for PREREG-k6-bespoke-gemv."""

import argparse
import json
import sys

# AMENDMENT-k6-frame: the decision bars are the RATIOS the prereg's
# own parentheticals registered ("(>= 2x over baseline)" and
# "(20-50% gain)"); the absolute forms repeated the B2 frame defect
# and are retired. Receipts carrying baseline_pair_us adjudicate on
# the ratio; the absolute constants remain only for legacy receipts.
PASS_RATIO = 0.50
PARTIAL_RATIO = 0.80
PASS_US = 36.0
PARTIAL_US = 58.0
# AMENDMENT (RESULTS-k6-stageA): the correctness gate for bf16-input
# MMA is RELATIVE -- max|delta| <= max|ref| * 2**-7 -- because the
# registered mechanism rounds both operands to bf16 before the MMA and
# an absolute 1e-2 was unachievable by construction. Receipts must
# carry max_abs_ref; argmax agreement >= 0.99 is recorded alongside.
REL_BAR = 2.0 ** -7
ARGMAX_FLOOR = 0.99


def verdict(rep):
    s = rep.get("summary") or {}
    if not s.get("noise_gate_pass"):
        return ("REFUSE", "noise gate failed")
    base = s.get("baseline_pair_us")
    if not base or base <= 0:
        return ("REFUSE", "no baseline pair")
    for name, c in (rep.get("cells") or {}).items():
        g = c.get("gate")
        if g and "max_abs_ref" in g:
            rel_ok = (g["max_abs_delta"]
                      <= g["max_abs_ref"] * REL_BAR)
            am_ok = (g["argmax_agree"] / max(1, g["argmax_total"])
                     >= ARGMAX_FLOOR)
            g = dict(g)
            g["pass"] = bool(rel_ok and am_ok)
            c = dict(c)
            c["gate"] = g
            if not g["pass"] and c.get("dot_pad_best"):
                return ("REFUSE", f"{name}: amended relative gate "
                        f"failed (max|d| {g['max_abs_delta']:.3e} vs "
                        f"budget {g['max_abs_ref'] * REL_BAR:.3e}, "
                        f"argmax {g['argmax_agree']}/{g['argmax_total']})")
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
    if base:
        r = p / base
        if r <= PASS_RATIO:
            return ("PASS", f"pair ratio {r:.3f} <= {PASS_RATIO} "
                    f"({p:.1f} vs {base:.1f} us) -- register K6-B")
        if r <= PARTIAL_RATIO:
            return ("PARTIAL", f"pair ratio {r:.3f} in ({PASS_RATIO}, "
                    f"{PARTIAL_RATIO}] ({p:.1f} vs {base:.1f} us) -- "
                    "K6-B under the A/A condition")
        return ("REFUTED", f"pair ratio {r:.3f} > {PARTIAL_RATIO}: the "
                "compute wall stands; the kernel lane closes")
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
        # ratio frame (baseline present): 30/72.4=0.414 PASS;
        # 36.2/72.4=0.5 boundary PASS; 45/72.4=0.62 PARTIAL;
        # 57.9/72.4=0.7997 boundary PARTIAL; 65/72.4=0.898 REFUTED
        (_fab(30.0), "PASS"),
        (_fab(36.2), "PASS"),          # ratio boundary 0.50
        (_fab(45.0), "PARTIAL"),
        (_fab(57.9), "PARTIAL"),       # ratio boundary 0.80
        (_fab(65.0), "REFUTED"),
        # the re-gate receipts themselves, both boxes
        (_fab(71.2, base=120.9), "PARTIAL"),
        (_fab(46.4, base=69.5), "PARTIAL"),
        (_fab(30.0, noise=False), "REFUSE"),
        (_fab(30.0, gate=False), "REFUSE"),
        # one cell with NO passing config: the pair is incomplete and
        # the other cell's time must not be judged as the pair
        (_fab(30.0, missing_cell=True), "REFUSE"),
        (dict(_fab(30.0), summary={"baseline_pair_us": None,
                                   "dot_pad_pair_us": 30.0,
                                   "noise_gate_pass": True}), "REFUSE"),
    ]
    # amended relative gate: 0.5 delta on a 120-magnitude ref is INSIDE
    # the bf16-MMA budget (120 * 2^-7 = 0.9375) and must pass; the same
    # delta on a 30-magnitude ref is outside and must refuse
    ok = _fab(46.4)
    for c in ok["cells"].values():
        c["gate"] = {"pass": False, "max_abs_delta": 0.5,
                     "max_abs_ref": 120.0,
                     "argmax_agree": 4070, "argmax_total": 4096}
    got, why = verdict(ok)
    assert got == "PARTIAL", (got, why)
    bad = _fab(46.4)
    for c in bad["cells"].values():
        c["gate"] = {"pass": False, "max_abs_delta": 0.5,
                     "max_abs_ref": 30.0,
                     "argmax_agree": 4070, "argmax_total": 4096}
    got, why = verdict(bad)
    assert got == "REFUSE", (got, why)
    cases += [(None, None)] * 0
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
