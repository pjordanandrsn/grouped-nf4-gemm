# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Verdict calculator for PREREG-k7-gemv-round2.

Report shape (written by k7_bench.py on the box):
  {"summary": {"noise_gate_pass": bool,
               "dotpad_pair_us": float,     # same-box K6-B denominator
               "candidate_pair_us": float}, # best gated candidate
   "cells": {"gate_up": CELL, "down": CELL}}
  CELL = {"gate": {"max_abs_delta", "max_abs_ref",
                   "argmax_agree", "argmax_total",
                   "bitwise_repeat": bool},   # G2: two runs torch.equal
          "dotpad_us": float, "best_us": float}

Bars are RATIOS vs the same-box dot-pad pair (bars-follow-the-claim;
the SV2 frame registered the slice target): PASS <= 0.39,
PARTIAL <= 0.60, else REFUTED. Gates G1-G4 REFUSE per the prereg.
"""

import argparse
import json
import sys

REL_BAR = 2.0 ** -7          # AMENDMENT RESULTS-k6-stageA (same mechanism)
ARGMAX_FLOOR = 0.99
PASS_RATIO = 0.39            # SV2 frame: slice 2.469 -> <= 0.97 ms
PARTIAL_RATIO = 0.60
CELLS = ("gate_up", "down")


def verdict(rep):
    s = rep.get("summary") or {}
    if not s.get("noise_gate_pass"):
        return ("REFUSE", "G3: harness noise gate failed")
    for name in CELLS:
        c = (rep.get("cells") or {}).get(name)
        if not c:
            return ("REFUSE", f"G4: cell {name} missing -- the census "
                    "pair is incomplete and one cell must not be "
                    "judged as the pair")
        g = c.get("gate") or {}
        for k in ("max_abs_delta", "max_abs_ref", "argmax_agree",
                  "argmax_total", "bitwise_repeat"):
            if k not in g:
                return ("REFUSE", f"G1/G2: cell {name} gate missing "
                        f"{k}")
        if g["max_abs_delta"] > g["max_abs_ref"] * REL_BAR:
            return ("REFUSE", f"G1: {name} max|d| "
                    f"{g['max_abs_delta']:.3e} exceeds the mechanism "
                    f"band {g['max_abs_ref'] * REL_BAR:.3e} -- timing "
                    "an incorrect kernel certifies nothing")
        if g["argmax_agree"] / max(1, g["argmax_total"]) < ARGMAX_FLOOR:
            return ("REFUSE", f"G1: {name} argmax "
                    f"{g['argmax_agree']}/{g['argmax_total']} below "
                    f"{ARGMAX_FLOOR}")
        if not g["bitwise_repeat"]:
            return ("REFUSE", f"G2: {name} is not run-to-run bitwise "
                    "deterministic -- an order-nondeterministic "
                    "reduction breaks every downstream identity gate")
        if not c.get("dotpad_us") or c["dotpad_us"] <= 0:
            return ("REFUSE", f"G4: cell {name} has no same-box "
                    "dot-pad denominator")
    base = s.get("dotpad_pair_us")
    cand = s.get("candidate_pair_us")
    if not base or base <= 0:
        return ("REFUSE", "G4: no same-box dot-pad pair denominator")
    if not cand or cand <= 0:
        return ("REFUSE", "no gated candidate pair time")
    r = cand / base
    if r <= PASS_RATIO:
        return ("PASS", r)
    if r <= PARTIAL_RATIO:
        return ("PARTIAL", r)
    return ("REFUTED", r)


def render(v, rep):
    tag, x = v
    if tag == "REFUSE":
        return f"K7 VERDICT: REFUSE\n  {x}"
    s = rep["summary"]
    lines = [f"K7 VERDICT: {tag}  (ratio {x:.3f} = "
             f"{s['candidate_pair_us']:.1f} / {s['dotpad_pair_us']:.1f}"
             f" us; PASS <= {PASS_RATIO}, PARTIAL <= {PARTIAL_RATIO})"]
    for name in CELLS:
        c = rep["cells"][name]
        lines.append(f"  {name}: {c['dotpad_us']:.1f} -> "
                     f"{c['best_us']:.1f} us")
    if tag == "REFUTED":
        lines.append("  occupancy was not the binding constraint; "
                     "250-by-composition loses its dominant slice "
                     "(PREREG-k7)")
    return "\n".join(lines)


def _mk(ratio=0.35, noise=True, delta_x=1.0, agree=100, bitwise=True,
        drop_cell=None, dotpad_cell=60.0):
    base = 100.0
    rep = {"summary": {"noise_gate_pass": noise,
                       "dotpad_pair_us": base,
                       "candidate_pair_us": base * ratio},
           "cells": {}}
    for name in CELLS:
        rep["cells"][name] = {
            "gate": {"max_abs_delta": 0.004 * delta_x,
                     "max_abs_ref": 1.0,
                     "argmax_agree": agree, "argmax_total": 100,
                     "bitwise_repeat": bitwise},
            "dotpad_us": dotpad_cell,
            "best_us": dotpad_cell * ratio}
    if drop_cell:
        del rep["cells"][drop_cell]
    return rep


def self_test():
    assert verdict(_mk(0.35))[0] == "PASS"
    assert verdict(_mk(0.39))[0] == "PASS"        # boundary holds
    assert verdict(_mk(0.391))[0] == "PARTIAL"
    assert verdict(_mk(0.60))[0] == "PARTIAL"
    assert verdict(_mk(0.601))[0] == "REFUTED"
    assert verdict(_mk(0.75))[0] == "REFUTED"
    # 2^-7 band: delta 0.004 passes vs ref 1.0 (budget 7.8e-3); 3x fails
    assert verdict(_mk(delta_x=3.0))[1].startswith("G1")
    assert verdict(_mk(agree=98))[1].startswith("G1")
    assert verdict(_mk(bitwise=False))[1].startswith("G2")
    assert verdict(_mk(noise=False))[1].startswith("G3")
    assert verdict(_mk(drop_cell="down"))[1].startswith("G4")
    assert verdict(_mk(dotpad_cell=0.0))[1].startswith("G4")
    bad = _mk()
    del bad["cells"]["gate_up"]["gate"]["bitwise_repeat"]
    assert verdict(bad)[1].startswith("G1/G2")
    r = _mk(0.35)
    print(render(verdict(r), r))
    print("k7_verdict self-test OK (bands, both boundaries, and seven "
          "refusal directions)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report", nargs="?")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return
    if not a.report:
        ap.error("report path or --self-test")
    rep = json.load(open(a.report))
    v = verdict(rep)
    print(render(v, rep))
    if v[0] == "REFUSE":
        sys.exit(3)


if __name__ == "__main__":
    main()
