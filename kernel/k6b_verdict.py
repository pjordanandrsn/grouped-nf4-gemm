# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Verdict calculator for PREREG-k6b-productize Stage A."""

import argparse
import json
import sys
from collections import Counter

PASS_RATIO = 0.85
PARTIAL_RATIO = 0.95
MIN_DIVERGE_STEP = 32
GRAM, MAX_REP = 8, 6


def _degenerate(t):
    if len(set(t)) < 30:
        return f"only {len(set(t))} distinct tokens"
    grams = Counter(tuple(t[i:i + GRAM]) for i in range(len(t) - GRAM + 1))
    worst = max(grams.values(), default=0)
    if worst > MAX_REP:
        return f"an {GRAM}-gram repeats {worst}x"
    return None


def verdict(rep):
    off_a, off_b = rep.get("off_a") or {}, rep.get("off_b") or {}
    on = rep.get("on") or {}
    for name, arm in (("off_a", off_a), ("off_b", off_b), ("on", on)):
        if not arm.get("step_ms_clean") or not arm.get("tokens"):
            return ("REFUSE", f"{name}: missing step or tokens")
    t_off, t_on = off_a["tokens"], on["tokens"]
    why = _degenerate(t_off)
    if why:
        return ("REFUSE", f"OFF trace degenerate: {why} -- the quality "
                "gate would be vacuous")
    if off_a["tokens"] != off_b["tokens"]:
        return ("REFUSE", "OFF arms not token-identical -- the box is "
                "not deterministic and divergence cannot be attributed")
    base = min(off_a["step_ms_clean"], off_b["step_ms_clean"])
    spread = abs(off_a["step_ms_clean"] - off_b["step_ms_clean"])
    ratio = on["step_ms_clean"] / base
    if spread >= base * (1 - PASS_RATIO) / 2:
        return ("REFUSE", f"A/A spread {spread:.3f} ms >= half the PASS "
                "margin -- the frame cannot resolve the gain")
    n = min(len(t_off), len(t_on))
    div = next((i for i in range(n) if t_off[i] != t_on[i]), None)
    agree_to_div = div if div is not None else n
    quality_ok = div is None or div >= MIN_DIVERGE_STEP
    q = (f"first divergence at step "
         f"{'NONE' if div is None else div} of {n}, agreement "
         f"{agree_to_div}/{n}")
    if not quality_ok:
        return ("REFUTED-QUALITY", f"{q} -- diverging inside "
                f"{MIN_DIVERGE_STEP} steps is too hot to ship")
    if ratio <= PASS_RATIO:
        return ("PASS", f"step ratio {ratio:.3f} <= {PASS_RATIO} "
                f"({on['step_ms_clean']:.2f} vs {base:.2f} ms); {q} -- "
                "default flips ON with disclosure")
    if ratio <= PARTIAL_RATIO:
        return ("PARTIAL", f"step ratio {ratio:.3f} in ({PASS_RATIO}, "
                f"{PARTIAL_RATIO}]; {q} -- knob ships OFF, available")
    return ("REFUTED", f"step ratio {ratio:.3f} > {PARTIAL_RATIO}: the "
            "kernel's us win does not survive the full step")


def _fab(ratio, div=None, aa=0.005, base=7.39, off_ident=True,
         degen=False):
    t = ([1, 2] * 64) if degen else [(i * 37 + 11) % 997
                                     for i in range(128)]
    ton = list(t)
    if div is not None:
        ton[div] = -1
    return {"off_a": {"step_ms_clean": base, "tokens": list(t)},
            "off_b": {"step_ms_clean": base + aa,
                      "tokens": list(t) if off_ident else list(t)[::-1]},
            "on": {"step_ms_clean": base * ratio, "tokens": ton}}


def self_test():
    cases = [
        (_fab(0.82), "PASS"),
        (_fab(0.85), "PASS"),                       # boundary
        (_fab(0.90), "PARTIAL"),
        (_fab(0.95), "PARTIAL"),                    # boundary
        (_fab(0.97), "REFUTED"),
        (_fab(0.82, div=100), "PASS"),              # late divergence ok
        (_fab(0.82, div=31), "REFUTED-QUALITY"),    # hot divergence
        (_fab(0.82, div=5), "REFUTED-QUALITY"),
        (_fab(0.82, aa=0.9), "REFUSE"),             # frame cannot resolve
        (_fab(0.82, off_ident=False), "REFUSE"),
        (_fab(0.82, degen=True), "REFUSE"),
    ]
    for rep, want in cases:
        got, why = verdict(rep)
        assert got == want, (got, want, why)
    print(f"self-test PASS ({len(cases)} cases: all outcomes, both "
          "ratio boundaries, the divergence floor, and every refusal)")


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
    print(f"K6-B VERDICT: {v}\n  {why}")
    if v == "REFUSE":
        sys.exit(2)


if __name__ == "__main__":
    main()
