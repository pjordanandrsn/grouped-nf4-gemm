# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Verdict calculator for PREREG-k5-mtile-probe. Thresholds are fixed by
the prereg; run --self-test before pointing it at receipts."""

import argparse
import json
import sys

RATIO_WIN = 0.6      # mtile_sum <= 0.6 * gemv_sum  -> MTILE-WINS
RATIO_REFUTE = 0.9   # mtile_sum >= 0.9 * gemv_sum  -> STRUCTURE-REFUTED
NOISE_PCT = 5.0


def verdict(rep):
    for name, cell in rep["cells"].items():
        if not cell.get("noise_gate_pass"):
            return ("REFUSE", f"noise gate failed at {name}: "
                    f"{cell.get('noise_drift_pct'):.2f}% > {NOISE_PCT}%")
        if cell.get("mtile_best") is None:
            return ("REFUSE", f"no successful M-tile config at {name}")
        if cell.get("gemv_us", 0) <= 0:
            return ("REFUSE", f"non-positive GEMV time at {name}")
    s = rep["summary"]
    r = s.get("ratio_mtile_over_gemv")
    if r is None or r <= 0:
        return ("REFUSE", "summary ratio missing or non-positive")
    if r <= RATIO_WIN:
        return ("MTILE-WINS",
                f"ratio {r:.3f} <= {RATIO_WIN}: register K5-B "
                "(decode_via_mtile routing knob, P-fid gates)")
    if r >= RATIO_REFUTE:
        return ("STRUCTURE-REFUTED",
                f"ratio {r:.3f} >= {RATIO_REFUTE}: existing M-tile path "
                "does not beat the GEMV at M=1; a bespoke tl.dot GEMV "
                "needs its own prereg")
    return ("INCONCLUSIVE-PAUSE",
            f"ratio {r:.3f} in ({RATIO_WIN}, {RATIO_REFUTE}): kernel lane "
            "pauses; elementwise-fusion lane takes priority")


def _fab(ratio, noise=1.0, best=True):
    g = 70.0
    cell = lambda gu: {"gemv_us": gu, "noise_drift_pct": noise,  # noqa: E731
                       "noise_gate_pass": noise <= NOISE_PCT,
                       "mtile_best": {"us": gu * ratio} if best else None}
    return {"cells": {"gate_up": cell(g * 0.6), "down": cell(g * 0.4)},
            "summary": {"gemv_sum_us": g, "mtile_sum_us": g * ratio,
                        "ratio_mtile_over_gemv": ratio,
                        "noise_gate_pass": noise <= NOISE_PCT}}


def self_test():
    cases = [
        (_fab(0.45), "MTILE-WINS"),
        (_fab(0.60), "MTILE-WINS"),        # boundary: <= wins
        (_fab(0.75), "INCONCLUSIVE-PAUSE"),
        (_fab(0.90), "STRUCTURE-REFUTED"),  # boundary: >= refutes
        (_fab(1.30), "STRUCTURE-REFUTED"),
        (_fab(0.45, noise=7.2), "REFUSE"),
        (_fab(0.45, best=False), "REFUSE"),
    ]
    for rep, want in cases:
        got, why = verdict(rep)
        assert got == want, (got, want, why)
    print(f"self-test PASS ({len(cases)} cases, both directions + refusals)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report", nargs="?")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.report:
        sys.exit("need a report path (or --self-test)")
    rep = json.loads(open(args.report).read())
    v, why = verdict(rep)
    s = rep["summary"]
    mt = s.get("mtile_sum_us")
    mt_s = f"{mt:.1f}us" if mt is not None else "n/a"
    print(f"K5 VERDICT: {v}\n  {why}\n"
          f"  gemv_sum={s['gemv_sum_us']:.1f}us mtile_sum={mt_s}")
    if v == "REFUSE":
        sys.exit(2)


if __name__ == "__main__":
    main()
