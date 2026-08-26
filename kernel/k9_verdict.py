# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Verdict calculator for PREREG-k9-fused-decode-grouping.

Report shape (composed on the box):
  {"stage_a": {"argsort_calls_per_step": int, "layers": int,
               "builder_us_per_call": float, "builder_ms_per_step": float,
               "launches_per_call": int},
   "correctness": {"exhaustive_pass": bool, "random_pass": bool,
                   "edges_pass": bool, "order_matches_stable": bool,
                   "bitwise_repeat": bool, "cases": int},
   "fused": {"us_per_call": float},
   "step": {"base_a": ARM, "base_b": ARM, "fused_a": ARM, "fused_b": ARM},
   "cert_knob_ms": float}
  ARM = {"step_ms_clean": float, "tokens": [int, ...]}

Bars are FRACTIONS of Stage A's own measurement -- X is unknown until
the box runs, so an absolute bar would have been a guess (the K7
posture).
"""

import argparse
import json
import math
import sys

PASS_CUT = 0.60
PARTIAL_CUT = 0.30
AA_TOL = 0.02
ANCHOR_TOL = 0.05


def _pos_finite(v):
    return isinstance(v, (int, float)) and math.isfinite(v) and v > 0


def verdict(rep):
    out = {"gates": {}, "stage_a": {}, "verdict": None}

    a = rep.get("stage_a") or {}
    layers = a.get("layers")
    calls = a.get("argsort_calls_per_step")
    if not isinstance(layers, int) or layers <= 0:
        return _refuse(out, "stage A: layer count missing")
    if calls != layers:
        return _refuse(out, f"stage A attribution: {calls} argsort "
                            f"calls/step but {layers} layers -- the "
                            "bitonic sort is NOT this builder, so the "
                            "premise is wrong; refusing rather than "
                            "proceeding on a mis-attribution")
    for k in ("builder_us_per_call", "builder_ms_per_step"):
        if not _pos_finite(a.get(k)):
            return _refuse(out, f"stage A: {k} missing or non-positive")
    out["stage_a"] = {"builder_ms_per_step": a["builder_ms_per_step"],
                      "us_per_call": a["builder_us_per_call"],
                      "launches_per_call": a.get("launches_per_call")}

    c = rep.get("correctness") or {}
    for k in ("exhaustive_pass", "random_pass", "edges_pass",
              "order_matches_stable", "bitwise_repeat"):
        if k not in c:
            return _refuse(out, f"correctness: {k} not reported")
        if not c[k]:
            return _refuse(out, f"G1/G2/G3: {k} FAILED -- the fused "
                                "builder must be bitwise-identical; "
                                "there is no tolerance band because "
                                "there is no rounding in the mechanism")
    if not isinstance(c.get("cases"), int) or c["cases"] < 1:
        return _refuse(out, "correctness: no cases reported -- a gate "
                            "that ran zero cases passes vacuously")

    for label, (x, y) in (("base", ("base_a", "base_b")),
                          ("fused", ("fused_a", "fused_b"))):
        s = rep.get("step") or {}
        for k in (x, y):
            arm = s.get(k)
            if not arm or not _pos_finite(arm.get("step_ms_clean")):
                return _refuse(out, f"G-A/A: arm {k} missing or "
                                    "non-positive")
        p, q = s[x]["step_ms_clean"], s[y]["step_ms_clean"]
        spread = abs(p - q) / min(p, q)
        out["gates"][f"{label}_aa"] = spread
        if spread > AA_TOL:
            return _refuse(out, f"G-A/A: {label} spread "
                                f"{spread * 100:.2f}% > {AA_TOL * 100:.0f}%")
        if s[x].get("tokens") != s[y].get("tokens"):
            return _refuse(out, f"G-A/A: {label} token streams differ")

    s = rep["step"]
    base = (s["base_a"]["step_ms_clean"] + s["base_b"]["step_ms_clean"]) / 2
    fused = (s["fused_a"]["step_ms_clean"]
             + s["fused_b"]["step_ms_clean"]) / 2
    cert = rep.get("cert_knob_ms")
    if not _pos_finite(cert):
        return _refuse(out, "anchor: cert_knob_ms missing or non-finite")
    drift = abs(base - cert) / cert
    out["gates"]["anchor_drift"] = drift
    if drift > ANCHOR_TOL:
        return _refuse(out, f"anchor: base {base:.3f} ms is "
                            f"{drift * 100:.1f}% off the certified "
                            f"{cert:.3f} ms")

    f = rep.get("fused") or {}
    if not _pos_finite(f.get("us_per_call")):
        return _refuse(out, "no fused builder per-call time")
    cut = 1.0 - f["us_per_call"] / a["builder_us_per_call"]
    out["gates"]["builder_cut_frac"] = cut
    out["gates"]["step_base_ms"] = base
    out["gates"]["step_fused_ms"] = fused
    out["gates"]["step_delta_ms"] = base - fused

    if cut >= PASS_CUT:
        out["verdict"] = ("PASS", cut)
    elif cut >= PARTIAL_CUT:
        out["verdict"] = ("PARTIAL", cut)
    else:
        out["verdict"] = ("REFUTED", cut)
    return out


def _refuse(out, why):
    out["verdict"] = ("REFUSE", why)
    return out


def render(out):
    tag, x = out["verdict"]
    if tag == "REFUSE":
        return f"K9 VERDICT: REFUSE\n  {x}"
    g, a = out["gates"], out["stage_a"]
    return "\n".join([
        f"K9 VERDICT: {tag}  (builder cut {x * 100:.1f}%; "
        f"PASS >= {PASS_CUT * 100:.0f}%, PARTIAL >= {PARTIAL_CUT * 100:.0f}%)",
        f"  stage A: builder {a['us_per_call']:.2f} us/call x layers "
        f"= {a['builder_ms_per_step']:.3f} ms/step "
        f"({a['launches_per_call']} launches/call)",
        f"  step (RECORDED, not the bar): {g['step_base_ms']:.3f} -> "
        f"{g['step_fused_ms']:.3f} ms = {g['step_delta_ms']:+.3f} ms",
    ])


def _mk(fused_us=1.0, base_us=6.0, calls=48, layers=48, aa=0.0,
        tok_b=None, cert=6.476, base=6.46, cases=64, **flags):
    c = {"exhaustive_pass": True, "random_pass": True, "edges_pass": True,
         "order_matches_stable": True, "bitwise_repeat": True,
         "cases": cases}
    c.update(flags)
    toks = list(range(30))
    return {"stage_a": {"argsort_calls_per_step": calls, "layers": layers,
                        "builder_us_per_call": base_us,
                        "builder_ms_per_step": base_us * layers / 1000,
                        "launches_per_call": 26},
            "correctness": c,
            "fused": {"us_per_call": fused_us},
            "step": {"base_a": {"step_ms_clean": base, "tokens": toks},
                     "base_b": {"step_ms_clean": base * (1 + aa),
                                "tokens": tok_b if tok_b is not None else toks},
                     "fused_a": {"step_ms_clean": base - 0.2, "tokens": toks},
                     "fused_b": {"step_ms_clean": base - 0.2, "tokens": toks}},
            "cert_knob_ms": cert}


def self_test():
    # Assert either SIDE of each bar, never the knife edge: the cut is
    # a ratio of binary floats, and 1 - 4.2/6.0 evaluates to
    # 0.2999999999999999 -- a fixture sitting exactly on 0.30 would be
    # testing float representation, not the rule (same trap as K8).
    assert verdict(_mk(fused_us=1.0))["verdict"][0] == "PASS"     # 83%
    assert verdict(_mk(fused_us=2.3))["verdict"][0] == "PASS"     # 62%
    assert verdict(_mk(fused_us=2.5))["verdict"][0] == "PARTIAL"  # 58%
    assert verdict(_mk(fused_us=4.1))["verdict"][0] == "PARTIAL"  # 32%
    assert verdict(_mk(fused_us=4.3))["verdict"][0] == "REFUTED"  # 28%
    assert verdict(_mk(fused_us=7.0))["verdict"][0] == "REFUTED"  # slower
    # the attribution gate: a sort that is NOT this builder refuses
    r = verdict(_mk(calls=96))
    assert r["verdict"][0] == "REFUSE" and "attribution" in r["verdict"][1], r
    # every correctness flag refuses, including a vacuous zero-case gate
    for flag in ("exhaustive_pass", "random_pass", "edges_pass",
                 "order_matches_stable", "bitwise_repeat"):
        r = verdict(_mk(**{flag: False}))
        assert r["verdict"][0] == "REFUSE" and flag in r["verdict"][1], flag
    assert verdict(_mk(cases=0))["verdict"][1].startswith("correctness")
    for bad, why in ((_mk(aa=0.03), "G-A/A"),
                     (_mk(tok_b=[9]), "G-A/A"),
                     (_mk(cert=7.39), "anchor"),
                     (_mk(cert=float("nan")), "anchor"),
                     (_mk(base_us=0.0), "stage A"),
                     (_mk(fused_us=0.0), "no fused"),
                     (_mk(layers=0), "stage A")):
        r = verdict(bad)
        assert r["verdict"][0] == "REFUSE" and \
            r["verdict"][1].startswith(why), (why, r["verdict"])
    for line in render(verdict(_mk())).splitlines():
        print(f"[SELF-TEST FIXTURE, NOT A RESULT] {line}")
    print("k9_verdict self-test OK (both band boundaries, the "
          "attribution gate, five correctness flags, a vacuous "
          "zero-case gate, and seven refusal directions)")


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
    out = verdict(json.load(open(a.report)))
    print(render(out))
    if out["verdict"][0] == "REFUSE":
        sys.exit(3)


if __name__ == "__main__":
    main()
