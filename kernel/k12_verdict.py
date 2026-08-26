# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Verdict calculator for PREREG-k12-moe-tier-fusion.

Bars are ABSOLUTE ms cuts off the step, because that is the unit the
certified ladder is denominated in -- and because fewer launches that
do not move the step is not a win.

Report shape:
  {"arms": {"both_disabled": ARM, "moe_compiled": ARM,
            "both_compiled": ARM | None},
   "census": {"before": {row: calls}, "after": {row: calls}},
   "cert_gate": [lo, hi]}
  ARM = {"a": float, "b": float, "tokens_a": [...], "tokens_b": [...],
         "recompiles": int, "error": str | None}
"""

import argparse
import json
import math
import sys

PASS_MS = 0.40
PARTIAL_MS = 0.15
AA_TOL = 0.02
# the raw-ATen rows the census attributes to the excluded region
TRACKED = ("unrolled_elementwise", "indexSelect", "elementwise_kernel",
           "reduce_kernel")


def _pos_finite(v):
    return isinstance(v, (int, float)) and math.isfinite(v) and v > 0


def verdict(rep):
    out = {"gates": {}, "verdict": None}
    arms = rep.get("arms") or {}
    for name in ("both_disabled", "moe_compiled"):
        a = arms.get(name)
        if not a:
            return _refuse(out, f"arm {name} missing")
        if a.get("error"):
            return _refuse(out, f"{name} raised: {str(a['error'])[:90]}")
        for k in ("a", "b"):
            if not _pos_finite(a.get(k)):
                return _refuse(out, f"{name}: step {k} missing or "
                                    "non-positive")
        spread = abs(a["a"] - a["b"]) / min(a["a"], a["b"])
        out["gates"][f"{name}_aa"] = spread
        if spread > AA_TOL:
            return _refuse(out, f"{name}: A/A spread "
                                f"{spread * 100:.2f}% > {AA_TOL * 100:.0f}%")
        if a.get("tokens_a") != a.get("tokens_b"):
            return _refuse(out, f"{name}: token streams differ between "
                                "identical runs")
        if a.get("recompiles", 0) != 0:
            return _refuse(out, f"{name}: {a['recompiles']} recompiles "
                                "inside the timed window")

    # compiling a region must not change what the model says
    if arms["both_disabled"].get("tokens_a") != \
            arms["moe_compiled"].get("tokens_a"):
        return _refuse(out, "moe_compiled changed the greedy token "
                            "stream -- compiling a region must not "
                            "change what the model says")

    # ATTRIBUTION: the tracked raw-ATen rows must actually fall
    cen = rep.get("census") or {}
    before, after = cen.get("before"), cen.get("after")
    if not isinstance(before, dict) or not isinstance(after, dict):
        return _refuse(out, "no replay census for both arms -- the "
                            "895.5 us attribution is unproven without "
                            "it")
    moved = {}
    for row in TRACKED:
        b = sum(v for k, v in before.items() if row in k)
        a2 = sum(v for k, v in after.items() if row in k)
        moved[row] = {"before": b, "after": a2}
    out["gates"]["census"] = moved
    if not any(m["after"] < m["before"] for m in moved.values()):
        return _refuse(out, "attribution: no tracked raw-ATen row fell "
                            "in count, so any step delta came from "
                            "somewhere other than the mechanism this "
                            "cycle registered")

    gate = rep.get("cert_gate")
    if not (isinstance(gate, (list, tuple)) and len(gate) == 2
            and all(_pos_finite(g) for g in gate)):
        return _refuse(out, "no committed anchor gate")

    b = (arms["both_disabled"]["a"] + arms["both_disabled"]["b"]) / 2
    m = (arms["moe_compiled"]["a"] + arms["moe_compiled"]["b"]) / 2
    # The gate was being validated for SHAPE and then never applied --
    # an anchor check that does not check anything (caught by this
    # file's own self-test). The baseline is knob-ON, so it sits below
    # the knob-OFF gate by design; what the gate excludes is a box
    # whose knob-OFF class is an outlier, so compare the arm to the
    # gate SCALED by the certified knob ratio (6.476 / 7.369).
    KNOB = 6.476 / 7.369
    lo, hi = gate[0] * KNOB, gate[1] * KNOB
    out["gates"]["baseline_gate"] = [lo, hi]
    if not (lo <= b <= hi):
        return _refuse(out, f"anchor: knob-ON baseline {b:.3f} ms is "
                            f"outside [{lo:.3f}, {hi:.3f}] -- the "
                            "committed gate scaled by the certified "
                            "knob ratio")
    cut = b - m
    out["gates"].update({"baseline_ms": b, "compiled_ms": m,
                         "cut_ms": cut})

    bc = arms.get("both_compiled")
    if bc is not None:
        out["gates"]["both_compiled_raised"] = bool(bc.get("error"))

    if cut >= PASS_MS:
        out["verdict"] = ("PASS", cut)
    elif cut >= PARTIAL_MS:
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
        return f"K12 VERDICT: REFUSE\n  {x}"
    g = out["gates"]
    lines = [f"K12 VERDICT: {tag}  (cut {x:+.3f} ms; PASS >= {PASS_MS}, "
             f"PARTIAL >= {PARTIAL_MS})",
             f"  step {g['baseline_ms']:.3f} -> {g['compiled_ms']:.3f} ms"]
    for row, mv in g["census"].items():
        lines.append(f"    {row:<24} {mv['before']:>5.0f} -> "
                     f"{mv['after']:>5.0f} calls/step")
    if g.get("both_compiled_raised") is False:
        lines.append("  NOTE: both_compiled did NOT raise -- F1's "
                     "attention exclusion may itself be stale")
    return "\n".join(lines)


def _mk(cut=0.5, aa=0.001, tok_differ=False, rec=0, err=None,
        census_moves=True, gate=(7.004, 7.906), drop=None):
    t = list(range(30))
    base = 6.48
    def arm(ms, e=None, tk=None):
        return {"a": ms, "b": ms * (1 + aa), "tokens_a": t,
                "tokens_b": tk if tk is not None else t,
                "recompiles": rec, "error": e}
    arms = {"both_disabled": arm(base),
            "moe_compiled": arm(base - cut, err,
                                (t if not tok_differ else [9] + t[1:])),
            "both_compiled": arm(base, "CompilationError: m_i")}
    if tok_differ:
        arms["moe_compiled"]["tokens_a"] = [9] + t[1:]
        arms["moe_compiled"]["tokens_b"] = [9] + t[1:]
    if drop:
        del arms[drop]
    after = {"unrolled_elementwise_kernel": 40 if census_moves else 218,
             "indexSelectS": 145, "elementwise_kernel<128,4>": 96,
             "reduce_kernel": 48}
    return {"arms": arms, "cert_gate": list(gate),
            "census": {"before": {"unrolled_elementwise_kernel": 218,
                                  "indexSelectS": 145,
                                  "elementwise_kernel<128,4>": 96,
                                  "reduce_kernel": 48},
                       "after": after}}


def self_test():
    assert verdict(_mk(cut=0.50))["verdict"][0] == "PASS"
    assert verdict(_mk(cut=0.41))["verdict"][0] == "PASS"
    assert verdict(_mk(cut=0.30))["verdict"][0] == "PARTIAL"
    assert verdict(_mk(cut=0.16))["verdict"][0] == "PARTIAL"
    assert verdict(_mk(cut=0.10))["verdict"][0] == "REFUTED"
    assert verdict(_mk(cut=-0.20))["verdict"][0] == "REFUTED"
    # a speedup with NO census movement is unattributed -> REFUSE
    r = verdict(_mk(cut=0.50, census_moves=False))
    assert r["verdict"][0] == "REFUSE" and "attribution" in r["verdict"][1], r
    # compiling must not change the model's output
    r = verdict(_mk(tok_differ=True))
    assert r["verdict"][0] == "REFUSE" and "token stream" in r["verdict"][1], r
    for bad, why in ((_mk(aa=0.03), "A/A spread"),
                     (_mk(rec=2), "recompiles"),
                     (_mk(err="boom"), "raised"),
                     (_mk(drop="moe_compiled"), "arm moe_compiled missing"),
                     (_mk(gate=(1.0, 2.0)), "anchor")):
        rr = verdict(bad)
        assert rr["verdict"][0] == "REFUSE", why
        if why:
            assert why in rr["verdict"][1], (why, rr["verdict"][1])
    nocen = _mk(); del nocen["census"]
    assert "no replay census" in verdict(nocen)["verdict"][1]
    for line in render(verdict(_mk())).splitlines():
        print(f"[SELF-TEST FIXTURE, NOT A RESULT] {line}")
    print("k12_verdict self-test OK (both band boundaries, the "
          "attribution gate that refuses an unexplained speedup, the "
          "output-invariance gate, and six refusal directions)")


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
