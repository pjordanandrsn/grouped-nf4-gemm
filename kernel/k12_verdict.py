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

#: The knob ratio's two ends, MEASURED ON THE SAME BOX (K6-B / SV1):
#: knob-ON 6.476 ms and its own knob-OFF partner 7.25 ms. Kept as a
#: pair, at module scope, so the self-test can assert the pairing and
#: a future edit cannot quietly substitute the three-box anchor
#: median for the denominator the way the first version did.
KNOB_PAIR = (6.476, 7.25)
PASS_MS = 0.40
PARTIAL_MS = 0.15
AA_TOL = 0.02
#: The raw-ATen rows the census attributes to the excluded region.
#: These must be DISJOINT matchers. "elementwise_kernel" as a bare
#: substring also catches `unrolled_elementwise_kernel` and
#: `vectorized_elementwise_kernel`, which move independently -- so the
#: gate could pass or refuse on the wrong family, and the reported
#: counts were the sum of three rows (review, gnf4#285). The plain
#: kernel is matched via its `::` prefix, which the two decorated
#: names do not carry.
TRACKED = ("unrolled_elementwise_kernel", "vectorized_elementwise_kernel",
           "::elementwise_kernel", "indexSelect", "reduce_kernel")


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
    rows_fell = any(m["after"] < m["before"] for m in moved.values())
    out["gates"]["tracked_rows_fell"] = rows_fell

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
    # gate SCALED by the knob ratio.
    #
    # That ratio must be PAIRED. It was 6.476 / 7.369, which is a
    # same-box knob-ON point over the THREE-BOX knob-OFF median --
    # two numbers never measured together, so their quotient is not a
    # knob ratio at all (review, gnf4#285). 6.476's own knob-OFF
    # partner, on that box, is 7.25; that pair is the ratio.
    # The correction is 1.6%, which on this gate is ~0.11 ms -- small
    # against a 12.6%-wide window, but a wrong denominator does not
    # become right by being applied to a wide gate.
    #
    # Standing caveat: the pair comes from ONE box, and M2 measured
    # 8.5% inter-box dispersion in absolute step time. Whether the
    # RATIO is box-invariant was never measured. So this stays what
    # the anchor always was -- an outlier excluder, not a
    # certification of the arm's class ([[bars-follow-the-claim]]).
    KNOB = KNOB_PAIR[0] / KNOB_PAIR[1]
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

    # ATTRIBUTION, scoped to a SPEEDUP -- which is what the prereg
    # registered: "if arm 2 gets FASTER while those rows are
    # unchanged, the speed came from somewhere else". This check ran
    # unconditionally, so an arm that came back SLOWER with unmoved
    # rows returned REFUSE where REFUTED is the honest answer: a
    # treatment that did not pay is a result, not an unattributable
    # one. Nothing is unattributed about a slowdown.
    #
    # Corrected while arm 2's TIMING was known and its census did not
    # yet exist, so it cannot be tuning toward an outcome -- and the
    # change can only turn a REFUSE into a REFUTED. It cannot produce
    # a PASS or a PARTIAL, so it can never favour the treatment.
    if cut > 0 and not rows_fell:
        return _refuse(out, "attribution: the arm got faster but no "
                            "tracked raw-ATen row fell in count, so "
                            "the speed came from somewhere other than "
                            "the mechanism this cycle registered")

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
    # REAL census row names -- a simplified fixture would not exercise
    # the overlap the matchers exist to avoid
    U = "void at::native::unrolled_elementwise_kernel<at::nat"
    V = "void at::native::vectorized_elementwise_kernel<4, at"
    E = "void at::native::elementwise_kernel<128, 4, at::nati"
    I = "void at::native::(anonymous namespace)::indexSelectS"
    R = "void at::native::reduce_kernel<128, 4, at::native::R"
    before = {U: 218, V: 72, E: 96, I: 145, R: 48}
    after = dict(before)
    if census_moves:
        after[U] = 40
    return {"arms": arms, "cert_gate": list(gate),
            "census": {"before": before, "after": after}}


def self_test():
    # The knob ratio must stay PAIRED. 6.476/7.369 -- a same-box
    # knob-ON point over the THREE-BOX knob-OFF median -- is not a
    # ratio of anything (review, gnf4#285). It reads only 1.6% off,
    # which is exactly the size of error a self-test has to catch,
    # because no verdict computed from it would look wrong.
    # Pin the VALUE, not the spelling: the first version of this
    # check scanned the source for "6.476 / 7.369" and fired on the
    # COMMENT that explains the fix. A test that cannot tell code
    # from prose about the code is not testing the code.
    import decode_anchor
    ratio = KNOB_PAIR[0] / KNOB_PAIR[1]
    assert abs(ratio - 6.476 / 7.25) < 1e-12, ratio
    assert abs(ratio - 6.476 / decode_anchor.ANCHOR_MS) > 1e-3, \
        ("the knob ratio is being computed against the three-box "
         "anchor median again -- that pair was never measured "
         "together")
    assert KNOB_PAIR[1] == 7.25, \
        "knob-OFF partner must be the same-box 7.25"
    assert abs(KNOB_PAIR[1] - decode_anchor.ANCHOR_MS) > 0.05, \
        ("the paired denominator must not BE the anchor median -- if "
         "those ever coincide, the pairing has been lost again")
    assert verdict(_mk(cut=0.50))["verdict"][0] == "PASS"
    assert verdict(_mk(cut=0.41))["verdict"][0] == "PASS"
    assert verdict(_mk(cut=0.30))["verdict"][0] == "PARTIAL"
    assert verdict(_mk(cut=0.16))["verdict"][0] == "PARTIAL"
    assert verdict(_mk(cut=0.10))["verdict"][0] == "REFUTED"
    assert verdict(_mk(cut=-0.20))["verdict"][0] == "REFUTED"
    # a SPEEDUP with no census movement is unattributed -> REFUSE
    r = verdict(_mk(cut=0.50, census_moves=False))
    assert r["verdict"][0] == "REFUSE" and "attribution" in r["verdict"][1], r
    # ...but a SLOWDOWN with no movement is a clean REFUTED, not an
    # unattributable one. Nothing is unattributed about a treatment
    # that did not pay, and the prereg scoped this gate to "if arm 2
    # gets FASTER". Running it unconditionally discarded exactly the
    # negative result the cycle exists to be able to report.
    r = verdict(_mk(cut=-0.67, census_moves=False))
    assert r["verdict"][0] == "REFUTED", r
    # and a slowdown WITH movement is equally REFUTED
    assert verdict(_mk(cut=-0.67, census_moves=True))["verdict"][0] \
        == "REFUTED"
    # a flat result with no movement is still REFUTED, not REFUSE
    assert verdict(_mk(cut=0.0, census_moves=False))["verdict"][0] \
        == "REFUTED"
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
    # the matchers must be DISJOINT on real census names: a bare
    # "elementwise_kernel" substring counted three families as one
    # (review, gnf4#285)
    cen = _mk()["census"]["before"]
    for name in cen:
        hits = [t for t in TRACKED if t in name]
        assert len(hits) == 1, (name, hits)
    r = verdict(_mk())
    mv = r["gates"]["census"]
    assert mv["unrolled_elementwise_kernel"]["before"] == 218, mv
    assert mv["::elementwise_kernel"]["before"] == 96, mv
    assert mv["vectorized_elementwise_kernel"]["before"] == 72, mv

    # The fixture's row names must stay REAL. They were simplified
    # once, and a simplified fixture cannot exercise the substring
    # overlap the matchers exist to avoid -- the same class as an
    # invented schema passing a test it could never fail.
    for name in _mk()["census"]["before"]:
        assert name.startswith("void at::native::"), name
        assert "<" in name or "namespace" in name, name

    nocen = _mk(); del nocen["census"]
    assert "no replay census" in verdict(nocen)["verdict"][1]
    for line in render(verdict(_mk())).splitlines():
        print(f"[SELF-TEST FIXTURE, NOT A RESULT] {line}")
    print("k12_verdict self-test OK (both band boundaries, the "
          "attribution gate that refuses an unexplained SPEEDUP while "
          "letting a slowdown refute, the "
          "output-invariance gate, DISJOINT row matchers, and six "
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
    out = verdict(json.load(open(a.report)))
    print(render(out))
    if out["verdict"][0] == "REFUSE":
        sys.exit(3)


if __name__ == "__main__":
    main()
