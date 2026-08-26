# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Verdict calculator for PREREG-k8-fp8-compute-attn.

Report shape (composed on the box):
  {"f32_a": ARM, "f32_b": ARM, "fp8_a": ARM, "fp8_b": ARM,
   "ppl": {"f32": float, "fp8": float,
           "tokens_f32": int, "tokens_fp8": int, "text_sha": str},
   "error_bound": {"mean": float, "p99": float, "max": float,
                   "passed": bool},
   "cert_knob_ms": float}
  ARM = {"step_ms_clean": float, "tokens": [int, ...]}

Speed bars are ABSOLUTE millisecond cuts off the same-box f32 arm,
because that is the unit the SV2 composition frame sums in; quality is
a perplexity delta, not an identity claim (the mechanism precludes
identity -- see the prereg).
"""

import argparse
import json
import math
import sys

AA_TOL = 0.02
ANCHOR_TOL = 0.05
PPL_EPS = 0.05          # TR2's quality epsilon, same question
PASS_CUT_MS = 0.15      # SV2 frame's low-end price for this lane
PARTIAL_CUT_MS = 0.05


def _pos_finite(v):
    """A duration must be finite AND positive. NaN slides through both
    `not v` and `v <= 0` (NaN is truthy, and every NaN comparison is
    False), and a zero denominator raises instead of refusing -- both
    fail PERMISSIVELY, which is the wrong direction for a gate
    (Bugbot, gnf4#267)."""
    return isinstance(v, (int, float)) and math.isfinite(v) and v > 0


def _arm_pair(rep, a, b, out, label):
    for k in (a, b):
        arm = rep.get(k)
        if not arm or not _pos_finite(arm.get("step_ms_clean")):
            return (f"G1: arm {k} missing, non-finite, or non-positive "
                    f"({(arm or {}).get('step_ms_clean')!r})")
    x, y = rep[a]["step_ms_clean"], rep[b]["step_ms_clean"]
    spread = abs(x - y) / min(x, y)
    out["gates"][f"{label}_aa_spread"] = spread
    if spread > AA_TOL:
        return (f"G1: {label} A/A spread {spread * 100:.2f}% > "
                f"{AA_TOL * 100:.0f}%")
    if rep[a].get("tokens") != rep[b].get("tokens"):
        return (f"G1: {label} token streams differ between identical "
                "runs (per-arm determinism)")
    return None


def verdict(rep):
    out = {"gates": {}, "quality": {}, "speed": {}, "verdict": None}

    for label, (a, b) in (("f32", ("f32_a", "f32_b")),
                          ("fp8", ("fp8_a", "fp8_b"))):
        why = _arm_pair(rep, a, b, out, label)
        if why:
            return _refuse(out, why)

    f32 = (rep["f32_a"]["step_ms_clean"] + rep["f32_b"]["step_ms_clean"]) / 2
    fp8 = (rep["fp8_a"]["step_ms_clean"] + rep["fp8_b"]["step_ms_clean"]) / 2
    out["gates"]["f32_ms"] = f32
    out["gates"]["fp8_ms"] = fp8

    cert = rep.get("cert_knob_ms")
    if not _pos_finite(cert):
        return _refuse(out, f"G2: certified knob point is missing, "
                            f"non-finite, or non-positive ({cert!r}) -- "
                            "nothing to anchor against")
    drift = abs(f32 - cert) / cert
    out["gates"]["anchor_drift"] = drift
    if drift > ANCHOR_TOL:
        return _refuse(out, f"G2: f32 arm {f32:.3f} ms is "
                            f"{drift * 100:.1f}% off the certified "
                            f"{cert:.3f} ms -- the ms bars do not transfer")

    eb = rep.get("error_bound") or {}
    if not eb.get("passed"):
        return _refuse(out, "G4: the cited fp8 error bound did not pass "
                            "ON THIS BOX -- the frame's tensor-level "
                            "guarantee must hold on the measured silicon")

    p = rep.get("ppl") or {}
    for k in ("f32", "fp8"):
        if not math.isfinite(p.get(k, float("nan"))):
            return _refuse(out, f"quality: {k} perplexity missing or "
                                "non-finite")
    if p.get("tokens_f32") != p.get("tokens_fp8"):
        return _refuse(out, f"G5: token budgets differ "
                            f"({p.get('tokens_f32')} vs "
                            f"{p.get('tokens_fp8')}) -- refuse rather "
                            "than normalise")
    if p.get("text_sha_f32", p.get("text_sha")) != \
            p.get("text_sha_fp8", p.get("text_sha")):
        return _refuse(out, "G5: the two arms evaluated different text")

    d_ppl = p["fp8"] - p["f32"]
    q_ok = d_ppl <= PPL_EPS
    out["quality"] = {"ppl_f32": p["f32"], "ppl_fp8": p["fp8"],
                      "delta": d_ppl, "eps": PPL_EPS, "pass": q_ok}

    cut = f32 - fp8
    out["speed"] = {"cut_ms": cut, "pass_bar": PASS_CUT_MS,
                    "partial_bar": PARTIAL_CUT_MS}

    if not q_ok:
        out["verdict"] = ("REFUTED", "quality")
    elif cut >= PASS_CUT_MS:
        out["verdict"] = ("PASS", "speed+quality")
    elif cut >= PARTIAL_CUT_MS:
        out["verdict"] = ("PARTIAL", "speed+quality")
    else:
        out["verdict"] = ("REFUTED", "speed")
    return out


def _refuse(out, why):
    out["verdict"] = ("REFUSE", why)
    return out


def render(out):
    tag, why = out["verdict"]
    if tag == "REFUSE":
        return f"K8 VERDICT: REFUSE\n  {why}"
    g, q, s = out["gates"], out["quality"], out["speed"]
    lines = [f"K8 VERDICT: {tag}  ({why})",
             f"  step: f32 {g['f32_ms']:.3f} -> fp8 {g['fp8_ms']:.3f} ms "
             f"= {s['cut_ms']:+.3f} ms "
             f"(PASS >= {s['pass_bar']}, PARTIAL >= {s['partial_bar']})",
             f"  quality: ppl {q['ppl_f32']:.4f} -> {q['ppl_fp8']:.4f} "
             f"= {q['delta']:+.4f} vs eps {q['eps']} -> "
             f"{'OK' if q['pass'] else 'FAILED'}"]
    if tag == "REFUTED" and why == "speed":
        lines.append("  the lane is dead as a 250 lever; SV2's pool "
                     "loses this slice (PREREG-k8)")
    if tag == "REFUTED" and why == "quality":
        lines.append("  fp8 compute degrades the model past the "
                     "registered epsilon; do not ship the knob")
    return "\n".join(lines)


def _mk(f32=6.46, fp8=6.28, ppl_d=0.01, aa=0.0, tok_b=None, cert=6.476,
        eb=True, budget=(4096, 4096), sha=("abc", "abc"), ppl_f32=8.0):
    toks = list(range(40))
    return {"f32_a": {"step_ms_clean": f32, "tokens": toks},
            "f32_b": {"step_ms_clean": f32 * (1 + aa), "tokens": toks},
            "fp8_a": {"step_ms_clean": fp8, "tokens": toks},
            "fp8_b": {"step_ms_clean": fp8,
                      "tokens": tok_b if tok_b is not None else toks},
            "ppl": {"f32": ppl_f32, "fp8": ppl_f32 + ppl_d,
                    "tokens_f32": budget[0], "tokens_fp8": budget[1],
                    "text_sha_f32": sha[0], "text_sha_fp8": sha[1]},
            "error_bound": {"mean": 1e-3, "p99": 2e-2, "max": 9e-2,
                            "passed": eb},
            "cert_knob_ms": cert}


def self_test():
    # Boundaries are asserted on either SIDE of each bar, never on the
    # knife edge: an absolute-ms threshold sits on binary floats, and
    # 6.46-0.05 evaluates to 0.04999999999999982 -- a fixture built by
    # subtraction would test the float, not the rule.
    assert verdict(_mk(fp8=6.28))["verdict"][0] == "PASS"        # 0.18
    assert verdict(_mk(f32=6.50, fp8=6.35))["verdict"][0] == "PASS"
    assert verdict(_mk(f32=6.50, fp8=6.36))["verdict"][0] == "PARTIAL"
    assert verdict(_mk(f32=6.50, fp8=6.44))["verdict"][0] == "PARTIAL"
    assert verdict(_mk(f32=6.50, fp8=6.46))["verdict"] == \
        ("REFUTED", "speed")
    assert verdict(_mk(fp8=6.50))["verdict"] == ("REFUTED", "speed")
    # quality dominates: a big speed win with a failed ppl gate REFUTES
    assert verdict(_mk(fp8=5.0, ppl_d=0.051))["verdict"] == \
        ("REFUTED", "quality")
    assert verdict(_mk(ppl_d=0.049))["verdict"][0] == "PASS"    # under eps
    # refusal directions
    for bad, why in ((_mk(aa=0.03), "G1"),
                     (_mk(tok_b=[1, 2]), "G1"),
                     (_mk(cert=7.39), "G2"),
                     (_mk(eb=False), "G4"),
                     (_mk(budget=(4096, 2048)), "G5"),
                     (_mk(sha=("abc", "def")), "G5"),
                     (_mk(ppl_f32=float("nan")), "quality"),
                     (_mk(f32=0.0), "G1"),               # would divide by 0
                     (_mk(fp8=-1.0), "G1"),
                     (_mk(f32=float("nan")), "G1"),
                     (_mk(cert=float("nan")), "G2"),     # would skip G2
                     (_mk(cert=float("inf")), "G2")):
        r = verdict(bad)
        assert r["verdict"][0] == "REFUSE" and \
            r["verdict"][1].startswith(why), (why, r["verdict"])
    missing = _mk()
    del missing["f32_b"]
    assert verdict(missing)["verdict"][1].startswith("G1")
    r = _mk()
    print(render(verdict(r)))
    print("k8_verdict self-test OK (both sides of both speed bars, "
          "the quality epsilon, quality-dominates-speed, and "
          "thirteen refusal directions)")


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
