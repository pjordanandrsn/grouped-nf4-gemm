# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Verdict calculator for PREREG-m3-default-on.

The speed is not in question -- both knobs carry committed receipts.
This decides whether the DEFAULTS may move, on a longer quality
horizon (8192 scored tokens vs K8's 1024).

Report shape:
  {"arms": {"off": ARM, "dotpad": ARM, "fp8": ARM, "both": ARM},
   "cert_gate": [lo_ms, hi_ms]}
  ARM = {"a": float, "b": float, "tokens_a": [...], "tokens_b": [...],
         "ppl": float, "ppl_tokens": int, "text_sha": str,
         "first_divergence": int | None}
"""

import argparse
import json
import math
import sys

PPL_EPS = 0.05
AA_TOL = 0.02
ARMS = ("off", "dotpad", "fp8", "both")
#: Which mechanism each arm MUST have exercised. An env var is a
#: request: GNF4_GEMV_DOTPAD=1 engages dot-pad only if the shape is
#: registered AND the part carries >= 160 SMs, so a "dotpad" arm can
#: silently run the certified scalar path. It would then match OFF in
#: step time and -- K6-B measured dot-pad token-IDENTICAL at 127 --
#: in perplexity too, and this cycle would read that as a free knob.
#:
#: AMENDMENT, added after registration and before the box: this gate
#: can only ever REFUSE. It cannot turn a REFUTED into a PASS or move
#: a bar, so adding it post-registration strengthens the cycle
#: without loosening anything it already committed to.
EXPECT = {"off":    {"dotpad": False, "fp8": False},
          "dotpad": {"dotpad": True,  "fp8": False},
          "fp8":    {"dotpad": False, "fp8": True},
          "both":   {"dotpad": True,  "fp8": True}}


def _pos_finite(v):
    return isinstance(v, (int, float)) and math.isfinite(v) and v > 0


def verdict(rep):
    out = {"gates": {}, "quality": {}, "speed": {}, "verdict": None}
    arms = rep.get("arms") or {}
    for name in ARMS:
        a = arms.get(name)
        if not a:
            return _refuse(out, f"arm {name} missing -- all four "
                                "combinations are required; a flip "
                                "cannot be judged from a subset")
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
        if not math.isfinite(a.get("ppl", float("nan"))):
            return _refuse(out, f"{name}: perplexity missing")

    # every quality arm must have scored the SAME text and budget
    shas = {arms[n].get("text_sha") for n in ARMS}
    budgets = {arms[n].get("ppl_tokens") for n in ARMS}
    if len(shas) != 1 or None in shas:
        return _refuse(out, f"quality arms scored different text "
                            f"({len(shas)} digests)")
    if len(budgets) != 1 or None in budgets:
        return _refuse(out, f"quality arms scored different budgets "
                            f"({sorted(b for b in budgets if b)})")
    budget = budgets.pop()
    if budget < 8192:
        return _refuse(out, f"quality window is {budget} tokens; the "
                            "prereg registered 8192 because every "
                            "prior receipt was short and that is the "
                            "question this cycle exists to answer")

    # mechanism receipts: what DISPATCHED, not what was requested
    for name in ARMS:
        a = arms[name]
        d, c = a.get("dispatch"), a.get("compute")
        if not isinstance(d, dict) or not isinstance(c, dict):
            return _refuse(out, f"{name}: no mechanism receipt -- the "
                                "arm cannot show which kernel ran, "
                                "only which env var was set")
        dot = d.get("dotpad", 0) + d.get("dotpad_splitk", 0)
        scal = d.get("scalar", 0) + d.get("scalar_splitk", 0)
        fp8 = c.get("fp8", 0)
        f32 = c.get("f32", 0)
        out["gates"][f"{name}_mech"] = {"dotpad": dot, "scalar": scal,
                                        "fp8": fp8, "f32": f32}
        if dot + scal == 0:
            return _refuse(out, f"{name}: GEMV tally is all zero -- "
                                "the decode path never dispatched, so "
                                "the arm proves nothing about it")
        if fp8 + f32 == 0:
            return _refuse(out, f"{name}: attention tally is all zero")
        want = EXPECT[name]
        if want["dotpad"] and dot == 0:
            return _refuse(out, f"{name}: names dot-pad but dispatched "
                                f"it {dot} times ({scal} scalar) -- the "
                                "knob was requested and ignored")
        if not want["dotpad"] and dot != 0:
            return _refuse(out, f"{name}: dispatched dot-pad {dot} "
                                "times but does not name it")
        if want["fp8"] and fp8 == 0:
            return _refuse(out, f"{name}: names fp8 compute but "
                                f"dispatched it {fp8} times")
        if not want["fp8"] and fp8 != 0:
            return _refuse(out, f"{name}: dispatched fp8 compute "
                                f"{fp8} times but does not name it")

    gate = rep.get("cert_gate")
    if not (isinstance(gate, (list, tuple)) and len(gate) == 2
            and all(_pos_finite(g) for g in gate)):
        return _refuse(out, "no committed anchor gate to check against")
    off = (arms["off"]["a"] + arms["off"]["b"]) / 2
    out["gates"]["off_ms"] = off
    if not (gate[0] <= off <= gate[1]):
        return _refuse(out, f"anchor: OFF/OFF arm {off:.3f} ms is "
                            f"outside the committed gate "
                            f"[{gate[0]}, {gate[1]}]")

    base = arms["off"]["ppl"]
    for name in ("dotpad", "fp8", "both"):
        d = arms[name]["ppl"] - base
        out["quality"][name] = {"delta": d, "pass": d <= PPL_EPS}
    ms = {n: (arms[n]["a"] + arms[n]["b"]) / 2 for n in ARMS}
    out["speed"] = ms
    noise = max(out["gates"][f"{n}_aa"] for n in ARMS)

    # S constrains shipping BOTH as the default -- it does NOT gate
    # whether a solo knob may flip, so its failure demotes to the
    # PARTIAL path rather than refuting the cycle (review, gnf4#284).
    q = out["quality"]
    slower_than_part = any(
        (ms["both"] - ms[n]) / ms[n] > noise for n in ("dotpad", "fp8"))
    composed_licensed = q["both"]["pass"] and not slower_than_part
    out["gates"]["composed_licensed"] = composed_licensed
    out["gates"]["both_slower_than_part"] = slower_than_part

    if composed_licensed and q["dotpad"]["pass"] and q["fp8"]["pass"]:
        out["verdict"] = ("PASS", "flip both")
        return out

    # The composition is NOT licensed, so AT MOST ONE default may move
    # -- flipping both IS the composed configuration that Q2/S just
    # refused, and returning both names here would ship exactly what
    # the bar forbade (review, gnf4#284, High).
    cands = [n for n in ("dotpad", "fp8") if q[n]["pass"]]
    if not cands:
        out["verdict"] = ("REFUTED", "quality: no knob holds at the "
                                     "registered horizon")
    elif len(cands) == 1:
        out["verdict"] = ("PARTIAL", cands)
    else:
        # registered tie-break: quality already holds for both, so the
        # larger measured step cut decides
        pick = min(cands, key=lambda n: ms[n])
        out["gates"]["tiebreak"] = {"reason": "composition unlicensed; "
                                              "larger step cut wins",
                                    "cuts": {n: ms["off"] - ms[n]
                                             for n in cands}}
        out["verdict"] = ("PARTIAL", [pick])
    return out


def _refuse(out, why):
    out["verdict"] = ("REFUSE", why)
    return out


def render(out):
    tag, x = out["verdict"]
    if tag == "REFUSE":
        return f"M3 VERDICT: REFUSE\n  {x}"
    lines = [f"M3 VERDICT: {tag}  ({x if isinstance(x, str) else ', '.join(x)})"]
    for n in ARMS:
        q = out["quality"].get(n)
        lines.append(f"  {n:<7} {out['speed'][n]:.3f} ms"
                     + (f"   ppl {q['delta']:+.4f} "
                        f"{'OK' if q['pass'] else 'FAILED'}" if q else
                        "   (baseline)"))
    return "\n".join(lines)


def _mk(off_ppl=8.0, d_dot=0.01, d_fp8=0.01, d_both=0.02, aa=0.001,
        budget=8192, sha="abc", both_ms=6.28, drop=None,
        gate=(7.004, 7.906), dotpad_ms=6.48, fp8_ms=7.14,
        mech=None):
    t = list(range(30))
    ms = {"off": 7.35, "dotpad": dotpad_ms, "fp8": fp8_ms,
          "both": both_ms}
    ppl = {"off": off_ppl, "dotpad": off_ppl + d_dot,
           "fp8": off_ppl + d_fp8, "both": off_ppl + d_both}
    arms = {n: {"a": ms[n], "b": ms[n] * (1 + aa), "tokens_a": t,
                "tokens_b": t, "ppl": ppl[n], "ppl_tokens": budget,
                "text_sha": sha, "first_divergence": None,
                "dispatch": {"dotpad": 96 if EXPECT[n]["dotpad"] else 0,
                             "dotpad_splitk": 0,
                             "scalar": 0 if EXPECT[n]["dotpad"] else 96,
                             "scalar_splitk": 0},
                "compute": {"fp8": 48 if EXPECT[n]["fp8"] else 0,
                            "f32": 0 if EXPECT[n]["fp8"] else 48}}
            for n in ARMS}
    if mech:
        arms[mech[0]].update(mech[1])
    if drop:
        del arms[drop]
    return {"arms": arms, "cert_gate": list(gate)}


def _mech_refusal(arm, patch):
    r = verdict(_mk(mech=(arm, patch)))
    assert r["verdict"][0] == "REFUSE", (arm, patch, r["verdict"])
    return r["verdict"][1]


def self_test():
    assert verdict(_mk())["verdict"][0] == "PASS"

    # ---- mechanism receipts -------------------------------------
    # The fixture above supplies CORRECT receipts for every arm, so
    # it exercises the gate in exactly the direction that cannot
    # catch anything. These drive it the other way. (This is the SV2
    # lesson: a fixture that repeats the instrument's own assumption
    # is not a test of the instrument.)
    ZERO = {"dotpad": 0, "dotpad_splitk": 0, "scalar": 0,
            "scalar_splitk": 0}

    # the case the whole receipt exists for: a "dotpad" arm that
    # silently ran the certified scalar path. Its step time and its
    # perplexity BOTH match OFF, so every other gate in this file
    # reads it as a knob that costs nothing.
    silent = _mk(d_dot=0.0, dotpad_ms=7.35,
                 mech=("dotpad", {"dispatch": dict(ZERO, scalar=96)}))
    r = verdict(silent)
    assert r["verdict"][0] == "REFUSE" and "ignored" in r["verdict"][1], r
    # ...and prove the REFUSAL came from the receipt, not from the
    # numbers: identical step times and identical perplexity, but
    # with a receipt showing dot-pad really dispatched, is not
    # refused. Without this the test could be passing for the wrong
    # reason ([[check-the-result-could-have-failed]]).
    honest = _mk(d_dot=0.0, dotpad_ms=7.35)
    assert verdict(honest)["verdict"][0] != "REFUSE", verdict(honest)

    assert "no mechanism receipt" in _mech_refusal(
        "off", {"dispatch": None})
    assert "all zero" in _mech_refusal("off", {"dispatch": dict(ZERO)})
    assert "does not name it" in _mech_refusal(
        "off", {"dispatch": dict(ZERO, scalar=90, dotpad=6)})
    assert "does not name it" in _mech_refusal(
        "dotpad", {"compute": {"fp8": 3, "f32": 45}})
    assert "names fp8" in _mech_refusal(
        "fp8", {"compute": {"fp8": 0, "f32": 48}})
    assert "ignored" in _mech_refusal(
        "both", {"dispatch": dict(ZERO, scalar=96)})
    # split-K dot-pad counts as dot-pad
    assert verdict(_mk(mech=("dotpad", {"dispatch": dict(
        ZERO, dotpad_splitk=96)})))["verdict"][0] == "PASS"
    assert verdict(_mk(d_dot=0.06, d_fp8=0.06, d_both=0.06))["verdict"][0] \
        == "REFUTED"

    # Q2 fails while BOTH solo knobs pass. PARTIAL must name exactly
    # ONE knob: returning both would flip both defaults, which IS the
    # composed configuration Q2 just refused (review, gnf4#284).
    r = verdict(_mk(d_both=0.051))
    assert r["verdict"][0] == "PARTIAL", r
    assert len(r["verdict"][1]) == 1, \
        "a Q2 failure must not ship the composed configuration"
    assert r["verdict"][1] == ["dotpad"], r      # larger cut wins

    # S failure demotes to PARTIAL, it does not refute the cycle: S
    # constrains shipping BOTH, not whether a solo knob may flip.
    r = verdict(_mk(both_ms=6.60))
    assert r["verdict"][0] == "PARTIAL", r
    assert r["verdict"][1] == ["dotpad"], r
    assert r["gates"]["both_slower_than_part"] is True

    # the registered tie-break is by measured cut, not by name order
    r = verdict(_mk(d_both=0.051, dotpad_ms=7.30, fp8_ms=6.40))
    assert r["verdict"] == ("PARTIAL", ["fp8"]), r

    # PARTIAL names only the knob that held when the other fails Q
    r = verdict(_mk(d_fp8=0.06, d_both=0.06))
    assert r["verdict"] == ("PARTIAL", ["dotpad"]), r
    for bad, why in ((_mk(drop="fp8"), "arm fp8 missing"),
                     (_mk(aa=0.03), "A/A spread"),
                     (_mk(budget=1024), "quality window"),
                     (_mk(gate=(6.0, 6.5)), "anchor")):
        rr = verdict(bad)
        assert rr["verdict"][0] == "REFUSE" and why in rr["verdict"][1], why
    mixed = _mk(); mixed["arms"]["fp8"]["text_sha"] = "zzz"
    assert "different text" in verdict(mixed)["verdict"][1]
    mixed2 = _mk(); mixed2["arms"]["fp8"]["ppl_tokens"] = 4096
    assert "different budgets" in verdict(mixed2)["verdict"][1]
    # THE invariant the Q2 hole violated: only PASS may move two
    # defaults. Asserted over a sweep rather than at the one fixture
    # that happened to expose it, so a future edit cannot reintroduce
    # a two-knob PARTIAL through some other path.
    for dd in (0.01, 0.06):
        for df in (0.01, 0.06):
            for db in (0.01, 0.06):
                for bm in (6.28, 6.60):
                    r = verdict(_mk(d_dot=dd, d_fp8=df, d_both=db,
                                    both_ms=bm))
                    tag, names = r["verdict"]
                    if tag == "PARTIAL":
                        assert len(names) == 1, (dd, df, db, bm, names)
                    elif tag == "PASS":
                        assert db <= PPL_EPS and dd <= PPL_EPS \
                            and df <= PPL_EPS, (dd, df, db, bm)

    for line in render(verdict(_mk())).splitlines():
        print(f"[SELF-TEST FIXTURE, NOT A RESULT] {line}")
    print("m3_verdict self-test OK (PASS/PARTIAL/REFUTED bands, the "
          "one-knob PARTIAL rule, the registered tie-break, the "
          "8192-token horizon, shared-text and "
          "shared-budget gates, the mechanism receipts -- including "
          "the silently-scalar arm every OTHER gate passes -- and "
          "eleven refusal directions)")


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
