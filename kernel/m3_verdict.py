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

    # S: BOTH must not be slower than either part, outside noise
    slower_than_part = any(
        (ms["both"] - ms[n]) / ms[n] > noise for n in ("dotpad", "fp8"))
    if slower_than_part:
        out["verdict"] = ("REFUTED", "composition: BOTH is slower than "
                                     "a single knob outside A/A noise")
        return out

    q = out["quality"]
    if q["both"]["pass"] and q["dotpad"]["pass"] and q["fp8"]["pass"]:
        out["verdict"] = ("PASS", "flip both")
    elif q["dotpad"]["pass"] or q["fp8"]["pass"]:
        keep = [n for n in ("dotpad", "fp8") if q[n]["pass"]]
        out["verdict"] = ("PARTIAL", keep)
    else:
        out["verdict"] = ("REFUTED", "quality: no knob holds at the "
                                     "registered horizon")
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
        budget=8192, sha="abc", both_ms=6.28, drop=None, gate=(7.004, 7.906)):
    t = list(range(30))
    ms = {"off": 7.35, "dotpad": 6.48, "fp8": 7.14, "both": both_ms}
    ppl = {"off": off_ppl, "dotpad": off_ppl + d_dot,
           "fp8": off_ppl + d_fp8, "both": off_ppl + d_both}
    arms = {n: {"a": ms[n], "b": ms[n] * (1 + aa), "tokens_a": t,
                "tokens_b": t, "ppl": ppl[n], "ppl_tokens": budget,
                "text_sha": sha, "first_divergence": None}
            for n in ARMS}
    if drop:
        del arms[drop]
    return {"arms": arms, "cert_gate": list(gate)}


def self_test():
    assert verdict(_mk())["verdict"][0] == "PASS"
    assert verdict(_mk(d_both=0.051))["verdict"][0] == "PARTIAL"
    assert verdict(_mk(d_dot=0.06, d_fp8=0.06, d_both=0.06))["verdict"][0] \
        == "REFUTED"
    # composition: BOTH slower than a part refutes even with good ppl
    r = verdict(_mk(both_ms=6.60))
    assert r["verdict"] == ("REFUTED", "composition: BOTH is slower than "
                                       "a single knob outside A/A noise"), r
    # PARTIAL names only the knob that held
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
    for line in render(verdict(_mk())).splitlines():
        print(f"[SELF-TEST FIXTURE, NOT A RESULT] {line}")
    print("m3_verdict self-test OK (PASS/PARTIAL/REFUTED bands, the "
          "composition bar, the 8192-token horizon, shared-text and "
          "shared-budget gates, and six refusal directions)")


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
