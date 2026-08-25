# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Verdict calculator for PREREG-f2-tail (graph-step tail: fused
combine T1 + fused QKV T2).

Bars verbatim from the prereg (gain frame, anchor class 7.35 ms):
PASS combined cut >= 0.35 ms with the identity gate green on every
arm; PARTIAL cut >= 0.15 (ships whichever single treatment carries a
gain wider than the A/A spread); REFUTED < 0.15. The prereg's
"(ratio <= 0.952)" is the 0.35/7.35 restatement rounded to 3 places --
inside the anchor window the exact ratio at cut=0.35 spans
0.951..0.954, so the CUT is the binding bar and the ratio is reported,
never enforced (enforcing the rounding would silently tighten the
registered bar).
REFUSE: T1 divergence (bitwise treatment, no numerics excuse),
T2/T1+T2 divergence before step 32, T2 projection outside the
max|ref|*2^-7 frame, A/A spread strictly wider than half the PASS
margin (0.175 ms), anchor outside 7.35 +/- 3%, degenerate traces.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PASS_CUT_MS = 0.35
PASS_RATIO_REPORTED = 0.952   # restatement only, never enforced
PARTIAL_CUT_MS = 0.15
AA_BAR_MS = PASS_CUT_MS / 2          # REFUSE iff spread > this
ANCHOR_MS = 7.35
ANCHOR_TOL = 0.03
MIN_DIVERGE_STEP = 32
T2_REL_BAR = 2.0 ** -7
GRAM, MAX_REP = 8, 6
EPS = 1e-9        # decimal-representation guard on bar comparisons only


def _degenerate(t):
    if len(t) >= 64 and len(set(t)) < 30:
        return f"only {len(set(t))} distinct tokens"
    grams = Counter(tuple(t[i:i + GRAM]) for i in range(len(t) - GRAM + 1))
    worst = max(grams.values(), default=0)
    if worst > MAX_REP:
        return f"an {GRAM}-gram repeats {worst}x"
    return None


def _prefix(a, b):
    """Identical leading tokens between two streams."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def _streams(arm):
    """tokens is {stream_id: [ids]}; normalize to sorted items."""
    return sorted(arm["tokens"].items())


def verdict(rep):
    need = ("aa", "off", "t1", "t2", "both")
    for k in need:
        arm = rep.get(k)
        if not arm:
            return ("REFUSE", f"missing arm {k!r}")
        if k == "aa":
            if len(arm.get("step_ms", [])) != 2:
                return ("REFUSE", "aa needs two step_ms entries")
        elif not arm.get("step_ms") or not arm.get("tokens"):
            return ("REFUSE", f"{k}: missing step_ms or tokens")

    off = rep["off"]
    # anchor: the OFF graph step must sit in the certified class
    lo, hi = ANCHOR_MS * (1 - ANCHOR_TOL), ANCHOR_MS * (1 + ANCHOR_TOL)
    if not (lo <= off["step_ms"] <= hi):
        return ("REFUSE", f"anchor non-compliance: OFF step "
                f"{off['step_ms']:.3f} ms outside [{lo:.3f}, {hi:.3f}]")

    a1, a2 = rep["aa"]["step_ms"]
    spread = abs(a1 - a2)
    if spread > AA_BAR_MS + EPS:
        return ("REFUSE", f"A/A spread {spread:.3f} ms wider than half "
                f"the PASS margin ({AA_BAR_MS:.3f}) -- the frame cannot "
                "resolve the gain")

    # degeneracy: every stream in every armed run (check-traces law)
    for k in ("off", "t1", "t2", "both"):
        for sid, toks in _streams(rep[k]):
            why = _degenerate(toks)
            if why:
                return ("REFUSE", f"{k}/{sid} trace degenerate: {why}")

    # identity gates vs OFF
    off_s = dict(_streams(off))
    idmsg = []
    for k, bitwise in (("t1", True), ("t2", False), ("both", False)):
        for sid, toks in _streams(rep[k]):
            if sid not in off_s:
                return ("REFUSE", f"{k}/{sid}: no matching OFF stream")
            p = _prefix(off_s[sid], toks)
            full = p == min(len(off_s[sid]), len(toks))
            if bitwise and not full:
                return ("REFUSE", f"T1 divergence at step {p} on {sid} "
                        "-- bitwise treatment has no numerics excuse")
            if not full and p < MIN_DIVERGE_STEP:
                return ("REFUSE", f"{k}/{sid} diverges at step {p} < "
                        f"{MIN_DIVERGE_STEP} -- mechanism bug, not "
                        "reorder noise")
            idmsg.append(f"{k}/{sid}:{'identical' if full else f'div@{p}'}")

    # T2 projection frame
    proj = rep.get("t2_proj")
    if not proj:
        return ("REFUSE", "missing t2_proj projection receipt")
    for name, cell in sorted(proj.items()):
        bar = cell["max_abs_ref"] * T2_REL_BAR
        if cell["max_abs_delta"] > bar:
            return ("REFUSE", f"t2_proj[{name}]: max|delta| "
                    f"{cell['max_abs_delta']:g} > max|ref|*2^-7 = {bar:g}")

    cut = off["step_ms"] - rep["both"]["step_ms"]
    ratio = rep["both"]["step_ms"] / off["step_ms"]
    cut_t1 = off["step_ms"] - rep["t1"]["step_ms"]
    cut_t2 = off["step_ms"] - rep["t2"]["step_ms"]
    detail = (f"combined cut {cut:.3f} ms (ratio {ratio:.3f}; "
              f"{rep['both']['step_ms']:.3f} vs {off['step_ms']:.3f}); "
              f"T1 alone {cut_t1:+.3f}, T2 alone {cut_t2:+.3f}, "
              f"A/A spread {spread:.3f}; identity {', '.join(idmsg)}")
    if cut >= PASS_CUT_MS - EPS:
        return ("PASS", detail + " -- ships both as defaults")
    if cut >= PARTIAL_CUT_MS - EPS:
        real = [n for n, c in (("T1", cut_t1), ("T2", cut_t2))
                if c > spread]
        ship = " + ".join(real) if real else "NEITHER singly"
        return ("PARTIAL", detail + f" -- single treatment(s) beating "
                f"the A/A spread: {ship}")
    return ("REFUTED", detail + " -- the tail is not addressable at "
            "this cost")


# ---------------------------------------------------------------- self-test
def _mk(off=7.35, t1=7.20, t2=7.15, both=6.95, aa=(7.35, 7.36),
        n=64, div=None, div_arm=None, proj_delta=1e-3, proj_ref=10.0):
    base = list(range(100, 100 + n))
    def arm(step, diverge=False):
        t = list(base)
        if diverge is not False:
            t[diverge:] = [x + 50000 for x in t[diverge:]]
        return {"step_ms": step, "tokens": {"s0": t}}
    rep = {"aa": {"step_ms": list(aa)},
           "off": arm(off),
           "t1": arm(t1, div if div_arm == "t1" else False),
           "t2": arm(t2, div if div_arm == "t2" else False),
           "both": arm(both, div if div_arm == "both" else False),
           "t2_proj": {p: {"max_abs_delta": proj_delta,
                           "max_abs_ref": proj_ref}
                       for p in ("q", "k", "v")}}
    return rep


def _self_test():
    v = verdict
    assert v(_mk())[0] == "PASS", v(_mk())
    # boundary: cut exactly 0.35 with ratio at bar passes
    r = _mk(off=7.35, both=7.00)
    assert v(r)[0] == "PASS", v(r)
    # cut 0.3499 just under the bar -> PARTIAL, not PASS
    r = _mk(off=7.35, both=7.0001)
    assert v(r)[0] == "PARTIAL", v(r)
    # PARTIAL: cut 0.2, T2 carries it (T1's 0.01 is inside the 0.05
    # A/A spread, so it is not a real gain under A/A)
    r = _mk(both=7.15, t1=7.34, t2=7.16, aa=(7.35, 7.40))
    out = v(r)
    assert out[0] == "PARTIAL" and "T2" in out[1] and "T1 +" not in out[1], out
    # PARTIAL with neither treatment beating spread
    r = _mk(both=7.15, t1=7.30, t2=7.30, aa=(7.35, 7.50))
    out = v(r)
    assert out[0] == "PARTIAL" and "NEITHER" in out[1], out
    # REFUTED
    r = _mk(both=7.25, t1=7.33, t2=7.33)
    assert v(r)[0] == "REFUTED", v(r)
    # A/A boundary: spread exactly 0.175 is NOT "wider than" -- allowed
    r = _mk(aa=(7.30, 7.475))
    assert v(r)[0] == "PASS", v(r)
    r = _mk(aa=(7.30, 7.4751))
    out = v(r)
    assert out[0] == "REFUSE" and "A/A" in out[1], out
    # anchor gate two-sided
    for bad in (7.05, 7.60):
        out = v(_mk(off=bad))
        assert out[0] == "REFUSE" and "anchor" in out[1], out
    # T1 divergence anywhere refuses
    out = v(_mk(div=50, div_arm="t1"))
    assert out[0] == "REFUSE" and "T1 divergence" in out[1], out
    # T2 divergence at 31 refuses, at 32 allowed
    out = v(_mk(div=31, div_arm="t2"))
    assert out[0] == "REFUSE" and "step 31" in out[1], out
    assert v(_mk(div=32, div_arm="t2"))[0] == "PASS"
    # both-arm early divergence refuses
    assert v(_mk(div=10, div_arm="both"))[0] == "REFUSE"
    # projection frame
    out = v(_mk(proj_delta=0.09, proj_ref=10.0))   # bar = 10*2^-7 = 0.078
    assert out[0] == "REFUSE" and "t2_proj" in out[1], out
    assert v(_mk(proj_delta=0.078, proj_ref=10.0))[0] == "PASS"
    # degenerate trace refuses
    r = _mk()
    r["off"]["tokens"]["s0"] = [1, 2] * 32
    out = v(r)
    assert out[0] == "REFUSE" and "degenerate" in out[1], out
    # missing pieces refuse
    r = _mk()
    del r["t2_proj"]
    assert v(r)[0] == "REFUSE"
    r = _mk()
    del r["both"]
    assert v(r)[0] == "REFUSE"
    print("f2_verdict self-test: OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--receipts", default=None,
                    help="dir with aa/off/t1/t2/both/t2_proj .json")
    a = ap.parse_args()
    if a.self_test:
        _self_test()
        return
    d = Path(a.receipts)
    rep = {}
    for k in ("aa", "off", "t1", "t2", "both", "t2_proj"):
        p = d / f"{k}.json"
        if p.exists():
            rep[k] = json.loads(p.read_text())
    v, why = verdict(rep)
    print(f"F2 VERDICT: {v}\n{why}")
    if v == "REFUSE":
        sys.exit(2)


if __name__ == "__main__":
    main()
