# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Verdict calculator for PREREG-f2-tail (graph-step tail: fused
combine T1 + fused QKV T2).

Report shape = the b1d graph-loop receipts as produced (K6-B house
shape): arms off_a/off_b/t1/t2/both each {"step_ms_clean", "tokens"
(flat greedy list)}, plus t2_proj {proj: {max_abs_delta,
max_abs_ref}}. Baseline is min(off_a, off_b) -- conservative for a
gain claim.

Bars verbatim from the prereg (gain frame, anchor class 7.35 ms):
PASS combined cut >= 0.35 ms with the identity gate green on every
arm; PARTIAL cut >= 0.15 (ships whichever single treatment carries a
gain wider than the A/A spread); REFUTED < 0.15. The prereg's
"(ratio <= 0.952)" is the 0.35/7.35 restatement rounded to 3 places --
inside the anchor window the exact ratio at cut=0.35 spans
0.951..0.954, so the CUT is the binding bar and the ratio is reported,
never enforced (enforcing the rounding would silently tighten the
registered bar).
REFUSE: A/A tokens not identical, T1 divergence (bitwise treatment,
no numerics excuse), T2/T1+T2 divergence before step 32, T2 projection
outside the max|ref|*2^-7 frame, A/A spread strictly wider than half
the PASS margin (0.175 ms), anchor outside 7.35 +/- 3%, degenerate
traces.
"""

import argparse
import json
import sys
from collections import Counter

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


def verdict(rep):
    arms = ("off_a", "off_b", "t1", "t2", "both")
    for k in arms:
        arm = rep.get(k)
        if not arm:
            return ("REFUSE", f"missing arm {k!r}")
        if not arm.get("step_ms_clean") or not arm.get("tokens"):
            return ("REFUSE", f"{k}: missing step_ms_clean or tokens")

    off_a, off_b = rep["off_a"], rep["off_b"]
    if off_a["tokens"] != off_b["tokens"]:
        return ("REFUSE", "OFF arms not token-identical -- the box is "
                "not deterministic and divergence cannot be attributed")

    base = min(off_a["step_ms_clean"], off_b["step_ms_clean"])
    spread = abs(off_a["step_ms_clean"] - off_b["step_ms_clean"])
    lo, hi = ANCHOR_MS * (1 - ANCHOR_TOL), ANCHOR_MS * (1 + ANCHOR_TOL)
    if not (lo <= base <= hi):
        return ("REFUSE", f"anchor non-compliance: OFF step {base:.3f} "
                f"ms outside [{lo:.3f}, {hi:.3f}]")
    if spread > AA_BAR_MS + EPS:
        return ("REFUSE", f"A/A spread {spread:.3f} ms wider than half "
                f"the PASS margin ({AA_BAR_MS:.3f}) -- the frame cannot "
                "resolve the gain")

    for k in arms:
        why = _degenerate(rep[k]["tokens"])
        if why:
            return ("REFUSE", f"{k} trace degenerate: {why}")

    t_off = off_a["tokens"]
    idmsg = []
    for k, bitwise in (("t1", True), ("t2", False), ("both", False)):
        toks = rep[k]["tokens"]
        p = _prefix(t_off, toks)
        full = p == min(len(t_off), len(toks))
        if bitwise and not full:
            return ("REFUSE", f"T1 divergence at step {p} -- bitwise "
                    "treatment has no numerics excuse")
        if not full and p < MIN_DIVERGE_STEP:
            return ("REFUSE", f"{k} diverges at step {p} < "
                    f"{MIN_DIVERGE_STEP} -- mechanism bug, not "
                    "reorder noise")
        idmsg.append(f"{k}:{'identical' if full else f'div@{p}'}")

    proj = rep.get("t2_proj")
    if not proj:
        return ("REFUSE", "missing t2_proj projection receipt")
    for name in ("q", "k", "v"):
        cell = proj.get(name)
        if not cell:
            return ("REFUSE", f"t2_proj missing projection {name!r}")
        bar = cell["max_abs_ref"] * T2_REL_BAR
        if cell["max_abs_delta"] > bar:
            return ("REFUSE", f"t2_proj[{name}]: max|delta| "
                    f"{cell['max_abs_delta']:g} > max|ref|*2^-7 = {bar:g}")

    both_ms = rep["both"]["step_ms_clean"]
    cut = base - both_ms
    ratio = both_ms / base
    cut_t1 = base - rep["t1"]["step_ms_clean"]
    cut_t2 = base - rep["t2"]["step_ms_clean"]
    detail = (f"combined cut {cut:.3f} ms (ratio {ratio:.3f}; "
              f"{both_ms:.3f} vs {base:.3f}); "
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
def _mk(off_a=7.35, off_b=7.36, t1=7.20, t2=7.15, both=6.95,
        n=64, div=None, div_arm=None, proj_delta=1e-3, proj_ref=10.0):
    base = list(range(100, 100 + n))

    def arm(step, name):
        t = list(base)
        if div is not None and div_arm == name:
            t[div:] = [x + 50000 for x in t[div:]]
        return {"step_ms_clean": step, "tokens": t}

    return {"off_a": arm(off_a, "off_a"), "off_b": arm(off_b, "off_b"),
            "t1": arm(t1, "t1"), "t2": arm(t2, "t2"),
            "both": arm(both, "both"),
            "t2_proj": {p: {"max_abs_delta": proj_delta,
                            "max_abs_ref": proj_ref}
                        for p in ("q", "k", "v")}}


def _self_test():
    v = verdict
    assert v(_mk())[0] == "PASS", v(_mk())
    # boundary: cut exactly 0.35 passes (baseline is min(off_a, off_b))
    r = _mk(off_a=7.35, off_b=7.36, both=7.00)
    assert v(r)[0] == "PASS", v(r)
    # cut 0.3499 just under the bar -> PARTIAL, not PASS
    r = _mk(off_a=7.35, off_b=7.36, both=7.0001)
    assert v(r)[0] == "PARTIAL", v(r)
    # PARTIAL: cut 0.2, T2 carries it (T1's 0.01 is inside the 0.05
    # A/A spread, so it is not a real gain under A/A)
    r = _mk(off_a=7.35, off_b=7.40, both=7.15, t1=7.34, t2=7.16)
    out = v(r)
    assert out[0] == "PARTIAL" and "T2" in out[1] \
        and "T1 +" not in out[1], out
    # PARTIAL with neither treatment beating the spread
    r = _mk(off_a=7.35, off_b=7.50, both=7.15, t1=7.30, t2=7.30)
    out = v(r)
    assert out[0] == "PARTIAL" and "NEITHER" in out[1], out
    # REFUTED
    r = _mk(both=7.25, t1=7.33, t2=7.33)
    assert v(r)[0] == "REFUTED", v(r)
    # A/A boundary: spread exactly 0.175 is NOT "wider than" -- allowed
    assert v(_mk(off_a=7.30, off_b=7.475))[0] == "PASS"
    out = v(_mk(off_a=7.30, off_b=7.476))
    assert out[0] == "REFUSE" and "A/A" in out[1], out
    # anchor gate two-sided (baseline = min of the pair)
    for a, b in ((7.05, 7.06), (7.60, 7.61)):
        out = v(_mk(off_a=a, off_b=b))
        assert out[0] == "REFUSE" and "anchor" in out[1], out
    # A/A determinism: differing OFF tokens refuse
    r = _mk()
    r["off_b"]["tokens"] = r["off_b"]["tokens"][:-1] + [1]
    out = v(r)
    assert out[0] == "REFUSE" and "deterministic" in out[1], out
    # T1 divergence anywhere refuses
    out = v(_mk(div=50, div_arm="t1"))
    assert out[0] == "REFUSE" and "T1 divergence" in out[1], out
    # T2 divergence at 31 refuses, at 32 allowed
    out = v(_mk(div=31, div_arm="t2"))
    assert out[0] == "REFUSE" and "step 31" in out[1], out
    assert v(_mk(div=32, div_arm="t2"))[0] == "PASS"
    # both-arm early divergence refuses
    assert v(_mk(div=10, div_arm="both"))[0] == "REFUSE"
    # projection frame: bar = 10 * 2^-7 = 0.078125
    out = v(_mk(proj_delta=0.09, proj_ref=10.0))
    assert out[0] == "REFUSE" and "t2_proj" in out[1], out
    assert v(_mk(proj_delta=0.078, proj_ref=10.0))[0] == "PASS"
    # degenerate trace refuses
    r = _mk()
    r["off_a"]["tokens"] = [1, 2] * 32
    r["off_b"]["tokens"] = [1, 2] * 32
    out = v(r)
    assert out[0] == "REFUSE" and "degenerate" in out[1], out
    # missing pieces refuse
    r = _mk()
    del r["t2_proj"]["k"]
    out = v(r)
    assert out[0] == "REFUSE" and "'k'" in out[1], out
    r = _mk()
    del r["t2_proj"]
    assert v(r)[0] == "REFUSE"
    r = _mk()
    del r["both"]
    assert v(r)[0] == "REFUSE"
    print("f2_verdict self-test: OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report", nargs="?", default=None,
                    help="composed report json (off_a/off_b/t1/t2/"
                         "both/t2_proj)")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        _self_test()
        return
    rep = json.load(open(a.report))
    v, why = verdict(rep)
    print(f"F2 VERDICT: {v}\n{why}")
    if v == "REFUSE":
        sys.exit(2)


if __name__ == "__main__":
    main()
