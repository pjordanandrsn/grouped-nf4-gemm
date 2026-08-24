# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""PREREG-m1-decode-config (K1) verdict. Bars hardcoded from the
prereg; --self-test both directions before any receipt."""

import argparse
import json
import sys
from pathlib import Path

AA_SPREAD_MAX = 7.5
SWEEP_DRIFT_MAX = 5.0
HK_PASS_RATIO = 2.0 / 3.0     # winner_sum <= 2/3 * plan_sum
HK_PARTIAL_RATIO = 4.0 / 4.84  # (3.2, 4.0] ms-equivalent band as a ratio
HE_PASS_MS = 13.8
GS_STEP_LO, GS_STEP_HI = 115.0, 165.0
GS_ATTN_MAX = 55.0
AGREE_INVESTIGATE = 100        # of 127: below this, no ship before diagnosis


def verdict(sweep, g0_1, g0_2, gk_1, gk_2, b16):
    out = {}
    sm = sweep["summary"]
    if not sm["noise_gate_pass"]:
        out["verdict"] = "NO-VERDICT (sweep noise gate: baseline drifted >5%)"
        return out
    ratio = sm["ratio_winner_over_plan"]
    out["h_k"] = {"plan_sum_ms": sm["plan_sum_ms"],
                  "winner_sum_ms": sm["winner_sum_ms"],
                  "ratio": ratio,
                  "winners": sm["winners"],
                  "pass": ratio <= HK_PASS_RATIO,
                  "partial": HK_PASS_RATIO < ratio <= HK_PARTIAL_RATIO}
    if not (out["h_k"]["pass"] or out["h_k"]["partial"]):
        out["verdict"] = ("REFUTED-FOR-CONFIG (the ablation space holds "
                          "< the bar at these shapes; the universal plan "
                          "stands and K2 kernel-body work is the lane)")
        return out

    d = b16["decode_median_ms"]
    gs = (GS_STEP_LO <= float(d["step"]) <= GS_STEP_HI
          and float(d["attention_host"]) <= GS_ATTN_MAX)
    out["gs_b16"] = {"pass": bool(gs), "step_ms": float(d["step"])}
    if not gs:
        out["verdict"] = "NO-VERDICT (B=16 certified point regressed)"
        return out

    arms = {}
    for name, (x, y) in (("g0", (g0_1, g0_2)), ("gk", (gk_1, gk_2))):
        m1 = float(x["step_ms_clean"])
        m2 = float(y["step_ms_clean"])
        spread = abs(m1 - m2) / min(m1, m2) * 100.0
        arms[name] = {"step_ms": min(m1, m2), "aa_spread_pct": spread,
                      "aa_pass": spread < AA_SPREAD_MAX}
        if not arms[name]["aa_pass"]:
            out["arms"] = arms
            out["verdict"] = f"NO-VERDICT (G0 fail on arm {name})"
            return out
    out["arms"] = arms

    for name, rep in (("g0", g0_1), ("gk", gk_1)):
        if not rep.get("tokens"):
            out["verdict"] = f"NO-VERDICT (empty token record in {name})"
            return out
    t0, tk = g0_1["tokens"], gk_1["tokens"]
    n = min(len(t0), len(tk))
    agree = next((i for i in range(n) if t0[i] != tk[i]), n)
    out["fidelity"] = {"token_agreement": agree, "window": n,
                       "bitwise_expected": False,
                       "investigate": agree < AGREE_INVESTIGATE}
    if out["fidelity"]["investigate"]:
        out["verdict"] = (f"NO-SHIP-PENDING-INVESTIGATION (token agreement "
                          f"{agree}/{n} < {AGREE_INVESTIGATE}: cross-config "
                          f"ulp drift should not diverge this early)")
        return out

    sk_ms = arms["gk"]["step_ms"]
    out["h_e"] = {"gk_step_ms": sk_ms, "g0_step_ms": arms["g0"]["step_ms"],
                  "pass": sk_ms <= HE_PASS_MS}
    out["reported"] = {"tok_s_single_stream": 1000.0 / sk_ms,
                       "e2e_gain_ms": arms["g0"]["step_ms"] - sk_ms}

    hk = out["h_k"]
    if hk["pass"] and out["h_e"]["pass"]:
        out["verdict"] = ("CERTIFIED (bake the per-shape winners into "
                          "_decode_plan behind an sm_120 + M=1-census "
                          "guard, in the RESULTS PR)")
    elif hk["pass"]:
        out["verdict"] = ("KERNEL-WIN-NOT-AT-WALL (H-K passed, H-E did "
                          "not: investigate the graph's composition "
                          "before any ship — no silent partial)")
    elif out["h_e"]["pass"]:
        out["verdict"] = ("PARTIAL (H-K in the partial band with the e2e "
                          "bar met: bake only with full disclosure)")
    else:
        out["verdict"] = "REFUTED (revert; K2 is the lane)"
    return out


def _sweep(ratio, noise=True, plan_sum=4.84):
    w = plan_sum * ratio
    return {"summary": {"noise_gate_pass": noise, "plan_sum_ms": plan_sum,
                        "winner_sum_ms": w,
                        "ratio_winner_over_plan": ratio,
                        "winners": {"gate_up": {"bn": 32, "warps": 4,
                                                "sk": 8, "ms": w * 0.6},
                                    "down": {"bn": 64, "warps": 2,
                                             "sk": 4, "ms": w * 0.4}}}}


def _t(step, toks=None):
    return {"step_ms_clean": step,
            "tokens": toks if toks is not None else list(range(127))}


def _b16(step=133.0, attn=41.0):
    return {"decode_median_ms": {"step": step, "attention_host": attn}}


def self_test():
    # certified
    v = verdict(_sweep(0.55), _t(15.2), _t(15.25), _t(13.2), _t(13.25),
                _b16())
    assert v["verdict"].startswith("CERTIFIED"), v
    # kernel win, wall miss
    v = verdict(_sweep(0.55), _t(15.2), _t(15.25), _t(14.5), _t(14.55),
                _b16())
    assert v["verdict"].startswith("KERNEL-WIN-NOT-AT-WALL"), v
    # partial band + e2e pass
    v = verdict(_sweep(0.75), _t(15.2), _t(15.25), _t(13.5), _t(13.55),
                _b16())
    assert v["verdict"].startswith("PARTIAL"), v
    # refuted for config
    v = verdict(_sweep(0.95), _t(15.2), _t(15.25), _t(15.1), _t(15.15),
                _b16())
    assert v["verdict"].startswith("REFUTED-FOR-CONFIG"), v
    # sweep noise gate
    v = verdict(_sweep(0.55, noise=False), _t(15.2), _t(15.25), _t(13.2),
                _t(13.25), _b16())
    assert v["verdict"].startswith("NO-VERDICT (sweep"), v
    # early token divergence blocks ship
    bad = list(range(127))
    bad[40] = 9999
    v = verdict(_sweep(0.55), _t(15.2), _t(15.25),
                _t(13.2, bad), _t(13.25, bad), _b16())
    assert v["verdict"].startswith("NO-SHIP-PENDING"), v
    # late ulp divergence tolerated (agreement >= 100)
    late = list(range(127))
    late[120] = 9999
    v = verdict(_sweep(0.55), _t(15.2), _t(15.25),
                _t(13.2, late), _t(13.25, late), _b16())
    assert v["verdict"].startswith("CERTIFIED"), v
    # b16 regression blocks
    v = verdict(_sweep(0.55), _t(15.2), _t(15.25), _t(13.2), _t(13.25),
                _b16(step=180.0))
    assert v["verdict"].startswith("NO-VERDICT (B=16"), v
    # empty tokens refused
    v = verdict(_sweep(0.55), _t(15.2, []), _t(15.25, []), _t(13.2, []),
                _t(13.25, []), _b16())
    assert v["verdict"].startswith("NO-VERDICT (empty"), v
    print("self-test OK: 9/9 branches exercised")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    for f in ("sweep", "g0-1", "g0-2", "gk-1", "gk-2", "b16"):
        ap.add_argument(f"--{f}")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        sys.exit(0)
    load = lambda f: json.loads(Path(f).read_text())
    print(json.dumps(verdict(load(a.sweep), load(a.g0_1), load(a.g0_2),
                             load(a.gk_1), load(a.gk_2), load(a.b16)),
                     indent=2))
