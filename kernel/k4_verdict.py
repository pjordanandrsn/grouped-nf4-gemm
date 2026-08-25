# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""PREREG-k4-wide-loads verdict. Bars hardcoded; --self-test both
directions before any receipt."""

import argparse
import json
import sys
from pathlib import Path

AA_SPREAD_MAX = 7.5
HM_FLOOR_RATIO = 0.5      # wide loads-only floor <= 0.5x scalar floor
HK_PASS_US = 40.0
HK_PARTIAL_US = 55.0
HE_PASS_MS = 12.0
GS_STEP_LO, GS_STEP_HI = 115.0, 165.0
GS_ATTN_MAX = 55.0


def verdict(bench, floors, g1_, g2_, gl1, gl2, b16):
    """bench: k2-style bitwise+timing at configs (legacy vs wide);
    floors: {cell: {scalar_floor_us, wide_floor_us}} from the peel."""
    out = {}
    bm = bench["summary"]
    out["g_b"] = {"bitwise_all": bool(bm["bitwise_all"])}
    if not bm["bitwise_all"]:
        out["verdict"] = ("REFUTED (G-B: outputs differ on CUDA -- the "
                          "construction claim is false; revert regardless "
                          "of speed)")
        return out

    hm = {}
    for cell, f in floors.items():
        hm[cell] = {"scalar_us": f["scalar_floor_us"],
                    "wide_us": f["wide_floor_us"],
                    "ratio": f["wide_floor_us"] / f["scalar_floor_us"]}
    out["h_m"] = {"cells": hm,
                  "pass": all(c["ratio"] <= HM_FLOOR_RATIO
                              for c in hm.values())}
    if not out["h_m"]["pass"]:
        out["verdict"] = ("REFUTED-FOR-MECHANISM (the wide streaming "
                          "floor did not drop: width does not address "
                          "the named wall -- escalate to k-strip-major "
                          "repacking, no partial credit)")
        return out

    d = b16["decode_median_ms"]
    gs = (GS_STEP_LO <= float(d["step"]) <= GS_STEP_HI
          and float(d["attention_host"]) <= GS_ATTN_MAX)
    out["gs_b16"] = {"pass": bool(gs), "step_ms": float(d["step"])}
    if not gs:
        out["verdict"] = "NO-VERDICT (B=16 certified point regressed)"
        return out

    arms = {}
    for name, (x, y) in (("wide", (g1_, g2_)), ("legacy", (gl1, gl2))):
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

    for name, rep in (("wide", g1_), ("legacy", gl1)):
        if not rep.get("tokens"):
            out["verdict"] = f"NO-VERDICT (empty token record in {name})"
            return out
    if g1_["tokens"] != gl1["tokens"]:
        out["verdict"] = ("REFUTED (G-B e2e: token divergence between "
                          "paths that must be value-identical)")
        return out
    out["identity_e2e"] = {"bitwise": True, "n": len(g1_["tokens"])}

    wide_us = bm["wide_sum_us"]
    out["h_k"] = {"legacy_pair_us": bm["legacy_sum_us"],
                  "wide_pair_us": wide_us,
                  "pass": wide_us <= HK_PASS_US,
                  "partial": HK_PASS_US < wide_us <= HK_PARTIAL_US}
    sw = arms["wide"]["step_ms"]
    out["h_e"] = {"wide_step_ms": sw,
                  "legacy_step_ms": arms["legacy"]["step_ms"],
                  "pass": sw <= HE_PASS_MS}
    out["reported"] = {"tok_s_single_stream": 1000.0 / sw,
                       "kernel_speedup":
                           bm["legacy_sum_us"] / wide_us}

    hk, he = out["h_k"], out["h_e"]["pass"]
    if hk["pass"] and he:
        out["verdict"] = ("CERTIFIED (wide becomes the default and the "
                          "wide-winner configs bake into _decode_plan, "
                          "both in the RESULTS PR)")
    elif hk["pass"]:
        out["verdict"] = ("KERNEL-WIN-NOT-AT-WALL (investigate the "
                          "graph's composition before any ship)")
    elif hk["partial"] and he:
        out["verdict"] = "PARTIAL (ship with full disclosure)"
    else:
        out["verdict"] = ("REFUTED-FOR-WIDTH (legacy stays default; "
                          "k-strip-major repacking is the lane)")
    return out


def _bench(wide_us, legacy_us=72.8, bitwise=True):
    return {"summary": {"legacy_sum_us": legacy_us, "wide_sum_us": wide_us,
                        "bitwise_all": bitwise}}


def _floors(r1=0.3, r2=0.3):
    return {"gate_up": {"scalar_floor_us": 56.5,
                        "wide_floor_us": 56.5 * r1},
            "down": {"scalar_floor_us": 23.5, "wide_floor_us": 23.5 * r2}}


def _t(step, toks=None):
    return {"step_ms_clean": step,
            "tokens": toks if toks is not None else list(range(127))}


def _b16(step=130.0, attn=42.0):
    return {"decode_median_ms": {"step": step, "attention_host": attn}}


def self_test():
    # certified
    v = verdict(_bench(32.0), _floors(), _t(11.4), _t(11.45),
                _t(13.4), _t(13.45), _b16())
    assert v["verdict"].startswith("CERTIFIED"), v
    # mechanism refuted dominates (floor did not drop)
    v = verdict(_bench(32.0), _floors(0.9, 0.85), _t(11.4), _t(11.45),
                _t(13.4), _t(13.45), _b16())
    assert v["verdict"].startswith("REFUTED-FOR-MECHANISM"), v
    # bitwise breach dominates all
    v = verdict(_bench(32.0, bitwise=False), _floors(), _t(11.4),
                _t(11.45), _t(13.4), _t(13.45), _b16())
    assert v["verdict"].startswith("REFUTED (G-B:"), v
    # e2e divergence
    v = verdict(_bench(32.0), _floors(), _t(11.4, [1]), _t(11.45, [1]),
                _t(13.4, [2]), _t(13.45, [2]), _b16())
    assert v["verdict"].startswith("REFUTED (G-B e2e"), v
    # kernel win, wall miss
    v = verdict(_bench(32.0), _floors(), _t(12.8), _t(12.85),
                _t(13.4), _t(13.45), _b16())
    assert v["verdict"].startswith("KERNEL-WIN-NOT-AT-WALL"), v
    # partial
    v = verdict(_bench(48.0), _floors(), _t(11.9), _t(11.95),
                _t(13.4), _t(13.45), _b16())
    assert v["verdict"].startswith("PARTIAL"), v
    # refuted for width
    v = verdict(_bench(68.0), _floors(0.45, 0.45), _t(13.3), _t(13.35),
                _t(13.4), _t(13.45), _b16())
    assert v["verdict"].startswith("REFUTED-FOR-WIDTH"), v
    # gates
    v = verdict(_bench(32.0), _floors(), _t(11.4), _t(11.45),
                _t(13.4), _t(13.45), _b16(step=190.0))
    assert v["verdict"].startswith("NO-VERDICT (B=16"), v
    v = verdict(_bench(32.0), _floors(), _t(11.4, []), _t(11.45, []),
                _t(13.4, []), _t(13.45, []), _b16())
    assert v["verdict"].startswith("NO-VERDICT (empty"), v
    print("self-test OK: 9/9 branches exercised")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    for f in ("bench", "floors", "g1", "g2", "gl1", "gl2", "b16"):
        ap.add_argument(f"--{f}")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        sys.exit(0)
    load = lambda f: json.loads(Path(f).read_text())
    print(json.dumps(verdict(load(a.bench), load(a.floors), load(a.g1),
                             load(a.g2), load(a.gl1), load(a.gl2),
                             load(a.b16)), indent=2))
