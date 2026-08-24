# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""PREREG-k2-vectorized-nibbles verdict. Bars hardcoded; --self-test
both directions before any receipt."""

import argparse
import json
import sys
from pathlib import Path

AA_SPREAD_MAX = 7.5
HK_PASS_US = 45.0            # vec pair-time (µs) at the winner configs
HK_PARTIAL_US = 58.0
HE_PASS_MS = 12.6
GS_STEP_LO, GS_STEP_HI = 115.0, 165.0
GS_ATTN_MAX = 55.0


def verdict(bench, g1_, g2_, gl1, gl2, b16):
    out = {}
    bm = bench["summary"]
    out["g_b"] = {"bitwise_all": bool(bm["bitwise_all"]),
                  "cells": {k: v["bitwise"]
                            for k, v in bench["cells"].items()}}
    if not bm["bitwise_all"]:
        out["verdict"] = ("REFUTED (G-B: legacy and vectorized outputs "
                          "differ on CUDA — the construction claim is "
                          "false; revert regardless of speed)")
        return out

    d = b16["decode_median_ms"]
    gs = (GS_STEP_LO <= float(d["step"]) <= GS_STEP_HI
          and float(d["attention_host"]) <= GS_ATTN_MAX)
    out["gs_b16"] = {"pass": bool(gs), "step_ms": float(d["step"])}
    if not gs:
        out["verdict"] = "NO-VERDICT (B=16 certified point regressed)"
        return out

    arms = {}
    for name, (x, y) in (("vec", (g1_, g2_)), ("legacy", (gl1, gl2))):
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

    for name, rep in (("vec", g1_), ("legacy", gl1)):
        if not rep.get("tokens"):
            out["verdict"] = f"NO-VERDICT (empty token record in {name})"
            return out
    ident = g1_["tokens"] == gl1["tokens"]
    out["identity_e2e"] = {"bitwise": bool(ident),
                           "n": len(g1_["tokens"])}
    if not ident:
        out["verdict"] = ("REFUTED (G-B e2e: token divergence between "
                          "paths that must be value-identical)")
        return out

    vec_us = bm["vec_sum_ms"] * 1000.0
    out["h_k"] = {"legacy_pair_us": bm["legacy_sum_ms"] * 1000.0,
                  "vec_pair_us": vec_us,
                  "pass": vec_us <= HK_PASS_US,
                  "partial": HK_PASS_US < vec_us <= HK_PARTIAL_US}
    sv = arms["vec"]["step_ms"]
    out["h_e"] = {"vec_step_ms": sv,
                  "legacy_step_ms": arms["legacy"]["step_ms"],
                  "pass": sv <= HE_PASS_MS}
    out["reported"] = {"tok_s_single_stream": 1000.0 / sv,
                       "kernel_speedup":
                           bm["legacy_sum_ms"] / bm["vec_sum_ms"]}

    hk, he = out["h_k"], out["h_e"]["pass"]
    if hk["pass"] and he:
        out["verdict"] = ("CERTIFIED (vectorized stays the default; "
                          "RESULTS records it)")
    elif hk["pass"]:
        out["verdict"] = ("KERNEL-WIN-NOT-AT-WALL (investigate the "
                          "graph's composition before any ship)")
    elif hk["partial"] and he:
        out["verdict"] = "PARTIAL (ship with full disclosure)"
    else:
        out["verdict"] = ("REFUTED-FOR-VARIANT (legacy stays default; "
                          "the lane escalates to the packing-layout "
                          "question with a fresh registration)")
    return out


def _bench(vec_us, legacy_us=72.4, bitwise=True):
    return {"summary": {"legacy_sum_ms": legacy_us / 1000.0,
                        "vec_sum_ms": vec_us / 1000.0,
                        "ratio_vec_over_legacy": vec_us / legacy_us,
                        "bitwise_all": bitwise},
            "cells": {"gate_up": {"bitwise": bitwise},
                      "down": {"bitwise": True}}}


def _t(step, toks=None):
    return {"step_ms_clean": step,
            "tokens": toks if toks is not None else list(range(127))}


def _b16(step=130.0, attn=42.0):
    return {"decode_median_ms": {"step": step, "attention_host": attn}}


def self_test():
    # certified
    v = verdict(_bench(40.0), _t(12.2), _t(12.25), _t(13.4), _t(13.45),
                _b16())
    assert v["verdict"].startswith("CERTIFIED"), v
    # bitwise breach dominates
    v = verdict(_bench(40.0, bitwise=False), _t(12.2), _t(12.25),
                _t(13.4), _t(13.45), _b16())
    assert v["verdict"].startswith("REFUTED (G-B:"), v
    # e2e token divergence dominates speed
    v = verdict(_bench(40.0), _t(12.2, [1, 2]), _t(12.25, [1, 2]),
                _t(13.4, [1, 3]), _t(13.45, [1, 3]), _b16())
    assert v["verdict"].startswith("REFUTED (G-B e2e"), v
    # kernel win, wall miss
    v = verdict(_bench(40.0), _t(13.4), _t(13.45), _t(13.5), _t(13.55),
                _b16())
    assert v["verdict"].startswith("KERNEL-WIN-NOT-AT-WALL"), v
    # partial band
    v = verdict(_bench(50.0), _t(12.4), _t(12.45), _t(13.4), _t(13.45),
                _b16())
    assert v["verdict"].startswith("PARTIAL"), v
    # refuted for variant
    v = verdict(_bench(65.0), _t(13.3), _t(13.35), _t(13.4), _t(13.45),
                _b16())
    assert v["verdict"].startswith("REFUTED-FOR-VARIANT"), v
    # b16 regression + empty tokens + arm spread
    v = verdict(_bench(40.0), _t(12.2), _t(12.25), _t(13.4), _t(13.45),
                _b16(step=190.0))
    assert v["verdict"].startswith("NO-VERDICT (B=16"), v
    v = verdict(_bench(40.0), _t(12.2, []), _t(12.25, []), _t(13.4, []),
                _t(13.45, []), _b16())
    assert v["verdict"].startswith("NO-VERDICT (empty"), v
    v = verdict(_bench(40.0), _t(12.2), _t(14.0), _t(13.4), _t(13.45),
                _b16())
    assert v["verdict"].startswith("NO-VERDICT (G0"), v
    print("self-test OK: 9/9 branches exercised")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    for f in ("bench", "g1", "g2", "gl1", "gl2", "b16"):
        ap.add_argument(f"--{f}")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        sys.exit(0)
    load = lambda f: json.loads(Path(f).read_text())
    print(json.dumps(verdict(load(a.bench), load(a.g1), load(a.g2),
                             load(a.gl1), load(a.gl2), load(a.b16)),
                     indent=2))
