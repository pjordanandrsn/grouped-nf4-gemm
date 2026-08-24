# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""PREREG-k3-attribution decision calculator. Emits the preregistered
branch decision from the peel-battery receipts (+ optional NCU facts).
--self-test both directions before any receipt."""

import argparse
import json
import sys
from pathlib import Path

NOISE_MAX = 5.0
REPLICA_TOL = 15.0
SINGLE_COMPONENT_MIN = 25.0    # % of the cell's full time
SPLITK_MIN = 20.0
FLOOR_X_ROOFLINE = 3.0
ROOFLINE_US = {"gate_up": 8.4, "down": 4.4}   # logical bytes / 1573.9 GB/s


def verdict(rep):
    out = {"cells": {}}
    triggers = []
    for name, c in rep["cells"].items():
        if c["noise_drift_pct"] > NOISE_MAX:
            out["verdict"] = f"NO-VERDICT (noise gate failed on {name})"
            return out
        if c["replica_vs_product_pct"] > REPLICA_TOL:
            out["verdict"] = (f"NO-VERDICT (replica mismatch on {name}: "
                              f"{c['replica_vs_product_pct']:.1f}% from the "
                              f"product kernel -- the battery measured "
                              f"something else)")
            return out
        f = c["peel_us"]["full"]
        shares = {k: max(v, 0.0) / f * 100.0
                  for k, v in c["components_us"].items()}
        shares["residual"] = max(c["residual_us"], 0.0) / f * 100.0
        floor_x = c["components_us"]["loads_floor"] / ROOFLINE_US[name]
        cell_out = {"full_us": f, "shares_pct":
                    {k: round(v, 1) for k, v in shares.items()},
                    "loads_floor_x_roofline": round(floor_x, 2)}
        if name == "gate_up" and "product_sk16_us" in c:
            sk_delta = (c["product_sk16_us"] - c["product_sk1_us"])
            cell_out["splitk_delta_pct"] = round(
                sk_delta / c["product_sk16_us"] * 100.0, 1)
        out["cells"][name] = cell_out
        for comp in ("lut_gather", "absmax", "activation"):
            if shares[comp] >= SINGLE_COMPONENT_MIN:
                triggers.append((shares[comp], f"{name}:{comp}"))
        if floor_x >= FLOOR_X_ROOFLINE:
            triggers.append((floor_x * 10, f"{name}:access_pattern"))

    gu = out["cells"].get("gate_up", {})
    if gu.get("splitk_delta_pct", -100) >= SPLITK_MIN:
        triggers.append((gu["splitk_delta_pct"], "gate_up:splitk"))

    if not triggers:
        out["verdict"] = ("PARK (diffuse account: no component >= 25%, "
                          "floor within 3x roofline, split-K under 20% -- "
                          "the kernel lane parks at 74.3 tok/s and the "
                          "elementwise-fusion lane takes priority)")
        return out
    triggers.sort(reverse=True)
    names = [t[1] for t in triggers]
    mapping = {
        "lut_gather": "register the register-LUT GEMV variant "
                      "(tl.gather codebook-in-registers, the M-tile "
                      "VARIANT-1 treatment never ported to decode)",
        "access_pattern": "register the packing-layout question (the "
                          "streaming floor itself is the wall)",
        "splitk": "register the reduction restructure",
        "absmax": "register the absmax-layout line",
        "activation": "register the activation-reuse line",
    }
    decisions = []
    seen = set()
    for _, t in triggers:
        key = t.split(":")[1]
        if key not in seen:
            seen.add(key)
            decisions.append(mapping[key])
    out["triggers"] = names
    out["verdict"] = "BRANCH: " + " | ".join(decisions)
    return out


def _cell(full, lut, absx, act, floor, drift=1.0, mismatch=3.0,
          sk1=None, sk16=None):
    c = {"peel_us": {"full": full, "no_lut": full - lut,
                     "no_absmax": full - absx, "no_act": full - act,
                     "loads_only": floor},
         "noise_drift_pct": drift,
         "replica_vs_product_pct": mismatch,
         "product_sk1_us": sk1 if sk1 is not None else full,
         "components_us": {"lut_gather": lut, "absmax": absx,
                           "activation": act, "loads_floor": floor}}
    c["residual_us"] = full - (max(lut, 0) + max(absx, 0)
                               + max(act, 0) + floor)
    if sk16 is not None:
        c["product_sk16_us"] = sk16
    return c


def self_test():
    # LUT-dominant -> register-LUT branch
    r = {"cells": {"gate_up": _cell(44.0, 15.0, 2.0, 1.0, 20.0,
                                    sk1=44.0, sk16=46.0),
                   "down": _cell(28.0, 9.0, 1.5, 1.0, 12.0)}}
    v = verdict(r)
    assert v["verdict"].startswith("BRANCH: register the register-LUT"), v
    # access-pattern floor dominant
    r = {"cells": {"gate_up": _cell(44.0, 4.0, 2.0, 1.0, 30.0,
                                    sk1=44.0, sk16=45.0),
                   "down": _cell(28.0, 2.0, 1.5, 1.0, 16.0)}}
    v = verdict(r)
    assert "packing-layout" in v["verdict"], v
    # split-K dominant
    r = {"cells": {"gate_up": _cell(30.0, 2.0, 2.0, 1.0, 16.0,
                                    sk1=30.0, sk16=44.0),
                   "down": _cell(28.0, 2.0, 1.5, 1.0, 12.0)}}
    v = verdict(r)
    assert "reduction restructure" in v["verdict"], v
    # diffuse -> PARK
    r = {"cells": {"gate_up": _cell(44.0, 4.0, 3.0, 2.0, 18.0,
                                    sk1=44.0, sk16=46.0),
                   "down": _cell(28.0, 3.0, 2.0, 1.0, 11.0)}}
    v = verdict(r)
    assert v["verdict"].startswith("PARK"), v
    # replica mismatch blocks
    r = {"cells": {"gate_up": _cell(44.0, 15.0, 2.0, 1.0, 20.0,
                                    mismatch=25.0, sk1=44.0, sk16=46.0)}}
    v = verdict(r)
    assert v["verdict"].startswith("NO-VERDICT (replica"), v
    # noise blocks
    r = {"cells": {"gate_up": _cell(44.0, 15.0, 2.0, 1.0, 20.0,
                                    drift=9.0, sk1=44.0, sk16=46.0)}}
    v = verdict(r)
    assert v["verdict"].startswith("NO-VERDICT (noise"), v
    print("self-test OK: 6/6 branches exercised")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--attr")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        sys.exit(0)
    print(json.dumps(verdict(json.loads(Path(a.attr).read_text())),
                     indent=2))
