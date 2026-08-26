# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Verdict calculator for PREREG-k11-mrow-feasibility.

Stage A adjudicates whether a qualifying M-filler EXISTS. The
criterion was fixed before any candidate was judged: a mapping
qualifies iff it strictly increases useful MACs per MMA issued
WITHOUT increasing weight bytes read per output. Both halves are
required -- filling M by re-reading weights trades a compute win for
a bandwidth loss on a kernel already at 3.8x its streaming floor.

Report shape:
  {"candidates": [{"name": str, "mac_ratio_before": float,
                   "mac_ratio_after": float,
                   "weight_bytes_ratio": float,
                   "available": bool, "note": str}, ...],
   "toolchain": {"sparse_mma": bool, "sub16_m": bool,
                 "triton": str, "probed": bool},
   "kscatter": {"equivalent": bool, "max_abs_delta": float,
                "max_abs_ref": float,
                "step_base_ms": float, "step_scatter_ms": float,
                "aa_noise": float} | None}
"""

import argparse
import json
import math
import sys

REL_BAR = 2.0 ** -7          # the K6-B mechanism band


def _finite(v):
    return isinstance(v, (int, float)) and math.isfinite(v)


def qualifies(c):
    """The registered criterion, applied to one candidate."""
    if not c.get("available"):
        return False
    for k in ("mac_ratio_before", "mac_ratio_after", "weight_bytes_ratio"):
        if not _finite(c.get(k)):
            return False
    return (c["mac_ratio_after"] > c["mac_ratio_before"]
            and c["weight_bytes_ratio"] <= 1.0)


def verdict(rep):
    out = {"qualifying": [], "verdict": None}
    cands = rep.get("candidates") or []
    if not cands:
        return _refuse(out, "no candidates enumerated -- a feasibility "
                            "verdict over an empty set is vacuous")
    tc = rep.get("toolchain") or {}
    if not tc.get("probed"):
        return _refuse(out, "toolchain not probed: the two OPEN rows "
                            "(sparse MMA, sub-16 M) must be resolved "
                            "against the INSTALLED build, not from "
                            "documentation")

    out["qualifying"] = [c["name"] for c in cands if qualifies(c)]

    ks = rep.get("kscatter")
    if ks is not None:
        if not _finite(ks.get("max_abs_delta")) or \
                not _finite(ks.get("max_abs_ref")):
            return _refuse(out, "kscatter: equivalence numbers missing")
        band = ks["max_abs_ref"] * REL_BAR
        if ks["max_abs_delta"] > band:
            return _refuse(out, f"kscatter is outside the K6-B band "
                                f"({ks['max_abs_delta']:.3e} > "
                                f"{band:.3e}) -- a wrong kernel "
                                "measures nothing")
        b, s = ks.get("step_base_ms"), ks.get("step_scatter_ms")
        if _finite(b) and _finite(s):
            noise = ks.get("aa_noise") or 0.0
            gain = (b - s) / b
            out["kscatter_gain"] = gain
            out["kscatter_noise"] = noise
            if gain > noise:
                out["verdict"] = ("REOPENED", gain)
                return out

    if out["qualifying"]:
        out["verdict"] = ("CANDIDATE-FOUND", out["qualifying"])
    else:
        out["verdict"] = ("REFUTED-INFEASIBLE",
                          f"{len(cands)} candidates, none qualifying")
    return out


def _refuse(out, why):
    out["verdict"] = ("REFUSE", why)
    return out


def render(out):
    tag, x = out["verdict"]
    if tag == "REFUSE":
        return f"K11 VERDICT: REFUSE\n  {x}"
    if tag == "REOPENED":
        return ("K11 VERDICT: REOPENED\n"
                f"  K-scatter beat the shipped path by {x * 100:.2f}% "
                f"(noise {out['kscatter_noise'] * 100:.2f}%) -- the "
                "MAC-ratio argument this prereg rests on is FALSIFIED")
    if tag == "CANDIDATE-FOUND":
        return f"K11 VERDICT: CANDIDATE-FOUND\n  {', '.join(x)}"
    return ("K11 VERDICT: REFUTED-INFEASIBLE\n"
            f"  {x}; the M dimension has no filler at T=1 on this "
            "toolchain, so 250-by-composition closes")


def _mk(after=6.2, wbytes=1.0, avail=True, probed=True, ks=None,
        n=3):
    c = [{"name": f"cand{i}", "mac_ratio_before": 6.2,
          "mac_ratio_after": after, "weight_bytes_ratio": wbytes,
          "available": avail, "note": ""} for i in range(n)]
    return {"candidates": c,
            "toolchain": {"sparse_mma": False, "sub16_m": False,
                          "triton": "3.7.1", "probed": probed},
            "kscatter": ks}


def self_test():
    # nothing qualifies -> the lane is infeasible
    assert verdict(_mk())["verdict"][0] == "REFUTED-INFEASIBLE"
    # a real gain with no extra weight traffic qualifies
    assert verdict(_mk(after=12.4))["verdict"][0] == "CANDIDATE-FOUND"
    # ...but not if it re-reads weights to get there
    assert verdict(_mk(after=12.4, wbytes=1.6))["verdict"][0] \
        == "REFUTED-INFEASIBLE"
    # ...nor if the toolchain does not offer it
    assert verdict(_mk(after=12.4, avail=False))["verdict"][0] \
        == "REFUTED-INFEASIBLE"
    # weight_bytes_ratio exactly 1.0 is allowed; above it is not
    assert qualifies({"name": "x", "mac_ratio_before": 6.2,
                      "mac_ratio_after": 9.0,
                      "weight_bytes_ratio": 1.0, "available": True})
    assert not qualifies({"name": "x", "mac_ratio_before": 6.2,
                          "mac_ratio_after": 9.0,
                          "weight_bytes_ratio": 1.01, "available": True})
    # K-scatter beating the shipped path FALSIFIES the prereg
    ks = {"equivalent": True, "max_abs_delta": 1e-3, "max_abs_ref": 1.0,
          "step_base_ms": 6.48, "step_scatter_ms": 6.20,
          "aa_noise": 0.002}
    assert verdict(_mk(ks=ks))["verdict"][0] == "REOPENED"
    # ...but a gain inside the noise floor does not
    ks2 = dict(ks, step_scatter_ms=6.475)
    assert verdict(_mk(ks=ks2))["verdict"][0] == "REFUTED-INFEASIBLE"
    # a wrong kernel measures nothing
    ks3 = dict(ks, max_abs_delta=1.0)
    assert verdict(_mk(ks=ks3))["verdict"][0] == "REFUSE"
    # refusals
    for bad, why in ((_mk(n=0), "no candidates"),
                     (_mk(probed=False), "toolchain not probed")):
        r = verdict(bad)
        assert r["verdict"][0] == "REFUSE" and why in r["verdict"][1], why
    for line in render(verdict(_mk())).splitlines():
        print(f"[SELF-TEST FIXTURE, NOT A RESULT] {line}")
    print("k11_verdict self-test OK (the criterion's BOTH halves, the "
          "weight-traffic boundary, the REOPEN path that falsifies the "
          "prereg, the noise floor, and three refusal directions)")


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
