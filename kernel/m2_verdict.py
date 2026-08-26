# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Verdict calculator for PREREG-m2-anchor-recert.

This cycle certifies an INSTRUMENT, not a treatment, so there is no
PASS/REFUTED band -- it either produces a defensible constant or
REFUSES. The decision rule was fixed before any box ran:

  new anchor = median of per-box A/A medians
  window     = +/-3%, unless inter-box spread > 6%, then +/-(spread/2)
               and the RESULTS must say the population is too
               dispersed for a 3% gate
  REFUSE     if any box's A/A spread > 2%, or fewer than 3 boxes

Report shape:
  {"boxes": [{"id": str, "a": float, "b": float,
              "tokens_identical": bool, "recompiles": int,
              "driver": str, "torch": str}, ...],
   "prior_anchor": 7.35, "harness_constant": 7.39}
"""

import argparse
import json
import math
import statistics
import sys

MIN_BOXES = 3
AA_TOL = 0.02
BASE_WINDOW = 0.03
DISPERSION_TRIGGER = 0.06


def _pos_finite(v):
    return isinstance(v, (int, float)) and math.isfinite(v) and v > 0


def verdict(rep):
    out = {"boxes": [], "verdict": None}
    boxes = rep.get("boxes") or []
    if len(boxes) < MIN_BOXES:
        return _refuse(out, f"only {len(boxes)} boxes completed; the "
                            f"rule requires {MIN_BOXES} before a "
                            "median means anything")
    meds = []
    for i, b in enumerate(boxes):
        for k in ("a", "b"):
            if not _pos_finite(b.get(k)):
                return _refuse(out, f"box {b.get('id', i)}: arm {k} "
                                    "missing or non-positive")
        x, y = b["a"], b["b"]
        spread = abs(x - y) / min(x, y)
        if spread > AA_TOL:
            return _refuse(out, f"box {b.get('id', i)}: A/A spread "
                                f"{spread * 100:.2f}% > "
                                f"{AA_TOL * 100:.0f}% -- it is not "
                                "measuring itself consistently, so it "
                                "cannot measure the class")
        if b.get("tokens_identical") is not True:
            return _refuse(out, f"box {b.get('id', i)}: token streams "
                                "differ between identical runs")
        if b.get("recompiles", 0) != 0:
            return _refuse(out, f"box {b.get('id', i)}: "
                                f"{b.get('recompiles')} recompiles in "
                                "the window")
        m = (x + y) / 2
        meds.append(m)
        out["boxes"].append({"id": b.get("id", i), "median_ms": m,
                             "aa_spread": spread})

    anchor = statistics.median(meds)
    spread = (max(meds) - min(meds)) / min(meds)
    if spread > DISPERSION_TRIGGER:
        window = spread / 2
        note = (f"population spread {spread * 100:.1f}% exceeds "
                f"{DISPERSION_TRIGGER * 100:.0f}%: the window is "
                f"widened to +/-{window * 100:.1f}% and the class is "
                "too dispersed for a 3% gate")
    else:
        window = BASE_WINDOW
        note = None

    prior = rep.get("prior_anchor")
    if not _pos_finite(prior):
        return _refuse(out, "no prior anchor to compare against")
    aa_noise = max(b["aa_spread"] for b in out["boxes"])
    delta = (anchor - prior) / prior
    out.update({"anchor_ms": anchor, "window": window,
                "inter_box_spread": spread, "dispersion_note": note,
                "prior_anchor": prior, "delta_vs_prior": delta,
                "aa_noise": aa_noise,
                "ladder_correction_required": abs(delta) > aa_noise,
                "harness_constant": rep.get("harness_constant")})
    out["verdict"] = ("CERTIFIED", anchor)
    return out


def _refuse(out, why):
    out["verdict"] = ("REFUSE", why)
    return out


def render(out):
    tag, x = out["verdict"]
    if tag == "REFUSE":
        return f"M2 VERDICT: REFUSE\n  {x}"
    lines = [f"M2 ANCHOR CERTIFIED: {out['anchor_ms']:.3f} ms "
             f"+/-{out['window'] * 100:.1f}% "
             f"(n={len(out['boxes'])}, inter-box spread "
             f"{out['inter_box_spread'] * 100:.1f}%)",
             f"  vs prior {out['prior_anchor']:.2f}: "
             f"{out['delta_vs_prior'] * 100:+.2f}% "
             f"(A/A noise {out['aa_noise'] * 100:.2f}%)"]
    if out["dispersion_note"]:
        lines.append(f"  NOTE: {out['dispersion_note']}")
    if out["ladder_correction_required"]:
        d = out["delta_vs_prior"]
        lines.append(f"  LADDER CORRECTION REQUIRED: the shift exceeds "
                     f"A/A noise, so the published default entry and "
                     f"every tok/s figure derived from it must be "
                     f"restated ({'faster' if d < 0 else 'slower'} "
                     f"than published)")
    else:
        lines.append("  ladder entry stands: the shift is inside A/A "
                     "noise")
    hc = out.get("harness_constant")
    if hc is not None and abs(hc - out["anchor_ms"]) / out["anchor_ms"] > 0.005:
        lines.append(f"  harness constant {hc} is "
                     f"{100 * (hc - out['anchor_ms']) / out['anchor_ms']:+.1f}% "
                     f"off the certified value and must be corrected")
    return "\n".join(lines)


def _mk(meds=(7.20, 7.24, 7.22), aa=0.001, tok=True, rec=0,
        prior=7.35, n=None):
    boxes = []
    for i, m in enumerate(meds[:n] if n else meds):
        boxes.append({"id": f"box{i}", "a": m, "b": m * (1 + aa),
                      "tokens_identical": tok, "recompiles": rec,
                      "driver": "580", "torch": "2.13"})
    return {"boxes": boxes, "prior_anchor": prior,
            "harness_constant": 7.39}


def self_test():
    r = verdict(_mk())
    assert r["verdict"][0] == "CERTIFIED", r
    assert abs(r["anchor_ms"] - 7.2212) < 0.01, r["anchor_ms"]
    # a shift larger than A/A noise obliges a ladder correction
    assert r["ladder_correction_required"] is True
    # ...and one inside it does not. Fixtures sit clearly on ONE SIDE
    # of every threshold, never on it: aa=0.02 against AA_TOL=0.02
    # evaluates to 0.020000000000000035 and refuses, which would test
    # float representation rather than the rule.
    tight = verdict(_mk(meds=(7.36, 7.36, 7.36), aa=0.005, prior=7.35))
    assert tight["verdict"][0] == "CERTIFIED", tight
    assert tight["ladder_correction_required"] is False, tight
    # dispersion widens the window and says so
    disp = verdict(_mk(meds=(7.0, 7.3, 7.6)))
    assert disp["window"] > BASE_WINDOW and disp["dispersion_note"], disp
    # refusals
    for bad, why in ((_mk(n=2), "only 2 boxes"),
                     (_mk(aa=0.03), "A/A spread"),
                     (_mk(tok=False), "token streams"),
                     (_mk(rec=1), "recompiles"),
                     (_mk(prior=0), "no prior anchor")):
        rr = verdict(bad)
        assert rr["verdict"][0] == "REFUSE" and why in rr["verdict"][1], \
            (why, rr["verdict"])
    bad = _mk(); bad["boxes"][1]["a"] = float("nan")
    assert verdict(bad)["verdict"][0] == "REFUSE"
    for line in render(r).splitlines():
        print(f"[SELF-TEST FIXTURE, NOT A RESULT] {line}")
    print("m2_verdict self-test OK (median rule, the ladder-correction "
          "trigger either side of A/A noise, the dispersion widening, "
          "and six refusal directions)")


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
