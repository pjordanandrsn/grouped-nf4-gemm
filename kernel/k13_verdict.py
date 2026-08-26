# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Verdict calculator for PREREG-k13 Stage A.

Stage A is a CENSUS, not a treatment: there is no PASS/PARTIAL/REFUTED
band, because nothing is being tested against a bar. The calculator's
whole job is to REFUSE a census that cannot support the conclusion
Stage B would be registered on -- and to render the ranked break list
so a human reads what dynamo said rather than what someone inferred.

Report shape (written on the box):
  {"cells": {"both_disabled": CELL, "moe_compiled": CELL},
   "moe_frames": [substrings identifying the MoE tier's frames]}
  CELL = {"breaks": [BREAK, ...],
          "dispatch": {...}, "compute": {...}}   # K12 mechanism receipt
  BREAK = {"file": str, "line": int, "func": str, "reason": str,
           "count": int, "phase": "trace" | "step"}

`phase` is recorded, never scored. Under CUDA-graph capture a break
that fires once AT CAPTURE still shapes every replay, so "trace" does
not mean "free" and this file refuses to let it be rendered as such.
"""

import argparse
import json
import sys

CELLS = ("both_disabled", "moe_compiled")


def _refuse(out, why):
    out["verdict"] = ("REFUSE", why)
    return out


def _in_moe(brk, frames):
    hay = f"{brk.get('file', '')}::{brk.get('func', '')}"
    return any(f in hay for f in frames)


def verdict(rep):
    out = {"gates": {}, "census": {}, "verdict": None}
    cells = rep.get("cells") or {}
    frames = rep.get("moe_frames") or []
    if not frames:
        return _refuse(out, "no moe_frames given -- 'is this break "
                            "inside the MoE tier?' cannot be answered "
                            "against an empty definition")
    for name in CELLS:
        c = cells.get(name)
        if not c:
            return _refuse(out, f"cell {name} missing; the census "
                                "needs the untraced baseline to show "
                                "it attributes nothing there")
        if not isinstance(c.get("breaks"), list):
            return _refuse(out, f"cell {name} has no break list")
        # K12's mechanism receipt: these cells must be the knob-ON
        # configuration K12 measured, not some other one
        d = c.get("dispatch") or {}
        if d.get("dotpad", 0) + d.get("dotpad_splitk", 0) <= 0:
            return _refuse(out, f"cell {name}: dot-pad did not "
                                "dispatch, so this is not the knob-ON "
                                "frame K12 measured")

    base = cells["both_disabled"]["breaks"]
    moe = cells["moe_compiled"]["breaks"]
    out["census"] = {"both_disabled": len(base), "moe_compiled": len(moe)}

    base_moe = [b for b in base if _in_moe(b, frames)]
    if base_moe:
        return _refuse(out, f"baseline attributes {len(base_moe)} "
                            "break(s) to the MoE tier, which is not "
                            "traced there -- the census mis-attributes")

    hits = [b for b in moe if _in_moe(b, frames)]
    out["gates"]["moe_breaks"] = len(hits)
    if not moe:
        return _refuse(out, "empty census: K12 OBSERVED a break under "
                            "--compile-moe-tier, so finding none means "
                            "the instrument did not see what K12 saw. "
                            "Resolve the disagreement before banking")
    if not hits:
        return _refuse(out, "no break inside the MoE tier's frames -- "
                            "that contradicts K12's measurement rather "
                            "than refining it")

    hits.sort(key=lambda b: -int(b.get("count", 0)))
    top = hits[0]
    if top.get("phase") not in ("trace", "step"):
        return _refuse(out, "top break does not record whether it "
                            "fires at trace or per step; under graph "
                            "capture a once-at-capture break still "
                            "shapes every replay, so the distinction "
                            "may not be left blank")
    out["gates"]["top"] = top
    out["verdict"] = ("CENSUS", top)
    return out


def render(out):
    tag, x = out["verdict"]
    if tag == "REFUSE":
        return f"K13 VERDICT: REFUSE\n  {x}"
    lines = [f"K13 STAGE A: CENSUS  ({out['gates']['moe_breaks']} break(s) "
             f"in the MoE tier; baseline "
             f"{out['census']['both_disabled']} total)"]
    for b in [x]:
        lines.append(f"  TOP  {b['file']}:{b['line']} in {b['func']}")
        lines.append(f"       {b['reason']}")
        lines.append(f"       count={b['count']} phase={b['phase']}"
                     + ("   (fires once at capture -- still shapes "
                        "every replay)" if b["phase"] == "trace" else ""))
    lines.append("  Stage A names the break. It does NOT establish it "
                 "is removable, and does not license re-running K12's "
                 "treatment.")
    return "\n".join(lines)


def _mk(moe_breaks=None, base_breaks=None, frames=("hot_residency", "hybrid"),
        dotpad=384, phase="step", drop=None):
    def cell(brks, dp=dotpad):
        return {"breaks": list(brks),
                "dispatch": {"dotpad": dp, "dotpad_splitk": 0,
                             "scalar": 0, "scalar_splitk": 0},
                "compute": {"f32": 192, "fp8": 0}}
    if moe_breaks is None:
        moe_breaks = [{"file": "experts4bit_qlora/engines/hot_residency.py",
                       "line": 226, "func": "_all_hot",
                       "reason": "Graph break from `Tensor.item()`",
                       "count": 1, "phase": phase}]
    rep = {"cells": {"both_disabled": cell(base_breaks or []),
                     "moe_compiled": cell(moe_breaks)},
           "moe_frames": list(frames)}
    if drop:
        del rep["cells"][drop]
    return rep


def self_test():
    r = verdict(_mk())
    assert r["verdict"][0] == "CENSUS", r
    assert r["gates"]["top"]["func"] == "_all_hot"

    # REFUSE directions, each driven by a report that reaches it
    assert "cell both_disabled missing" in verdict(
        _mk(drop="both_disabled"))["verdict"][1] or "missing" in verdict(
        _mk(drop="both_disabled"))["verdict"][1]
    assert "empty census" in verdict(_mk(moe_breaks=[]))["verdict"][1]
    # a break OUTSIDE the MoE frames does not count as one inside
    outside = [{"file": "torch/nn/modules/module.py", "line": 1, "func": "x",
                "reason": "r", "count": 1, "phase": "step"}]
    assert "no break inside" in verdict(_mk(moe_breaks=outside))["verdict"][1]
    # baseline must not attribute breaks to an untraced region
    assert "mis-attributes" in verdict(
        _mk(base_breaks=[{"file": "hot_residency.py", "line": 9,
                          "func": "_fwd", "reason": "r", "count": 1,
                          "phase": "step"}]))["verdict"][1]
    assert "knob-ON" in verdict(_mk(dotpad=0))["verdict"][1]
    assert "empty definition" in verdict(_mk(frames=()))["verdict"][1]
    # phase must be stated
    assert "trace or per step" in verdict(_mk(phase=None))["verdict"][1]

    # ranking picks the largest count, not the first listed
    many = [{"file": "hot_residency.py", "line": 1, "func": "a",
             "reason": "r", "count": 2, "phase": "step"},
            {"file": "hot_residency.py", "line": 2, "func": "b",
             "reason": "r", "count": 9, "phase": "step"}]
    assert verdict(_mk(moe_breaks=many))["gates"]["top"]["func"] == "b"

    # a trace-phase break renders its caveat rather than reading free
    t = verdict(_mk(phase="trace"))
    assert "still shapes" in render(t)

    print(render(r))
    print("k13_verdict self-test OK (census render, ranking by count, "
          "the trace-phase caveat, and seven refusal directions)")


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
