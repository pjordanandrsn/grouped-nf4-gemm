#!/usr/bin/env python3
"""Reduce the C2 capturability regression gate.

    python reduce_capturability_gate.py <evidence-dir>

Reads the six receipts a gate run produces -- {pub1,cap,pub2} x offload{1,0} --
and grades the change against the band registered in
`kernel/prereg_capturability_scope.json`.

WHAT IT COMPARES, AND WHY IT IS NOT A COMPARISON AGAINST THE PUBLISHED NUMBERS.
The published e2e ratios were measured on a 4090 and an H100, on other pods, at
another e4b version. Comparing a new pod's ratio against those would be a
cross-run comparison, which this program's rules forbid. So the gate runs three
sweeps on ONE card in one session:

    pub1   published grouped-nf4-gemm wheel      (pre-change call path)
    cap    the same wheel with nf4_grouped.py and nf4_qlora.py overwritten by
           the changed files, nothing else touched
    pub2   published files restored              (version-level self-pair)

`pub2/pub1` is the instrument's own self-pair. Same code twice: if THAT is not
in band, the box drifted and `cap/pub1` says nothing about the change. It is
checked first and it can void the run, exactly as the per-cell self-pair gates
every other leg here.

An empty comparison is UNUSABLE, never a pass. A gate that reports "all 0 cells
inside the band" is the most dangerous output this script could produce.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

BAND = (0.967, 1.032)          # kernel/prereg_capturability_scope.json
ARMS = ("fast_train", "fast_train_dgrad")
MIN_CELLS = 4                  # 2 data modes x 2 offload settings, per arm


def load(evid: Path, label: str):
    """{(mode, offload, arm): speedup} plus the per-sweep reference self-pairs."""
    speed, selfpair, problems = {}, {}, []
    for off in (1, 0):
        p = evid / f"e2e_{label}-off{off}.json"
        if not p.exists():
            problems.append(f"missing {p.name}")
            continue
        try:
            d = json.loads(p.read_text())
        except Exception as e:
            problems.append(f"{p.name}: {type(e).__name__}: {e}")
            continue
        for mode, arms in (d.get("cells") or {}).items():
            for name, r in arms.items():
                if r.get("INVALID_no_modules_patched"):
                    problems.append(f"{label}/{mode}/{name}: patched 0 modules")
                if r.get("frozen_changed"):
                    problems.append(f"{label}/{mode}/{name}: frozen bytes CHANGED")
                if name in ARMS and r.get("speedup_vs_reference"):
                    speed[(mode, off, name)] = r["speedup_vs_reference"]
            sp = (arms.get("reference_selfpair") or {}).get("speedup_vs_reference")
            selfpair[(mode, off)] = sp
            if sp is None:
                problems.append(f"{label}/{mode}/off{off}: no reference self-pair")
    return speed, selfpair, problems


def ratios(a, b):
    return {k: b[k] / a[k] for k in sorted(set(a) & set(b)) if a[k] and b[k]}


def block(title, d):
    print(f"\n--- {title} ---")
    for (mode, off, name), v in d.items():
        flag = "" if BAND[0] <= v <= BAND[1] else "   <-- OUTSIDE BAND"
        print(f"  {mode:<7} offload={off} {name:<18} {v:.4f}{flag}")
    if d:
        print(f"  median {st.median(d.values()):.4f}   n={len(d)}")
    else:
        print("  (no comparable cells)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("evidence", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    sweeps = {l: load(a.evidence, l) for l in ("pub1", "cap", "pub2")}
    problems = [p for _, _, probs in sweeps.values() for p in probs]

    print("=== per-sweep reference self-pairs (drift inside each sweep) ===")
    for lbl, (_, sp, _) in sweeps.items():
        for (mode, off), v in sorted(sp.items()):
            print(f"  {lbl:<5} {mode:<7} offload={off} "
                  f"{'None' if v is None else format(v, '.4f')}")

    inst = ratios(sweeps["pub1"][0], sweeps["pub2"][0])
    gate = ratios(sweeps["pub1"][0], sweeps["cap"][0])
    block("INSTRUMENT self-pair  pub2/pub1 (same code twice)", inst)
    block("GATE  cap/pub1 (the capturability change)", gate)

    inst_bad = {k: v for k, v in inst.items() if not BAND[0] <= v <= BAND[1]}
    gate_bad = {k: v for k, v in gate.items() if not BAND[0] <= v <= BAND[1]}
    worst = lambda d: max(d.values(), key=lambda x: abs(1 - x))  # noqa: E731

    # ORDER MATTERS. Emptiness is checked BEFORE the band, on both comparisons,
    # because "no cells were outside the band" is trivially true of no cells and
    # would otherwise read as a pass.
    if problems:
        v = f"VERDICT: UNUSABLE — {len(problems)} receipt problem(s): {problems[:4]}"
    elif len(inst) < MIN_CELLS * len(ARMS) or len(gate) < MIN_CELLS * len(ARMS):
        v = (f"VERDICT: UNUSABLE — expected {MIN_CELLS * len(ARMS)} comparable "
             f"cells per comparison, got instrument={len(inst)} gate={len(gate)}. "
             f"An empty or partial comparison is not a pass.")
    elif inst_bad:
        v = (f"VERDICT: VOID — the INSTRUMENT self-pair left the band on "
             f"{len(inst_bad)}/{len(inst)} cells (worst {worst(inst_bad):.4f}). "
             f"The box drifted; cap/pub1 is not readable, and this is NOT a "
             f"statement about the change.")
    elif gate_bad:
        v = (f"VERDICT: GATE FAILED — {len(gate_bad)}/{len(gate)} cells outside "
             f"{BAND[0]}-{BAND[1]} (worst {worst(gate_bad):.4f}). The "
             f"capturability change moved throughput; per the registered stop "
             f"rule it is reverted.")
    else:
        v = (f"VERDICT: GATE PASSED — all {len(gate)} cells inside "
             f"{BAND[0]}-{BAND[1]} (median {st.median(gate.values()):.4f}), "
             f"instrument self-pair clean on all {len(inst)}.")
    print("\n" + v)
    if a.out:
        a.out.write_text(json.dumps(
            {"band": BAND, "instrument_selfpair": {str(k): v for k, v in inst.items()},
             "gate": {str(k): v for k, v in gate.items()},
             "problems": problems, "verdict": v}, indent=1))
    return 0 if v.startswith("VERDICT: GATE PASSED") else 1


if __name__ == "__main__":
    sys.exit(main())
