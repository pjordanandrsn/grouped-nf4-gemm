#!/usr/bin/env python3
"""Reduce the GPU-bound capturability gate (tokbudget_4096).

    python reduce_capturability_kernel_gate.py <evidence-dir>

Grades `kernel/prereg_capturability_gate_tokbudget.json` (OTS-stamped pre-data).
Reads receipts named `dequant_forward_{pub1,cap,pub2}-<model>.json`.

The graded quantity is the FUSED arm's own per-step time, `ms.g_a`, because both
arms call the identical `lora_delta_grouped` and this change touches that too --
so `d_over_g` partially cancels the thing under test. It is reported, not graded.

Order of checks is the point of this file:
  P1  every graded cell must be measurement_class == kernel (both arms >= 50%
      GPU-busy). A cell that is not is dropped -- the premise of this gate is
      that the cell is GPU-bound.
  --  a cell whose own g_selfpair left the band is dropped (the leg's own rule).
  P2  pub2/pub1 -- the IDENTICAL code twice -- must hold the band. Checked BEFORE
      the gate ratio is read, and it can VOID the run. This is the prediction
      that matters: if a GPU-bound cell cannot hold 3% either, no gate of this
      shape is measurable on rented shared hardware.
  P3  cap/pub1 -- the change.

Emptiness is checked before the band, on both comparisons: "no cells outside the
band" is trivially true of no cells.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

BAND = (0.967, 1.032)          # carried over unchanged from the scope prereg
SELFPAIR_BAND = (0.967, 1.032)  # the leg's own per-cell g_selfpair rule
BUSY_BAR = 0.50


def load(evid: Path, label: str):
    """{(model, proj): row} for every ok cell of one sweep, across all models."""
    cells, problems = {}, []
    hits = sorted(evid.glob(f"dequant_forward_{label}-*.json"))
    if not hits:
        problems.append(f"no receipts for sweep {label!r}")
    for p in hits:
        try:
            d = json.loads(p.read_text())
        except Exception as e:
            problems.append(f"{p.name}: {type(e).__name__}: {e}")
            continue
        for r in d.get("rows", []):
            if r.get("status") != "ok":
                problems.append(f"{label}/{r.get('model')}/{r.get('proj')}: "
                                f"status={r.get('status')} {r.get('reason','')[:60]}")
                continue
            key = (r["model"].split("/")[-1], r["proj"])
            cells[key] = r
    return cells, problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("evidence", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    sweeps, problems = {}, []
    for lbl in ("pub1", "cap", "pub2"):
        c, pr = load(a.evidence, lbl)
        sweeps[lbl] = c
        problems += pr

    common = sorted(set(sweeps["pub1"]) & set(sweeps["cap"]) & set(sweeps["pub2"]))
    print(f"=== cells present in all three sweeps: {len(common)} ===")

    dropped, graded = {}, []
    print("\n=== per-cell gates (P1 premise, and the leg's own self-pair) ===")
    for k in common:
        why = []
        for lbl in ("pub1", "cap", "pub2"):
            r = sweeps[lbl][k]
            mc = r.get("measurement_class")
            if mc != "kernel":
                why.append(f"{lbl}:measurement_class={mc}"
                           f"(min_busy={r.get('min_busy_fraction')})")
            sp = r.get("g_selfpair")
            if sp is None or not (SELFPAIR_BAND[0] <= sp <= SELFPAIR_BAND[1]):
                why.append(f"{lbl}:g_selfpair={sp}")
        busy = {lbl: (sweeps[lbl][k].get("min_busy_fraction") or 0) for lbl in sweeps}
        print("  %-24s %-8s busy(min) %s  %s" % (
            k[0], k[1], " ".join(f"{lbl}={100*v:.0f}%" for lbl, v in busy.items()),
            "DROPPED: " + "; ".join(why) if why else "graded"))
        (dropped.setdefault(k, why) if why else graded.append(k))

    def ratio(num, den, key):
        x, y = sweeps[den][key]["ms"]["g_a"], sweeps[num][key]["ms"]["g_a"]
        return y / x if x else None

    inst = {k: ratio("pub2", "pub1", k) for k in graded}
    gate = {k: ratio("cap", "pub1", k) for k in graded}
    comp = {k: (sweeps["cap"][k]["d_over_g"] / sweeps["pub1"][k]["d_over_g"])
            for k in graded if sweeps["pub1"][k].get("d_over_g")}

    def block(title, d):
        print(f"\n--- {title} ---")
        for k, v in sorted(d.items()):
            flag = "" if v is not None and BAND[0] <= v <= BAND[1] else "   <-- OUTSIDE BAND"
            print("  %-24s %-8s %.4f%s" % (k[0], k[1], v, flag))
        if d:
            print("  median %.4f   n=%d" % (st.median(d.values()), len(d)))
        else:
            print("  (none)")

    block("P2 INSTRUMENT  pub2/pub1 of ms.g_a (same code twice)", inst)
    block("P3 GATE        cap/pub1 of ms.g_a (the change)", gate)
    block("secondary, NOT graded: d_over_g cap/pub1 (partially cancels)", comp)

    inst_bad = {k: v for k, v in inst.items() if not BAND[0] <= v <= BAND[1]}
    gate_bad = {k: v for k, v in gate.items() if not BAND[0] <= v <= BAND[1]}
    worst = lambda d: max(d.values(), key=lambda x: abs(1 - x))  # noqa: E731

    if problems:
        v = f"VERDICT: UNUSABLE — {len(problems)} receipt problem(s): {problems[:4]}"
    elif not graded:
        v = (f"VERDICT: UNUSABLE — 0 of {len(common)} cells survived the per-cell "
             f"gates. P1 FALSIFIED if the reason is measurement_class: "
             f"tokbudget_4096 is not the GPU-bound cell it was chosen to be.")
    elif len(graded) < 2:
        v = (f"VERDICT: UNUSABLE — only {len(graded)} cell(s) graded; too few to "
             f"adjudicate. Not a pass.")
    elif inst_bad:
        v = (f"VERDICT: VOID — P2 FALSIFIED. The instrument self-pair left the "
             f"band on {len(inst_bad)}/{len(inst)} cells (worst "
             f"{worst(inst_bad):.4f}) on a GPU-BOUND cell. The redesign fails "
             f"too: the e2e gate's collapse was not special to that leg, and no "
             f"gate of this shape is measurable on rented shared hardware. NOT a "
             f"statement about the change.")
    elif gate_bad:
        v = (f"VERDICT: GATE FAILED — P3. {len(gate_bad)}/{len(gate)} cells "
             f"outside {BAND[0]}-{BAND[1]} (worst {worst(gate_bad):.4f}). The "
             f"change moved throughput at a kernel-bound size; per the stop rule "
             f"it is reverted.")
    else:
        v = (f"VERDICT: GATE PASSED — all {len(gate)} graded cells inside "
             f"{BAND[0]}-{BAND[1]} (median {st.median(gate.values()):.4f}), "
             f"instrument clean on all {len(inst)} (median "
             f"{st.median(inst.values()):.4f}). Does NOT close the e2e gate, "
             f"which stays OPEN and unmet.")
    print("\n" + v)
    if a.out:
        a.out.write_text(json.dumps(
            {"band": BAND, "graded": [list(k) for k in graded],
             "dropped": {f"{k[0]}/{k[1]}": w for k, w in dropped.items()},
             "instrument": {f"{k[0]}/{k[1]}": v for k, v in inst.items()},
             "gate": {f"{k[0]}/{k[1]}": v for k, v in gate.items()},
             "d_over_g_secondary": {f"{k[0]}/{k[1]}": v for k, v in comp.items()},
             "problems": problems, "verdict": v}, indent=1))
    return 0 if v.startswith("VERDICT: GATE PASSED") else 1


if __name__ == "__main__":
    sys.exit(main())
