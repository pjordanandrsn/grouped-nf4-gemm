#!/usr/bin/env python3
"""Mechanical reducer for PREREG-unsloth-head-to-head. Reads receipts, applies
the frozen criteria, prints a verdict. No judgement calls live here.

Cell value = median across reps of the per-rep PAIRED ratio, taken within one
cell of one rep so drift cancels. Worst rep reported alongside, because a
median that hides a bad rep is how a claim survives that should not have.

    python bench/phase1/reduce_unsloth_h2h.py <receipt.json> [...] [--json out]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

# --- frozen criteria params (mirror of the prereg; single source is the JSON) -
P1_BAR, P1_MIN, P1_TOTAL = 1.0, 7, 8
P2_BAR, P2_MIN, P2_TOTAL = 1.0, 5, 8
P1_BAND = (2.0, 6.0)
P2_BAND = (1.2, 3.0)
SELF_PAIR_BAND = (0.97, 1.03)
DECODE_REGIMES = ("decode_bs1", "decode_m8")
PREFILL_REGIMES = ("prefill_s2048",)

BASE = "fused_nf4"
RATIOS = {
    # name -> (numerator backend, denominator backend). >1 means the fused
    # kernel is faster, for every row, so the table reads one direction only.
    "unsloth_native_vs_fused": ("unsloth_native", BASE),
    "unsloth_bf16_vs_fused": ("unsloth_native_bf16", BASE),
    "dequant_vs_fused": ("dequant_grouped", BASE),
    "proxy_vs_native": ("unsloth", "unsloth_native"),
    # Under --paired-base the BASE row carries its own adjacent base-vs-base
    # ratio, so the dedicated selfpair alias is redundant. Both keys are read;
    # whichever the receipt carries is used.
    "self_pair": (BASE, BASE),
    "self_pair_bracketed": ("fused_nf4_selfpair", BASE),
}


def load(paths):
    reps = []
    for p in paths:
        d = json.loads(Path(p).read_text())
        idx = {}
        for c in d.get("cells", []):
            if c.get("ms_median") is None:
                continue
            idx[(c["model"], c["proj"], c["regime"], c["backend"])] = c
        reps.append({"env": d.get("env", {}), "idx": idx, "src": str(p)})
    return reps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("receipts", nargs="+")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    reps = load(args.receipts)
    if not reps:
        print("no receipts")
        return 2
    env = reps[0]["env"]

    cells = sorted({(k[0], k[1], k[2]) for r in reps for k in r["idx"]})
    per_ratio = defaultdict(dict)   # ratio -> cell -> {median, worst, n}
    for name, (num, den) in RATIOS.items():
        for cell in cells:
            vals = []
            for r in reps:
                a = r["idx"].get((*cell, num))
                b = r["idx"].get((*cell, den))
                # PREFERRED: the true paired ratio, where the base was re-timed
                # immediately before this backend. Only valid when the
                # denominator IS the paired base.
                if a is not None and den == BASE and a.get("paired_ratio"):
                    vals.append(a["paired_ratio"])
                elif num == den:
                    # Degenerate: dividing a backend's single timing by itself
                    # is exactly 1.0 and would let an UNPAIRED receipt sail
                    # through the self-pair validity gate. Emit nothing, so Q2
                    # falls back to the bracketed key or reports "no data"
                    # rather than a fabricated pass.
                    continue
                elif a and b and a["ms_median"] and b["ms_median"]:
                    # Fallback for unpaired receipts: numerator_time /
                    # denominator_time, so >1 == fused faster, matching the
                    # prereg's "paired unsloth_native/fused_nf4 >= 1.0" wording.
                    # (A first draft divided the other way and scored P1/P2
                    # backwards on correct data.) This path is NOT paired and
                    # any verdict resting on it says so.
                    vals.append(a["ms_median"] / b["ms_median"])
            if vals:
                per_ratio[name][cell] = {
                    "median": statistics.median(vals),
                    "worst": min(vals),
                    "n": len(vals),
                }

    out = {
        "gpu": env.get("gpu"),
        "capability": env.get("capability"),
        "torch": env.get("torch"),
        "unsloth_native_env": env.get("unsloth_native"),
        "n_reps": len(reps),
    }

    # ---- Q3 engagement (positive control) ----------------------------------
    impls = {
        r["idx"][k].get("impl")
        for r in reps
        for k in r["idx"]
        if k[3] == "unsloth_native" and r["idx"][k].get("impl")
    }
    fp = env.get("unsloth_native") or {}
    tma = fp.get("supports_tma") if isinstance(fp, dict) else None
    cap = (env.get("capability") or "0.0").split(".")[0]
    expect_tma = cap.isdigit() and int(cap) >= 9
    out["Q3_impls_seen"] = sorted(i for i in impls if i)
    out["Q3_tma_live"] = tma
    out["Q3_tma_expected"] = expect_tma
    out["Q3_PASS"] = bool(
        impls
        and all("unsloth.kernels.moe" in i for i in impls if i)
        and tma == expect_tma
    )

    # ---- Q2 self-pair (instrument validity) --------------------------------
    # Prefer the ADJACENT base-vs-base pair; fall back to the older bracketed
    # placement only if this receipt has no paired data at all.
    sp_key = "self_pair" if per_ratio["self_pair"] else "self_pair_bracketed"
    sp = [v["median"] for v in per_ratio[sp_key].values()]
    out["Q2_kind"] = "adjacent" if sp_key == "self_pair" else "bracketed(cell-spanning)"
    out["Q2_self_pair_range"] = [min(sp), max(sp)] if sp else None
    out["Q2_worst_cells"] = sorted(
        (
            {"cell": "|".join(c), "median": round(v["median"], 4)}
            for c, v in per_ratio[sp_key].items()
            if not (SELF_PAIR_BAND[0] <= v["median"] <= SELF_PAIR_BAND[1])
        ),
        key=lambda d: abs(d["median"] - 1.0),
        reverse=True,
    )[:5]
    out["Q2_PASS"] = bool(sp) and all(
        SELF_PAIR_BAND[0] <= v <= SELF_PAIR_BAND[1] for v in sp
    )
    if sp and not out["Q2_PASS"]:
        out["Q2_NOTE"] = (
            "VOID, not FAILED: the instrument drifted mid-cell, so every ratio "
            "measured beside it is noise. Re-run; do not report."
        )

    # ---- P1 / P2 ------------------------------------------------------------
    def score(regimes, bar, need, total, band, key):
        rows = {c: v for c, v in per_ratio["unsloth_native_vs_fused"].items()
                if c[2] in regimes}
        won = [c for c, v in rows.items() if v["median"] >= bar]
        meds = [v["median"] for v in rows.values()]
        med = statistics.median(meds) if meds else None
        out[f"{key}_cells_measured"] = len(rows)
        out[f"{key}_cells_at_or_above_bar"] = len(won)
        out[f"{key}_median_ratio"] = med
        out[f"{key}_predicted_band"] = list(band)
        out[f"{key}_band_hit"] = bool(med is not None and band[0] <= med <= band[1])
        # The bar is scored over cells MEASURED; NOT-RUN cells are reported, never
        # backfilled. If fewer than `total` ran, that is disclosed, not smoothed.
        out[f"{key}_PASS"] = bool(rows) and len(won) >= min(need, len(rows)) and len(rows) >= need
        return med

    score(DECODE_REGIMES, P1_BAR, P1_MIN, P1_TOTAL, P1_BAND, "P1_decode")
    score(PREFILL_REGIMES, P2_BAR, P2_MIN, P2_TOTAL, P2_BAND, "P2_prefill")

    out["VERDICT"] = (
        "H2H_CONFIRMED"
        if (out.get("P1_decode_PASS") and out.get("P2_prefill_PASS")
            and out["Q2_PASS"] and out["Q3_PASS"])
        else "NOT_CONFIRMED"
    )

    # ---- report -------------------------------------------------------------
    print(f"== {out['gpu']} (cc {out['capability']}) reps={out['n_reps']} "
          f"TMA={out['Q3_tma_live']} ==")
    hdr = f"{'cell':52s} {'uns/fused':>10s} {'bf16/fused':>11s} {'proxy/nat':>10s} {'self':>6s}"
    print(hdr)
    for c in cells:
        def g(n):
            v = per_ratio[n].get(c)
            return f"{v['median']:.3f}" if v else "  -  "
        label = f"{c[0].split('/')[-1][:26]:26s} {c[1]:8s} {c[2]:14s}"
        print(f"{label} {g('unsloth_native_vs_fused'):>10s} "
              f"{g('unsloth_bf16_vs_fused'):>11s} {g('proxy_vs_native'):>10s} "
              f"{g('self_pair'):>6s}")
    for k in ("Q2_PASS", "Q3_PASS", "P1_decode_PASS", "P1_decode_median_ratio",
              "P1_decode_band_hit", "P2_prefill_PASS", "P2_prefill_median_ratio",
              "P2_prefill_band_hit", "VERDICT"):
        print(f"{k}: {out.get(k)}")
    if "Q2_NOTE" in out:
        print(out["Q2_NOTE"])

    out["table"] = {f"{c[0]}|{c[1]}|{c[2]}": {n: per_ratio[n].get(c) for n in RATIOS}
                    for c in cells}
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
