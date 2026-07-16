# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Mechanical verdicts from frozen criteria (CHARTER §3.3 / §3.4).

reduce_ceiling.py ceiling <runs.json>   — ladder verdict per (family, band, delta)
reduce_ceiling.py gate    <gate.json>   — fixture-gate 4/4 grading (Phase-0 exit)

runs.json: {"family":…, "band":…, "delta":…, "rungs":[{"name":…, "train_h":…,
"heldout_h":…}, … in ladder order]}
gate.json: {"fixtures":[{"name":…, "kind":"planted"|"null", "target":…,
"chance":…, "best_heldout_h":…}, …]}

Verdicts are arithmetic. This file takes no flags that change thresholds; the
thresholds live in procedure.yaml and nowhere else.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent


def _load_yaml(path):
    # minimal flat-yaml reader (no dependency): supports the procedure.yaml subset
    import re
    out, stack = {}, [(-1, None)]
    cur = out
    parents = {0: out}
    prev_indent, holders = 0, {0: out}
    for raw in Path(path).read_text().splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        key_part = line.strip()
        m = re.match(r"^([\w.]+):\s*(.*)$", key_part)
        if not m:
            continue
        key, val = m.group(1), m.group(2)
        holder = holders.get(indent)
        if holder is None:
            # attach to nearest shallower holder
            cand = max(i for i in holders if i < indent)
            holder = holders[cand]
        if val == "":
            holder[key] = {}
            holders[indent + 2] = holder[key]
        else:
            holder[key] = _scalar(val)
    return out


def _scalar(v):
    v = v.strip()
    if v.startswith("[") and v.endswith("]"):
        inner = v[1:-1].strip()
        return [] if not inner else [_scalar(x) for x in inner.split(",")]
    if v.startswith("{") and v.endswith("}"):
        d = {}
        for part in v[1:-1].split(","):
            if ":" in part:
                k, vv = part.split(":", 1)
                d[k.strip()] = _scalar(vv)
        return d
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v.strip("'\"")


def proc():
    return _load_yaml(HERE / "procedure.yaml")


def verdict_ceiling(run: dict, crit: dict) -> dict:
    rungs = run["rungs"]
    hs = [r["heldout_h"] for r in rungs]
    gain = float(crit["plateau_gain_abs"])
    gap_min = float(crit["train_heldout_gap_min"])
    rv_h = float(crit["runtime_viable_h"])
    top = rungs[-1]
    verdicts = []
    # runtime-viable: any rung >= threshold held-out at delta==primary
    if int(run.get("delta", 1)) == 1 and any(h >= rv_h for h in hs):
        verdicts.append("runtime-viable")
    top_gain = hs[-1] - hs[-2]
    prev_gain = hs[-2] - hs[-3] if len(hs) >= 3 else float("inf")
    if top_gain >= gain:
        verdicts.append("probe-limited (ceiling not established)")
    elif (top_gain < gain and prev_gain < gain
          and (top["train_h"] - top["heldout_h"]) >= gap_min):
        verdicts.append("model-limited")
    else:
        verdicts.append("plateau-without-overfit-gap (no verdict; extend data or ladder)")
    return {"family": run.get("family"), "band": run.get("band"),
            "delta": run.get("delta"), "heldout_by_rung": hs,
            "verdict": verdicts}


def verdict_gate(gate: dict, fx: dict) -> dict:
    band = float(fx["recovery_band_abs"])
    nullm = float(fx["null_margin_abs"])
    rows, passes = [], 0
    for f in gate["fixtures"]:
        if f["kind"] == "planted":
            ok = abs(f["best_heldout_h"] - f["target"]) <= band
            rows.append({"name": f["name"], "target": f["target"],
                         "best_heldout_h": f["best_heldout_h"],
                         "pass": bool(ok)})
        else:
            ok = f["best_heldout_h"] <= f["chance"] + nullm
            rows.append({"name": f["name"], "chance": f["chance"],
                         "best_heldout_h": f["best_heldout_h"],
                         "pass": bool(ok), "leakage_alarm": not ok})
        passes += bool(ok)
    return {"gate": f"{passes}/{len(rows)}",
            "exit_phase0": passes == len(rows), "rows": rows}


def main():
    mode, path = sys.argv[1], sys.argv[2]
    p = proc()
    data = json.loads(Path(path).read_text())
    if mode == "ceiling":
        out = verdict_ceiling(data, p["criteria"])
    elif mode == "gate":
        out = verdict_gate(data, p["fixture_gate"])
    else:
        raise SystemExit("mode must be 'ceiling' or 'gate'")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
