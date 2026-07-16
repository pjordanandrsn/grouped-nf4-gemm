# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Phase-0 exit runner (CHARTER §3.4): build the four fixtures, run the full
ladder on each through the contract dataloader, write receipts, and let the
committed reducer grade the gate. Exploratory tier; labels in-band.

Usage: python3 run_fixture_gate.py [--fast]
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from capture.streams import ContractLoader           # noqa: E402
from fixtures.planted import make_null, make_planted  # noqa: E402
from probes.ladder import train_eval_rung             # noqa: E402
from reduce.reduce_ceiling import proc                 # noqa: E402

P = proc()
FX = P["fixture_gate"]
DIMS = {k: int(v) for k, v in FX["fixture_dims"].items()}
N = {k: int(v) for k, v in FX["samples"].items()}
SEED = int(FX["seed"])
TRAIN = P["training"]
LADDER = [
    {"name": "linear", "kind": "linear"},
    {"name": "mlp_d", "kind": "mlp", "width_mult": 1},
    {"name": "mlp_4d", "kind": "mlp", "width_mult": 4},
    {"name": "attn2", "kind": "attn", "heads": 4, "layers": 2},
]

def run():
    t0 = time.time()
    date = time.strftime("%Y%m%d")
    rdir = Path(__file__).parent / "receipts" / date
    rdir.mkdir(parents=True, exist_ok=True)
    work = Path(__file__).parent / "receipts" / date / "fixture_streams"
    n_total = N["train"] + N["heldout"] + 1
    fixtures = []
    specs = [(f"planted_{int(h*100)}", h) for h in FX["planted_levels"]] + [("null", None)]
    for name, h in specs:
        d = work / name
        if h is not None:
            target = make_planted(d, h, DIMS, n_total, SEED)
        else:
            target = make_null(d, DIMS, n_total, SEED + 7)
        loader = ContractLoader(d, delta=1)
        tX, ty, hX, hy = loader.split(heldout=N["heldout"], seed=SEED)
        rungs = []
        for rung in LADDER:
            r = train_eval_rung(rung, tX, ty, hX, hy, loader.E, loader.k, TRAIN)
            rungs.append({"name": rung["name"], **r})
            print(f"[{name}] {rung['name']:8s} train={r['train_h']:.4f} heldout={r['heldout_h']:.4f}", flush=True)
        best = max(r["heldout_h"] for r in rungs)
        fixtures.append({
            "name": name, "kind": "planted" if h is not None else "null",
            "target": target if h is not None else None,
            "chance": None if h is not None else target,
            "best_heldout_h": best, "rungs": rungs,
        })
    gate = {"tier": "EXPLORATORY", "charter": "router_probe/CHARTER.md",
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_s": round(time.time() - t0, 1), "fixtures": fixtures}
    gate_path = rdir / "EXPLORATORY_fixture_gate.json"
    gate_path.write_text(json.dumps(gate, indent=1))
    print(f"\nreceipt: {gate_path}")
    print("--- committed reducer verdict ---", flush=True)
    subprocess.run([sys.executable, str(Path(__file__).parent / "reduce" / "reduce_ceiling.py"),
                    "gate", str(gate_path)], check=True)


if __name__ == "__main__":
    run()
