# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Reduce one sweep card's hw_harness.json to its cross-arch table row:
fused/dequant speedup (min-max, median) over the 8 census decode cells, plus
the prefill gate_up and down bands. Validated against the published B200/L40
rows before use on new cards (anchor-before-extend)."""
import json
import statistics
import sys


def reduce(path):
    d = json.load(open(path))
    cells = [c for c in d["cells"] if c.get("status") == "ok"]

    def t(model, proj, regime, backend):
        for c in cells:
            if (c["model"] == model and c["proj"] == proj
                    and c["regime"] == regime and c["backend"] == backend):
                return c["ms_median"]
        return None

    keys = sorted({(c["model"], c["proj"]) for c in cells})
    out = {}
    for regime in ("decode_bs1", "prefill_s2048"):
        ratios = {}
        for (m, p) in keys:
            dq, fu = t(m, p, regime, "dequant_grouped"), t(m, p, regime, "fused_nf4")
            if dq and fu:
                ratios[(m.split("/")[-1], p)] = dq / fu
        out[regime] = ratios
    return out


def band(r, lo=None):
    v = sorted(r.values())
    return f"{v[0]:.2f}-{v[-1]:.2f} (med {statistics.median(v):.2f})"


if __name__ == "__main__":
    out = reduce(sys.argv[1])
    dec = out["decode_bs1"]
    print(f"decode census: {band(dec)}   [{len(dec)} cells]")
    pre = out["prefill_s2048"]
    gu = {k: v for k, v in pre.items() if k[1] == "gate_up"}
    dn = {k: v for k, v in pre.items() if k[1] == "down"}
    print(f"prefill gate_up: {band(gu)}")
    print(f"prefill down:    {band(dn)}")
    for (m, p), v in sorted(dec.items()):
        print(f"  decode {m:28s} {p:8s} {v:.2f}x")
