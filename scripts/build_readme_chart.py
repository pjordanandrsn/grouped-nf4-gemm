#!/usr/bin/env python3
"""Build the README's first-screen chart from docs/claims.json, so the chart cannot drift from the register.

    python scripts/build_readme_chart.py          # write docs/assets/speed-vs-unsloth-rtx4090-h100.svg
    python scripts/build_readme_chart.py --check  # exit 1 if the committed SVG differs from a fresh build

Every number comes from two claims' own sentences, parsed with strict patterns: a reworded claim fails the build
instead of silently charting a stale figure.
  gnf4.kernel.h2h-unsloth              decode vs Unsloth's MoE kernel (4-bit storage), and Unsloth's bf16-resident
                                       prefill win, on the same chart so the chart never shows only the wins
  gnf4.kernel.e2e-training-real-prose  OLMoE QLoRA on real prose vs this project's per-expert loop: a different
                                       baseline, so its own panel
Bars sit on a log2 axis centred on 1x (equal speed): right = grouped-nf4-gemm faster, left = Unsloth faster.
Colours: the dataviz reference diverging pair (blue / red), validated for CVD and contrast in light and dark mode.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets" / "speed-vs-unsloth-rtx4090-h100.svg"

H2H_DECODE = re.compile(r"decode (\d+\.\d+)x on H100 \(their TMA path live\) and (\d+\.\d+)x on RTX 4090")
H2H_PREFILL = re.compile(r"Unsloth wins its own bf16-resident regime by (\d+\.\d+)-(\d+\.\d+)x at prefill on the H100")
E2E = re.compile(r"runs (\d+\.\d+)x on a 4090 and (\d+\.\d+)x on an H100 on real prose")


def figures(claims_path: Path) -> dict:
    claims = {c["id"]: c for c in json.loads(claims_path.read_text())["claims"]}
    h2h, e2e = claims["gnf4.kernel.h2h-unsloth"]["claim"], claims["gnf4.kernel.e2e-training-real-prose"]["claim"]
    found = {"decode": H2H_DECODE.search(h2h), "prefill": H2H_PREFILL.search(h2h), "e2e": E2E.search(e2e)}
    missing = [k for k, m in found.items() if not m]
    if missing:
        sys.exit(f"FAIL: the claim wording changed, so these figures could not be read: {missing}. "
                 "Update the patterns in scripts/build_readme_chart.py together with the claim.")
    d, p, e = found["decode"].groups(), found["prefill"].groups(), found["e2e"].groups()
    return {"decode_h100": d[0], "decode_4090": d[1], "prefill_lo": p[0], "prefill_hi": p[1],
            "e2e_4090": e[0], "e2e_h100": e[1]}


W, LABEL_W, PAD = 840, 250, 24
AXIS_L, AXIS_R = LABEL_W + 40, W - PAD - 70
MID = (AXIS_L + AXIS_R) / 2
SPAN = 3.0  # log2 units each side: 1x .. 8x


def x(ratio: float, side: int) -> float:
    return MID + side * (math.log2(ratio) / SPAN) * (AXIS_R - MID)


def bar(y: float, ratio: float, side: int, cls: str, h: float = 16) -> str:
    """A bar from the 1x centre to `ratio` on `side` (+1 right, -1 left); 4px rounded data end, square at the base."""
    x0, x1 = MID, x(ratio, side)
    lo, hi = min(x0, x1), max(x0, x1)
    r = 4
    if side > 0:
        path = f"M{lo:.1f},{y} H{hi - r:.1f} Q{hi:.1f},{y} {hi:.1f},{y + r} V{y + h - r} Q{hi:.1f},{y + h} {hi - r:.1f},{y + h} H{lo:.1f} Z"
    else:
        path = f"M{hi:.1f},{y} H{lo + r:.1f} Q{lo:.1f},{y} {lo:.1f},{y + r} V{y + h - r} Q{lo:.1f},{y + h} {lo + r:.1f},{y + h} H{hi:.1f} Z"
    return f'<path class="{cls}" d="{path}"/>'


def build(f: dict) -> str:
    out = []
    def text(x_, y_, s, cls="t2", anchor="start", size=13, weight=400):
        out.append(f'<text x="{x_:.1f}" y="{y_:.1f}" class="{cls}" text-anchor="{anchor}" font-size="{size}" font-weight="{weight}">{s}</text>')

    def axis(y_top, y_bot, sides):
        for side in sides:
            for k in (1, 2, 4, 8):
                if k == 1 and side < 0:
                    continue
                xx = x(k, side)
                out.append(f'<line class="grid" x1="{xx:.1f}" x2="{xx:.1f}" y1="{y_top}" y2="{y_bot}"/>')
                text(xx, y_bot + 16, f"{k}×", "t3", "middle", 11)
        out.append(f'<line class="base" x1="{MID:.1f}" x2="{MID:.1f}" y1="{y_top - 4}" y2="{y_bot}"/>')

    H = 380
    text(PAD, 30, "grouped-nf4-gemm against Unsloth's MoE kernel", "t1", size=16, weight=700)
    text(PAD, 50, "Same pod and process. Log scale; 1× is equal speed.", "t3", size=12)
    # legend (two series: always present, and the bars are direct-labelled too)
    lx = PAD
    out.append(f'<rect class="fast" x="{lx}" y="62" width="12" height="12" rx="3"/>'); text(lx + 18, 73, "grouped-nf4-gemm faster", "t2", size=12)
    out.append(f'<rect class="slow" x="{lx + 190}" y="62" width="12" height="12" rx="3"/>'); text(lx + 208, 73, "Unsloth faster", "t2", size=12)

    rows = [("RTX 4090 · decode, weights stored in 4-bit", float(f["decode_4090"]), +1, f'{f["decode_4090"]}×'),
            ("H100 · decode, weights stored in 4-bit", float(f["decode_h100"]), +1, f'{f["decode_h100"]}×'),
            ("H100 · prefill, weights resident in bf16", None, -1, f'{f["prefill_lo"]}–{f["prefill_hi"]}×')]
    y0, step = 96, 34
    axis(y0 - 6, y0 + step * len(rows) - 8, (+1, -1))
    for i, (label, ratio, side, val) in enumerate(rows):
        y = y0 + i * step
        text(LABEL_W, y + 12, label, "t2", "end", 12)
        if ratio is not None:
            out.append(bar(y, ratio, side, "fast"))
            text(x(ratio, side) + 6, y + 12, val, "t1", "start", 12, 700)
        else:  # a reported range: solid to the low end, the rest of the range lighter
            lo, hi = float(f["prefill_lo"]), float(f["prefill_hi"])
            out.append(bar(y, hi, side, "slow range"))
            out.append(bar(y, lo, side, "slow"))
            text(x(hi, side) - 6, y + 12, val, "t1", "end", 12, 700)

    y1 = y0 + step * len(rows) + 52
    text(PAD, y1 - 14, "OLMoE QLoRA on real prose, against this project's per-expert loop (a different baseline)", "t1", size=13, weight=700)
    rows2 = [("RTX 4090 · end-to-end training step", float(f["e2e_4090"]), f'{f["e2e_4090"]}×'),
             ("H100 · end-to-end training step", float(f["e2e_h100"]), f'{f["e2e_h100"]}×')]
    axis(y1 - 6, y1 + step * len(rows2) - 8, (+1,))
    for i, (label, ratio, val) in enumerate(rows2):
        y = y1 + i * step
        text(LABEL_W, y + 12, label, "t2", "end", 12)
        out.append(bar(y, ratio, +1, "fast"))
        text(x(ratio, +1) + 6, y + 12, val, "t1", "start", 12, 700)
    text(PAD, H - 12, "From docs/claims.json: gnf4.kernel.h2h-unsloth, gnf4.kernel.e2e-training-real-prose.", "t3", size=11)

    style = """
  .bg{fill:#fcfcfb} .t1{fill:#0b0b0b} .t2{fill:#52514e} .t3{fill:#6f6e69}
  .fast{fill:#2a78d6} .slow{fill:#e34948} .range{opacity:.5}
  .grid{stroke:#e6e5e1;stroke-width:1} .base{stroke:#8a8984;stroke-width:1.5}
  @media (prefers-color-scheme: dark){
    .bg{fill:#1a1a19} .t1{fill:#ffffff} .t2{fill:#c3c2b7} .t3{fill:#9d9c94}
    .fast{fill:#3987e5} .slow{fill:#e66767} .grid{stroke:#33332f} .base{stroke:#7a7973}
  }"""
    title = (f"grouped-nf4-gemm against Unsloth's MoE kernel: decode {f['decode_4090']}x faster on RTX 4090 and "
             f"{f['decode_h100']}x on H100 with weights stored in 4-bit; Unsloth faster by {f['prefill_lo']}-{f['prefill_hi']}x "
             f"at H100 prefill with weights resident in bf16. OLMoE QLoRA on real prose, against this project's per-expert loop: "
             f"{f['e2e_4090']}x on RTX 4090, {f['e2e_h100']}x on H100.")
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" '
            f'aria-labelledby="t" font-family="-apple-system,Segoe UI,Helvetica,Arial,sans-serif">\n'
            f'<title id="t">{title}</title>\n<style>{style}\n</style>\n'
            f'<rect class="bg" width="{W}" height="{H}" rx="8"/>\n' + "\n".join(out) + "\n</svg>\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    svg = build(figures(ROOT / "docs" / "claims.json"))
    if a.check:
        if not OUT.exists() or OUT.read_text() != svg:
            print(f"FAIL: {OUT.relative_to(ROOT)} is stale -- run: python scripts/build_readme_chart.py")
            return 1
        print(f"OK: {OUT.relative_to(ROOT)} is current")
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(svg)
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
