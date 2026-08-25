# AMENDMENT — K6's decision bars were absolute; the claim was a ratio

Registered 2026-08-25, after the re-gate receipt and before any
re-adjudication under this frame.

## What the re-gate established

On box 11 (`receipts-k6-bespoke/regate/`; RTX 5090 at a 500 W power
limit, driver 580.95):

1. **The amended correctness gate PASSES both cells.** max|Δ| = 0.500
   against recorded references of 147.0 (gate_up) and 84.5 (down) —
   inside the 2⁻⁷ budgets (1.148 / 0.660) — with argmax agreement
   99.4%. The dot-pad mechanism is numerically certified at Stage A's
   level; K6-B's token gates remain ahead.
2. **The absolute timing bars repeated the B2 frame defect.** This
   box runs the SAME binaries ~1.7× slower than box 10 (power cap:
   baseline pair 120.9 µs vs 69.5). Against PASS ≤ 36 / REFUTED > 58,
   dot-pad's 71.2 µs adjudicates REFUTED — while its RATIO to the
   same-box baseline is **0.589**, BETTER than box 10's disclosed
   0.668. An absolute bar rejected the stronger result on the slower
   box, exactly as `bars-follow-the-claim` records for F1-B2.

## The frame, from the prereg's own text

PREREG-k6 justified its numbers as ratios: PASS 36 µs was written
"(≥ 2× over baseline; more than half the gap to the floor closed)"
and PARTIAL 58 µs as "(20–50% gain)". The amendment adopts those
registered ratios as the bars, with the absolute forms retired:

- **PASS** — pair ratio ≤ 0.50 (the "≥ 2×" the prereg wrote).
- **PARTIAL** — ratio ≤ 0.80 (the "≥ 20% gain"), ships only under the
  A/A condition as registered.
- **REFUTED** — ratio > 0.80.

## Re-adjudication on the SAME receipts

- Box 11: 71.2 / 120.9 = **0.589 → PARTIAL** (noise gate PASS).
- Box 10 (disclosed in RESULTS-k6-stageA): 46.4 / 69.5 = 0.668 →
  PARTIAL. **Both boxes agree** under the ratio frame — the frame,
  not the kernel, was what disagreed.

K6 Stage A therefore closes **PARTIAL**: the dot-pad reduces the
census pair by 33–41% with certified Stage A numerics, and **K6-B
(productization with end-to-end token gates) is licensed** under the
PARTIAL-ships-if-gain-real condition, which both boxes' noise gates
satisfy.
