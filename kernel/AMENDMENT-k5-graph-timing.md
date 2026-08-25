# AMENDMENT 2 — K5: eager-harness timing is host-CPU-sensitive; the probe moves to CUDA-graph replay

Registered 2026-08-24, before any graph-timed data exists.

## The finding (attempts 2–3, receipts in RESULTS)

Two independent driver-580+ boxes (drivers 590.48 and 595.58, different
hosts) measured the K1-winner GEMV pair at **94.5 and 94.6 µs** —
agreeing with each other to 0.15% with the noise gate passing, while
sitting +30% above the 72.4 µs K1 receipt. That is deterministic
host-class difference, not GPU health. The decomposition: gate_up read
+10% (44.5 → 48.5–49.8 µs) while down read **+60%** (27.9 → 44.8–46.0
µs). down is the short cell — and in K1's own table it shows a
25-of-48-config plateau within 10% of best, the signature of a cell
whose eager-harness time is dominated by **per-call host enqueue
overhead**, not kernel execution. The eager wrapper harness (python
call + plan lookup + `torch.empty` per iteration) measures
kernel + host-gap; the gap scales with host single-thread speed
(EPYC 9755 Turin at K1 vs 9654/9B14 Genoa here). The AMENDMENT-1
anchor gate (eager pair within ±10% of 72.9 µs) therefore rejects
healthy GPUs on slower-CPU hosts indefinitely — an instrument defect,
not a box defect.

## Amended timing basis

Production decode does not launch eagerly: the b1d loop replays the
whole step as a captured CUDA graph. The probe adopts the same basis:

- **Graph-replay timing is the registered basis.** Each arm is
  captured as a CUDA graph of 32 back-to-back calls (allocations
  inside capture ride the graph's private pool — the exact pattern
  b1d certified at scale) and timed over replays; per-call time =
  median chunk / (replays × 32). Host enqueue cost is excluded, as it
  is in the product.
- **Eager timing is still recorded once per cell** (K1 continuity,
  disclosed as kernel+host-gap; not a gate).
- **The registered ratio** (M-tile best sum / GEMV sum, thresholds
  0.6/0.9 unchanged) is computed on graph-basis numbers.
- **Gates:** driver ≥ 580 (rent); graph-GEMV start/end drift ≤ 5%
  (noise, refusal); structural sanity graph ≤ eager per cell
  (refusal if violated). The AMENDMENT-1 ±10%-of-72.9 eager anchor
  gate is RETIRED as defective — replaced by the host-insensitive
  graph basis itself. Triad remains recorded-advisory (GPU-pure
  cycle).

The first passing box's graph-basis GEMV pair becomes the K5 graph
anchor, recorded in RESULTS beside the eager pair.
