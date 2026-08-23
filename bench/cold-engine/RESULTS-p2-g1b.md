# RESULTS — P2-G1b: the feasible window is empty at m = 16 (calibration finding)

Registered in [PREREG-p2-g1b.md](PREREG-p2-g1b.md) (#208). Run 2026-08-23 on
a rented EPYC 9J14 + RTX 5090 (driver 610.57, same host class as G1), repo at
`eb4b18b`.

**Outcome: the registered box-gate fired — the feasible burst window
`[p_min, LINK_CAP]` is EMPTY at m = 16, so no wall was claimed.** This is the
prereg's third registered category ("the box class cannot host the
controller's promote actuator at this m"), not a PASS and not a REFUTED. The
harness printed the window math and refused the arms
([p2-g1b-2026-08-23-window.log](p2-g1b-2026-08-23-window.log)).

## The numbers (all from this box's calibration, before any arm)

Calibration ([elastic-2026-08-23-e3-g1b.json](elastic-2026-08-23-e3-g1b.json)):
n* = 3.85 ∈ [2,5] (in-gate), B_cpu 179.8 GB/s (64 threads), B_link 56.3,
B_gpu 1053.8, hide = 1.0. Derived rows: t_cpu_row 49.0 µs, t_gpu_row 8.4 µs,
t_link_row 156.4 µs, per-row gain 40.7 µs.

First live output of the new `bench_dispatch()` probe — the serial host cost
of one amended burst, warm, idle GPU:

| p | 1 | 2 | 3 | 4 | 8 |
|---|---|---|---|---|---|
| C_disp (µs) | 113.9 | 132.9 | 153.1 | 173.7 | 252.3 |

The fit is exactly linear: **C_disp[p] = 94.1 + 19.8·p µs** (max residual
< 1 µs across all five points). Two readings:

* **Fixed cost F ≈ 94 µs/burst** — `want()` bookkeeping, tag + event
  create/record, compute-stream wait, the pre-staged-id launch (~21 µs of it),
  end-sync. Paid once per burst.
* **Marginal cost s ≈ 19.8 µs/row** — two `cudaMemcpyAsync` enqueues
  (packed + scales are separate host tensors) plus a pinned staging write,
  per row. **Half the 40.7 µs/row gain is eaten by per-row enqueue glue.**

## Why the window closes at m = 16

Feasibility needs some p with `p·(gain − s) > F` **and** `p ≤ LINK_CAP(m)`:

```
dispatch floor:  p · (40.7 − 19.8) > 94.1   →   p ≥ 5
link cap:        LINK_CAP(16) = floor(16 · 49.0 / 205.4) = 3
→ [5, 3] = ∅
```

The registered gate printed `p_min=8` (the smallest *swept* p clearing the
floor); the analytic crossing is p ≥ 5. Either way it sits above the m = 16
link cap.

## The m-dependence — what the boundary map says

`LINK_CAP(m) = 0.2385·m` on this box, so **the window opens at m ≥ 21** with
the glue exactly as measured. And m = 16 was G1's toy: one layer's cold
queue. The spec's controller (§5) acts on **step walls aggregated across all
layers** — the engine-regime CPU queue is layers × k (~128 for a 16-layer ×
top-8 shape), where this box's own constants predict a window of
**{5 … 30}**: wide open with no further engineering. The next gate (P2-G1c,
spec §11) is therefore the same amended mechanics at engine-regime m = 128 —
registered from this run's calibration *before* any new measurement, so the
m = 16 finding stands as the boundary map that motivates it rather than a
goalpost moved after the fact.

Secondary options the map also prices, if G1c were to fail: fusing each row's
packed+scales into one contiguous host copy halves s (→ floor at p ≥ 4.5);
trimming F to ~50 µs (vectorized staging, event reuse) opens {3} even at
m = 16. Neither is claimed — they are priced, not measured.

## Provenance notes

* Same host class as G1's box; B_gpu measured 1053.8 here vs 756.6 on the
  G1 rental (same GPU model — vast-side variance). It enters Δ only through
  t_gpu_row (8.4 vs 11.6 µs) and does not drive the finding.
* Box pre-gated per the NUMA lesson (2 nodes, triad 144.7 GB/s) before any
  spend. Box destroyed after receipts; zero instances; ~$0.15 this run.
