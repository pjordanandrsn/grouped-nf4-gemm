# PREREG — P2-G1c: the amended mechanics at the engine-regime queue

Registered before any measurement, from G1b's calibration alone
([RESULTS-p2-g1b.md](RESULTS-p2-g1b.md)). G1b's registered box-gate found the
feasible window **empty at m = 16**: C_disp = 94.1 + 19.8·p µs against a
40.7 µs/row gain puts the dispatch floor (p ≥ 5) above LINK_CAP(16) = 3.
But m = 16 was G1's toy — one layer's cold queue. The spec's controller (§5)
acts on step walls aggregated across **all** layers; the engine-regime CPU
queue is layers × k ≈ **128** for a 16-layer top-8 shape. On the measured
box class, G1b's own boundary map predicts LINK_CAP(128) = 30 and an open
window **{5 … 30}** with the mechanics exactly as measured. G1c tests that
prediction. The regime change is registered here, from prior calibration,
before any G1c measurement — G1b's finding stands untouched.

## Design

[p2_g1b_promotion.py](p2_g1b_promotion.py) with the registered G1c
parameters: `--m 128 --steps 5 --repeats 10 --sweep 2,4,8,16,24,30`.
`steps × m = 640 = E` — each repeat consumes the whole arena exactly once
(the no-reuse stream, asserted in-harness). Calibration runs first with
`--disp-sweep 2,4,8,16,24,30` so `C_disp[p]` covers the sweep. Same shape,
same scrub, same correctness gates, same CPU thread policy (the
calibration's best), `Transient` margin sized to the burst (`k = max(p, 8)`,
I1). All other mechanics identical to G1b (spec §4, I9).

## The registered prediction

With `Δ = t_cpu_row − (1 − hide)·t_link_row − t_gpu_row` and this box's
`C_disp[p]`, window `[p_min, LINK_CAP(128)]` as in the spec (§9):

> **Bar:** realised step saving `wall_A − wall_B` ≥ **0.70 × (p·Δ −
> C_disp[p])** on the medians at **every** swept `p` inside the window.
> Refuted at any in-window `p` ⇒ the amended mechanics do not pay even in
> the engine regime; the controller stays unbuilt and the next step is
> mechanics engineering (the G1b map's priced exits), not another m.

* Below-floor sweep points ({2, 4} expected) are reported, not gated; a win
  there is a model miss in the good direction, reported as such.
* **Box gates:** `n* ∈ [2, 5]`; feasible window ∩ sweep non-empty; NUMA
  pre-gate (≤ 2 nodes, triad ≥ 100 GB/s) before any spend.
* If this box class's calibration shifts such that the predicted in-window
  points {8, 16, 24} land outside the actual window, the actual window rules
  — the bar applies to the calibrated window, not the predicted one.

## Falsifiability — same two spoilers, both must fail their bar in-window

1. Synchronous copies (hide destroyed, burst kept).
2. The un-amended G1 serial mechanics verbatim (per-row events, list ids,
   launch after the CPU tier returns).

## Correctness and validation — identical to G1b

Committed-contract correctness (voids walls on failure); `p = 0` reduction
(walls within 10%, zero copies); exact counter accounting `p × steps` per
arm; the burst's single event covers the id staging.

## What would count as a miss

* Refuted at any in-window p ⇒ per-p report; controller unbuilt; the priced
  exits (fused row copy halving s; F-trim to ~50 µs) become the next
  registered work.
* Either spoiler clears its bar ⇒ uninformative, not scored.
* Empty actual window at m = 128 on an in-gate box ⇒ the boundary map was
  wrong in the direction that matters; reported as a model refutation of
  §9's window formula, with the measured C_disp/link constants beside the
  G1b ones.
