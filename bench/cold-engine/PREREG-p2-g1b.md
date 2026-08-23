# PREREG — P2-G1b: the amended mechanics must pay where the model says they can

Registered before any measurement. Successor to [PREREG-p2-g1.md](PREREG-p2-g1.md),
whose verdict was **REFUTED** ([RESULTS-p2-g1.md](RESULTS-p2-g1.md)): serial
per-row dispatch and the link budget, neither charged by the model. Both are
now in the spec ([SPEC-elastic-phase2.md](SPEC-elastic-phase2.md) §4, §5, §9,
I9 — amended #207), and this gate re-tests the mechanics **as amended**.

## What changed in the mechanics (spec §4, I9)

One stream-ordered **burst** per step: a single `want()`, per-row copies on
the side stream, **one** burst event covering copies + id staging, slot ids
**pre-staged device-side** (pinned int32 staging, async copy — never a python
list at launch), and the burst's GPU work enqueued — event-gated — **before**
the step's CPU execution begins. The host never waits between tiers.

## Design (unchanged from G1 where not named)

Same box class, same A/B pairing, same shape (`gptossish_gateup`, m = 16,
32 steps, ≥ 5 repeats, medians), same no-reuse stream, same 1 GiB
single-threaded scrub before each timed arm, CPU tier at the calibration's
best thread count. Sweep `p ∈ {1, 2, 3, 4, 8}` (3 added: the predicted
feasible window on a 9J14-class box is {2, 3}).

## The registered prediction

From this box's `elastic_e3.py` receipt (run first, gated `n* ∈ [2, 5]`),
with `Δ = t_cpu_row − (1 − hide)·t_link_row − t_gpu_row` and `C_disp[p]` the
calibrated per-burst dispatch cost:

```
feasible window: p_min ≤ p ≤ LINK_CAP
  p_min    = min p : p · (t_cpu_row − t_gpu_row) > C_disp[p]
  LINK_CAP = floor(m · t_cpu_row / (t_cpu_row + t_link_row))
```

> **Bar:** realised step saving `wall_A − wall_B` ≥ **0.70 × (p·Δ −
> C_disp[p])** on the medians, at **every** swept `p` inside the feasible
> window. Refuted at any in-window `p` ⇒ the amended mechanics still do not
> pay; the controller stays unbuilt.

* Out-of-window `p` are swept and **reported, not gated** — the floor and cap
  predict they lose. An out-of-window `p` that *pays* is reported as a model
  miss (the good direction), not a gate failure.
* **Box gate additions:** the feasible window ∩ sweep must be non-empty, else
  the box is rejected before the arms run (spec §9: an empty window fails
  calibration for promotion exactly as an out-of-range n* does).

## Falsifiability — two spoilers, both must fail their own bar

1. **Synchronous copies** (hide spoiler, as G1): default-stream blocking
   copies, sync before the CPU tier. Destroys the hide, keeps the burst.
2. **The un-amended G1 mechanics verbatim** (dispatch spoiler): per-row
   events, python-list ids, GPU work launched only after the CPU tier
   returns. The refuted configuration must stay refuted — if it clears the
   G1b bar, the amendment explains nothing and G1b is uninformative.

Both spoilers are scored against the same in-window bar; each must fall
below it at every in-window `p`, before the real measurement is read.

## Correctness — identical to G1, the committed contracts

Weight bytes identical; CPU bit-exact vs `ref_gemv_grouped` (numpy in); GPU
within the committed 2e-2 with the reference fed the same bf16-rounded
activations; retention zero-H2D. Any failure voids the wall numbers.

## Harness validation, before trust

* `p = 0` reduces Arm B to Arm A: walls within repeat spread, zero H2D.
* Counter accounting: H2D row count = `p × steps` exactly, per arm.
* The burst's single event covers the id staging (the gemm must never read
  `ids_dev` before the async stage lands — same event, same gate).

## What would count as a miss

* Refuted at any in-window `p` ⇒ reported per-`p`; the controller is not
  built; the overhead is characterized further before any third attempt.
* Either spoiler clearing its bar ⇒ uninformative, not scored.
* An empty feasible window on an in-gate box ⇒ the box class cannot host the
  controller's promote actuator at this `m`; reported as a calibration
  finding, no wall claimed.
