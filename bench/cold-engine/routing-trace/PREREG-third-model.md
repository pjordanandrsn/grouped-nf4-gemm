# Preregistration — two threshold rules, tested out of sample

Both rules below were derived from **OLMoE-1B-7B** and **Granite-3.0-3B-A800M**
(`RESULTS-crossover.md`, `RESULTS-demand.md`). Both papers name the same
limit: 8 trace × model combinations, capacities within a trace not
independent, and a third model needed to test the thresholds properly.

This registers the predictions **before** the third model is captured. Written
and committed first so the numbers cannot be chosen after seeing them.

## The model

**Qwen1.5-MoE-A2.7B** — 24 layers × 60 experts, **top-4**. Chosen because
every trace so far is top-8; this is the first different routed-set size, and
the two rules are stated in terms of it.

| | layers | experts | top-k | arena | rows per step |
|---|---|---|---|---|---|
| OLMoE (derived on) | 16 | 64 | 8 | 1024 | 128 |
| Granite (derived on) | 32 | 40 | 8 | 1280 | 256 |
| **Qwen1.5-MoE (test)** | **24** | **60** | **4** | **1440** | **96** |

Four prompts as before: prose, code, math, dialogue. 512 autoregressive decode
steps, 64-token prompt, profile on steps 0–255 and score on 256–511.

## P1 — the device row cache crosses at one decode step

> The expert-keyed device cache makes **more** transfers than the engine's
> positional cache when `steps_held = capacity ÷ (layers × top-k) < 1`, and
> **fewer** when `steps_held ≥ 1`.

For this model `layers × top-k` = **96**, so the crossover is predicted at
**96 rows** — not at any fraction of the 1440-row arena, and lower in absolute
terms than either model it was derived from.

* **Confirmed** if every swept capacity below 96 rows gives ratio > 1 and
  every capacity at or above 96 gives ratio < 1.
* **Refuted** if any capacity below 96 gives ratio < 1, or any at/above gives
  > 1.
* Swept at `steps_held` ∈ {0.5, 0.75, 0.9, 1.0, 1.25, 1.5} → capacities
  {48, 72, 86, 96, 120, 144}.

**P1b** — below the threshold the cache takes **exactly zero** hits under LRU
(transfers == routed row-slots), because the pattern is near-cyclic. Refuted
by any nonzero hit count below 96 rows.

## P2 — demand-paging beats placement iff capacity covers the working set

> `headroom = scored working set ÷ capacity ≤ 1` predicts that demand-paging
> (LRU) makes fewer transfers than static placement, with **no false
> positives**.

* **Confirmed** if, across the 12 swept capacities × 4 prompts (48 cells),
  every cell with `headroom ≤ 1` has demand beating static.
* **Refuted** by a single cell with `headroom ≤ 1` where static wins — the
  claim derived on 96 cells was *zero* false positives, so one is a
  refutation, not noise.
* The rule is **sufficient, not necessary**: cells above the threshold where
  demand also wins are expected (8 of 28 on the derivation set) and are not
  evidence against it.
* Capacities: fractions {0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
  0.9, 1.0} of the 1440-row arena.

## What would count as a miss

Stated so it cannot be renegotiated afterwards:

* P1 fails if the crossover is not at 96 rows — including if it lands at some
  fraction of the *arena* instead, which is the hypothesis
  `RESULTS-concentration.md` corrected and would resurrect.
* P1b fails on any hit below threshold, which would mean the zero-hit region
  is not LRU's cyclic pathology but something specific to the two models it
  was seen on.
* P2 fails on any false positive.
* A model that cannot be captured, or whose router the probe cannot read, is
  **not** a result either way and will be reported as a failed capture.

## Reporting

Both outcomes get the same treatment. If a rule is refuted the derivation
documents get a banner, exactly as `RESULTS-concentration.md` and
`RESULTS-crossover.md` already carry for explanations of mine that failed.
