# RESULTS — Reclaimable residency (R1–R10), first measurement

Registered in [`PREREG-tribrid-stage3.md`](PREREG-tribrid-stage3.md)
(stamped `7bf5b2be…`) as a hypothesis **separate** from the base tribrid
gate: *the interval between logical eviction and physical overwrite
contains measurable reusable information.* Receipts in
`gate1-5090-zen5/reclaim*.json`, `armA_*.json`, `armB_*.json`.

Same box, model, arena, routing trace and workload as
[`RESULTS-tribrid-gate1.md`](RESULTS-tribrid-gate1.md). The only variable
is `protected_rows` — the cold tier's capacity-ownership budget. At
`protected_rows == hot_rows` nothing is ever reclaimable and the tier is
the pre-Stage-3 tier exactly.

## Verdict

| prediction | result |
|---|---|
| **R1** — 5–20% of logical evictions reused before overwrite | **CONFIRMED, exceeded** — 11–60% measured |
| **R5** — soft eviction ≤ hard eviction in cost | **CONFIRMED** — flat uncontended, *faster* contended |
| **R6** — best gains at moderate capacity pressure | **shape confirmed** — see below |
| ≥10% fewer physical NVMe reads (Arm A vs Arm B) | **CONFIRMED** — −14.9% and −29.6% |
| no numerical/correctness difference | **CONFIRMED** — token-identical, every arm |
| no increase in protected-memory usage | **CONFIRMED** by construction |
| 5–15% lower exposed cold-path wall | **MISS** — measured 3.4–4.4%, below the band |

## Uncontended: 265 cold rows, 384-slot pool

| protected | median ms | win reads | logical evict | resurrections | P(reuse) | bytes saved |
|---|---|---|---|---|---|---|
| 384 (control) | 50.81 | 26 | 0 | 0 | — | 0 |
| 256 | 51.11 | 26 | 741 | 0 | 0.0000 | 0 |
| 192 | 50.85 | 26 | 1052 | 247 | **0.2872** | 874 MB |
| 128 | 50.90 | 26 | 1496 | 627 | **0.5056** | 2.22 GB |
| 96 | 52.20 | 26 | 1808 | 907 | **0.5967** | 3.21 GB |

**The ghost working set is real.** Cutting protected capacity by 75%
(384→96) produced **zero** additional disk reads. The tier behaves as
though it still owns 384 rows while protecting 96 — roughly a **4×**
effective multiplier, where the prereg guessed 1.2–1.4×.

Stated with its limit: at this cold mass the working set fits the pool, so
nothing ever contends for the reclaimable slots and every one survives to
be resurrected. That is the mechanism working, not the hard case.

## Contended: 505 cold rows, 128-slot pool

P(reuse) falls to 0.11–0.32 from 0.29–0.60 — reclaimable slots are taken
faster when something wants them, which is R6's predicted shape.

## The registered comparison: Arm A (hard) vs Arm B (reclaimable)

Identical protected budgets, 20% cold mass. Arm A sets
`hot_rows == protected_rows` so nothing can be reclaimable; Arm B keeps a
128-slot pool so `128 − protected` slots hold reclaimable rows.

| protected | Arm A reads | Arm B reads | Δ reads | Arm A wall | Arm B wall | Δ wall |
|---|---|---|---|---|---|---|
| 96 | 3077 | 2618 | **−14.9%** | 79.02 ms | 76.31 ms | −3.4% |
| 64 | 3720 | 2618 | **−29.6%** | 80.46 ms | 76.91 ms | −4.4% |
| 32 | **infeasible** | 2618 | — | — | 75.37 ms | — |

The read reduction clears the registered ≥10% bar at both feasible points
and grows as the protected budget shrinks. The **wall** improvement is
real but **below** the registered 5–15% band, and is reported as a miss.

### Reclaimable residency extends the feasible operating range

Arm A at `protected=32` does not run. It fails with the tier's own named
refusal — `request of 36 unique rows exceeds hot_rows=32` — because a
single layer routed 36 unique cold experts into a 32-slot pool. Arm B at
the same protected budget runs fine: the *pool* is 128, only *ownership*
is 32, so the demand window is satisfied by slots the policy does not
protect.

That is capacity ownership and information retention coming apart in the
most concrete way available: the same protected budget is infeasible under
hard eviction and unremarkable under reclaimable residency. It was not a
registered prediction and is reported as an observation, not a scored
clause.

## Correction — R1's numbers are contaminated and are withdrawn as measured

A Bugbot finding on gnf4#112, **read late and merged past**, invalidates the
headline number in this document.

`ColdCpuView.ensure` materialized a batch by calling `segment_into` once per
expert per segment, and `segment_into` issued its **own demand
`ColdTier.ensure`** for that single expert. Every one of those replaced the
demand window and ran the demotion pass — so materializing a batch logically
evicted its own siblings, which the next outer expert's hit then
"resurrected". Both counters were inflated by the measurement path itself.

Every run in this document (`reclaim.json`, `reclaim_contended.json`,
`armA_*`, `armB_*`) took that path, because the direct landing did not exist
yet. So:

* **P(reuse before overwrite) = 11–60% is withdrawn.** The true rate is
  lower by an unknown amount; the inflation is self-inflicted and scales
  with segments-per-expert (four here), so it is not a small correction.
* **R1 is NOT confirmed by this document.** It reverts to untested.
* The **Arm A vs Arm B read counts** (−14.9% / −29.6% physical NVMe reads)
  are *less* affected — they count real disk reads, not eviction
  bookkeeping — but both arms ran the same buggy path, so the comparison is
  fair while the absolute rates are not trustworthy.
* The **ghost working set** observation (protected capacity cut 75% with
  zero additional disk reads) rests on read counts, not resurrection
  counts, and survives.
* The **feasibility finding** (hard eviction at `protected=32` cannot run at
  all) is structural and survives.

Fixed in the follow-up: `segment_into` takes `ensure=False`, and the view
passes it, so a batch no longer demotes itself. **Re-measuring R1 needs a
box and is not done here.** Recording the contamination rather than quietly
re-running is the point — the number was published, and this is what
corrects it.

## What this does not establish

- **H2D refills were not measured.** These runs used `cold_dest="cpu"`,
  which has no H2D by construction, so the registered "≥5% fewer H2D
  expert refills" clause is untested rather than passed. It needs a
  cold→GPU arm — currently blocked behind
  [e4b#171](https://github.com/pjordanandrsn/experts4bit-qlora/issues/171),
  since that path does not produce the reference tokens.
- **R2, R3, R4, R7–R10 are untested.** No VRAM-side reclaimable arm exists
  (this is DRAM only), no burst-vs-uniform routing comparison, no
  promotion-churn measurement.
- **One workload shape.** `order="tail"` only; the `head` (bursty) order
  that R4 points at was not swept.
- The P(reuse) numbers are specific to this trace's locality. Instrument
  law 7 applies to the wall numbers as always.
