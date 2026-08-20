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
| **R5** — soft eviction ≤ hard eviction in cost | **CONFIRMED** — flat uncontended, *faster* contended — ⚠️ the *contended* half is scored on capacity-unmatched arms; see the reconciliation |
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

> **⚠️ The arms below do not hold the same amount of memory.** Arm A's pool is
> `protected` rows; Arm B's is 128. Matched-capacity controls reverse the sign
> of every result in this section — reads (#145), the knee (#147), and wall on
> real NVMe (#153) — and the feasibility extension turns out to be a pool-size
> fact that hard eviction reproduces when given the same 128 rows. The
> measurements reproduce; the attribution does not. See
> [`reconciliation/RESULTS-r5-reconciled.md`](reconciliation/RESULTS-r5-reconciled.md).
> The numbers are kept exactly as published.

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

---

# Re-measurement (2026-08-20) — R1 measured on a decode-only window

The section above said re-measuring R1 "needs a box and is not done here."
This is that run.

**Box:** RTX 5090 + AMD EPYC 9655 (Zen5). Measured ceilings: B_vram
1573.1 GB/s, B_dram 425.27 GB/s, B_nvme 5.11 GB/s **sequential**, G0
proceed. gnf4 `48c78a1` (fixed) and `0a10eab` (pre-#130) for the control
arm; e4b at the `cold_stats` denominator fix. Same model, same arena
geometry, and the **committed routing trace** (`olmoe_profile.jsonl`, 1010
rows), so the placement solve reproduces the original cold sets exactly:
**265 experts at 5%** (achieved 0.0503) and **505 at 20%** (achieved
0.2003). Receipts in `r1-2026-08-20/`.

## The measurement window, and why the first pass of this run was wrong

`run_steps` performs six warmup prefills, twenty-four warmup decode steps,
and one more prefill to build the KV cache before the timed loop. A prefill
routes 62–64 of 64 experts per layer — the whole arena — which is precisely
the access shape that function exists to keep out of a decode measurement.

The first pass of this re-measurement snapshotted the tier counters
**before** `run_steps`, so every one of those prefills landed inside the
"window": wall was decode-only while reads, evictions, resurrections and
overwrites were prefill-plus-decode. Caught by Bugbot on gnf4#132.

It is not a small effect. Uncontended reads: **218 → 4**. Contended Arm B
reads: **3434 → 538**. Roughly six sevenths of what the first pass called
cold-path disk traffic was warmup.

`run_steps` now takes an `on_measure_start` callback fired at the true
boundary, every counter is differenced across that window, and
`reuse_before_overwrite` is **recomputed from the differenced terms** rather
than read off the tier, which reports it over the process lifetime. Two
numbers below are therefore reversals of the first pass, and are flagged as
such.

**The same defect is in `run_gate1.py`** — under a comment that explicitly
claimed the opposite. It is fixed there too, but **the published gate-1 read
counts in `RESULTS-tribrid-gate1.md` predate the fix and are uncorrected**;
they need a re-run. Gate 1's MISS verdict rests on prefetch coverage rather
than on those counts, so the verdict is not in question — the counts are.

Instrument notes: proxy-ssh box (`ports: None`, law 4), acceptable since the
only bulk transfer is box→HuggingFace. Law 7: different host class from the
withdrawn runs, so **wall** is scored only against same-session matched
pairs.

## R1, measured

**Contended — 505 cold rows, 128-slot pool** (the informative regime):

| protected | reads | logical evict | resurrections | overwritten | **P(reuse)** |
|---|---|---|---|---|---|
| 128 (control) | 527 | 0 | 0 | 0 | — |
| 96 | 538 | 622 | 84 | 538 | **0.135** |
| 64 | 538 | 713 | 175 | 538 | **0.245** |
| 32 | 538 | 823 | 280 | 538 | **0.342** |

Registered R1 is **5–20%**. One point lands inside it and two sit above:
**R1 is CONFIRMED, and exceeded at the tighter ownership budgets.**

This **reverses the first pass of this run**, which reported 6.7 / 16.9 /
24.4% and called R1 "confirmed, modestly". Those numbers were diluted by
prefill traffic. The corrected values also sit close to the withdrawn
document's contended figures (0.109 / 0.218 / 0.321) — so on the contended
arm the withdrawn numbers were closer to right than the first correction
suggested, for the wrong reason.

**Uncontended — 265 cold rows, 384-slot pool: cannot test R1 at all.**

| protected | reads | logical evict | resurrections | overwritten | P(reuse) |
|---|---|---|---|---|---|
| 384 / 256 | 4 | 0 | 0 | 0 | — |
| 192 | 4 | 11 | 7 | 0 | 1.000 |
| 128 | 4 | 60 | 56 | 0 | 1.000 |
| 96 | 4 | 103 | 99 | 0 | 1.000 |

`reclaimable_overwritten` is **zero at every point** — the working set fits
the pool, so no reclaimable row is ever contended for and P collapses to
1.000 by construction. That is the absence of the event R1 asks about, not a
100% reuse rate. A configuration without capacity pressure cannot score this
clause in either direction.

The **ghost working set is confirmed and is starker on a clean window**:
**4 disk reads**, flat, while ownership is cut 4× (384→96).

## The bug's effect, isolated

Pre-fix tree (`0a10eab`) against the fixed tree, **same box, same trace,
same harness, same window**, direct landing off so the buggy path is
reachable at all:

| protected | P buggy | P fixed | relative inflation | reads buggy | reads fixed |
|---|---|---|---|---|---|
| 96 | 0.155 | **0.135** | +15% | 519 | 538 |
| 64 | 0.278 | **0.245** | +13% | 519 | 538 |
| 32 | 0.367 | **0.342** | +7% | 519 | 538 |

The self-ensure bug **does** inflate P, consistently, by 7–15% relative —
the direction #130 claimed, at a magnitude well short of what would explain
11–60%. The withdrawal was correct; its stated reasoning ("not a small
correction … scales with segments-per-expert") still overstates it.

**Under the current default the bug is unreachable.** `cold_direct` defaults
to `True` and `build_cold_view(direct=True)` attaches an external landing
that bypasses `segment_into`, so buggy and fixed trees return identical
counters at the default.

### The "unexplained gap" is a windowing artifact, and is now moot

The first pass recorded an open discrepancy: published Arm B reads 2618
against 3430–3495 measured. Both are **warmup-inclusive** numbers, and the
corrected decode-only figure is **538**. The published absolutes were taken
on the pre-fix window and are not comparable to anything here. A residual
difference between the published figures and a warmup-inclusive
reproduction remains unaccounted for, but it no longer matters: neither is
the measurement the prereg asks for. **No read count in this document may
be compared with one from before the window fix.**

## Arm A vs Arm B — the read claim holds

Arm A is hard eviction (`hot_rows == protected_rows`); Arm B keeps a
128-slot pool so `128 − protected` rows are reclaimable. Both at 20% cold,
`cold_direct=True`.

| protected | A reads | B reads | **Δ reads** | A wall | B wall | Δ wall |
|---|---|---|---|---|---|---|
| 96 | 622 | 538 | **−13.5%** | 72.48 ms | 70.70 ms | −2.5% |
| 64 | 713 | 538 | **−24.5%** | 76.94 ms | 70.89 ms | **−7.9%** |
| 32 | **infeasible** | 538 | — | — | 70.74 ms | — |

* **≥10% fewer physical NVMe reads: PASSES at both feasible points**
  (−13.5%, −24.5%). This **reverses the first pass of this run**, which
  measured −6.4% / −16.4% on the contaminated window and concluded the
  claim "does not survive". It does. **#130 was right to keep it**, and the
  first correction was wrong to contradict it — though not for the reason
  #130 gave, since the code change does move read counts in both arms.
* **5–15% lower exposed cold-path wall: PARTIAL** — −7.9% at protected=64
  is inside the band, −2.5% at 96 is below it. The withdrawn document
  scored this a flat MISS at 3.4–4.4%.
* **Feasibility survives verbatim.** Arm A at protected=32 still fails with
  the identical named refusal — `request of 36 unique rows exceeds
  hot_rows=32` — while Arm B at the same ownership budget runs normally.
* **Token-identical in every arm** (`all_tokens_identical: true`, 19 arms
  across 6 receipts). Reclaimable residency remains pure bookkeeping.

## Corrected scoreboard

| prediction | withdrawn claim | measured now |
|---|---|---|
| **R1** — 5–20% reuse before overwrite | 11–60%, "exceeded" | **CONFIRMED, exceeded at 2 of 3** — 13.5 / 24.5 / 34.2% |
| **R5** — soft eviction ≤ hard | confirmed | **CONFIRMED** — B ≤ A on reads and wall at both feasible points |
| **R6** — best gains at moderate pressure | shape confirmed | **CONFIRMED** — P rises monotonically as ownership tightens |
| ≥10% fewer physical NVMe reads | confirmed at both | **CONFIRMED at both** — −13.5% / −24.5% |
| 5–15% lower exposed cold wall | MISS (3.4–4.4%) | **PARTIAL** — −7.9% at 64, −2.5% at 96 |
| ghost working set | ~4× | **CONFIRMED** — 4× ownership cut, 4 reads flat |
| feasibility extension | observed | **CONFIRMED** — identical named refusal |
| no numerical difference | confirmed | **CONFIRMED** — 19/19 arms token-identical |

## What this still does not establish

Everything in "What this does not establish" above stands unchanged: no
cold→GPU arm, no VRAM-side reclaimable arm (R2, R3, R4, R7–R10 remain
untested), `order="tail"` only, one trace's locality. Added by this run:

- **The uncontended regime cannot score R1** and must not be quoted for it.
  Only configurations where the routed set exceeds the pool produce the
  overwrite event the prediction is about.
- **Gate 1's published read counts are uncorrected** for the same window
  defect and need a re-run.
- **Read counts from before the window fix are not comparable** to anything
  in this section, in either direction.
