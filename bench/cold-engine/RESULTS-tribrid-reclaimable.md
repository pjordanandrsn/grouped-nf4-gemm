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

---

# Re-measurement (2026-08-20) — R1 measured, and the withdrawal's reasoning corrected

The section above said re-measuring R1 "needs a box and is not done here."
This is that run.

**Box:** RTX 5090 + AMD EPYC 9755 (Zen5). Measured ceilings: B_vram
1572.7 GB/s, B_dram 470.99 GB/s, B_nvme 5.96 GB/s **sequential**, G0
proceed. gnf4 `48c78a1` (fixed) and `0a10eab` (pre-#130, for the control
arm); e4b `36c0aee` plus the `cold_stats` forwarding fix this run required.
Same model, same arena geometry, and the **committed routing trace**
(`olmoe_profile.jsonl`, 1010 rows) — so the placement solve reproduces the
original cold sets exactly: **265 experts at 5%** (achieved 0.0503) and
**505 at 20%** (achieved 0.2003). Receipts: `r1-2026-08-20/`.

Instrument notes. (1) This box is proxy-ssh — `ports: None`, instrument
law 4 — acceptable here because the only bulk transfer is box→HuggingFace.
(2) Instrument law 7: different host class from the withdrawn runs, so
**wall** numbers are scored only against same-session matched pairs and
never against the old document. Read counts are trace-and-policy
determined and are the load-bearing measurement here.

## R1, measured

**Contended — 505 cold rows, 128-slot pool** (the informative regime):

| protected | reads | logical evict | resurrections | overwritten | **P(reuse)** |
|---|---|---|---|---|---|
| 128 (control) | 3293 | 0 | 0 | 0 | — |
| 96 | 3434 | 3564 | 236 | 3296 | **0.067** |
| 64 | 3430 | 4039 | 673 | 3302 | **0.169** |
| 32 | 3430 | 4462 | 1064 | 3302 | **0.244** |

Registered R1 is **5–20%**. Two of three points land inside it, one above:
**R1 is CONFIRMED, modestly.** The withdrawn "11–60%, exceeded" is gone and
nothing here restores it.

**Uncontended — 265 cold rows, 384-slot pool: cannot test R1 at all.**

| protected | reads | logical evict | resurrections | overwritten | P(reuse) |
|---|---|---|---|---|---|
| 384 / 256 | 218 | 0 | 0 | 0 | — |
| 192 | 218 | 388 | 362 | 0 | 1.000 |
| 128 | 218 | 785 | 695 | 0 | 1.000 |
| 96 | 218 | 1049 | 927 | 0 | 1.000 |

`reclaimable_overwritten` is **zero at every point** — the working set fits
the pool, so no reclaimable row is ever contended for and P collapses to
1.000 by construction. That is not a 100% reuse rate; it is the absence of
the event whose probability R1 asks about. A configuration without capacity
pressure cannot score this clause in either direction, and the withdrawn
document's uncontended P values were reporting overwrites that only the bug
created.

The **ghost working set survives and is sharper**: reads are flat at **218**
while ownership is cut 4× (384→96). Zero additional disk reads for a 75%
smaller protected budget.

## The bug's actual effect, isolated

The withdrawal attributed the published numbers to gnf4#112. That is
testable: run the pre-fix tree (`0a10eab`) against the fixed tree on the
**same box, same trace, same harness**, with the direct landing off so the
buggy path is reachable at all.

| protected | P buggy | P fixed | reads buggy | reads fixed |
|---|---|---|---|---|
| 96 | 0.082 | **0.067** | 3495 | 3434 |
| 64 | 0.159 | **0.169** | 3492 | 3430 |
| 32 | 0.248 | **0.244** | 3488 | 3430 |

**The contamination is real but small.** It inflates P by ~22% at the
tightest ownership budget, and at the looser two the difference is within
what the instrument moves anyway. Reads are inflated ~1.8% throughout. The
withdrawal was the right call — the number was not clean — but its stated
reasoning ("not a small correction … scales with segments-per-expert")
**overstated the magnitude**. The bug is not what produced 11–60%.

**Under the current default the bug is unreachable.** `cold_direct` defaults
to `True`, and `build_cold_view(direct=True)` attaches an external landing
that bypasses `segment_into` entirely — so buggy and fixed trees return
identical counters at the default, and the first pass of this
re-measurement accidentally demonstrated that by finding no difference at
all. The contamination applies only to the non-direct path, which is no
longer how this code runs.

### What is still unexplained

Neither tree reproduces the withdrawn absolutes. Published Arm B reads were
**2618**; every configuration measured here reads **3430–3495**, and the
buggy tree does not close the gap. So the published figures differ from
anything reproducible today for reasons **beyond** #112 — most plausibly
e4b-side changes to how rows are requested between then and now. Recorded
as an open discrepancy, not resolved. It is why this is presented as a
fresh measurement of current code rather than a restoration of the old
number, and why no counter here should be compared across the two
documents.

## Arm A vs Arm B — the "surviving" read claim does not survive

Arm A is hard eviction (`hot_rows == protected_rows`); Arm B keeps a
128-slot pool so `128 − protected` rows are reclaimable. Both at 20% cold,
`cold_direct=True`.

| protected | A reads | B reads | **Δ reads** | A wall | B wall | Δ wall |
|---|---|---|---|---|---|---|
| 96 | 3670 | 3434 | **−6.4%** | 76.52 ms | 75.85 ms | −0.9% |
| 64 | 4103 | 3430 | **−16.4%** | 80.08 ms | 77.38 ms | −3.4% |
| 32 | **infeasible** | 3430 | — | — | 74.37 ms | — |

Against the registered clauses:

* **≥10% fewer physical NVMe reads: MISSES at protected=96 (−6.4%), PASSES
  at protected=64 (−16.4%).** The withdrawn document scored this "CONFIRMED
  at both feasible points" on −14.9% / −29.6%. It does not hold at both.
  #130 listed this among the claims that survive because they "rest on read
  counts rather than eviction bookkeeping" — **that reasoning was wrong**:
  the code change moved the read counts too, in both arms.
* **5–15% lower exposed cold-path wall: MISS at both points** (−0.9%,
  −3.4%), consistent with the withdrawn document's own verdict of MISS
  though not with its magnitudes. Treat the wall clause as **unresolved
  rather than scored**: deltas this small sit near run-to-run variation,
  and a first pass on a different host class produced a visibly different
  wall picture from the same read counts. Only the receipted box is
  reported here.
* **Feasibility survives verbatim.** Arm A at protected=32 still fails with
  the identical named refusal — `request of 36 unique rows exceeds
  hot_rows=32` — while Arm B at the same ownership budget runs normally.
  Capacity ownership and information retention come apart exactly as
  before.
* **Token-identical in every arm of every run** (`all_tokens_identical:
  true`, 19 arms across 6 receipts). Reclaimable residency remains pure
  bookkeeping.

## Corrected scoreboard

| prediction | withdrawn claim | measured now |
|---|---|---|
| **R1** — 5–20% reuse before overwrite | 11–60%, "exceeded" | **CONFIRMED** — 6.7 / 16.9 / 24.4% |
| **R5** — soft eviction ≤ hard | confirmed | **CONFIRMED** — B ≤ A on reads and wall at both feasible points |
| **R6** — best gains at moderate pressure | shape confirmed | **CONFIRMED** — P rises monotonically as ownership tightens |
| ≥10% fewer physical NVMe reads | confirmed at both | **PARTIAL** — miss at 96 (−6.4%), pass at 64 (−16.4%) |
| 5–15% lower exposed cold wall | MISS (3.4–4.4%) | **UNRESOLVED** — −0.9% / −3.4% here, not stable across hosts |
| ghost working set | ~4× | **CONFIRMED** — 4× cut in ownership, 0 extra reads |
| feasibility extension | observed | **CONFIRMED** — identical named refusal |
| no numerical difference | confirmed | **CONFIRMED** — 19/19 arms token-identical |

## What this still does not establish

Everything in "What this does not establish" above stands unchanged: no
cold→GPU arm, no VRAM-side reclaimable arm (R2, R3, R4, R7–R10 remain
untested), `order="tail"` only, one trace's locality. Added by this run:

- **The uncontended regime cannot score R1** and must not be quoted for it.
  Only configurations where the routed set exceeds the pool produce the
  overwrite event the prediction is about.
- **The gap to the withdrawn absolutes is open.** Until it is explained, no
  counter in this document may be compared against the pre-correction one.
- **The wall clause needs an instrument that resolves single-digit
  percentages** before it can be scored either way.
