# Stage 3 — what the tribrid measurements concluded

Written after gates 1 and 2 and the reclaimable-residency arm, from the
receipts in `bench/cold-engine/`. Both preregs are stamped
([base](../../bench/cold-engine/PREREG-tribrid-stage3.md) `7bf5b2be…`,
[amendment 1](../../bench/cold-engine/PREREG-tribrid-stage3-amendment1.md)
`0d5f9dbe…`), so every prediction below was registered before the data
existed.

## The thesis, and what happened to it

> **Placement decides where bytes should live. Timing decides where an
> invocation should execute. Tribrid scheduling decides how cold bytes reach
> that execution before they become the critical path.**
>
> — the directive's core claim

**The scheduling half of that is not supported by these measurements.** Cold
handling on this workload is dominated by *retention* and by *per-call
software cost*, neither of which is a scheduling problem. Two of three
scheduling gates missed, and the clear win came from a memory mechanism.

| | verdict | why |
|---|---|---|
| **Gate 1** — can cold mass be admitted without proportional wall growth? | **MISS** | The hide-ratio clause is unreachable *by construction*: storage is a minority of cold-path cost at 1–10% cold mass, so a perfect prefetcher could remove only that fraction of the exposure. The published 5–11% is **6–12%** once charged at the box's real sequential ceiling, and lower still once the read counts are re-taken on the fixed window — see the correction in `RESULTS-tribrid-gate1.md`. The verdict rests on prefetch coverage (<1% of demand misses), a ratio inside the window and unaffected by either error |
| **Gate 2** — does choosing a destination by deadline beat a threshold? | **MISS** | Backlog changed 0 of 1975 decisions in the regime built to provoke it; the rule's own syncs cost 11.7–15.6% for routing identical to fixed-CPU |
| **Reclaimable residency (R1, R5)** | **CONFIRMED** (withdrawn, then re-measured) | P(reuse before overwrite) is **13.5 / 24.5 / 34.2%** at protected 96/64/32 on a decode-only window — one point inside the registered 5–20% band and two above it. Reads **−13.5% / −24.5%** at the two feasible budgets, wall −2.5% / −7.9%. The earlier 11–60% was withdrawn for a nested-ensure defect that does inflate P, but only by 7–15% relative; the real distortion was a measurement window that counted warmup prefills against decode-only wall. See `RESULTS-tribrid-reclaimable.md`. **No read count here is comparable to one published before that window fix** |
| **VRAM reclaimable residency** | **Validated, and it needed fixing** | Bitwise-equal to the uncached engine on GPU, and now measured on a real 512-step OLMoE decode routing sequence rather than a fixture that held every expert. **As shipped it LOST to the positional cache the engine already had** (108.3% of its transfers at 12.5% capacity). Two policy defects: `protected` defaulted to half the arena, and `VramSlots` incremented a `_clock` it never read, so eviction was by slot index. Fixed, it tracks ideal LRU everywhere and removes 23% of the positional cache's transfers at 12.5% capacity. Untimed. `bench/cold-engine/routing-trace/` |
| **Gate 3** — is the placement loop worth closing? (scored offline, not run) | **YES, and the obvious policy is not the right one** | Adaptive re-placement beats static by **6–41%** fewer reads at matched capacity, with migrations under 1% of the reads they save. Controls: it is not a stale profile (doubling the profile gains 1.8–13.5% and adaptive still gains on top), not noise (the top-384 set moves, Jaccard 0.558 between halves), and **4–30% above the best achievable fixed set**, widening with capacity. Demand-paging with no placement at all is *worse* than static below 512 rows. One prompt. `bench/cold-engine/routing-trace/RESULTS-gate3.md` |
| **Re-placement policy** (follow-on, not registered) | **A third of the headroom taken** | Decaying the frequency counts at each re-placement (EWMA, half per period) wins at every capacity and closes **31–44%** of the gap between gate 3's policy and the best achievable fixed set. Interior optimum at 0.5 — worse on both sides — with the curve turning back up as it approaches the short window R4 refuted. A pinned/demand-paged **hybrid lost**: it improves monotonically as it stops being a hybrid. Two thirds of the gap remain, and the oracle is a fixed set, so what is left is choosing better rather than more often. `bench/cold-engine/routing-trace/RESULTS-policies.md` |
| **R7** — reclaimable residency moves the NVMe knee outward 20–50% | **REFUTED** | 14 capacities from 64 to 1024 rows, matched memory. Hard knee 1024, soft knee 1024, movement **+0.0%**; soft is worse below 768 and indistinguishable above. Not threshold-sensitive — the curves are within 2% everywhere. And there is barely a knee to move: the read-vs-capacity curve is smooth and concave, marginal value falling 119→9 reads per row. `bench/cold-engine/routing-trace/RESULTS-r7.md` |
| **R10** — reclaimable residency cuts churn and refills | **REFUTED** | 10 of 10 at **matched capacity** — both arms holding the same physical rows, differing only in whether ownership is capped below that number. Reads and churn both move the wrong way, +0.7% to +1.5%. Given 128 rows, owning all 128 beats owning 96 and letting 32 be reclaimable: reclaimable rows lose every allocation contest, so they are overwritten first. **Implies a control R1's read clause never had** — its arms were not capacity-matched, and the matched version reverses the sign. `bench/cold-engine/routing-trace/RESULTS-r10.md` |
| **R3** — DRAM resurrection rate exceeds VRAM | **UNDETERMINED** | Measured matched for the first time: one captured trace through both state machines at the same capacity and budget. The verdict **inverts with the protected budget**, which R3 never pins — holds 5/5 at `rows/2`, refuted 4/5 at `rows−k`. At 128 rows the VRAM rate moves 0.0%→33.9% and the DRAM rate 13.5%→0.4% on that one setting. Left undetermined rather than refuted because amending a registered prediction to match a result is what preregistration prevents. `bench/cold-engine/routing-trace/RESULTS-r3.md` |
| **R4** — short-window recurrence beats long-run frequency | **REFUTED as stated** | Scored on the captured OLMoE decode sequence, six capacities × six windows. At genuinely short windows frequency wins **5 of 5** signal-bearing capacities (w=4 and w=8); recurrence only starts winning at w≥16 and only at the two smallest. Recency's ρ rises monotonically with window width everywhere, converging on frequency from below — it predicts better the more it behaves like frequency, which is the opposite of the claim. Gate 3's loop should be frequency-driven; a windowed predictor, if kept, wants a WIDE window. Headroom is limited either way (best ρ = 0.476 outside the smallest capacity). `bench/cold-engine/routing-trace/RESULTS-r4.md` |
| **Direct scatter** (implementation, not a registered gate) | **Real, regime-bound** | −43% on the fill path in isolation; −12.5% end-to-end at 20% cold mass; **null** at 5% |

## The one prediction that mattered most was the directive's own

> *"NVMe itself will cease being the interesting bottleneck surprisingly
> quickly. Once reads overlap and hot misses get retained, the next wall may
> become staging/copy orchestration or the compute destination's available
> slack."*
>
> — the prediction the directive named as most worth falsifying

**Confirmed, almost verbatim.** Retention did it: the cold tier caches
effectively enough that at 5% cold mass a served step issues **37 disk reads
across 128 steps**. What remained was staging and copy orchestration, which
is exactly what the direct scatter attacked and where the only end-to-end
speedup came from.

The consequence is that the program's central metaphor inverted. "Make cold
storage latency schedulable" presumes storage latency is the wall. Once the
tier retains, it is not — and what replaces it is not schedulable either,
because it is per-call software cost, which you fix rather than schedule.

## The uncomfortable headline

**Fixed `cold_dest="cpu"` beat every policy tried.** At 20% cold mass, on
both load-asymmetry regimes, the best destination policy was no policy. The
threshold lost to it by ~8%; the deadline estimator lost by more.

A scheduler earns its instrumentation only where the right answer varies.
Here it barely does: the predicted CPU and GPU join times are far enough
apart at these shapes that neither backlog nor routing shape moves the
answer often enough to pay for asking.

## What is *not* falsified, and should not be over-read

- **Prediction 3** (a cold expert should sometimes run on the slower engine
  because the faster one is committed) is **not** refuted. It was *observed*
  — 58 flips in the cpu-loaded regime — just too rarely, and too cheaply
  approximated by group shape, to pay for the machinery.
- **Prediction 2** (>70% hide ratio) was never given a working hiding
  mechanism to be tested against; the shipped prefetcher covers <1% of
  critical-path reads because the tier has already cached what it predicts.
  Scoring it as refuted would be scoring a prediction against an instrument
  that was not running.
- **One box class, one model, one workload shape** for every number here.
  Instrument law 7 applies throughout.

## What the evidence actually points at

The receipts favour **residency over scheduling**, in three ways:

1. **Reclaimable residency is the strongest result in Stage 3** and the only
   one that improved a registered metric by more than its predicted band.
   It also *extends the feasible operating range*: hard eviction at
   `protected=32` cannot run at all, while reclaimable at the same budget is
   unremarkable.
2. **Retention is what removed storage from the critical path** — not
   prefetch, not destination choice. The tier's cache did it.
3. **The remaining cold cost is implementation, not policy**, and yielded to
   an implementation fix (`preadv` scatter) where it yielded to nothing else.

Gate 3 (adaptive residency — promotion and demotion driven by observed
reuse) is **now scored offline and the answer is yes** — see the verdict
table. It remains the live thread, and R2, R3, R4 and R7–R10 remain
untested — **R4 is now scored and REFUTED**, see the verdict table. The VRAM
side of reclaimable residency now has that real routing trace
(`bench/cold-engine/routing-trace/olmoe_routing_seq.jsonl`, 512 autoregressive
decode steps of OLMoE), and it was worth taking: sized far below the expert
count, the cache as shipped was *worse than the positional one already in the
engine*. Two policy defects explain it and both are fixed. The trace is also
the first real captured routing **sequence** in this repo, which is what R4 was
registered against; R4 is scored above.

**One correction still outstanding.** The measurement-window defect that
distorted R1 is also present in `run_gate1.py`, and is fixed there — but
**gate 1's published read counts were taken with it and are uncorrected.**
They are warmup-inclusive where they claim decode-only, by roughly the
factor R1 measured (six sevenths of the traffic was warmup). A second,
smaller error was found in the same addendum and *is* corrected: disk time
was charged at 6.26 GB/s, described as the box's sequential ceiling, which
is in fact its **random** qd16 rate — the sequential ceiling is 5.51.
`run_gate1.py` had been reading a `seq_best_gbs` key that no calibration in
this repo produces, so every gate-1 receipt carries `b_nvme_gbs: null` and
the constant was chosen by hand. It now derives the ceiling from the
sequential points and raises when there are none.

Gate 1's MISS does not rest on those absolutes — it rests on prefetch
coverage, a ratio taken inside the window — so the verdict stands, and
correcting the reads makes the reframing stronger rather than weaker. **No
read count in that document should be quoted until it is re-run.**

## Reclaimable residency does not pay at matched capacity

Three predictions about the mechanism have now been scored against one
captured decode trace, and the picture is consistent:

* **R10 — REFUTED**, 10/10. Capping ownership below capacity costs ~1% more
  reads and churn.
* **R7 — REFUTED**. The knee does not move at all (+0.0%).
* **R3 — UNDETERMINED**. Its verdict is decided by the protected budget,
  which it never pins.

The mechanism is sound and the state machine is correct — that was settled
separately, bitwise, on GPU. What the measurements say is that **at matched
capacity there is nothing to buy**: reclaimable rows lose every allocation
contest, so they are overwritten first, and owning a row outright retains it
strictly better than leaving it reclaimable.

**This is one trace, uncontended, on a synthetic arena, `ColdTier` only.**
The gap it does not close is contention: R5 reports soft eviction *faster*
than hard when the tier is contended, which is exactly the regime where a
ghost row that survives long enough to be resurrected could pay for itself.
That measurement has not been made, and it is the one that would change this
conclusion if anything does.

## Two arms that were never capacity-matched

**Three** results in `RESULTS-tribrid-reclaimable.md` rest on one comparison:
Arm A's pool is `protected` rows, Arm B's is **128**. Arm B has 33%, 100% or
300% more memory depending on the point.

| claim | as recorded | matched control |
|---|---|---|
| R5 — soft ≤ hard, *"faster contended"* | wall 79.02→76.31, 80.46→76.91 ms | **soft +2.3% to +9.9% SLOWER** on real NVMe (#153) |
| ≥10% fewer NVMe reads | −14.9%, −29.6% | **+0.7% to +1.5% worse**, 10 of 10 (#145) |
| feasibility extension | Arm A at protected=32 "does not run" | **hard eviction runs it fine** given the same 128 rows |

Every sign reverses. The feasibility one settles without a trace: the refusal
is `request of 36 unique rows exceeds hot_rows=32`, a statement about the
**pool**, and `hot_rows=128, protected=128` serves the same request.

**The measurements reproduce; the attribution does not.** A-vs-B varies
capacity and ownership together and cannot separate them.

R5 is refuted by the half its escape clause does not cover: the clause
exempts regressions "attributable to metadata/sync", and the +0.8–8.7% wall
residual *is* the `_demote` walk — but the +1.1–1.5% extra **reads** are
eviction quality, not metadata. Worth recording that the clause exempts one
of the only two channels through which this mechanism could ever lose, which
is the same defect shape as R3's unpinned budget and R2's two denominators.

Full working: `bench/cold-engine/reconciliation/RESULTS-r5-reconciled.md`.
R6 is scored in `RESULTS-r6.md` (confirmed as registered, but the band is
worth +0.73% against −1.50% elsewhere; the earlier CONFIRMED is vacated), and
the uncontended half of R5 stands.

## Two models, four prompts: one conclusion survives

Every offline result here replayed one trace of one model. Eight traces now
exist across two architectures — OLMoE (16×64, top-8) and Granite-3.0-3B-A800M
(32×40, top-8) — compared at equal *fractions* of the arena rather than equal
row counts.

| conclusion | four prompts | + second model |
|---|---|---|
| **R4 refuted** (frequency > short-window recurrence) | holds 20/22 | **holds 18/18 — 38 of 40 overall** |
| **Device row cache beats the positional cache** | holds | **BREAKS — 123–130% of positional at 12.5%** |
| **Gate 3** — adaptive beats static | direction holds | **BREAKS — +0.0% on Granite math** |
| **EWMA is the better policy** | refuted | adaptive wins 13 of 24 |
| **Placement beats demand-paging when the tier is scarce** | refuted | — |

**One of five survived both axes.** R4 is the only result that looks like a
property of MoE routing rather than of a trace.

The device row cache failed in exactly the way its own results document said
it might — *"it pays only if re-routing to a new position is common"* — and
four prompts of one model could not reach that failure mode, while one prompt
of a second model did.

**That "concentration" explanation was then scored, and does not hold.**
Coverage of the arena predicts the cache outcome at ρ = 0.276 and entropy at
0.363 — nothing. Against the gate-3 gain it reaches ρ = +0.468, the right sign
but under the 0.5 every hypothesis was held to: weak, not supporting. It was
a story fitted to two salient cells.

**What separates is arithmetic: `steps_held` = capacity ÷ (layers × top-k)**,
the rows one decode step asks for. Every configuration where the device cache
loses has `steps_held < 1`; every one where it wins has ≥ 1 — **24 of 24**,
ρ = −0.895.

**And the reason is LRU, not capacity** — the second explanation to be scored
and corrected here. Routing per step is a near-cyclic scan of those rows, and
LRU below the cycle length is the textbook zero-hit case: below one step LRU
retains *nothing* in 24 of 24 cells, FIFO likewise, while **random eviction is
zero-hit in 0 of 24**. Above one step LRU is best again in 22 of 24. The
failure is a **cliff, not a slope**, so adding rows below the threshold buys
nothing — which is what a reader needs to know when they cannot size it.
Guidance is unchanged (size to one step): random only beats the engine's
positional cache in 7 of 24 sub-threshold cells, all at the boundary.

**And the question left open there is now answered.** Demand-paging beats
static placement whenever **capacity covers the scored working set** —
`headroom = working set ÷ capacity ≤ 1` classifies 96 configurations at 91.7%
against a 70.8% base rate, with **20 true positives and zero false
positives**. Sufficient, not necessary: eight cells win between 1.07 and 1.93
as well. If the fast tier holds every row the window asks for, LRU takes only
compulsory misses and no placement can beat compulsory.

That correction is worth more than the rule. The earlier test reported "not
supported, ρ = 0.055" on **three** positives; with twenty-eight, ρ = −0.720.
And ρ was the wrong statistic either way — a threshold hypothesis scored by
rank correlation, across a range where 21 of 24 cells sat on one side of the
threshold, could not have seen a step. **Score threshold claims as
classifiers against a base rate**; a correlation coefficient does not report
when it had no power.
`bench/cold-engine/routing-trace/RESULTS-demand.md`.

That also corrects how the two models were compared. Granite routes 256 rows
per step from a 1280 arena; OLMoE routes 128 from 1024. **Fraction of the
arena was the wrong normaliser** — at "12.5%" Granite held 0.62 of a step and
OLMoE held exactly 1.00, so they were never at the same pressure.
`DevRowCache.stats()` now reports `steps_held` and `too_small_to_retain` so a
deployment below 1.0 says so.

**What this says about how the next stage should be registered.** Three
predictions here (R3, R5, R2) were unfalsifiable because their verdict turned
on a parameter they never pinned. These five were falsifiable but keyed to the
wrong variable. A clause of the form *"policy X beats Y below N rows"* cannot
be scored across workloads; *"below N rows **relative to the scored working
set**"* can.

Full working: `bench/cold-engine/routing-trace/RESULTS-generalization.md`.
## A metric this campaign leaned on does not carry weight

R1–R3 all use **resurrection rate** as though higher were better. Measured
against physical refills on one trace, it is not reliably coupled to cost:

* At 128 VRAM rows the rate rises **0.0% → 33.9%** while refills *improve*,
  65,536 → 43,338. Rate up, cost down.
* Between victim rules at 256 rows the rate falls **266 → 0** while refills
  also improve, 54,819 → 43,338. Rate down, cost down.

It rises with quality in one comparison and falls with it in another. A
resurrection is a **capacity-relative bookkeeping event, not a saving** — it
counts rows that were demoted and then needed again, which a cache that
demoted better would never have demoted at all.

**Report physical refills.** The resurrection rate cannot carry a claim by
itself. R1 is undisturbed: its operational half was measured in reads
(−13.5% / −24.5%), not in this rate.

## Cost of the campaign

Six rented boxes, ~$3.06 total. Every box destroyed and verified
(SSH-refused plus API-null), except one where the API confirmed teardown but
the SSH probe raced it — recorded as such rather than claimed clean.

One receipt was lost to operator error (scp chained with teardown) and the
point re-run rather than cited from scrollback; both runs agreed, and only
the re-run is cited.
