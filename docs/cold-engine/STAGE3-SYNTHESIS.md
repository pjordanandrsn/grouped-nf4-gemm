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
| **Gate 1** — can cold mass be admitted without proportional wall growth? | **MISS** | The hide-ratio clause is unreachable *by construction*: storage is 5–11% of cold-path cost at 1–10% cold mass, so a perfect prefetcher could remove at most ~10% of the exposure |
| **Gate 2** — does choosing a destination by deadline beat a threshold? | **MISS** | Backlog changed 0 of 1975 decisions in the regime built to provoke it; the rule's own syncs cost 11.7–15.6% for routing identical to fixed-CPU |
| **Reclaimable residency (R1, R5)** | **PARTLY WITHDRAWN** | The 11–60% P(reuse) is **contaminated** by a nested-ensure defect in the measurement path (see the correction in `RESULTS-tribrid-reclaimable.md`) and reverts to untested. What survives: physical reads −14.9/−29.6% at matched budgets, the ~4× ghost working set, and the feasibility finding — all of which rest on read counts or structure, not resurrection counts |
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
reuse) is therefore the live thread, and R2, R3, R4 and R7–R10 remain
untested, as does the VRAM side of reclaimable residency entirely.

## Cost of the campaign

Six rented boxes, ~$3.06 total. Every box destroyed and verified
(SSH-refused plus API-null), except one where the API confirmed teardown but
the SSH probe raced it — recorded as such rather than claimed clean.

One receipt was lost to operator error (scp chained with teardown) and the
point re-run rather than cited from scrollback; both runs agreed, and only
the re-run is cited.
