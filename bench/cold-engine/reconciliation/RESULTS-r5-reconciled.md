# R5 vs everything since: three claims, one unmatched comparison

`RESULTS-tribrid-reclaimable.md` records three results in favour of
reclaimable residency. Everything measured since contradicts all three. This
reconciles them, and the answer is the same in each case: **the arms did not
hold the same amount of memory.**

No new hardware. Committed receipts plus one structural check that runs
anywhere.

## The three claims, and the arms behind them

> Arm A sets `hot_rows == protected_rows` so nothing can be reclaimable;
> Arm B keeps a **128-slot pool** so `128 − protected` slots hold reclaimable
> rows.

| protected | Arm A rows | Arm B rows |
|---|---|---|
| 96 | **96** | **128** |
| 64 | **64** | **128** |
| 32 | **32** | **128** |

Arm B has 33%, 100% and 300% more memory. Every result below is scored across
that gap:

| claim | as recorded | measured against |
|---|---|---|
| **R5** — soft ≤ hard in cost | CONFIRMED — "faster contended", wall 79.02→76.31 ms and 80.46→76.91 ms | Arm A vs Arm B |
| **≥10% fewer NVMe reads** | CONFIRMED — −14.9%, −29.6% | Arm A vs Arm B |
| **feasibility extension** | observed — Arm A at protected=32 "does not run" | Arm A vs Arm B |

## What matched controls say

| control | result |
|---|---|
| reads, offline, matched rows ([#145](../routing-trace/RESULTS-r10.md)) | soft **+0.7% to +1.5% worse**, 10 of 10 |
| knee, 14 capacities, matched ([#147](../routing-trace/RESULTS-r7.md)) | knee moves **+0.0%** |
| **wall, real NVMe, matched** ([#153](../wall-hard-vs-soft/RESULTS-wall-hard-vs-soft.md)) | soft **+2.3% to +9.9% slower** |

Every sign reverses.

## The feasibility extension is a pool-size fact

This one needs no trace at all. The refusal Arm A hit is
`request of 36 unique rows exceeds hot_rows=32` — a statement about the
*pool*, not about eviction policy. Running the same 36-row request three
ways:

| configuration | result |
|---|---|
| `hot_rows=32, protected=32` — Arm A as run | **REFUSED** — "request of 36 unique rows exceeds hot_rows=32" |
| `hot_rows=128, protected=32` — Arm B as run | OK |
| `hot_rows=128, protected=128` — **hard eviction, matched pool** | **OK** |

Hard eviction serves the request fine when given the same 128 rows Arm B had.
Reclaimable residency did not extend the feasible range; **a bigger pool
did.** The document already says the right thing about what it observed —
"capacity ownership and information retention coming apart" — but the
comparison it observed it through varied capacity, not just ownership.

## Is R5 refuted, or exempted?

R5's refutation condition is narrow:

> refuted if **measurable regression not attributable to metadata/sync**

The wall regression splits in two:

| component | size | attributable to |
|---|---|---|
| extra reads | +1.1% to +1.5% | **eviction quality** — reclaimable rows lose every allocation contest, so they are overwritten before reuse |
| residual wall | +0.8% to +8.7% | **metadata** — `_demote` walks the ACTIVE set on every request |

The residual is exactly what the clause exempts. **The read component is
not.** Extra disk reads are not metadata and not sync; they are the mechanism
retaining worse. So R5 is refuted on its own terms, by the component its
escape clause does not cover.

**And the clause is a preregistration defect worth recording.** Soft eviction
differs from hard in precisely two ways: it keeps more bookkeeping, and it
retains differently. Exempting "metadata/sync" exempts one of the only two
channels through which the mechanism could ever lose. A refutation condition
that excludes the dominant failure mode is close to unfalsifiable — the same
shape as R3's unpinned budget and R2's two denominators.

## What this does not claim

- **The original measurements are not wrong.** They reproduce. What fails is
  the attribution: A-vs-B varies capacity and ownership together, so it cannot
  separate them.
- **R6 is untouched.** "Best gains at moderate pressure" is a shape across
  protected budgets within one pool, not an A/B comparison.
- **The uncontended half of R5 stands** — "flat uncontended" agrees with the
  matched controls, which find the gap smallest at the smallest pools.
- One trace and one arena for the read and wall controls; the feasibility
  check is structural and depends on neither.
