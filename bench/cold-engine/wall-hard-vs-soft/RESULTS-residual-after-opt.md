# The residual SURVIVES the tier optimisations, and is now entirely outside the read

**Registered outcome: SURVIVES.** Cutting 61% of the tier's non-read work
removed only 16% of the soft-hard gap. The optimisation took work *both* arms
did; the asymmetry it was supposed to explain is still there.

Registered in [`PREREG-residual-after-opt.md`](PREREG-residual-after-opt.md)
before the box was rented, with DISSOLVED / SURVIVES / PARTIAL named in advance.

## Result

`rows=256`, `protected=248`, `qd=1`, **9 repeats**, pinned, O_DIRECT.

| | hard | soft | Δ |
|---|---|---|---|
| wall | 49923.4 ms | 53399.5 ms | **+6.96%** |
| `read_ns` | 44748.4 ms | 45302.9 ms | **+1.24%** |
| `non_read_ns` | 5175.0 ms | 8096.6 ms | **+56.46%** |
| reads | 32169 | 32605 | +1.36% |

Decomposing the 3476.1 ms of extra wall:

| term | ms | share of residual |
|---|---|---|
| explained by 436 more reads, inside | 606.5 | |
| explained by 436 more reads, outside | 70.1 | |
| **residual** | **2799.5** | |
| — inside reads | **−52.0** | **−1.9%** |
| — **outside reads** | **2851.5** | **101.9%** |

## Against the pre-optimisation measurement

| | before | now |
|---|---|---|
| residual | 5.08 pts | **5.61 pts** |
| `Δread_ns` | +2.19% | **+1.24%** |
| `Δnon_read_ns` | +26.59% | **+56.46%** |
| `non_read` share of hard wall | 17.5% | **10.4%** |
| residual outside the read | 86.5% | **101.9%** |

Two things happened at once, and only together do they answer the question.

**The optimisation worked.** Hard-arm non-read work fell from 13159.4 ms to
5175.0 ms, **−61%**, and its share of wall from 17.5% to 10.4%.

**It did not touch the mechanism.** The soft-hard non-read *gap* fell only from
3498.8 ms to 2921.6 ms, **−16%**. So `_victim` — sweep and all — was worth at
most a sixth of the residual, and what remains is asymmetric work the LFU heap
does not do.

This is exactly the reading the preregistration warned about: `non_read_ns`
shrinking in absolute terms is what the optimisation *did* and is not evidence
either way. The delta is the residual, and the delta **grew** (+26.59% →
+56.46%) because the denominator shrank faster than the gap.

## The per-read storage cost is gone

Before, 13.5% of the residual sat inside the reads. Now `Δread_ns` is **+1.24%
against +1.36% more reads** — marginally *below* count, so the soft arm's reads
are individually no more expensive, and the inside-reads term is **−1.9%**.

Every candidate that could live in the storage path is now excluded by
measurement rather than by argument: locality (same read order, identical
medians), per-read cost (absent), read count (subtracted), queue depth
(invariant), and the victim sweep (≤16%).

## Instrument

The sharpest in the campaign: **IQR 0.38–0.41%** on 9 repeats against an 8.4%
effect — a 20× margin, against #153's 17× and the previous box's ~5×.

Drift bracket: leading control **+8.36%**, trailing **+7.92%**, 0.44 points
apart. Stable across the session.

Preconditions: O_DIRECT confirmed to *bypass* (64 MB read twice: 32.6 then
32.5 ms, ratio **1.00**), arena row bytes 3342336, read counts 32169/32605
matching `RESULTS-r10.md`.

## Two boxes were rejected before this one, which is the point of the gate

**Attempt 1** (A4000, 1028 MB/s advertised) drifted 10 points *under* the
measurement: leading control +7.50%, trailing +17.83%, intra-run spread
4.5–15.5%. Without the trailing control I would have had a leading control at
+6.14 pts and a timed run at +17.88 pts, and "the residual grew" would have
looked like a finding. It was host noise. Advertised `disk_bw` predicted none
of this — the probe measured 2.45 GB/s, and the real problem was contention.

**Attempt 2** was this box, refused by a gate that then turned out to be
measuring wrong. The gate now runs the leading control first and aborts before
the expensive timed run unless the effect clears spread by 3×; on n=5 with
max−min it read hard 0.44% / soft 4.41% and refused. The soft arm's four other
repeats spanned 1.1% — one outlier decided it. The **estimator** was changed to
IQR/median and repeats raised to 9; the **threshold stayed at 3×**. That
distinction is deliberate: loosening the threshold after seeing a refusal would
have been fitting the gate to the answer.

## What this does not show

One trace, one capacity, one queue depth, and a **different box** from the
pre-optimisation run, so the absolute residual (5.08 → 5.61 pts) should not be
read as an increase — cross-box comparison of absolute wall is exactly what
this campaign keeps punishing. What transfers is the *within-run* partition:
101.9% outside the read, and the 61%-vs-16% asymmetry, both measured against
their own arms in the same process.

"Outside the read" remains a region, not a mechanism. The next suspect on
evidence is the per-read dispatch machinery — a Future, an Event and several
lock round-trips per read, ~28% of tier CPU after the heap — but that is a
hypothesis this run does not test.

## Receipts

`residual-after-opt/` — `a_ctl_lead`, `a_timed_qd1`, `a_ctl_trail`, `warm`,
plus `refused_box_ctl` from the gate refusal. Two boxes destroyed; total spend
for this measurement **$0.32**.
