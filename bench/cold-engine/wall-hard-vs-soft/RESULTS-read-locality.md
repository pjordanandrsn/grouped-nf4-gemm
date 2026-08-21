# Read locality is not the residual either. Five candidates down.

**REFUTED, as preregistered.** The two arms read the arena in the *same*
order, to the digit on the median. Locality cannot carry a ~5% per-read cost
that isn't there.

Registered in [`PREREG-read-locality.md`](PREREG-read-locality.md) before the
instrument ran. Scored by
[`../routing-trace/score_locality.py`](../routing-trace/score_locality.py).

## The measurement

`|Δoffset|` between consecutive **physical** reads, in rows (the policy-level
quantity; the real arena's stride is 3.3 MB). Hard = ownership uncapped, soft =
capped 8 below, at matched capacity. `qd=1`, so the replay is deterministic.

| rows | arm | reads | mean (rows) | median | IQR | Δ mean | Δ median |
|---|---|---|---|---|---|---|---|
| 128 | hard | 44298 | 23.28 | 9.0 | 4–17 | | |
| 128 | soft | 44964 | 22.94 | 9.0 | 4–17 | **−1.5%** | **0.0** |
| 256 | hard | 32169 | 31.58 | 12.0 | 5–23 | | |
| 256 | soft | 32605 | 31.18 | 12.0 | 5–22 | **−1.3%** | **0.0** |
| 384 | hard | 21890 | 45.77 | 17.0 | 7–33 | | |
| 384 | soft | 22150 | 45.25 | 17.0 | 7–33 | **−1.1%** | **0.0** |
| 512 | hard | 13616 | 71.28 | 25.0 | 11–53 | | |
| 512 | soft | 13720 | 70.72 | 25.0 | 11–52 | **−0.8%** | **0.0** |

`rows=256, protected=248` is the qd probe's own configuration, so the headline
row is measured where the residual was measured.

## Why this refutes it

The registered falsifier was "medians within a few percent and quantiles
overlapping." The medians are **identical at every capacity**, the IQRs are
identical or off by one row, and the means differ by ~1% — **in the wrong
direction**. Soft is marginally *more* local than hard, not less.

For locality to explain the residual, soft's reads would have to be materially
*further apart*, enough to make each read ~5% more expensive. They are not
further apart at all. This was preregistered as a necessary-condition test:
finding no gap rules the hypothesis out, because a per-read cost attributed to
locality requires the offsets to actually differ.

I do not have an explanation for the consistent ~1% in soft's favour and am
not going to invent one. It is small, it is the opposite of the hypothesis,
and nothing here rests on it.

## Instrument checks

Two, both passed before the numbers were read:

- **Captured reads == the tier's own miss counter**, exactly, in all eight
  runs (44298, 44964, 32169, 32605, 21890, 22150, 13616, 13720). The spy saw
  every physical read and invented none.
- **Those counts match `RESULTS-r10.md`** at the same capacities, which was
  scored independently and earlier. The replay is reproducing known behaviour,
  not a private variant of it.

Reads are captured by wrapping the **reader**, not by calling `ensure()` one
expert at a time. Splitting the per-layer call would change the tier's
batching, and batching is part of what decides the read order this measures —
the instrument would have altered the thing it was there to observe.

## Where the residual stands

Five candidates eliminated, and they were the plausible ones:

| # | candidate | how it died |
|---|---|---|
| 1 | the demote sort | real-NVMe A/B (#161) |
| 2 | the whole demote path | O(over) probe bounded it at ≤0.8 of 4.1 pts |
| 3 | "lower effective bandwidth" | algebraically a restatement, not a cause |
| 4 | CPU–I/O overlap | flat at qd=1 and qd=4 (#163) |
| 5 | **read locality** | **same read order, identical medians (here)** |

What survives is unchanged and now sharper: a **per-read** cost, invariant to
queue depth, not bookkeeping, and not explained by which rows are read, how
many, or in what order. Those are most of the things a storage-side
explanation can be made of.

I am not proposing a sixth candidate. The useful artifact is that the next
person does not have to re-run these five, and that the surviving shape is
narrow enough to be worth a targeted instrument rather than another guess.

## What this does not show

The offset *sequence* is geometry-independent — it is decided by policy — so it
transfers from the toy arena to the real one. **Absolute** seek cost does not,
and is not claimed: this says the two arms issue the same read order, not what
any read costs. A real-NVMe measurement could still find a per-read difference,
but it would not be attributable to locality, because there is no locality
difference to attribute it to.

One trace, one model, four capacities, one queue depth.

## Receipts

`../routing-trace/locality.json` (the qd-probe configuration) and
`locality_{128,384,512}.json`. Offline, no GPU, no spend.
