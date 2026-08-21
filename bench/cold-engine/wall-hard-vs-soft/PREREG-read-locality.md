# Preregistration: is #153's residual read *locality*?

Registered before running the instrument. Config is fixed to the qd probe's
(`rows=256`, `protected=248`), so this measures the same configuration the
residual was measured on.

## The residual

| qd | hard | soft | wall Δ | reads Δ | residual |
|---|---|---|---|---|---|
| 4 | 24279 ms | 25799 ms | +6.3% | +1.3% | **+5.0** |
| 1 | 36303 ms | 38660 ms | +6.5% | +1.4% | **+5.1** |

Soft costs ~6.4% more wall while issuing only ~1.35% more reads. About **5
points** are unexplained by read *count*, and the gap does not move with queue
depth. Four candidates are eliminated (demote sort, the whole demote path,
"lower effective bandwidth" as a restatement, CPU–I/O overlap). What remains
is a **per-read** cost: soft's reads are individually more expensive.

## Hypothesis

Read **locality**. The two arms read different rows in different orders. If
soft's consecutive reads land further apart in the arena, each read costs more
— longer seeks, worse readahead, fewer same-erase-block hits — at identical row
size and count. Nothing in the campaign records offsets, so this has never been
looked at.

## Instrument

The offset sequence is decided by **policy**, not by row size: which row is
read when is a function of the trace and the eviction rules. So it can be
recovered offline from the toy arena and mapped onto the real geometry
(1024 rows x 3.3 MB) for interpretation. Distances are reported in **rows**
(the policy-level quantity) and in bytes at the real 3.3 MB stride.

Metric: `|Δoffset|` between **consecutive physical reads** (misses only — hits
touch no disk).

## Predictions, registered

- **CONFIRMED** if soft's central tendency is materially above hard's — enough
  that a per-read cost difference of the observed size is plausible. Soft
  issues ~1.35% more reads and costs ~6.4% more wall, so per-read cost must be
  up ~5%. A locality gap that could carry that should be visible as a clear
  separation in median/mean `|Δ|`, not a fractional one.
- **REFUTED** if the two distributions are indistinguishable — medians within a
  few percent and quantiles overlapping. Then locality is the fifth eliminated
  candidate and the residual is something else again.

## Stated in advance

This is a **necessary-condition** test, not a sufficient one. Finding a
locality gap would not prove it causes the residual; it would make locality the
first candidate not yet ruled out, and would need a real-NVMe A/B to close.
Finding *no* gap does rule it out, because a per-read cost attributed to
locality requires the offsets to actually differ.

The toy arena cannot speak to absolute seek cost — only to whether the two
arms' offset *sequences* differ. That is exactly the question, and it is
geometry-independent.
