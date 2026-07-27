# RESULTS-routed-residual — ADDENDUM 1

**Amends the reading of `RESULTS-routed-residual.md` (sha `5252e645…`,
committed `e5bc707`). The original is NOT edited; its stamp stands.**

## What changed

`a74691a` (same day, parallel session) found the actual defect behind the
impossible fraction, and it was **not the ceiling**:

- A size-swept re-probe on the same pod gave **12.65–18.46 GB/s**, so the
  in-run 14.68 was about right. The original's diagnosis — contended probe,
  pinned-vs-pageable as invalidator, NUMA speculation — is **withdrawn**. (The
  standalone 26.0 best-case reading is unconfirmed against the sweep;
  probe-methodology disagreement is noted, unresolved, and no longer
  load-bearing.)
- **The divisor was broken**: `offload_stats_report` divided link bytes by
  summed per-stage copy WINDOWS around `non_blocking` enqueues, which do not
  bound the transfer. The bytes were always correct (independent model
  7.984 GB/token vs harness 7.98). Fixed in e4b `aa3e948`
  (`fix/routed-gbps-wall`).

## Corrected numbers (post-hoc; labelled, not registered)

Sound transport rate = 7.984 GB / 0.9061 s = **8.81 GB/s**. Fraction of ceiling:

| ceiling estimate | fraction | vs bar ≤0.70 |
|---|---|---|
| 12.65 (sweep min) | 0.70 | holds (at the edge) |
| 14.68 (in-run)    | 0.60 | holds |
| 18.46 (sweep max) | 0.48 | holds |
| 26.0 (standalone) | 0.34 | holds |

**R4-corrected: HOLDS at every ceiling estimate available.** This is a
bugfix-grade recompute of campaign-1 data, performed after the fact — it is
**corrected-exploratory evidence, not a registered confirmation**. Campaign 1's
registered R4 remains UNADJUDICATED as the original states; the registered test
of the corrected instrument is PREREG-routed-residual-2 as amended.

## Retraction

The original's exploratory paragraph ("fraction lands 0.85–0.99 … pointing
toward *transfers already near-efficient; the coalescer would not pay*") is
**RETRACTED** — it was arithmetic on the broken numerator. The sound recompute
points the other way: ~30–50% of the link is unclaimed during the routed step,
i.e. toward building the coalescer, pending the registered re-run.

## Instrument consequence

The harness no longer trusts any stats-layer rate: the registered numerator is
computed in `routed_residual_verdicts.py` from receipt fields
(`by_policy.routed.bytes / decode_wall_s`), commit `1f5740e`. A
campaign-1-shaped receipt refuses to adjudicate rather than yielding a
build-dependent verdict. Discontinuity note per `a74691a`: receipts produced
before/after e4b `aa3e948` disagree on what `gbps` means; only the
harness-computed number is comparable across builds.
