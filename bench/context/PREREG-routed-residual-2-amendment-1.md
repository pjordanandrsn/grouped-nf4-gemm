# PREREG-routed-residual-2 — AMENDMENT 1

**Amends:** `PREREG-routed-residual-2.md`, OTS-stamped sha `d9a32256…`.
**Status: PRE-DATA for this registration.** No qualifying host has ever been
provisioned (Latitude `g3.h100.small` stock has not appeared; nothing fired).
The original is NOT edited; its stamp stands and the two can be diffed.

## Why an amendment, hours after stamping

`a74691a` + e4b `aa3e948` (parallel session, same day) showed the premise
the original was written on is wrong: the campaign-1 defect was never the
ceiling probe — it was `offload_stats_report` dividing bytes by copy-window
event brackets that do not bound a `non_blocking` transfer. Two consequences
the original cannot survive unamended:

1. **As stamped, it would re-run the broken instrument.** It pins gnf4 commits
   but never pins the e4b side, and its numerator is the stats-layer gbps.
2. **Its disclosed expectation is inverted.** The sound recompute of campaign-1
   data holds at every ceiling estimate (0.34–0.70); the authors now expect
   **R4 to HOLD**, not to be falsified.

Because the recompute is post-hoc on campaign-1 data, the registered test of
the corrected instrument is this amended prereg — new data, instrument fixed
before any of it exists.

## A1 — The numerator (replaces the original's implicit stats-layer gbps)

Registered numerator: **`by_policy.routed.bytes / decode_wall_s`**, both
accumulated over the identical decode region (stats reset after prefill; wall
summed over the same forwards), computed in `routed_residual_verdicts.py`
@ `1f5740e` — build-independent by construction. The stats-reported gbps is a
diagnostic and never adjudicates. Deployed e4b build must nonetheless carry
`aa3e948` or later, so diagnostics do not mislead operators mid-run.

## A2 — Contamination disclosure, updated

The authors expect **P2/R4 to HOLD** (sound fraction ≈0.5–0.7 on
campaign-1-class hosts). Bias direction is now toward *confirming a build
decision*; the guards are unchanged and symmetric: the 0.70 bar and P3's
binding rule in both directions carry over verbatim, and a falsifying result
would be believed and recorded.

## A3 — Ceiling: gates re-scoped, robustness added

- Ceiling = best-of-N direct pinned probe, **pre-load** (pristine link), all
  readings in the receipt. (Unchanged in method; now explicitly pre-load, with
  the post-load re-probe demoted to P4's exploratory comparison.)
- **I1 (pinned ≥ pageable) is demoted from STOP to recorded diagnostic.** The
  ceiling was not the defect; I1 failing on bare metal is a finding to record
  (P4's question), not grounds to void R4.
- **New robustness clause (registered):** if the P2 verdict is not invariant
  across the measured spread of pre-load pinned readings, there is **no
  verdict** — the box cannot resolve R4. This forecloses post-hoc denominator
  selection permanently. (Campaign-1 note: at sound 8.81 GB/s the verdict is
  invariant across 12.65–26.0, with 12.65 landing exactly at the bar.)
- I2 (stability) carries over unchanged.

## Unchanged

Fixture (Qwen3-235B, NF4 pinned, routed-only, spec/cache OFF, reps 6 ABBA, one
process); host class (root bare-metal H100, Latitude first / DO fallback after
2026-08-10) — the out-of-sample rationale stands even though the
container-anomaly rationale weakened, and P4 still wants bare-metal
introspection; P1 identity gate; P3's decision rule in both directions; cost
caps and teardown discipline; everything under "Not claimed".
