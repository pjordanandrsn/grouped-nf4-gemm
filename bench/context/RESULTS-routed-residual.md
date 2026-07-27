# RESULTS — PREREG-routed-residual (`ba7a29ce` + amendment-1 `2397612e`)

**Adjudication: R1/R2/R3 PASS · R6 no-measurable-effect · R4 UNADJUDICATED
(instrument invalid on this host class) · R5 NO DECISION REGISTERED.**

Two runs on RunPod 2×A100-SXM4-80GB (same pod `v1t2jcd4fep8ly`, image
`runpod/pytorch:2.4.0-cu12.4`), 2026-07-27. Run 1 complete (24 arms, receipt
`routed_residual_reps6.json`). Run 2 carried the decode-only + settled-ceiling
fixes and was killed ~12/24 arms in by an **external RunPod-side pod
termination** (~2h15m uptime, ~2h before the armed teardown backstop; same
failure class as the two unexplained A100 terminations noted 2026-07-26). The
continuous off-box puller preserved its log; per-arm timings and all ceiling
probes survived. Total spend ≈ $6.

## Registered outcomes

| pred | outcome | evidence |
|---|---|---|
| R1 bit-identity | **PASS** | greedy ids identical across C/T1, both runs |
| R2 engagement | **PASS** | run 1: T1 `host:1410/device:0`, C exact mirror; run 2: `1316/0` |
| R3 no-regression | **PASS** | 1.0033 vs self-pair spread 0.0161 |
| R6 magnitude | **no measurable effect** | run 1 ratio **1.0033**, run 2 (partial) **0.9964** — the sign flips across runs; the effect is inside noise. R6's literal verdict at 1.0033 is "REGRESSION" only because the registered band has a hard edge at 1.00; with the sign flip in hand, reading that label as a real regression would be wrong. Registered miss of the [0.95, 1.00] band edge, honest reading: **zero**. |
| R4 decomposition | **UNADJUDICATED** | see below |
| R5 decision | **NONE** | R5 consumes R4; no verdict may be manufactured |

Harness gates: `arm_fidelity` PASS both runs (C on dict plan, T1 on flat, exact
mirrors). `logit_identity` (not registered; #24's standard) max |Δlogit| = **0.0**
in run 1. `position_balance` 69/69/69/69. `prefill_separated`: run 1 **None**
(pre-fix receipt cannot testify), run 2 **confirmed** by counters (1316 = 94×14,
prefill excluded).

## Why R4 could not be adjudicated

Two independent defects, either fatal alone:

1. **Numerator (fixed mid-arc):** `routed_gbps` blended prefill with decode.
   Prefill routed ~52 unique experts/layer (vs decode's 8, under
   `_routed_max`=64), so ~32% of expert-stages arrived in copies 6.5× larger,
   inflating the figure. Decode-only: **22.83 → 22.05 GB/s** (−3.4%). The ~52 is
   itself corroborating data — the uniform null predicts ~117 for this prompt;
   the access-pattern law (0.5-crossing at 57.5 tok) predicts ~52.
2. **Denominator (NOT fixed; structural):** the registered ceiling instrument
   (`report_offload_environment`, a mean over 20 copies) read pinned
   **13.9–14.8 GB/s** while pageable sat at **18.5–18.6** — pinned *below*
   pageable, which is physically backwards. Reproducible across 5 settled
   probes (spread 0.09 on pageable), i.e. **not contention**: the settle+max-of-N
   mitigation was built on a contention hypothesis and the data refuted it. A
   standalone direct probe on the same host: pinned best-of-10 **26.0 GB/s**
   (64 MB) / 24.3 (256 MB) with huge variance, pageable steady. Something
   structural in this containerized host (plausibly NUMA placement of pinned
   allocations against ~123 GB of already-pinned model homes) makes the pinned
   path bimodal and any mean-based probe unusable.

With the registered instrument, fraction = 22.05/14.78 ≈ 1.49 > 1 → the
plausibility gate (added mid-arc, commit `807793c`) correctly refuses to
adjudicate. **No substitute denominator is used** — swapping in a post-hoc
instrument whose verdict is already known from the data would be instrument
selection, not measurement.

**Exploratory, plainly labelled:** against every candidate true ceiling (#22's
22.21, standalone 24.3/26.0), the decode-only fraction lands **0.85–0.99** —
above the 0.70 bar in all cases, pointing toward "transfers already
near-efficient; the coalescer would not pay". Suggestive; not a registered
result; the fresh registration below exists to test it properly.

## Disposition

- R4/R5 close **unadjudicated** here. Follow-up: `PREREG-routed-residual-2.md`
  (fresh registration: best-of-N direct ceiling from the start, decode-only from
  the start, bare-metal host class stated) — not an amendment, because the
  authors have seen these arms and amendment-under-knowledge is post-hoc.
- The ceiling-instrument anomaly is written up separately (finding: mean-based
  pinned probes are unusable on this host class; pageable>pinned is the tell).
- Evidence: `~/e4b-evidence/routed-residual/` (run 1 receipt + log),
  `~/e4b-evidence/r4-rerun/` (run 2 rescued log). OTS on commit of this file.
