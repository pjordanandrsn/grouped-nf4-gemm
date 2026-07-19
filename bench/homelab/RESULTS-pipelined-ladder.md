# RESULTS — pipelined hybrid ladder, graded against the stamped slip

Grades `PREREG-pipelined-hybrid.md` (OTS-stamped, sha256 `5639bf6e…82f3`).
Pod: DO H200 585786245 (atl1), destroyed 404-verified. Raw state verbatim:
`receipts-pipelined-ladder-585786245.txt`. Code under test: e4b `7864955` +
`e7f8535` + `8911e0d` + **`be5fe53`** (the pre-declared post-stamp
arena-allocation cache; content-refresh untouched; A2000 26/26).

**Measured constants (substituted into the slip's band arithmetic per its
substitution rule):** pipe 45.3 GB/s (not 55.2 — bands re-derived at 45.3),
HBM 4.21 TB/s (rw), RAPL: only `dtpm` present (no CPU energy domain —
absence recorded). Hot sets + capture derived on-pod by the committed
reducer (`tools/hot_sets_from_receipts.py`, `c4d6e2a`) from the stamped
stream sha256[:16]=45c071e689b9f173: capture K=8 .235 / 16 .373 / 32 .572 /
64 .799.

## The ladder (eager arm; r=3 medians, 24-token greedy decode)

| K | tok/s | ms/tok | b_rel | cold GiB/tok | hot D2D MiB/tok | J/tok |
|---:|---:|---:|---:|---:|---:|---:|
| 0   | 18.74 | 53.4 | 0.0063 | 1.636 | 0    | 12.4 |
| 8   | 19.56 | 51.1 | 0.0063 | 1.525 | 114  | 12.3 |
| 16  | 22.52 | 44.4 | 0.0063 | 1.197 | 449  | 11.4 |
| 32  | 23.62 | 42.3 | 0.0063 | 0.648 | 1012 | 10.6 |
| 64  | 23.83 | 42.0 | 0.0063 | 0.170 | 1501 | 10.3 |
| 128 | 23.93 | 41.8 | 0.0063 | 0.000 | 1675 | 10.3 |

**GRAPHS arm: demoted to eager at every K** — capture failed
deterministically at one site: `RuntimeError: Cannot copy between CPU and
CUDA tensors during CUDA graph capture unless the CPU tensor is pinned`
(reported per the P1 demotion clause, not hidden; the 140-char truncation
lost the traceback — the follow-up session captures it in full). The
engine's own capture is proven (unit + A2000 gates); the copy is upstream
in the full-model forward.

llama frozen arms, same box, pinned flags: resident **240.75 ± 6.52**,
ncmoe32 **24.25 ± 0.18** — both on their pins (24.34/23.28 prior).

## Grades

| P | prediction (slip) | outcome | grade |
|---|---|---|---|
| P1 | b_rel < 3e-2 every K, both arms | eager 0.0063 at every K; graphs **demoted+reported** | **GREEN** (eager); demotion handled per clause |
| P2 | graphs K=128 ∈ [6,36] ms | arm demoted | **UNGRADED-AS-STATED** (adjacent datum: eager K=128 = 41.8 ms vs v0's 61) |
| P3 | graphs K=0 ∈ [39,70] ms | arm demoted | **UNGRADED-AS-STATED** |
| P4 | eager K=0 ∈ [55.2, 117.5] ms (at 45.3 GB/s) | **53.4 ms** — faster than the band floor | **RED (over-band)** — readings below |
| P5 | graphs crossing K* ≤ 64 (capture ≥ .40 ✓ active) | arm demoted; eager plateaus **23.93 vs pin 24.25** (−1.3%), no crossing | **UNGRADED-AS-STATED**; eager near-miss recorded |
| P6 | tok/s non-decreasing in K | 18.74→19.56→22.52→23.62→23.83→23.93, strictly | **GREEN** |
| P7 | K=0 cold ∈ [95%,100%] of 1.842 GiB | 1.636 GiB = **88.8%** | **RED** (favorable direction, out of band) |
| P8 | fidelity, measurement-only | ours: nll 3.0208 / ppl 20.51 (655 tok, single-pass). llama tool refused (<1024 tok) | recorded; llama side ABSENT (text too short — fixed in follow-up) |

## Pre-committed readings applied

**P4 over-band** (we were *faster* than predicted). Two named overestimates:
1. **Traffic below naive** — measured 1.636 vs 1.842 GiB/tok: the slot
   cache re-hits ~11% of fetches. The "near-uniform routing" basis
   underestimated short-range route repetition. (This is also exactly P7's
   red — one cause, two bands.)
2. **OH_e/U2_e widths high** — overhead+non-expert at K=0 is 9.7 ms by slip
   arithmetic (53.4 − 43.7), below the predicted 11.5 ms minimum: the
   launch-share multiplier ran below its [0.5, 2.0] width on this silicon
   (H200 launches are cheaper than consumer-Ada-derived bands allowed).

**Demotion cascade** (P2/P3/P5): a single deterministic capture-blocking
CPU→CUDA copy upstream of the engine. Not one of the three under-band
suspects — its own reported event with its own repair path.

**Arm methodology lessons (recorded, not excused):**
- The **decomposition arm's sign came out negative** (isolated layer-step
  1501 μs × 36 = 54.0 ms > the 44.4 ms whole token at K=16): the isolated
  probe's per-call synchronize makes it an UPPER bound, not an additive
  term — in-model layers pipeline their enqueues. U2 was NOT pinned by this
  arm as designed.
- The **sync-audit counted its own instrument**: 9 warnings in 5 steps ≈ the
  harness's own 2-per-step timing `synchronize()`. Redone without timing
  syncs in the follow-up.

## The two closing numbers (directive)

1. **Achieved fraction of transfer floor at K=0: 81.8%** by the slip's
   arithmetic (43.7 ms naive-bytes floor / 53.4 measured); 72.6% against
   measured bytes. **v0 was 13.6%** — the 215 ms bounty is collected:
   250 → 53.4 ms/token at K=0 (4.7×), 61 → 41.8 at K=128.
2. **Crossing of llama's pinned 24.3: not achieved.** The eager arm
   plateaus at 23.93 (−1.3%) from K=32 up — the binding term is no longer
   transfer (4 ms at K=64) but the eager launch bundle + non-expert share
   (~42 ms floor), which is precisely what the demoted graphs arm exists to
   cut. The capture repair is the identified, single lever.

## Observations for the record

- **b_rel identical (0.0063) at every K** — placement-invariant math:
  hot/cold split changes where bytes compute, never what they compute. The
  "K is config, not code path" design claim, observed.
- Energy: 12.4 → 10.3 J/tok, monotone down with residency (PCIe activity
  falls). llama arms were not energy-instrumented this session (recorded;
  follow-up wraps them).
- Fast-shelf watch-item quantified at scale: hot D2D reaches 1.675
  GiB/token at K=128 — trivial at 4.21 TB/s HBM (~0.8 ms) but ~12 ms/token
  on an A2000-class 288 GB/s card: the in-place hot GEMM path is warranted
  before any small-card serving claim.
- Reducer capture at K=16 (.373) vs the reanalysis's ≈.30: split
  methodology (8/12-prompt train split here); grading used reducer values
  per U3, as stamped.

## Follow-up session (operator-authorized "follow up work on DO")

One session: (1) reproduce the capture error with full traceback, fix the
unpinned copy, re-run the **graphs arms** across the ladder under the same
frozen protocol; (2) matched text extended ≥1200 tokens so llama's ppl tool
runs — both stacks scored; (3) sync-audit without self-syncs; (4) pynvml
wrap around the llama arms. Everything else stays frozen.
