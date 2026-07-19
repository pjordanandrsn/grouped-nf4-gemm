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

---

# SESSION 2 — graphs arms landed (amendment; prior stamp preserved as
# `.pre-session2.ots`)

Pod: DO H200 585792200 (atl1), destroyed 404-verified. Raw state verbatim:
`receipts-pipelined-ladder-s2-585792200.txt`. Pipe re-measured 45.2 GB/s.
Same frozen protocol; the capture blocker was reproduced and repaired for
zero pod spend on the A2000 (five rounds): three sites —
`masking_utils.eager_mask`'s per-step `torch.tensor(0.0, device=cuda)`,
the sliding-window cache layer's `torch.tensor([-1], …)` in its roll
branch, and (repro-only) HF's stock MoE fallback, which the e4b engine
replaces on the pod. **Recipe** (harness-level, this file is its
documentation): a contained const-shim serving cached device scalars/small
int-lists, plus **warm past the sliding window before capture** — the
sliding layer's branch is chosen by a *python int*, so capture freezes the
live branch; warming past the window freezes the roll branch, which is the
semantically-correct steady state (and steady-state timing). Validated on
the A2000 tiny-model rig at b_rel 0.00000 before any pod spend.

## Session-2 ladder (graphs r=3; eager r=1 continuity anchors reproduced
## session 1 within noise)

| K | resGB | graphs tok/s | ms/tok (sd) | b_rel (state-matched) | eager anchor |
|---:|---:|---:|---:|---:|---:|
| 0   | 0.0  | 20.76 | 48.2 (1.2) | **0.0000** | 18.93 |
| 8   | 1.8  | 22.90 | 43.7 (2.3) | **0.0000** | 19.73 |
| 16  | 7.4  | **24.84** | 40.3 (2.2) | **0.0000** | 22.51 |
| 32  | 14.7 | **30.21** | 33.1 (2.3) | **0.0000** | 22.91 |
| 64  | 29.5 | **40.44** | 24.7 (1.4) | **0.0000** | 22.69 |
| 128 | 58.9 | **62.06** | 16.1 (0.0) | **0.0000** | 21.97 |

llama frozen arms, same box, same session: resident 239.50 ± 6.88, ncmoe32
**24.43 ± 0.06**. Fidelity, matched 1108-token text (measurement-only, as
stamped; different chunking methodologies and different quantization
formats — GGUF-native shipped quant vs NF4-converted — so no claim is
attached): ours nll 3.0957 / ppl 22.10 single-pass full-context; llama
chunked-512 PPL 19.80 ± 2.68. llama arm GPU energy: 5803 J (resident) /
5107 J (ncmoe32) per whole bench run incl. model load — attribution
approximate, recorded not scored.

**Harness attribution artifact (recorded):** session-2's per-token cold-GiB
and J/tok columns for the graphs arms divide a window that includes the
2×133-step warm prefixes by the 72 replays — they are warm-inclusive and
NOT per-replay traffic. Session-1's eager traffic numbers (e.g. 1.636
GiB/tok at K=0) remain the clean per-token measurements; physics bounds the
K=0 replay traffic at ≤ 48.2 ms × 45.2 GB/s = 2.18 GB. Sync-audit-v2
(no self-syncs): 7 warnings in a window that includes the reference-path
prefill (T>1 falls back by design); the decode step's sync-freedom is
separately proven by the unit gate under `set_sync_debug_mode("error")`.

## Amended grades (P2/P3/P5 now gradeable)

| P | prediction | outcome | grade |
|---|---|---|---|
| P1 | b_rel < 3e-2 every K, both arms | eager 0.0063; graphs **0.0000 at every K** | **GREEN** |
| P2 | graphs K=128 ∈ [6, 36] ms | **16.1 ms** (v0: 61) | **GREEN** |
| P3 | graphs K=0 ∈ [49.7, 79.7] ms (at 45.2 GB/s) | **48.2 ms** — 3% under the floor of the band | **RED (over-band)** — same two named overestimates as P4: slot-cache traffic below naive + launch share below width |
| P5 | graphs crossing K* ≤ 64 | **K\* = 16**: 24.84 > 24.43 (same-session frozen) | **GREEN** |
| P6 | monotone dial | graphs 20.76→22.90→24.84→30.21→40.44→62.06, strict | **GREEN** |

(P4/P7 grades from session 1 stand; P8 now recorded for both stacks.)

## The two closing numbers — FINAL

1. **Achieved fraction of transfer floor at K=0: 90.7%** (graphs; slip
   arithmetic 43.7/48.2). v0: 13.6%. Eager session-1: 81.8%.
2. **The hybrid crosses llama's pinned 24.3 at K\* = 16** (7.4 GB resident
   experts, ~matched to ncmoe32's ~6.6 GB) and pulls away on the dial:
   1.24× at K=32, 1.65× at K=64, 62.1 tok/s at K=128. Honest bound on the
   other end: resident-vs-resident, llama's all-GPU path (239.5) remains
   ~3.9× our all-hot engine (62.1) — the fused-GEMV-per-layer granularity +
   non-expert eager stack is our next ceiling, and it is llama's home turf,
   not the constrained-VRAM regime this instrument is for.

Deviations, documented: arena-allocation cache (`be5fe53`, pre-declared);
the capture recipe above (harness-level, zero engine change); reducer
capture-split methodology note (session 1). Both sessions' pods destroyed,
404-verified; total program pod spend this arc ≈ $11.
