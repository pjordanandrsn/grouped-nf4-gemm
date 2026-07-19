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

---

# SESSION 3 — robustness + steelman (amendment 2; prior stamp preserved as
# `.pre-session3.ots`; grades PREREG-note-session3.md `7ad357e`, pre-data)

Pod: DO H200 585846927, destroyed 404-verified. Receipts verbatim:
`receipts-pipelined-s3-585846927.txt`.

## E-grades (greens and reds)

| E | prediction | outcome | grade |
|---|---|---|---|
| E1 | llama ncmoe32 `-t24` ∈ [28, 55]; K\* moves right but survives | **45.34 ± 0.18** — in band; lands the pre-committed **">45" branch by 0.34** (boundary case, ruled UP per the branch text) | **GREEN**, branch `>45` bound |
| E2 | panel K=16 median ∈ [22, 27]; worst ≥ 20 | median **28.71** (over-band); worst **23.58** ✓ | worst-clause **GREEN**; median **RED (over-band)** — s2's single prompt sat at the route-diverse end (only p3 colder); explained below |
| E3 | soak halves within 3% | first-half 35.35, second-half **62.99** (−43.9% "drift") | **RED** — not instability: the 512-token greedy continuation degenerates into a repetitive loop, routing collapses, cold→0.04 GiB/tok, speed rises to the overhead floor. b_rel 0.0000 throughout. Protocol fix for S4: **teacher-forced soak** (content-stationary by construction) |
| E4 | K=0 per-replay cold ∈ [1.3, 1.9] GiB/tok | typical prompts 1.59–1.68; panel median ≈ 1.61; degenerate p6 = 0.34 (included, flagged) | **GREEN** |
| E5 | chunked ppl ratio ours/llama ∈ [0.9, 1.3] | ours 32.76 (4 chunks incl. a 289-token tail) vs llama 21.76 ± 2.36 (3 full chunks) ⇒ **1.51** | **RED** — causes named: (a) real NF4 re-quant error vs the native-precision checkpoint (expected direction, size unknown), (b) chunk-accounting mismatch (our short-context tail chunk inflates ours; llama floors to full chunks), (c) text composition. **Resolution = the A4 reference-standard arm in session 4; the fidelity axis is UNRESOLVED until then and every speed claim carries this caveat** |
| E6 | thread ladder monotone; `-t1` < 8 tok/s | 3.98 / 7.65 / 13.93 / 24.25 / 35.36 / 45.34 — monotone, near-linear to 16t | **GREEN** — the crossover curve, measured |
| — | decode-only sync audit | **0 warnings in 5 full-model steps** (session-2's 7 were the reference-path prefill, by design) | clean |

## The one-law panel (why the "discrepancy" isn't one)

Every panel row fits **t ≈ 12.5 ms + cold_bytes / 45.2 GB/s** (p3: 30.3+12.1
= 42.4 ✓; p0: 28.3+12.1 = 40.4 ✓; p6: 3.0+13.2 = 16.2 ✓ — and p6's 61.76
equals the K=128 floor because zero cold traffic makes every K look like
K=128). Prompt-to-prompt cold bytes at K=16 span 10× (0.13–1.28 GiB/tok):
generic/templated continuations route into the expert-frequency head —
exactly what the train-derived hot set covers — while diverse technical
prose routes into the tail. Session 2's single prompt (p0) reproduces to
0.3% and sits near the *worst* case, so the earlier crossing figure was
conservative sampling, not flattery. Locked stat remains
median-of-prompts; worst-prompt reported beside it always.

## RESTATED HEADLINE (executing the pre-committed ">45" branch)

**The blanket "crossing at K\*=16" claim is withdrawn as a general fat-host
statement and restated as a function of host CPU strength** — which the
data now measures rather than assumes. Same box, llama at each host
class's best config, our panel medians (host-CPU-independent by
construction):

| host-CPU class (llama config) | llama tok/s | ours ≥ llama at | resident-expert GB there |
|---|---:|---|---:|
| 1 thread  | 3.98  | K=0 (20.45, **5.1×**) | 0 |
| 2 threads | 7.65  | K=0 (2.7×) | 0 |
| 4 threads | 13.93 | K=0 (1.5×) | 0 |
| 8 threads | 24.25 | **K=16** (28.71) | 7.4 |
| 16 threads | 35.36 | K=64 (45.21) | 29.5 |
| 24 vCPU (box best) | **45.34** | K=64 is a statistical tie (45.21 ± ~1.4); clear at **K=128** (61.69) | 58.9 |

Durable claims, unchanged by the restatement: host-CPU-independence of the
hybrid (its column is flat in threads); the weak-host regime win (1.5–5×
at ≤4 cores with **zero** resident-expert VRAM); placement-invariant
correctness (b_rel 0.0000 at every K, every prompt, and through the
512-replay soak); a sync-free decode loop (0 warnings, decode-only);
the transfer-law fit; energy trending 9→4 J/tok with residency. The
fat-host equal-VRAM comparison now honestly reads: **llama's CPU-offload
is the right tool on many-core hosts; the hybrid is the right tool when
host cores are few, VRAM is the binding constraint, or the host CPU is
needed for other work.** Fidelity caveat: E5 red pending the A4
reference-standard measurement — no speed claim should be quoted without
it.

Session-4 binds from this session: [S3-BIND-1] llama best `-t` = **24**
(box max; B3 probes its own 16 metal cores); [S3-BIND-2] branch **">45"**
— S4's crossing predictions are per-host-class rows, not a single K\*.
Pod spend this session ≈ $4.4; arc total ≈ $15.5.

---

# SESSION 4 — confirmatory replication, B1+B2 graded (amendment 3; prior
# stamp preserved as `.pre-session4.ots`; grades the OTS-stamped
# `PREREG-session4-replication.md`, sha 157ac8eb…)

B1 = DO H200 atl1 585856812 (A4 assignment), B2 = DO H200 nyc2 585856813 —
both destroyed, 404-verified. Receipts verbatim:
`receipts-pipelined-s4-b1-585856812.txt`, `…-b2-585856813.txt`. **B3
(Latitude H100 metal): zero inventory at all nine US sites at fire time
(`in_stock=[]`; the plans API's `available` is deployability, not stock) —
a 30-min stock poller is armed and B3 folds in when metal exists. Same
wave, per-box paired protocol; delay is methodologically neutral.**

## Panel grades (graphs, median-of-8-prompts / worst; H2D 44.3 / 43.6 GB/s)

| K | band | B1 | B2 | grade |
|---:|---|---|---|---|
| 0   | [18.5, 22.5] | 20.45 / 19.52 | 20.46 / 19.53 | **GREEN** |
| 8   | (recorded)   | 25.99 / 21.42 | 26.00 / 21.43 | — |
| 16  | [24, 33]     | 28.70 / 23.57 | 28.72 / 23.57 | **GREEN** |
| 32  | [30, 40]     | 34.88 / 30.42 | 34.92 / 30.44 | **GREEN** |
| 64  | [38, 50]     | 48.26 / 41.83 | 48.37 / 41.91 | **GREEN** |
| 128 | [56, 67]     | 61.74 / 61.61 | 61.85 / 61.70 | **GREEN** |

**Cross-box consistency: max/min ≤ 1.0023 at every K** (gate was ≤ 1.15 —
passed with a factor-65 margin; session-3's numbers reproduce to 0.1%).
Degenerate-loop prompts flagged at K ≤ 32, included in medians, per the
locked metric. b_rel = 0.0000 on every capture point, both boxes.

## Crossing rows (bound branch, graded)

- vs the 8-core-class reference (pinned `-t 8` = 24.25): ours K=16 =
  **1.184×** on both boxes — band [1.05, 1.35] **GREEN**.
- vs box-best llama (`-t 24`, ncmoe32): K=64 ratio **B1 1.095 (RED-over by
  0.025)** / **B2 1.051 (GREEN)** — tie band was [0.93, 1.07]. Context the
  band math missed: llama's own ncmoe32@t24 repeats spread **40.24–46.04
  (±7%) within/across these boxes**, wider than our cross-box spread
  (0.2%); the substantive tie conclusion stands, the formal B1 red is
  reported. K=128 ratio 1.401 / 1.343 — band [1.20, 1.45] **GREEN**.
- **New honest observation (fine ladder):** llama's best offload config on
  these 24-vCPU boxes is **ncmoe28 ≈ 49.8–51.6 tok/s at ~13 GB resident**
  — at mid-range VRAM llama at its best out-paces our K=64 (29.5 GB,
  48.3). Our uncontested rows on fat hosts remain K=128 (61.8 beats their
  entire offload ladder), the flat-in-host-CPU column, and (pending A4)
  whatever fidelity says. The ladder is monotone as predicted.

## Other stamped arms

- **A1 placement invariance: 100.0% token agreement, K=0 vs 16 vs 128, both
  boxes.** The slip's publication clause is satisfied: *the dial changes
  speed, not answers* — now a measured sentence.
- **Teacher-forced soak:** first/second halves 26.17 → 27.15 tok/s on BOTH
  boxes (identical to the hundredth) — drift **−3.6%**, band was ±3% ⇒
  **RED by 0.6 pt**, favorable direction, systematic (clock ramp-up is the
  suspect; content is stationary by construction). The greedy-degeneration
  confound from session 3 is gone.
- **Energy (differential — provisioning cancels):** llama resident **1.64 /
  1.73 J/tok** (validates the method: ~240 tok/s at ~400 W ⇒ ~1.7);
  llama ncmoe32@t24 **3.15 / 2.79 J/tok GPU-side only** — the CPU term is
  unmeasured on these VMs (powercap exposes only `dtpm`), and llama's
  24-thread expert compute is precisely what it omits. Ours: **7.3 / 7.5
  J/tok at K=16 (soak, clean windows)** — inside the [5.5, 9.5] band,
  GREEN — and ours is GPU-dominated *total* (host near-idle during DMA
  streaming). No cross-stack energy verdict is claimed until a box with
  CPU energy visibility reports (B3 metal is that box).
- **Fidelity, exact-chunk-matched:** ours **29.27** vs llama **21.7559**
  (identical on both boxes; deterministic) ⇒ ratio **1.345** — the
  accounting fix narrowed session-3's 1.51 but the red stands. **A4 is the
  decider and is INCOMPLETE this wave:** the reference load OOM'd (the
  ladder process still held module refs + the 61 GB pinned arena; the
  dequant then attempted a monolithic 120 GiB allocation). Root-caused;
  completion design: subprocess-clean A4 on a fresh box, reference logits
  + KL vs the retained `ours_logits.npy` (617 MB, in hand), plus reference
  ppl to anchor both stacks to ground truth.
- **A2 cross-stack agreement: INCOMPLETE** — llama-cli hung 600 s on the
  first prompt under docker (interactive-mode edge); the timeout killed
  the block. Ours-side continuations are saved; completion uses one
  llama-server + 8 API calls instead.

## Completion run (operator GO; DO H200 585867877, destroyed 404-verified;
## receipts `receipts-pipelined-s4-completion-585867877.txt`)

Both incomplete arms finished with their root causes mechanized: ours-pass
and reference-pass ran in SEPARATE python processes (exit = guaranteed GPU
+ pinned-RAM release; no parent residue → the reference's dequant had the
full card), and A2 used one llama-server + 8 API calls (llama-cli's
interactive hang path never engaged).

**A4 — fidelity vs the reference standard (the decider).** Reference =
shipped checkpoint, native precision dequantized via HF
(`Mxfp4Config(dequantize=True)`, device_map auto, 119 s load), same
matched text, exact same 3×512 chunks. Precondition gate passed (identical
tokenization, 1825 tokens). Artifact `ref_logits.npy` sha256[:16]
=a7ca117747f8657f, shape (1536, 201088).

| stack | exact-chunk ppl | vs reference |
|---|---:|---|
| **reference (shipped precision)** | **26.75** | — (ground truth) |
| ours (NF4 re-quant) | 29.27 | KL 0.0657, top-1 **88.15%**, max-rel 0.249 |
| llama (its GGUF repack) | 21.76 | (below reference — see reading) |

**Reading — the E5/ppl red is RESOLVED and the pre-committed expectation
holds:** ours sits **+9.4% ppl above** the reference standard — a real,
modest NF4 re-quantization cost (mean KL 0.066, 88% top-1 to native), the
expected direction and now a *sized* number. **llama's 21.76 is 18.7%
BELOW the reference** — a repack does not add information, so a sub-
reference ppl is a scoring-convention artifact (llama-perplexity's
sliding-window context vs our independent-chunk cross-entropy), not
superior fidelity. So the earlier "ours/llama = 1.345" was never a fidelity
gap at all — it compared two *different distances-from-truth measured
different ways*. Against the common ground truth, ours is +9.4% and the
llama number is not a like-for-like fidelity figure. **The pre-committed
ordering (llama-KL ≤ ours-KL) is not evaluable as stated** — llama's
logits weren't extracted (only its self-scored ppl), so no llama-vs-
reference KL exists; recorded as a partial, and the honest fidelity claim
is now the absolute one: *ours is within KL 0.066 / 88% top-1 / +9.4% ppl
of the shipped-precision model.*

**A2 — cross-stack greedy agreement (temp 0, 8 prompts × 64 tok, common
prefix vs ours):** per-prompt 100/100/42/30/19/16/15/8%, **median 24%**,
against the pre-registered band **[70, 95]%** ⇒ **RED (below)**. Reading,
pre-committed style: the band was simply wrong for greedy decode across
two quantizations — deterministic greedy has NO error tolerance, so the
first near-tie logit flip forks the sequence permanently and prefix-
agreement collapses even between near-identical models. The *content*
tells the real story: every divergence is a legitimate paraphrase at a
genuine branch point (p1 "buried beneath the sand" vs "filled with ancient
coins"; p3 "photon gas. The" vs "photon gas in "), never a degeneration —
and the fully-determined prompts agree 100% (p2 quicksort, p7 French
translation). The KL 0.066 from A4 is the correct fidelity instrument;
A2's prefix-agreement is a decode-divergence-amplification artifact, and
the band is corrected to that understanding (a distributional metric like
A4, not sequence prefix, is what "agreement" should have meant). No
publication claim rests on A2.

## Wave status — B1/B2 COMPLETE, B3 stock-gated

Stamped bands: **all six panel rows GREEN, both boxes; replication 0.2%;
A1 placement-invariance 100%; the fidelity red RESOLVED** (ours within
KL 0.066 / +9.4% ppl of ground truth; the 1.345 "gap" was a cross-scoring
artifact). Remaining reds are narrow and understood: soak −3.6% vs ±3%
(favorable, clock-ramp); B1 K=64-vs-box-best 1.095 vs 1.07 (llama's own
±7% t24 spread swamps it); A2 prefix-agreement below a band that was
mis-specified for greedy decode (corrected; A4 KL is the right instrument).
Honest headline caveat retained: llama's true best on 24-vCPU boxes
(ncmoe28 ≈ 50 tok/s @ 13 GB) beats our mid-VRAM points; our uncontested
domains are weak/mid-host CPU, VRAM-at-parity, K=128, and fidelity-to-
native at a stated 9.4% cost. B3 (Latitude H100 metal) still
`in_stock=[]`; 30-min poller armed, folds in on restock. Wave spend ≈
$11.7; arc total ≈ $27.
