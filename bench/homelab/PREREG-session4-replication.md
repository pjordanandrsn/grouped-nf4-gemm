# PREREG — Session 4: confirmatory cross-provider replication (+ Latitude arm)

**Tier: CONFIRMATORY (pre-registered replication of the exploratory crossing
result). Status: DRAFT → bind the two [S3-BIND] slots from session-3 grades,
then STAMP-READY. NO box fires before the stamp.** Code under test: e4b
`7864955..be5fe53` (pushed; unchanged). Committed reducer + stamped stream
unchanged (`c4d6e2a`, sha 45c071e689b9f173).

## Design

Three boxes in parallel, one paired protocol each — every comparative
number is same-box, same-session:

| box | provider | silicon | role |
|---|---|---|---|
| B1 | DO atl1 | H200 141 GB (VM, 24 vCPU) | replication 1 |
| B2 | DO nyc2 | H200 141 GB (VM, 24 vCPU) | replication 2 (different host pool) |
| B3 | **Latitude** DAL/CHI | **H100 80 GB bare metal**, EPYC 9124 16c, 192 GB, `ubuntu24_ml_in_a_box` | cross-provider / cross-virtualization / third host-CPU point |

B3's VRAM caps its ladder at K ∈ {0, 8, 16} (bare 64.1 + hot 7.4 ≈ 75 GB;
K=32 does not fit) — the crossing point K\*=16 is tested exactly where it
lives. B1/B2 run K ∈ {0, 8, 16, 32, 64, 128}.

## Locked metric definitions

- Per-box per-K statistic: **median over 8 fixed prompts of
  median-over-64-replays** (graphs arm; one eager anchor at K=16 for
  continuity). Degenerate greedy loops are INCLUDED in the median and
  flagged (best/median > 1.5), excluded from nothing.
- **Provisioning-exclusion accounting (both stacks, symmetric):** per-token
  bytes and joules count steady-state decode windows only. Ours: counters +
  nvml snapshotted after capture, before replays (model load, quantize,
  arena pinning/fill, hot-stack placement, prefill, warm all outside).
  llama: **differential energy** — each config run at `-n 24` and `-n 512`,
  J/token = (E₅₁₂ − E₂₄)/(2×488); container start, gguf load, and pp cancel
  identically. Bytes columns are ours-only (engine-internal
  instrumentation); cross-stack comparison happens on tok/s and
  differential J/token.
- llama configs: resident anchor + **fine ncmoe ladder {28, 30, 32, 34,
  36} at its best thread count** ([S3-BIND-1]: `-t` = argmax of the
  session-3 thread ladder on 24 vCPU for B1/B2; B3 re-runs a mini
  thread probe {8, 16} on its 16 metal cores and uses its own argmax —
  each box gets llama at its own best). Any further llama tuning found
  mid-run is a documented steelman, never silent.
- Correctness voids, never loses: b_rel(state-matched) < 3e-2 per capture
  point; a capture failure demotes that point to eager and is reported.

## Predictions (bands from sessions 1–3 distributions; two-sided)

**Ours (DO H200 boxes B1/B2, graphs, panel median):**

| K | band tok/s | basis |
|---:|---|---|
| 0   | [18.5, 22.5] | s2 20.76 single; s3 panel 20.45 (worst 19.52) |
| 16  | [24, 33]     | s2 24.84 single; s3 panel 28.71 (worst 23.58) |
| 32  | [30, 40]     | s3 panel 34.87 (worst 30.42) |
| 64  | [38, 50]     | s2 40.44; s3 cont 45.21 |
| 128 | [56, 67]     | s2 62.06; s3 cont 61.69 |

**B3 (Latitude H100 metal) — ratio prediction, not absolute** (no prior
H100 observation; basis = H100/H200 spec ratios, HBM 3.35/4.8 TB/s, fewer
SMs, similar gen5 pipe): per shared K, **B3/median(B1,B2) ∈ [0.70, 1.05]**.
*Falsify below:* metal/H100 penalizes the engine beyond bandwidth ratios —
named investigation. *Falsify above:* VM overhead on DO was material —
worth its own note.

**llama (paired, per box):** resident B1/B2 ∈ [225, 255]; resident B3 ∈
[150, 205] (bandwidth-ratio basis). ncmoe fine ladder monotone in
GPU-resident layers. Best-config decode J/token ∈ [8, 30] (differential;
basis: ~40 ms/tok at 300–500 W class). Ours J/token at K=16 ∈ [5.5, 9.5]
(s3 clean windows: K=0 ≈ 8.8–9.2, K=64 5.0, K=128 3.9).

**Traffic (ours, B1/B2):** K=0 panel cold ∈ [1.45, 1.75] GiB/tok (s3:
1.59–1.63; slot-cache variation across prompts); K=128 cold = 0.

**Crossing predictions — BOUND (session-3 E1 = 45.34 ⇒ the ">45" branch
executed; the blanket fat-host K\*=16 claim is already withdrawn/restated
in RESULTS amendment f4fec4f). S4 predicts per-host-class rows, not one
K\*:**
- **B1/B2 vs box-best llama (-t 24):** K=64 within ±7% of llama-best
  (statistical tie band, [0.93, 1.07]×); K=128 ∈ [1.20, 1.45]× llama-best.
  *Falsify either side.*
- **B1/B2 vs the 8-core-class row (-t 8, retained as the thread-capped
  host reference):** ours K=16 panel median > llama(-t8) by [1.05, 1.35]×.
- **B3 (16 dedicated metal cores):** llama-best-B3 ∈ [30, 45] (basis: DO's
  shared-vCPU t16 = 35.4; dedicated Zen4 cores faster per-core). Ours-B3
  K=16 ∈ [0.70, 1.05]× of the DO K=16 median. B3's matched-VRAM verdict is
  a genuine open measurement: both readings pre-committed — ours(K16) >
  llama-best-B3 ⇒ the crossing holds on mid-size metal hosts at matched
  VRAM; ours(K16) ≤ llama-best-B3 ⇒ B3 joins the fat-host rows and the
  hybrid's case there rests on VRAM-at-parity + host-CPU-freedom + energy,
  stated plainly.

**Cross-box consistency:** for each shared K, max/min across B1,B2 ≤ 1.15
(same product, different hosts). *Falsify:* pod-to-pod variance larger
than reported — widen all published error bars accordingly.

**Agreement checks (fidelity, sharper than ppl — strict cross-stack
bit-exactness is a category error and is not claimed anywhere):**
- **A1 — placement invariance (ours, the design property):** greedy 64-token
  sequences at K ∈ {0, 16, 128} from the same prompt/state predicted
  **token-identical across K** — same NF4 bytes through the same kernel at
  every placement. Band: agreement ∈ [98, 100]%; the sub-100 allowance is
  named (fp32 `index_add_` atomics can flip a near-tie argmax) — any
  mismatch is attributed there or investigated, and the "dial changes
  speed, not answers" sentence is only published if this measures 100%.
- **A2 — cross-stack greedy agreement:** 8 prompts × 64 tokens, temp-0,
  both stacks; report % token agreement + first-divergence positions.
  Band: [70, 95]% (different quantization formats and engines; greedy
  amplifies epsilon at branch points — cross-format caveat applies to the
  sentence that reports this). Contextualizes the ppl rows.
- A3 (OPTIONAL, operator call, tone-gated): llama resident-vs-ncmoe token
  agreement — compute migration between their CPU and GPU kernels makes
  small divergence *expected and benign*; measured only to frame A1's
  symmetry, reported only with that exact framing, or dropped.

**A4 — fidelity vs the REFERENCE STANDARD (both stacks vs ground truth;
runs on B1):** the reference is the shipped checkpoint in its shipped
precision through the HF native path (~63 GB, fits B1; bf16-dequantized
disk-offload fallback — same represented values). One forward over the
matched text → a sha-recorded reference-logits artifact. Both stacks then
score against it at identical positions: ours = the bare NF4 forward
(engine-family representative; engine-vs-forward exactness is separately
proven), llama = its own saved logits on the same file. Metrics: mean KL
of next-token distributions, top-1 agreement, max relative logit error.
**Precondition gate:** tokenizer identity on the file (token counts must
match exactly; both stacks counted the prior matched file at 655).
**Pre-committed expectation, stated before data: llama-KL ≤ ours-KL** —
theirs is a near-lossless repack of the native values; ours is a
re-quantization with real error by construction. Rows recorded-not-scored;
the measured quantity of interest is the SIZE of our re-quant gap against
the standard. Strict bit-exactness is claimed by neither stack against the
reference (kernel accumulation orders differ); exactness claims remain
ours-internal (engine vs our own NF4 forward).

## Protocol

Per box: probes (pinned H2D, HBM D2D, RAPL/powercap inventory — B3 metal
may expose CPU energy; recorded either way) → reducer (hot sets + capture
from the stamped stream; shas cited) → panel (8 prompts × 64 replays,
graphs, per-K) → eager anchor K=16 → teacher-forced soak (256 forced-token steps from the matched text —
content-stationary by construction, per session-3's E3 lesson; halves
within 3%) → ours ppl chunked-512 mirroring llama's accounting EXACTLY (floor(N/512)
full chunks, tail dropped, per session-3's E5 lesson) → llama:
resident + fine ncmoe ladder at box-best `-t`, differential-energy pairs
(n=24/n=512) for resident and matched-VRAM point, ppl. Watchers armed;
independent hardcaps (DO: launchd DELETE; **Latitude: launchd DELETE
/servers/{id}** — new launcher, same lock/one-shot/destroy+verify laws;
`latitude_watch` HA alerts are the independent safety net). Artifacts land
before destruction; 404/absence-verified teardown both providers.

Budget: 2 × H200 ≈ $8.5 + 1 × H100 metal ≈ $4.5 ⇒ **≈ $13, hardcapped ≤
$20 total.** One parallel wave; re-fires only on operator GO.

## Grading

Committed-reducer style: per-box tables (greens and reds), cross-box
medians ± spread, K\* per box, ratio grades for B3, consistency grade,
energy comparison under the provisioning-exclusion rule. RESULTS +
`.ots`; any red amends the public claims before further publication.

## Bind log — COMPLETE; STAMP-READY

- [S3-BIND-1] llama best `-t` on 24 vCPU: **24** (45.34 ± 0.18); B3 probes
  its own 16 metal cores and uses its argmax.
- [S3-BIND-2] branch **">45"** — per-host-class predictions above replace
  the single-K\* form.
- Session-3 grades folded: RESULTS amendment **f4fec4f** (E5 red makes A4
  the load-bearing fidelity arm of this session).
