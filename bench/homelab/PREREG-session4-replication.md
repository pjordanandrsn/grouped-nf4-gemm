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

**The crossing (per box, the headline under test):** K\*(box) = min K with
ours(panel median) > llama(best config, that box). [S3-BIND-2] — the
prediction is conditional on session-3's E1 outcome and is pre-committed
per branch, to be bound to ONE branch before stamping:
- E1 lands ≤ 25 tok/s (threads don't help) → predict **K\* = 16 on B1, B2
  and B3**.
- E1 ∈ (25, 33] → predict **K\* = 16 or 32 on B1/B2**; B3 graded at its own
  llama-best (16 metal cores): K\* ∈ {8, 16} plausible if B3's llama-best
  is lower than DO's.
- E1 ∈ (33, 45] → predict **K\* ∈ {32, 64} on B1/B2**; the K=16 crossing
  claim is then restated as thread-capped-host-specific, BEFORE
  publication.
- E1 > 45 → the fat-host crossing claim is withdrawn as stated and the
  result is re-framed on the host-CPU axis (weak-host regime + energy),
  plainly.

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

## Protocol

Per box: probes (pinned H2D, HBM D2D, RAPL/powercap inventory — B3 metal
may expose CPU energy; recorded either way) → reducer (hot sets + capture
from the stamped stream; shas cited) → panel (8 prompts × 64 replays,
graphs, per-K) → eager anchor K=16 → soak-lite (256 replays, K=16, halves
within 3%) → ours ppl chunked-512 on the ~1.8k matched text → llama:
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

## Bind log (to complete before stamp)

- [S3-BIND-1] llama best `-t` on 24 vCPU: ____ (from E1)
- [S3-BIND-2] conditional-crossing branch selected: ____ (from E1)
- Session-3 grades folded into RESULTS-pipelined-ladder.md: commit ____
