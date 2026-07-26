# PREREG — how many cache slots per layer?

**Tier: CONFIRMATORY. Status: STAMPED before the pod was created.**
gnf4 @ `3510248`(e4b), both local, unpushed.

## Why

#34 measured the per-layer expert cache at **8 slots/layer**: hit rate **0.1322**,
step **1.120×**. That is far under the **0.4513** previous-token reuse #32
measured, and there is a specific reason to think the partition is simply too
small: with speculation on, each layer writes **~9 rows per token** (8 predicted
plus ~1 miss) into an 8-slot partition, so it **evicts within the token** — the
previous token's set is already partly gone before the next token asks for it.

Sweeping 8 / 16 / 24 / 32 in one load, so the cost buys a curve rather than a
point.

## Predictions

- **C1a — 16 slots clears the within-token eviction.** Hit rate at 16 ≥ **0.30**,
  more than double the 8-slot 0.1322. *Falsified below 0.20.*
- **C1b — and approaches the reuse ceiling.** Hit rate at 32 ∈ **[0.40, 0.55]**,
  bracketing #32's 0.4513, since 32 slots hold ~3.5 tokens of history and reuse
  beyond one token is small. *Falsified outside [0.30, 0.65].*
- **C1c — time follows, with diminishing returns.** Best step ≥ **1.25×** (vs
  8-slot's 1.120×), and the 24→32 improvement is smaller than 8→16.
  *Falsified if no setting beats 1.15×.*
- **C1d — GATE.** Every setting stays bit-identical: `max|Δlogit| = 0`.

**Reported, not predicted:** peak VRAM per setting. At 32 slots the pool alone is
~31.9 GB on this model, which may make the best hit rate the wrong choice.

## Pre-committed decision

Pick the setting with the best **time**, and if two are within 2%, the one using
less VRAM. If the best is 32, note explicitly that the recommendation costs
~32 GB of pool and is therefore a large-VRAM option, not a default.

## Outcome — hit rate scales exactly as predicted, and buys nothing measurable

Two passes. The first (2 reps) produced a tidy-looking curve; the second (6 reps,
with ranges) shows it was noise.

**Pass 1, 2 reps** — 8/16/24/32 slots gave 1.110× / 1.145× / 1.444× / 1.201×.
Non-monotonic: 24 beat 32 despite a *lower* hit rate. That is not a mechanism.

**Pass 2, 6 reps, link 25.94 GB/s:**

| slots | median | range | spread | hit rate | speedup [range] |
|---|---:|---|---:|---:|---|
| no cache | 0.8308 | 0.8013–0.8640 | 7.5% | — | 1.000× |
| 16 | 0.8776 | 0.7547–0.9199 | 18.8% | **0.3521** | 0.947× [0.903, 1.101] |
| 24 | 0.7314 | 0.5816–0.8368 | 34.9% | **0.4452** | 1.136× [0.993, 1.428] |
| 32 | 0.8187 | 0.7751–0.8539 | 9.6% | **0.5384** | 1.015× [0.973, 1.072] |

| prediction | predicted | measured | verdict |
|---|---|---|---|
| C1a hit@16 ≥ 0.30 | ≥0.30 | **0.3521** | **CONFIRMED** |
| C1b hit@32 ∈ [0.40, 0.55] | — | **0.5384** | **CONFIRMED** |
| C1c best step ≥ 1.25× | ≥1.25 | **1.136×**, range includes 1.0 | **FALSIFIED** |
| C1d bit-identical | max\|Δ\|=0 | all settings | **CONFIRMED** |

**The mechanism is exactly right and the payoff is absent.** Enlarging the
partition removes within-token eviction precisely as predicted — hit rate
0.132 → 0.352 → 0.445 → 0.538, monotonic and clean, because hit rates are counts
and counts are not noisy. **Every timing range overlaps the baseline.** 16 slots
has a median *worse* than no cache at all. Nothing here is rankable.

**Why a 0.54 hit rate buys nothing: speculation already got there.** It moves the
transfer off the critical path during the preceding layers' compute, so the bytes
the cache elides were already overlapped. The two mechanisms target the same term
and speculation reaches it first. #34 flagged this as an inference; it is now
measured.

**Retraction.** #34 reported the cache at **1.120×** on 2 reps. At 6 reps with
ranges that is not separable from noise, and the honest figure is **no
demonstrated gain**. The 235B ladder's end-to-end result is therefore **9.09×
(through speculation)**, not 10.21×.

**The pre-committed decision cannot execute.** "Pick the setting with the best
time" presumes the times are rankable. They are not, so the recommendation is to
**not enable the expert cache** on a configuration that already speculates — it
costs 8–32 GB of VRAM for no measured return.
