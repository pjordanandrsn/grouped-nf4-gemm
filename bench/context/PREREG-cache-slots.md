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
