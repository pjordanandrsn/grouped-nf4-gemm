# Preregistration — is top-k what decides whether frequency beats recency?

Registered before any capture. No data exists for the manipulation below.

## Where this comes from

[`RESULTS-policy-headroom.md`](RESULTS-policy-headroom.md) measured a
frequency-aware victim rule against LRU on four models. The per-model means
split cleanly, and they split on **top-k**:

| model | LFU ÷ LRU | top-k | E | k/E |
|---|---|---|---|---|
| OLMoE | 0.761 | **8** | 64 | 0.125 |
| Granite | 0.746 | **8** | 40 | 0.200 |
| Qwen1.5-MoE | 0.953 | **4** | 60 | 0.067 |
| gpt-oss-20b | 0.965 | **4** | 32 | 0.125 |

[`RESULTS-frequency-signal.md`](RESULTS-frequency-signal.md) tested popularity
skew, working-set pressure and reuse distance against this split and refuted
all three. **It did not consider top-k** — the string does not appear in its
prereg, its harness, or its results.

`k/E` is separately ruled out by the table above: OLMoE and gpt-oss share
`k/E` = 0.125 and sit at opposite ends (0.761 vs 0.965).

## Why this is a hypothesis and not a finding

**A binary variable splits four points 2–2 by chance about a third of the
time.** With n = 4 models this observation is worth exactly one experiment and
no assertions. It is registered here so the experiment cannot be reported as
having confirmed something that was fitted to it afterwards.

The plausible mechanism, stated so it can be wrong: each decode step samples
`k` experts per layer, so `k` is the sample size from which a frequency
estimate is built. Small `k` makes per-step frequency noisy relative to
recency; large `k` makes it converge. If that is the mechanism, the effect
should be **monotone in k** and should appear **within a single model**.

## The manipulation, which is the point

Cross-model comparison cannot separate `k` from architecture, training data,
expert count or router temperature. **Within one model it can**, because
`capture_routing.py` derives routing as `topk(linear(h, W_router), k)` from the
router's own weights — so `k` is a free parameter over a fixed model and a
fixed logit distribution.

Capture **OLMoE** (native k=8) and **gpt-oss-20b** (native k=4) at
**k ∈ {2, 4, 8, 16}**, four prompts each, 512 steps, everything else identical.
Score `LFU ÷ LRU` with the existing `policy_headroom.py` at `steps_held`
∈ {1.0, 1.5, 2.0}, capacity recomputed from each capture's own
`layers × k`.

**Disclosed:** at non-native `k` the model computes a different MLP output, so
the generated token stream diverges from native decoding. The routing is still
the model's own router logits — only how many are taken changes. This is a
study of how a cache responds to a routing pattern, not of the model's quality
at that `k`, and the results will say so. Native-k runs (OLMoE at 8, gpt-oss at
4) must reproduce the published per-model ratios; if they do not, the capture
is wrong and nothing else is scored.

## K1 — the ratio is monotone in k

> `LFU ÷ LRU` decreases monotonically as `k` rises, **within each model**.

* **Confirmed** if the four `k` values are strictly ordered
  `ratio(2) > ratio(4) > ratio(8) > ratio(16)` in both models, on the median
  over prompts and capacities.
* **Refuted** by any inversion in either model.
* A model where the ratio is flat across all four `k` (range < 0.02) refutes
  it as decisively as an inversion, and is reported as "k does not matter
  here" rather than as noise.

## K2 — k accounts for the cross-model split

> At **matched k**, the two models' `LFU ÷ LRU` differ by less than the
> published cross-model gap of **0.20** (0.761 vs 0.965).

* **Confirmed** if `|ratio_olmoe(k) − ratio_gptoss(k)| < 0.20` at every k.
* **Refuted** if the gap persists at matched k — which would mean top-k is not
  the variable, and the split at n=4 was the coincidence its prior probability
  says it usually is.

K1 and K2 are independent: k could drive the within-model trend without
explaining the between-model difference, and that outcome is a real result,
not a partial confirmation. It will be reported as "K1 confirmed, K2 refuted".

## What would count as a miss

* Native-k runs not reproducing the published ratios ⇒ capture is wrong,
  nothing is scored, reported as a failed capture.
* K1 refuted ⇒ the top-k story is dead and the split at n=4 was chance.
* K2 refuted with K1 confirmed ⇒ k matters within a model and something else
  separates models; the remaining candidates are then E, L, and router
  temperature, none of which this tests.
* The box is destroyed when the run ends and receipts are committed before it.
