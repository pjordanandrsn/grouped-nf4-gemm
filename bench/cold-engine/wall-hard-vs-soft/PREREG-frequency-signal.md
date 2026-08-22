# Preregistration: why does frequency carry no signal on gpt-oss?

Registered before measuring.

## The open question

[`RESULTS-policy-headroom.md`](../RESULTS-policy-headroom.md) measured a
frequency-aware victim rule against LRU on 48 cells:

| model | LFU ÷ LRU | gap closed |
|---|---|---|
| OLMoE | 0.775 | 49% |
| Granite | 0.745 | 49% |
| Qwen1.5-MoE | 0.893 | 30% |
| **gpt-oss-20b** | **0.992** | **2%** |

and states the consequence plainly: the one model where frequency buys nothing
is the one every wall measurement in this program was made on. It does not say
**why**, and that is what decides whether some *other* implementable policy
could work there, or whether gpt-oss's routing is simply not predictable from
what a cache can see at eviction time.

## Hypothesis

**Frequency helps exactly when expert popularity is skewed.** LFU beats LRU by
keeping rows that will be needed again because they are needed *often*; if
routing spreads mass evenly across experts, every resident row has the same
expected future and frequency is noise. Qwen's math prompt already shows the
degenerate case — a period-2 alternation makes frequency uniform, and the other
session recorded LFU losing 2.31× there.

So gpt-oss should route **more uniformly** than OLMoE and Granite.

## Predictions, registered

Measured per (model, prompt) on the same traces, over the routed key stream:

- **CONFIRMED** — a skew statistic (normalised entropy of the expert-visit
  distribution, and/or Gini) orders the four models the same way the measured
  `LFU ÷ LRU` does, with gpt-oss the most uniform. Specifically: Spearman
  |rho| ≥ 0.8 between per-cell skew and per-cell `LFU ÷ LRU` across all 48
  cells.
- **REFUTED** — no such ordering (|rho| < 0.5), or gpt-oss is *not* the most
  uniform. Then popularity skew is not the mechanism, and the reason LFU fails
  on gpt-oss is something else — which is a more interesting result, because it
  would mean the obvious explanation is wrong.
- **PARTIAL** — the ordering holds across models but not within them, or
  0.5 ≤ |rho| < 0.8. Skew is part of the story and not all of it.

## Stated in advance

This is diagnostic, not a proposal. Confirming it would NOT license shipping
LFU; it would say which models it can help and why, and predict the answer for
a model not yet traced. Refuting it is worth more, because every policy idea
in this thread so far has assumed popularity is the exploitable structure.

Skew will be computed from the **routed stream** the cache actually sees
(layer, expert) pairs, not from gate logits, because the cache can only act on
what it observes.
