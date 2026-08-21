# The concentration story was wrong; here is the rule that actually separates

Receipt: [`concentration.json`](concentration.json). Harness:
[`score_concentration.py`](score_concentration.py). Eight traces, two models,
24 configurations. No box.

[`RESULTS-generalization.md`](RESULTS-generalization.md) explained every
broken conclusion by **concentration** — how much of the arena a generation
touches. That explanation was offered *after* seeing the results, so it was
scored rather than believed. **It does not hold.**

Three measures of a generation, defined without reference to any outcome, and
the sign each predicts stated before the numbers:

| measure | outcome | predicted | ρ | verdict |
|---|---|---|---|---|
| **steps_held** | cache/positional | −1 | **−0.895** | **SUPPORTED** |
| headroom | cache/positional | +1 | +0.927 | supported, but see below |
| coverage | cache/positional | +1 | +0.276 | **not supported** |
| entropy | cache/positional | +1 | +0.363 | **not supported** |
| headroom | demand vs static | +1 | +0.055 | **not supported** |
| coverage | gate-3 gain | +1 | +0.468 | **not supported** (right sign, below threshold) |

**Coverage and entropy — the two measures that are actually about the
generation — predict nothing.** Concentration was a story fitted to two
salient cells (OLMoE mathematics, Granite code), not a rule.

The gate-3 row is the closest concentration comes to working: ρ = +0.468 is
the *right* sign (`gate3_gain` is negative when adaptive wins, so less
coverage predicting more gain is a positive correlation) and still short of
the 0.5 the other hypotheses were held to. Weak support is not support, but
it is not the refutation the first version of this document called it — that
version had the predicted sign inverted and reported a right-signed result as
wrong-signed.

`headroom` (working set ÷ capacity) correlates strongly, but within any one
trace it varies *only* with capacity, so it is largely restating "a bigger
cache helps more". It is reported for completeness, not as an explanation.

## The rule that does separate: one decode step

`steps_held` = capacity ÷ **rows one decode step asks for** (layers × top-k).

| | steps_held | cache vs positional |
|---|---|---|
| granite @ 12.5% | **0.62** | **1.23–1.30 — LOSES** |
| olmoe @ 12.5% | 1.00 | 0.61–0.77 |
| granite @ 37.5% | 1.88 | 0.48–0.54 |
| olmoe @ 37.5% | 3.00 | 0.09–0.47 |
| granite @ 50% | 2.50 | 0.28–0.36 |
| olmoe @ 50% | 4.00 | 0.06–0.33 |

**Every configuration where the cache loses has `steps_held < 1`; every one
where it wins has `steps_held ≥ 1`. 24 of 24.**

The mechanism is arithmetic, not statistics. A cache smaller than one step's
routed set is fully evicted before its own next request, so it retains
nothing across steps — and then the extra host→cache write it pays on every
miss (documented when it shipped) is pure loss against a positional cache
that at least skips same-position hits for free.

This is **structural**, not a property of the workload. Granite routes
32 × 8 = **256 rows per step** from an arena of 1280, so 12.5% of that arena
is 160 rows — less than one step. OLMoE routes 16 × 8 = 128 from 1024, so the
same 12.5% is exactly one step. The two models were never being compared at
the same pressure; *fraction of the arena* was the wrong normaliser, and
`layers × top-k` is the right one.

## Now instrumented rather than remembered

`DevRowCache.stats()` reports `per_step_rows`, `per_step_rows_max`,
`steps_held` and `too_small_to_retain`, learned from the requests that arrive
— the cache cannot be told how many layers share it, but it can count them. A
deployment below 1.0 is reporting that it cannot retain, instead of quietly
costing 1.2–1.3× the cache it replaced.

The count is per **step**, not per layer-ever-seen. The boundary is a layer
index that does not increase, because the engines walk layers in ascending
order once per step. That is load-bearing in two directions, because the
residency engine **skips `want()` for a layer with no cold experts** and which
layers appear therefore varies step to step: summing each layer's last-seen
count keeps a *silent* layer in the total forever, while watching only for
repeats folds a *newly active* lower-indexed layer into the previous step.
Both report demand spanning two steps as one.
`too_small_to_retain` is judged against the **worst** step seen rather than
the last, because capacity that cannot hold the heaviest step retains nothing
across it whatever the average does.

## What is still unexplained

**When demand-paging beats placement.** It wins in 3 of 24 cells (OLMoE math
at 37.5% and 50%, Granite code at 50%) and no measure here separates them —
`headroom` gives ρ = 0.055, and the one Granite cell has headroom 1.93 while
the two OLMoE cells have 0.98 and 0.74. Three positives is too few to fit a
rule to, and fitting one would repeat the mistake this document exists to
correct.

## Limits

- 24 configurations, two models, four prompts, 512 steps.
- Spearman on 24 points; ρ = −0.895 is strong but the perfect separation at
  `steps_held = 1` rests on **four** losing cells, all Granite at 12.5%.
  The threshold is where the mechanism says it should be, which is why it is
  stated as a rule rather than a fitted boundary — but a third model would
  test it properly.
- Counted, not timed.
