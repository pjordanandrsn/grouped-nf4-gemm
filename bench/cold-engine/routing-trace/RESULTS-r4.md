# R4 — REFUTED on a real routing sequence

Receipt: [`r4.json`](r4.json). Harness: [`score_r4.py`](score_r4.py). Trace:
[`olmoe_routing_seq.jsonl`](olmoe_routing_seq.jsonl). No box; this replays a
captured trace and runs anywhere.

## As registered

> **R4** — short-window recurrence predicts resurrection better than long-run
> expert frequency. **Refuted if global frequency predicts as well or
> better.** (`PREREG-tribrid-stage3.md`)

`reuse_profile.ReuseProfile` was built to answer this rather than assume it —
it computes both predictors and `predictor_scores` reports each as a
Spearman rank agreement against the resurrections an expert actually
accumulated. What was missing was ground truth on a real trace. It now exists.

## Result

**Frequency wins.** 512 autoregressive decode steps of OLMoE-1B-7B, swept
across six cache capacities × six recurrence windows:

| rows | resurrections | max abs ρ | recency wins | frequency wins |
|---|---|---|---|---|
| 192 | 134 | 0.283 | 3 | 3 |
| 256 | 148 | 0.335 | 0 | **6** |
| 384 | 284 | 0.374 | 0 | **6** |
| 512 | 417 | 0.359 | 0 | **6** |
| 768 | 110 | **0.036** | 6 | 0 |

Restricted to cells where either predictor has any signal at all
(max |ρ| ≥ 0.15): **frequency 21, recency 3.**

**Recency's only clean sweep is at 768 rows, where the largest correlation of
either predictor is 0.036.** That is not a regime where recency wins; it is a
regime where nothing predicts anything, and one noise value happens to sit
above another. Counting it as support for R4 would be reading the sign of
noise.

In the three capacities carrying the most resurrections — 148, 284 and 417
events — frequency wins **all eighteen** cells.

## The mechanism, which is the part worth keeping

Recency's ρ rises monotonically with the window at every capacity:

| rows | w=4 | w=8 | w=16 | w=32 | w=64 | w=128 | frequency |
|---|---|---|---|---|---|---|---|
| 384 | 0.167 | 0.252 | 0.304 | 0.329 | 0.342 | 0.367 | **0.375** |
| 512 | 0.097 | 0.121 | 0.191 | 0.234 | 0.256 | 0.302 | **0.359** |

**Short-window recurrence improves only as it stops being short**, converging
towards long-run frequency from below and never reaching it — at w=128, a
quarter of the whole trace, it is still behind. The prediction was that a
*locally* hot expert is worth retaining over a uniformly warm one. On this
trace the local signal is strictly the weaker one, and the shorter the
window, the weaker it gets.

## What this costs the design, and what it does not

Gate 3 is the loop *placement → execution → observed reuse → new placement*.
R4 was the argument for making that loop **recency-driven**. It should be
frequency-driven instead, which is both simpler and cheaper: a counter per
expert, no window, no deque.

That is a genuine simplification of `ReuseProfile.classify`, which currently
uses recency. It is **not** changed here — the classifier feeds gate 3, gate 3
has not been run, and swapping a policy on the strength of one trace would
repeat the mistake this measurement exists to catch.

**The headroom is small either way.** Frequency's best rank agreement is
**ρ = 0.374**. Neither predictor is strong, so an adaptive residency policy
driven by either has limited room before it is guessing. That is worth
knowing before gate 3 is designed around it.

## Limits

- **One model, one prompt, 512 decode steps.** OLMoE routes top-8 of 64 with
  high churn; a model with sharper routing locality could plausibly favour
  recency, and this does not test that.
- **Resurrection is capacity-relative.** It only exists against a cache of
  some size, which is why capacity is swept rather than fixed. At 128 rows
  there are **zero** resurrections — with a one-request demotable margin every
  demoted row is reclaimed by the next layer's request before its own expert
  comes round again — so that point is undefined, not a zero.
- **Rank agreement, not accuracy.** Spearman answers "does this predictor
  order experts the way the outcome did", which is the question a promotion
  policy asks. It is not a claim about calibration.
