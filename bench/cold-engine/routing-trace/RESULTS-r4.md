# R4 — REFUTED as stated: recurrence only competes once it stops being short

Receipt: [`r4.json`](r4.json). Harness: [`score_r4.py`](score_r4.py). Trace:
[`olmoe_routing_seq.jsonl`](olmoe_routing_seq.jsonl). No box; replays a
captured trace and runs anywhere.

## As registered

> **R4** — **short-window** recurrence predicts resurrection better than
> long-run expert frequency. **Refuted if global frequency predicts as well
> or better.** (`PREREG-tribrid-stage3.md`)

`reuse_profile.ReuseProfile` computed both predictors from the start so this
could be settled by a trace. The trace now exists; this scores it.

## The result turns on the word "short"

512 autoregressive decode steps of OLMoE-1B-7B, six cache capacities × six
recurrence windows. Counting only cells where either predictor has any signal
(max |ρ| ≥ 0.15):

| recurrence window | frequency wins | recency wins |
|---|---|---|
| **4 ticks** | **5 of 5** | 0 |
| **8 ticks** | **5 of 5** | 0 |
| 16 | 3 | 2 |
| 32 | 3 | 2 |
| 64 | 3 | 2 |
| 128 | 3 | 2 |

**At genuinely short windows, long-run frequency wins at every capacity that
carries signal.** Recurrence only starts winning at 16 ticks and up, and only
at the two smallest capacities. R4 is refuted on its own terms.

## Full grid — recency ρ by window, against frequency

| rows | events | max abs ρ | w=4 | w=8 | w=16 | w=32 | w=64 | w=128 | **frequency** |
|---|---|---|---|---|---|---|---|---|---|
| 128 | 22,198 | 0.848 | 0.771 | 0.811 | 0.825 | 0.834 | 0.847 | 0.848 | **0.822** |
| 192 | 265 | 0.384 | 0.253 | 0.323 | 0.362 | 0.376 | 0.384 | 0.381 | **0.353** |
| 256 | 266 | 0.375 | 0.212 | 0.290 | 0.326 | 0.358 | 0.365 | 0.371 | **0.375** |
| 384 | 552 | 0.476 | 0.232 | 0.320 | 0.375 | 0.420 | 0.439 | 0.466 | **0.476** |
| 512 | 740 | 0.445 | 0.088 | 0.149 | 0.222 | 0.273 | 0.307 | 0.365 | **0.445** |
| 768 | 128 | 0.038 | 0.012 | −0.018 | −0.012 | 0.007 | −0.004 | −0.011 | **−0.038** |

**Recency's ρ rises monotonically with window width at every single
capacity.** It converges on frequency from below. Where it does overtake —
128 and 192 rows — it does so only after the window has grown to a sixth or a
quarter of the entire trace, and by small margins (0.848 vs 0.822 at best).
Where retention matters most (384, 512 rows) frequency wins at every window
including 128.

So the mechanism is consistent across the whole grid: **recurrence predicts
better the more it is allowed to behave like frequency.** That is the
opposite of what R4 asserts.

The one place recency sweeps every window is 768 rows, where the largest
correlation of either predictor is **0.038** — nothing predicting anything,
with one noise value above another. Reported and discounted, not tallied as
support.

## Consequences stated, not acted on

R4 was the argument for making gate 3's loop **recency-driven**. On this trace
it should be frequency-driven — also simpler: a counter per expert, no window,
no deque.

`ReuseProfile.classify` still uses recency and is **deliberately unchanged.**
Gate 3 has not been run, and swapping a policy on one trace would repeat the
mistake this measurement exists to catch. What the data does support is
narrower: if a windowed predictor is kept, the window should be wide (≥64),
which is close to admitting frequency.

**Headroom is limited either way** at the capacities where the cache is worth
running. Frequency's best rank agreement outside the 128-row point is
**ρ = 0.476**. Neither predictor is strong, which is worth knowing before gate
3 is designed around one.

## Ground truth, and a correction to the first pass

A resurrection is a hit on a row that lost capacity ownership but was not yet
overwritten, so it exists only relative to a cache of some size — hence the
capacity sweep.

**The first version of this harness mislabelled it.** It read slot state
*before* `want`, but `want` settles the previous tag *first* and only then
resolves hits, so rows demoted by the previous request were still `RETIRING`
at the check and were never counted. That recorded **0** events at 128 rows
against the tier's own 22,198, and roughly half elsewhere — and it inverted
the verdict at the two smallest capacities. The harness now settles first,
mirroring `want`, and **asserts its per-expert labels equal
`VramSlots.resurrections`**, so the ranking cannot silently be built on a
different event than the one being predicted.

## Limits

- One model, one prompt, 512 decode steps. OLMoE routes top-8 of 64 with high
  churn; a model with sharper locality could favour recurrence, and this does
  not test that.
- The 128-row point dominates the event count (22,198 of 24,149). It is the
  regime where the cache is smallest and thrashes hardest — informative, but
  one point.
- Spearman is rank agreement, not calibration, which is the question a
  promotion policy actually asks.
