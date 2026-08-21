# Preregistration — separating top-k from k/E

[`RESULTS-third-model.md`](RESULTS-third-model.md) refuted the rule that the
device row cache crosses at `layers × top-k` rows. It crosses there on OLMoE
and Granite (8 traces of 8) and **not** on Qwen1.5-MoE, which needs two to
three rows more.

That document also ruled out its own first explanation. Raw step-to-step
routing overlap separates the winners from the loser 3× — but `k/E`, the
overlap expected from independent uniform routing, differs 3× on its own, and
once divided out Granite (wins, 2.18×) and Qwen (loses, 2.01×) are
indistinguishable. So the surviving candidates are `k` and `k/E`, and **every
model captured so far confounds them**:

| | layers | E | k | `k/E` | per-step rows | wins at one step |
|---|---|---|---|---|---|---|
| OLMoE | 16 | 64 | 8 | 12.5% | 128 | yes |
| Granite | 32 | ≥40 | 8 | 20.0% | 256 | yes |
| Qwen1.5-MoE | 24 | 60 | 4 | 6.7% | 96 | **no** |

Large `k` and large `k/E` always coincide. This registers the predictions for
a model where they do not, **before it is captured**.

## The model

**Mixtral-8x7B-Instruct-v0.1** — expected 32 layers × 8 experts, **top-2**.
It inverts the confound: the **smallest** `k` of any candidate and the
**largest** `k/E`.

| | layers | E | k | `k/E` | arena | per-step rows |
|---|---|---|---|---|---|---|
| **Mixtral (test)** | **32** | **8** | **2** | **25.0%** | **256** | **64** |

Geometry will be read from the router probe at capture and reported. If it
does not match — a different layer count, a different top-k — the predicted
capacities below are recomputed from the *observed* geometry, which is what
the third-model prereg did and what kept its P1 test honest.

Same protocol as the third model: four prompts (prose, code, math, dialogue),
512 decode steps, 64-token prompt, profile on 0–255 and score on 256–511.

**Additionally, and new:** the capture must record the generated token ids per
step. Qwen's math trace turned out to be a period-2 repetition loop that was
only detectable by inspecting lag-overlap after the fact
([`reuse_overlap.py`](reuse_overlap.py)). With tokens recorded a degenerate
loop is visible directly. A trace whose generation is a repetition loop is
reported and **excluded from the primary result**, with the leave-it-in
numbers reported alongside.

## P3 — which quantity governs the one-step crossover

The two hypotheses make **opposite** predictions on this model, which is the
entire reason for capturing it.

> **P3-a (`k/E` governs):** the cache beats the positional baseline at exactly
> `layers × top-k` = **64 rows**, as on OLMoE and Granite.
>
> **P3-b (`k` governs):** it does **not** — at 64 rows the ratio is > 1 and
> the crossover lands strictly above, as on Qwen.

* Swept at `steps_held` ∈ {0.5, 0.75, 0.9, 1.0, 1.25, 1.5} → capacities
  {32, 48, 58, 64, 80, 96}, plus a row-by-row probe from 64 upward to locate
  the crossover exactly.
* Scored against the **real `DevRowCache`**, not the LRU simulation in
  `score_crossover.py`. The two disagree at exactly this boundary on all 12
  traces captured so far, in both directions, and scoring the simulation would
  have recorded the third model's P1 as a pass it did not earn.
* **P3-a confirmed** if all four prompts give ratio < 1 at 64 rows;
  **P3-b confirmed** if all four give > 1.
* **Neither** is confirmed if the prompts split. That outcome is registered in
  advance as a real possibility and will be reported as "both refuted", not
  resolved by majority.

I am not predicting which. Writing down a guess and then reporting it as a
finding is the failure this file exists to prevent; the point is that the two
outcomes are distinguishable and the sweep is fixed beforehand.

## P4 — the two rules that survived, retested

Straight replications on a fourth model, same form as before.

> **P4a** — below one step, LRU takes **exactly zero** hits. Refuted by any
> nonzero hit count below 64 rows.
>
> **P4b** — `headroom = scored working set ÷ capacity ≤ 1` predicts
> demand-paging beats static placement with **no false positives**. Refuted by
> a single false positive.

* P4b capacities: fractions {0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
  0.8, 0.9, 1.0} of the 256-row arena.
* Note the arena here is **256 rows**, an order of magnitude smaller than
  Qwen's 1440. At the low fractions several capacities collapse to very few
  rows (0.05 → 13); cells whose capacity is below `top-k` = 2 would be
  degenerate, and none are, but the small-arena regime is itself new and any
  cell where static and demand **tie** is counted as *not* a demand win, which
  is the conservative direction for a no-false-positive claim.

## What would count as a miss

* P3 fails to resolve if the four prompts split; that is reported as such.
* P4a fails on any hit below threshold.
* P4b fails on any false positive.
* A model that cannot be captured, or whose router the probe cannot read, is
  **not** a result either way and will be reported as a failed capture.
* **Quantization caveat, registered now:** if the model has to be loaded at
  4-bit to fit the box, that is stated in the results, because quantization
  can flip borderline top-k routing decisions. A trace captured at reduced
  precision is reported as such and is not silently compared against the
  bf16-captured traces.

## Reporting

Both outcomes get the same treatment, and a refutation gets a banner on the
document whose rule it refutes — as
[`RESULTS-crossover.md`](RESULTS-crossover.md) and
[`RESULTS-concentration.md`](RESULTS-concentration.md) already carry.
