# Preregistration — separating top-k from k/E

> # ⛔ WITHDRAWN, before any box was rented
>
> Every question this file was written to answer turned out to be void or
> undecidable. Kept, unedited below the line, because a preregistration that
> quietly disappears when it stops being convenient is worth nothing.
>
> **1. Its premise was an artifact.** This file exists because Qwen1.5-MoE
> (top-4) appeared to refute the one-step crossover while the two top-8
> models did not, which made `k` and `k/E` the two candidate explanations.
> Qwen did not refute it. The replay harness left `DevRowCache(routed=...)`
> at its default of 8 while the model routes 4, sizing the demotion budget
> for a top-8 engine; corrected, the threshold holds on all three models.
> There is no anomaly left to explain, so there is nothing for P3 to
> discriminate between. See
> [`RESULTS-third-model.md`](RESULTS-third-model.md).
>
> **2. P3 could not have failed anyway.**
> [`structural_check.py`](structural_check.py) drives the same cache with
> synthetic routing from independent-uniform to 95%-sticky and flat to
> heavily skewed popularity, at all four geometries. The verdict moves in
> **0 of 24** conditions. The threshold is arithmetic on
> `protected = rows - k`, not a property of routing, so no captured model
> can test it — Mixtral included.
>
> **3. P4b is not testable at this geometry.** Mixtral's arena is
> `32 x 8 = 256` rows, which is also its entire key space, and 256 warm steps
> route essentially all of it. So `headroom = working set / capacity <= 1`
> only at `capacity = 256`, where static placement already holds every key,
> makes zero transfers, and demand-paging cannot beat it. The one cell where
> the rule makes a positive prediction is the one cell where the comparison
> is vacuous. Scored as registered, P2 would have "refuted" on four
> false positives that are all arithmetic. For the record the same check
> finds **0 vacuous cells in all 144 captured cells** across the three real
> models, so this is specific to a small-E geometry and does not touch any
> published result.
>
> **What would be worth capturing instead** — noted so the next
> preregistration starts from something real:
>
> * **P1b is not forced.** That LRU and FIFO take zero hits below one step
>   while random does not is a policy fact, not arithmetic, and it is the
>   part of the crossover story that survives. A geometry where the routed
>   set is small relative to the arena would stress it hardest.
> * **P2 is not forced either**, and it is the only rule here whose
>   confirmation carried real information. It needs a model whose eval
>   working set is meaningfully *smaller* than its arena — that is, large E —
>   which is the opposite of what this file selected for.
>
> Nothing below this line has been edited.

---


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

**Mixtral-8x7B-Instruct-v0.1** — 32 layers × 8 experts, **top-2**, read from
the published `config.json` (`num_hidden_layers` 32, `num_local_experts` 8,
`num_experts_per_tok` 2, `torch_dtype` bfloat16). It inverts the confound: the
**smallest** `k` of any candidate and the **largest** `k/E`.

| | layers | E | k | `k/E` | arena | per-step rows |
|---|---|---|---|---|---|---|
| **Mixtral (test)** | **32** | **8** | **2** | **25.0%** | **256** | **64** |

Geometry is still read from the router probe at capture and reported, and if
it does not match the config — a different layer count, a different top-k, a
router the probe reads differently — the capacities below are recomputed from
the *observed* geometry. That clause is what kept the third model's P1 test
honest and it stays, config or no config.

**Size, and the precision that follows from it.** Mixtral is 46.7B parameters:
~93 GB in bf16, which does not fit one 80 GB card. So the capture is either
bf16 across two GPUs or reduced precision on one. Registered in advance
because the choice is not free — quantization perturbs router logits and can
flip a borderline top-2 decision, and top-2 has the least margin of any model
captured. Preference order, and the results will say which was used:
**bf16 on two cards > 8-bit on one > 4-bit on one.**

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
  the crossover exactly. Those rows are `steps_capacity(64, ·)` — the 0.9 cell
  is **58**, `round(57.6)`, not 57. Mixtral is the first geometry where
  rounding and flooring differ, and a test pins the grid so the harness cannot
  drift off this file (`test_capacity.py`).
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
