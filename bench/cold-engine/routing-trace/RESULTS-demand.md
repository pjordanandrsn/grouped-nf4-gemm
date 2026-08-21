# When demand-paging beats placement — the question I twice declined to answer

> **Held out of sample.** Preregistered and retested on a third model in
> [`RESULTS-third-model.md`](RESULTS-third-model.md) — Qwen1.5-MoE-A2.7B, 48
> new cells: **15 TP / 0 FP / 3 FN / 30 TN**, ρ(headroom, wins) = **−0.792**
> against −0.720 here. Zero false positives at `headroom ≤ 1` again, and the
> misses are again all above the threshold. Sufficient, not necessary, on
> three models.


Receipt: [`demand.json`](demand.json). Harness:
[`score_demand.py`](score_demand.py). 96 configurations, two models, four
prompts. No box.

[`RESULTS-concentration.md`](RESULTS-concentration.md) tested this and
reported **"not supported, ρ = 0.055"**, then declined to say more:

> Three positives is too few to fit a rule to, and fitting one would repeat
> the mistake this document exists to correct.

Declining to fit a rule to three points was right. **Leaving it there was
not** — the fix for too few positives is more points, not a better story.
Twelve capacities instead of three: **96 cells with 28 positives**, against 24
with 3.

## The rule is sufficient, and has no exceptions

| rule | accuracy | TP | FP | FN | TN |
|---|---|---|---|---|---|
| headroom ≤ 0.9 | 0.833 | 12 | 0 | 16 | 68 |
| **headroom ≤ 1.0** | **0.917** | **20** | **0** | 8 | 68 |
| headroom ≤ 1.1 | 0.896 | 22 | 4 | 6 | 64 |
| headroom ≤ 1.5 | 0.823 | 26 | 15 | 2 | 53 |

`headroom` = scored working set ÷ capacity. Baseline (always say no): 0.708.

**At `headroom ≤ 1` there are twenty true positives and zero false
positives.** Every configuration where the fast tier can hold everything the
window will ask for, demand-paging beat static placement — no exceptions
across two models and four prompts.

That is a **sufficient** condition, not a necessary one: eight cells win above
the threshold too, all between 1.07 and 1.93, and five of the eight are
Granite code. So "capacity covers the working set → don't bother with
placement" is safe to act on; "capacity does not cover it → placement wins" is
not.

The mechanism is not subtle. If the cache holds every row the window asks
for, LRU takes only compulsory misses, and no placement policy can do better
than compulsory.

## Why the earlier test said the opposite

Two independent problems, and both were mine.

**Underpowered.** With three positives, ρ = 0.055. With twenty-eight,
**ρ = −0.720**. The first number measured the absence of data, not the absence
of an effect — and I reported it as "not supported" rather than "not tested".

**And ρ was the wrong statistic regardless.** The hypothesis is a *threshold*:
capacity covers the working set, or it does not. A rank correlation computed
across a range where 21 of 24 cells sit on one side of that threshold cannot
see a step. Even now, with the effect clearly present, ρ = −0.720 understates
what a single split at 1.0 does — 91.7% of cells classified correctly.

The lesson is narrower than "get more data": **score a threshold hypothesis as
a classifier against a base rate**, and a correlation coefficient will not
tell you when you have failed to.

## Limits

- 96 cells but only 8 trace × model combinations; capacities within one trace
  are not independent samples.
- The threshold is at exactly 1.0, which is where the mechanism puts it, not
  where the fit does — the best split was 0.9983 and rounding it to the
  mechanistic value costs nothing. That is the reason to trust it rather than
  a coincidence to celebrate.
- Reads counted, not timed.
- `static` is top-C by frequency, a weaker stand-in for `solve_placement`'s
  greedy balance, so the placement arm is if anything handicapped.
