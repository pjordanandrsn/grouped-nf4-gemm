# Rank predicts recurrence and cannot be spent: S1 refuted narrowly, S2 emphatically

Receipts: [`rank-2026-08-22/`](rank-2026-08-22/) — 16 traces carrying rank and
the near-miss band, plus
[`rank-policy.json`](rank-2026-08-22/rank-policy.json). Harness:
[`rank_policy.py`](routing-trace/rank_policy.py). Preregistration:
[`PREREG-router-rank.md`](PREREG-router-rank.md), merged before capture.
RTX 5090, **$0.72**, box destroyed.

| prediction | outcome |
|---|---|
| **S1** — rank 1 recurs ≥ 1.5× as often as rank k, on all four models | **REFUTED** (3 of 4 clear it) |
| **S2** — a rank-aware policy closes ≥ 30% of the gap | **REFUTED**, and it is *worse than LRU* |

## S1 — the signal is real, and misses the bar on one model

P(expert recurs at this layer's next visit | its rank at this visit):

| model | rank 1 | rank k | ratio | |
|---|---|---|---|---|
| OLMoE | 61.2% | 25.0% | **2.44×** | ok |
| Granite | 59.9% | 34.0% | 1.76× | ok |
| Qwen1.5-MoE | 16.4% | 12.0% | **1.37×** | under 1.5 |
| gpt-oss | 53.5% | 34.3% | 1.56× | ok |

Registered as "≥ 1.5 on all four", so this is **refuted** — by one model, at
1.37 against a 1.5 bar. The full profile is **monotone on every model**:

```
olmoe     0.61 0.49 0.44 0.40 0.35 0.31 0.28 0.25
granite   0.60 0.55 0.52 0.48 0.45 0.41 0.38 0.34
qwen      0.16 0.15 0.13 0.12
gptoss    0.53 0.48 0.42 0.34
```

The router's confidence does predict recurrence. Qwen's ratio is compressed
because its recurrence is low everywhere — 12–16% against 25–61% elsewhere,
which matches its step-to-step overlap of 13.4% against 41–50%. There is less
to predict, not a broken predictor.

## S2 — run anyway, exploratory, and it is worse than doing nothing

The preregistration said a refuted S1 makes S2 not worth running, on the
grounds that "a policy cannot exploit a signal that is not there". That
rationale does not survive the numbers above: the signal *is* there, monotone
on all four models and over the bar on three. So S2 was run and is reported
**as exploratory, not as a registered result**.

Fraction of the LRU-to-Belady gap closed:

| model | rank-aware | LFU | rank ÷ LRU |
|---|---|---|---|
| OLMoE | **−19%** | 43% | 1.087 |
| Granite | **−24%** | 44% | 1.121 |
| Qwen1.5-MoE | −6% | 22% | 1.020 |
| gpt-oss | **−41%** | 2% | 1.197 |

Negative throughout, beating LRU in **6 of 48** cells. A policy built on a
real signal is 2–20% *worse* than one that ignores it.

## The two results together, and what they close

Rank correlates with **whether** an expert returns. Recency correlates with
**when**. At these capacities a row survives only if it is needed almost
immediately, so imminence is what matters and a rule that trades it away for
a better long-run predictor loses — exactly what the numbers show.

That completes the ordering story and sharpens
[`RESULTS-cyclic-policy.md`](RESULTS-cyclic-policy.md), which found ordering
"nearly irrelevant". It is not symmetric:

| policy | lever | gap closed |
|---|---|---|
| LFU | **not an ordering rule** — pins recurring experts, shrinks the working set | 43 / 44 / 22 / 2 % |
| ARC | adaptive ordering | ~6% flat |
| cyclic | structural ordering | ~0% |
| **rank** | **router-signal ordering** | **−6 to −41%** |

LRU sits near a **broad local optimum** among ordering rules. Cyclic changes
93% of victim choices and moves transfers ~1%; ARC recovers 6%; and a rule
using genuinely better information about recurrence does actively worse. You
cannot beat it by reordering, and you can lose by trying.

Nine things have now been tried against the ~1.9× gap: five explanations of
when frequency wins, and four policies. The only one that moves the number is
the one that is not an ordering rule at all.

## Preconditions

**Implementation validated**: at capacity ≥ key space rank = LRU = Belady
(L = 4 and 16); Belady ≤ rank ≤ all-miss over 30 random workloads; and with
**k = 1**, where every resident's rank is 0, the policy reduces *exactly* to
its LRU tie-break — the degenerate case that catches a comparison with the
wrong sign.

**Capture invariants** asserted at write time on all 16 traces: rank order is
a permutation of the sorted routed set, and the near-miss band never
intersects the selection. Every trace also carries `repo_id` and the full
`env` block, so this is the first set that is regenerable from the repository
alone.

## What is left

The near-miss band was captured and is **not** used above — a policy over
"experts that nearly made the cut" is a different experiment, and folding it
in here would be fitting a second idea to the same data. It is the obvious
next candidate, and after nine failures the prior on it should be set
accordingly.
