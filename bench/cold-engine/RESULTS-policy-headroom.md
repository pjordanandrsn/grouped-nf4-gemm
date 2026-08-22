# A frequency-aware victim rule closes half the gap on two models and nothing on a third

Receipt: [`policy-headroom.json`](routing-trace/policy-headroom.json). Harness:
[`policy_headroom.py`](routing-trace/policy_headroom.py). 48 cells — four
models × four prompts × three capacities. No box.

[`RESULTS-oracle-headroom.md`](RESULTS-oracle-headroom.md) measured the
shipped cache at **1.90× Belady's optimum** and found it is plain LRU. Belady
is a bound, not a proposal. This asks what an **implementable** policy
recovers — one that consults only what is known at eviction time.

`VramSlots._claim` selects `min(cands, key=_used)`: pure LRU, no frequency
term. (The LFU heap in `nvme_residency.py` belongs to `ColdTier`, a different
tier, and is not this.)

## The headline, and why it is not a recommendation

| | median | |
|---|---|---|
| **LFU ÷ LRU** | **0.847** | LFU beats LRU in **42 of 48** cells |
| fraction of the LRU→optimal gap LFU closes | **33%** | |

That reads like a straightforward 15% transfer win. It is not, because the
benefit is not distributed evenly across models:

| model | LFU ÷ LRU | cells LFU wins | gap closed |
|---|---|---|---|
| OLMoE | 0.775 | 12/12 | **49%** |
| Granite | 0.745 | 12/12 | **49%** |
| Qwen1.5-MoE | 0.893 | 11/12 | 30% |
| **gpt-oss-20b** | **0.992** | **7/12** | **2%** |

**On gpt-oss the change is worth nothing** — and gpt-oss is the model whose
arena every wall measurement in this program was made on
([`RESULTS-wall-real-routing.md`](RESULTS-wall-real-routing.md)), and the one
captured most recently. Had this been run on OLMoE and Granite alone, the
conclusion would have been "switch to LFU for ~25%", and shipping that would
have bought 0.8% on the workload the engine is actually benchmarked against.

The per-capacity picture is stable (0.862 / 0.814 / 0.839 at steps_held 1.0 /
1.5 / 2.0), so this is a model axis, not a capacity axis.

## Six cells where LFU loses

| trace | held | cap | LRU | LFU |
|---|---|---|---|---|
| qwen math | 2.00 | 192 | 6,730 | **15,562** |
| gptoss code | 2.00 | 192 | 19,176 | 21,270 |
| gptoss math | 2.00 | 192 | 21,080 | 21,695 |
| gptoss prose | 2.00 | 192 | 16,076 | 16,664 |
| gptoss code | 1.00 | 96 | 31,545 | 31,774 |
| gptoss code | 1.50 | 144 | 26,154 | 26,208 |

The 2.31× blowup is qwen's math prompt, the period-2 repetition loop
documented in
[`routing-trace/RESULTS-third-model.md`](routing-trace/RESULTS-third-model.md)
— under a strict alternation, frequency is uniform and carries no signal while
recency carries all of it. Excluding all three of its cells moves the overall
median only 0.847 → 0.848, so it is not what drives the result either way.

## A prediction of mine that failed

Step-to-step routed-set overlap is 41–50% on three of the four models, 2.0–4.0×
chance, so "routed in the previous step" looked like information LRU throws
away. Two policies were built on it: evict rows the previous step did not
touch first, breaking ties by recency (`prevstep`) or by frequency
(`prevstep_lfu`).

**It does not work.** `prevstep ÷ LRU` has a median of **1.001** and wins in
7 of 48 cells — indistinguishable from LRU, and worse than LFU everywhere.

The reason is visible after the fact: a row is touched at most once per step,
so "not touched last step" and "least recently used" are nearly the same
predicate. The prior adds no information LRU did not already have, and its
coarse two-class split discards the fine ordering LRU has within the class.
Recorded because it was a plausible hypothesis with real supporting statistics
behind it, and the statistics were about the wrong quantity.

## What this establishes

A frequency-aware victim rule is **worth trying and is not a guaranteed win**.
It is cheap — `VramSlots` already keeps per-slot state, and the concurrent
`ColdTier` work shows the heap form is affordable — but it should be gated on
a measurement against the target model, not adopted on the median.

The ~1.9× gap to optimal is real and mostly still open: the best implementable
policy tested closes a third of it, and none of it on gpt-oss.

## Method

`policy_headroom.py`'s LRU is cross-checked against the independent LRU
simulation in `score_crossover.py` and agrees exactly at three capacities;
recency is an explicit monotone clock rather than a position scan, so every
policy breaks ties on the same quantity.
