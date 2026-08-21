# Out of sample on a third model: all three predictions held — after a harness bug nearly said otherwise

Receipt: [`RESULTS-third-model.json`](RESULTS-third-model.json). Harness:
[`score_third_model.py`](score_third_model.py). Preregistration:
[`PREREG-third-model.md`](PREREG-third-model.md), sha256
`523fe8b7…5df694a3a94` — unchanged since it was committed in #172, before the
model was captured. One A6000-hour for the capture; scoring is offline.

| prediction | outcome |
|---|---|
| **P1** — the device cache crosses at `layers × top-k` = 96 rows | **CONFIRMED**, but see below — it could not have failed |
| **P1b** — below the threshold LRU takes exactly zero hits | **CONFIRMED** |
| **P2** — `headroom ≤ 1` predicts demand-paging wins, no false positives | **CONFIRMED** |

> **This document first reported P1 as REFUTED.** That was wrong, and the
> reason is worth more than the result. The section
> [below](#the-refutation-was-mine-not-the-models) has it in full: the replay
> harness left `DevRowCache(routed=...)` at its default of **8** while Qwen
> routes **4**, so the cache's demotion budget was sized for a top-8 engine
> and it thrashed. Corrected, the threshold holds on all three models and all
> twelve traces.

## The model

**Qwen1.5-MoE-A2.7B**, 24 layers × 60 experts, **top-4** — the first top-4
trace; both derivation models are top-8. 512 decode steps per prompt,
64-token prompt, four prompts.

| prompt | steps | distinct (layer, expert) of 1440 |
|---|---|---|
| prose | 512 | 1439 |
| code | 512 | 1366 |
| math | 512 | 1167 |
| dialogue | 512 | 1394 |

Geometry came out exactly as registered: 24 routers, top_k 4, 60 experts,
arena 1440, 96 rows per step.

## P1 — confirmed, and the crossover is exactly at one step

Ratio of cache transfers to the engine's positional baseline. Above 1 the
cache loses, below 1 it wins; the prediction is > 1 below 96 rows and < 1 at
or above.

| prompt | 48 | 72 | 86 | **96** | 120 | 144 |
|---|---|---|---|---|---|---|
| prose | 1.090 | 1.090 | 1.090 | **0.917** | 0.917 | 0.917 |
| code | 1.068 | 1.068 | 1.068 | **0.924** | 0.924 | 0.924 |
| math | 1.046 | 1.046 | 1.046 | **0.937** | 0.937 | 0.937 |
| dialogue | 1.071 | 1.071 | 1.071 | **0.923** | 0.923 | 0.923 |

Not one violation in 24 cells. Run identically, the same holds on both
derivation models, so the threshold is now **12 traces of 12 across three
models** — and the crossover is *at* one step, not near it: every cell at
0.9 steps loses and every cell at 1.0 wins.

The win is smaller here than on the models it was derived from (0.92–0.94
against 0.61–0.77 for OLMoE and 0.70–0.74 for Granite). The threshold
generalizes; the magnitude is not claimed to.

## P1b — confirmed, 12 of 12

Below 96 rows the cache takes **exactly zero** hits — every routed row-slot is
a transfer, on all four prompts and all three swept capacities.

That it is a *policy pathology* rather than a capacity limit reproduces too
([`crossover-qwen.json`](crossover-qwen.json),
[`score_crossover.py`](score_crossover.py)):

| | cells | LRU zero-hit | FIFO zero-hit | RANDOM zero-hit |
|---|---|---|---|---|
| Qwen, below one step | 12 | **12** | **12** | **0** |

Random retains plenty in exactly the regime where LRU and FIFO retain
nothing — the same signature as the two derivation models, at a third cycle
length.

## P2 — confirmed, zero false positives

48 cells, 12 capacities × 4 prompts ([`demand-qwen.json`](demand-qwen.json),
[`score_demand.py`](score_demand.py)). Demand-paging wins in 18;
always-say-no scores 0.625.

Both receipts cited here derive every capacity from one shared `capacity()`
helper, so they agree on all 48 by construction. They did not at first: the
two harnesses used `int(arena * f)` and `int(round(arena * f))`, and
`1440 * 0.7` is `1007.9999999999999` in binary, so one said 1007 and the
other 1008 (Bugbot, #177). The rule is a floor and stays one; going through
`Fraction` on the decimal text removes the error, moves exactly one cell
across three arenas and twelve fractions, flips no win, and leaves the
derivation receipt `demand.json` byte-identical.

| rule | accuracy | TP | FP | FN | TN |
|---|---|---|---|---|---|
| headroom ≤ 0.9 | 0.854 | 11 | 0 | 7 | 30 |
| **headroom ≤ 1.0** | **0.938** | **15** | **0** | **3** | **30** |
| headroom ≤ 1.1 | 0.938 | 16 | 1 | 2 | 29 |
| headroom ≤ 1.5 | 0.833 | 17 | 7 | 1 | 23 |

**Zero false positives at the mechanistic threshold**, which is what the
preregistration required. Spearman(headroom, wins) = **−0.792** here against
−0.720 on the derivation set. The three misses are all *above* the threshold,
which the preregistration registered in advance as expected:

| prompt | frac | headroom | margin |
|---|---|---|---|
| prose | 0.90 | 1.08 | +3.3% |
| math | 0.15 | 1.81 | +89.8% |
| math | 0.20 | 1.35 | +97.3% |

Sufficient, not necessary, on a third model.

## P1 could not have come out any other way

Three models agreeing reads as generalization. It is not, and the check that
shows why costs nothing: [`structural_check.py`](structural_check.py), receipt
[`structural-check.json`](structural-check.json).

Drive the same cache with **synthetic** routing spanning the plausible space —
stickiness from 0 (independent uniform draws, no temporal structure at all) to
0.95, expert popularity from flat to heavily skewed — at all four geometries,
including Mixtral's. If the threshold were a property of routing, some corner
of that space should move it.

**It moves in 0 of 24 conditions.** Zero hits below one step and a win at one
step, every time, including for routing with no temporal structure whatsoever.

The threshold is arithmetic on `protected = rows - k`: below one step the
cache cannot hold a single step's working set, so each step evicts its own
rows before reusing them; at one step it can. The three captured models did
not independently confirm a rule — **they could not have refuted it.**

This does not make the threshold wrong or useless. It is the right sizing
rule and it is worth stating. It does mean two things, both of which cut
against how this line of work has been reported:

* P1's confirmation here is far weaker evidence than "held on a third model"
  suggests, and this document says so rather than banking it.
* Capturing a fourth model to test it buys nothing. See
  [`PREREG-fourth-model.md`](PREREG-fourth-model.md), which is **withdrawn**
  for that reason before any box was rented.

What is *not* settled by arithmetic is the part P1b measures — that below the
threshold **LRU and FIFO** take zero hits where **random** does not. That is a
policy fact, it is not forced, and it is the part of the crossover story that
survives.

## The refutation was mine, not the model's

`DevRowCache.__init__` takes `routed`, the engine's `k`, and derives
`protected = rows - routed` from it. It **defaults to 8**, and
`replay_dev_cache.replay()` never passed it. Every replay ever run therefore
sized the demotion budget for a top-8 engine — correct by accident for OLMoE
and Granite, wrong for Qwen, and wrong for anything else that is not top-8.

The failure is total rather than gradual. On a **perfectly static** working
set — the same `per_step` keys every step, in exactly `per_step` rows, where
the ideal is one fill per key and then hits forever:

| geometry | k | per_step | fills, `routed=8` | fills, `routed=k` | ideal |
|---|---|---|---|---|---|
| 32 × 2 | 2 | 64 | **4096** | 64 | 64 |
| 24 × 4 | 4 | 96 | **6144** | 96 | 96 |
| 16 × 8 | 8 | 128 | 128 | 128 | 128 |
| 32 × 8 | 8 | 256 | 256 | 256 | 256 |

Every access missed, on a workload that never changes. An over-large
`protected` leaves too few rows demotable, and `_claim` prefers RECLAIMABLE
over ABSENT, so the few unprotected rows thrash in a cycle while virgin rows
are never touched.

Corrected, Qwen's four P1 violations disappear and nothing else in the study
moves — OLMoE and Granite were already `routed == k`, so every number derived
from them is unaffected.

**Why it survived as long as it did.** It does not look like a bug. It looks
like a model that simply does not benefit from caching — a plausible,
publishable, *interesting* negative result, complete with a mechanism I had
already written down for it. It was caught only because a synthetic dry run
at a fourth geometry produced a number that could not be true: a cache holding
a working set that never changes, missing on every single access.

`replay()` now takes `routed` from the trace's `top_k`, and
`kernel/test_dev_row_cache.py` pins both directions — that a static working
set is retained at exactly one step for k ∈ {2, 4, 8}, and that sizing for 8
when the model routes fewer produces the exact total-thrash count.

## One trace is a degenerate repetition loop

Qwen's **math** trace alternates with period 2 — overlap of a step's routed
set with the step `lag` before it
([`reuse_overlap.py`](reuse_overlap.py), [`reuse-overlap.json`](reuse-overlap.json)):

| lag | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| qwen math | 10.5% | **84.6%** | 10.0% | **83.1%** | 10.0% | **82.0%** |

Even lags ~83%, odd lags ~10%. None of the other eleven traces does this.
Capture is greedy (`argmax`, no sampling) over 512 steps, which falls into
repetition loops readily. It is the likely cause of that trace's outliers,
including the two largest P2 misses (`math` at frac 0.15 and 0.20).

**Every verdict is robust to dropping it.** Rescored on the other three
prompts (`--prompts prose,code,dialogue`):

| | all four (registered) | without math |
|---|---|---|
| P1 | CONFIRMED, 0 violations | CONFIRMED, 0 violations |
| P1b | CONFIRMED, 12 of 12 | CONFIRMED, 9 of 9 |
| P2 | CONFIRMED, 0 FP of 48 | CONFIRMED, 0 FP of 36 |

The trace here does not record tokens, so the loop is inferred from routing
alone. `capture_routing.py` now records the generated token id per step, and
`reuse_overlap.py` names the period directly when they are present and falls
back to flagging an even-lag spike when they are not — across the twelve
captured traces it fires on exactly one, the right one.

## A separate finding: the simulation is not the cache

P1 is scored against the real `DevRowCache`. The standalone LRU simulation in
[`score_crossover.py`](score_crossover.py) — which is what
[`RESULTS-crossover.md`](RESULTS-crossover.md) reports — disagrees with the
shipped cache at capacity == one step on **all 12 traces**, always in the same
direction:

| model | sim − real, at one step |
|---|---|
| OLMoE | +6,359 … +9,487 |
| Granite | +14,004 … +15,259 |
| Qwen | +1,479 … +1,926 |

The simulation is pure LRU over `(layer, expert)` keys; the cache resurrects
reclaimable rows and picks victims LFU-then-LRU, so it beats pure LRU
everywhere. This does not overturn `RESULTS-crossover.md` — its threshold
claim is about eviction *policies*, it is stated as such, and its crossover
location matches the real cache on every model. But its hit *counts* describe
a simulation rather than the shipped cache, and that distinction was not drawn
in it.

## What holds, across three models and twelve traces

| claim | status |
|---|---|
| cache below one step never beats positional | holds, 36 of 36 |
| cache at one step beats positional | holds, 12 of 12 |
| LRU/FIFO take zero hits below one step, random does not | holds, 3 models |
| `headroom ≤ 1` ⇒ demand-paging wins, no false positives | holds, 3 models |
| the threshold is an arena fraction | refuted, 3 models |
