# The one-step threshold is real; my explanation of it was not

> **Held out of sample on a third model.** Qwen1.5-MoE-A2.7B (24L × 60E,
> **top-4**) was preregistered and tested in
> [`RESULTS-third-model.md`](RESULTS-third-model.md). Both claims held: the
> crossover is at one step on the third model too — 12 traces of 12 across
> three models, every cell below one step losing and every cell at it
> winning — and LRU and FIFO take zero hits below one step where random does
> not, 12 cells of 12 at a third cycle length.
>
> An earlier version of this banner said the crossover claim had been
> **refuted**. It had not. The replay harness left `DevRowCache(routed=...)`
> at its default of 8 while the model routes 4, sizing the demotion budget
> for a top-8 engine; the cache thrashed and the top-4 model looked like a
> counterexample. `RESULTS-third-model.md` has the full account.
>
> **Still true, and unrelated to that:** the counts below come from the
> pure-LRU simulation in `score_crossover.py`, not from the shipped
> `DevRowCache`. The two disagree at capacity == one step on all 12 traces —
> the simulation is pessimistic by 1.5k–15k, everywhere, because the cache
> resurrects reclaimable rows and picks victims LFU-then-LRU. The crossover
> *location* reported here matches the real cache on every model; the hit
> *counts* are a simulation's and are not labelled as such below.

Receipt: [`crossover.json`](crossover.json). Harness:
[`score_crossover.py`](score_crossover.py). Eight traces, two models, 48
configurations. No box.

[`RESULTS-concentration.md`](RESULTS-concentration.md) found that
`steps_held = capacity ÷ (layers × top-k)` separated every configuration where
the device cache helped from every one where it lost, and explained it as:

> A cache smaller than one step's routed set is fully evicted before its own
> next request, so it retains nothing across steps.

**The threshold is real and reproduces exactly. The explanation is wrong.**

## It is not capacity, it is LRU

Routing per step is a near-cyclic scan of the same `layers × top-k` rows, and
**LRU on a cyclic reference pattern with capacity below the cycle length is
the textbook worst case: zero hits.** So is FIFO. A policy without that
pathology should get hits in exactly the regime where these get none.

| | cells | LRU zero-hit | FIFO zero-hit | RANDOM zero-hit |
|---|---|---|---|---|
| **below one step** | 24 | **24** | **24** | **0** |

| | cells | LRU is best |
|---|---|---|
| **at or above one step** | 24 | **22** |

Below the threshold LRU and FIFO retain **literally nothing** — every routed
row is a transfer, on both models and all four prompts. Random retains
plenty. Above it, LRU is the right policy again in 22 of 24.

The sharpness follows: at `steps_held` 0.90 (OLMoE, cap 115) LRU gets **0**
hits; at 0.99 (cap 127) it gets 18,573; at 1.00 it gets 22,198. That is not a
capacity curve, it is a pathology switching off within two rows of the cycle
length.

## Crossing at one step on both models, at 2× the absolute capacity

| steps_held | 0.50 | 0.75 | 0.90 | 1.00 | 1.25 | 1.50 |
|---|---|---|---|---|---|---|
| olmoe prose (cache ÷ positional) | 1.162 | 1.162 | 1.162 | **0.768** | 0.768 | 0.757 |
| granite prose | 1.297 | 1.297 | 1.297 | **0.704** | 0.701 | 0.658 |

OLMoE crosses at **128** rows and Granite at **256** — the same predicted
point, at twice the absolute capacity. The identical values below the
threshold are the zero-hit region: the ratio is exactly `routed ÷ positional`,
because nothing is ever hit.

## The practical guidance does not change

Random has no zero-hit region, but it only beats the engine's existing
positional cache in **7 of 24** sub-threshold cells — all of them at
`steps_held = 0.90`, right at the boundary. At 0.50 and 0.75 it loses
everywhere. Above the threshold LRU beats positional in **24 of 24**.

So: **size the device cache to at least one decode step's routed set**, as
before. What changes is *why*, and that matters for what someone does when
they cannot: the failure is a policy pathology, not a capacity shortfall, so
adding 10% more rows below the threshold buys **nothing at all** — the cache
stays at zero hits until it reaches the cycle length. `too_small_to_retain`
is reporting a cliff, not a slope.

## What this says about my own change

I replaced slot-index victim selection with LRU in #157, on evidence that it
was worth 1.12–1.33× above the threshold. That was right, and it did not
create this: **FIFO is equally pathological, 24 of 24**, and slot-index order
behaves the same way. LRU did not introduce the zero-hit region; it also did
not fix it, and I did not look for it because every capacity I tested then
was at or above one step.

## Limits

- Two models, four prompts, 512 decode steps, 48 configurations.
- Counted, not timed. Random's extra hits are cheaper transfers, not
  necessarily cheaper wall.
- `random` here is uniform over resident rows with a fixed seed; no attempt
  was made to tune it, and it is not proposed as the shipped policy.
- The claim is about *this* access pattern being near-cyclic. A router with
  strong temporal locality within a step would not show it.
