# One prompt or four: which conclusions were about routing?

Receipt: [`generalization.json`](generalization.json). Harness:
[`score_generalization.py`](score_generalization.py). Traces:
`olmoe_{prose,code,math,dialogue}.jsonl`. No box beyond the capture.

Every offline result in this campaign replayed **one** captured trace, and
"one prompt" is the limit printed at the top of each of them. Four now exist —
same model, same decode shape, deliberately unlike generations. Each
conclusion re-run against all four.

**Two survive, one survives in direction only, two do not.**

| conclusion | verdict |
|---|---|
| **R4 refuted** — frequency beats short-window recurrence | **holds** — 20 of 22 signal-bearing cells, all four prompts |
| **Device row cache beats the positional cache** | **holds, and prose was the worst case** |
| **Gate 3** — adaptive re-placement beats static | **direction holds, magnitude does not** |
| **EWMA is the better policy** | **does not hold** — loses on code at every capacity |
| **Placement beats demand-paging when the tier is scarce** | **does not hold** — fails on math |

## The working set is what varies

| prompt | distinct (layer, expert) pairs in the scored window |
|---|---|
| prose | 899 of 1024 |
| code | 878 |
| dialogue | 783 |
| **math** | **377** |

Mathematics collapses to a third of the arena. That single number explains
most of what follows.

## Holds: R4

Frequency beats short-window recurrence in **20 of 22** signal-bearing cells
at w=4 and w=8 across all four prompts. Both exceptions are dialogue at 256
rows, by 0.018 and 0.056. The refutation was not a property of prose.

## Holds, conservatively: the device row cache

| rows | prose | code | math | dialogue |
|---|---|---|---|---|
| 128 | 76.8% | 72.3% | **61.1%** | 75.0% |
| 384 | 46.9% | 28.8% | **8.9%** | 44.2% |
| 512 | 30.9% | 16.4% | **5.5%** | 33.1% |

(transfers as a share of what the engine's positional cache still makes;
lower is better)

**Prose is the worst case at every capacity.** The published figure —
76.8% at 12.5% capacity — understates the cache on three of four prompts,
and on mathematics it removes 91% of the positional cache's transfers at 384
rows. This conclusion was conservative, not overfitted.

## Direction only: gate 3

Adaptive re-placement beats static on **all twelve** cells, so the premise
holds. The magnitude does not:

| prompt | adaptive vs static |
|---|---|
| code | **−0.5% to −2.6%** |
| dialogue | −2.1% to −6.9% |
| prose | −6.3% to −17.3% |
| math | −5.9% to **−63.0%** |

The published range was "6–41%". On code the loop is worth **half a percent**
— indistinguishable from not closing it. The claim that survives is *adaptive
never loses*; the number attached to it was prose's.

## Does not hold: EWMA

Decaying the counts won on prose at every capacity, which is where it was
found. Across four prompts no policy dominates:

| | ewma | adaptive | hybrid | demand |
|---|---|---|---|---|
| cells won (of 12) | 6 | 3 | 2 | 1 |

**On code, plain adaptive beats EWMA at every capacity.** That is a mechanism,
not noise: decay helps when the routing distribution drifts and hurts when it
is stationary, because discarding old observations discards signal that is
still valid. Prose drifts; code does not.

The recommendation "use EWMA" should be "measure whether your workload
drifts, and decay only if it does".

## Does not hold: placement beats demand-paging

Published as *"demand-paging is worse than static at every capacity up to 512
— profile knowledge is worth something LRU cannot recover, in exactly the
regime where the fast tier is scarce."*

That holds on prose, code and dialogue. **On mathematics it fails
completely**: at 384 rows `demand` does **54** reads against static's
**2,335**, and at 512 it does 41 against 1,347. The eval-window working set is
377 pairs, so capacity covers it and recency wins outright — the fast tier is
not scarce, whatever the nominal ratio says.

The real rule is not about capacity in rows. **It is whether capacity covers
the working set**, which is a property of the generation, not of the
configuration.

## Limits

- **One model.** All four traces are OLMoE-1B-7B, 16×64, top-8. Nothing here
  separates "about MoE routing" from "about this model's routing".
- Four prompts, one continuation each, 512 decode steps.
- Reads counted, not timed.
- The two surviving conclusions survive *on these four*; four is better than
  one and is not many.
