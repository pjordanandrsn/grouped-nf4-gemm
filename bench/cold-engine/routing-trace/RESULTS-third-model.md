# Out of sample on a third model: one rule held, one missed by three rows

Receipt: [`RESULTS-third-model.json`](RESULTS-third-model.json). Harness:
[`score_third_model.py`](score_third_model.py). Preregistration:
[`PREREG-third-model.md`](PREREG-third-model.md), sha256
`523fe8b7…5df694a3a94` — unchanged since it was committed in #172, before the
model was captured. No box for the scoring; one A6000-hour for the capture.

| prediction | outcome |
|---|---|
| **P1** — the device cache crosses at `layers × top-k` = 96 rows | **REFUTED** |
| **P1b** — below the threshold LRU takes exactly zero hits | **CONFIRMED** |
| **P2** — `headroom ≤ 1` predicts demand-paging wins, no false positives | **CONFIRMED** |

## The model

**Qwen1.5-MoE-A2.7B**, 24 layers × 60 experts, **top-4** — the first top-4
trace; both derivation models are top-8. 512 decode steps per prompt, 64-token
prompt, four prompts.

| prompt | steps | distinct (layer, expert) of 1440 |
|---|---|---|
| prose | 512 | 1439 |
| code | 512 | 1366 |
| math | 512 | 1167 |
| dialogue | 512 | 1394 |

The geometry came out exactly as registered: 24 routers, top_k 4, 60 experts,
arena 1440, 96 rows per step.

## P1 — refuted, by two to three rows

The prediction was that ratio > 1 below 96 rows and < 1 at or above. Below is
right on all 12 cells. **At exactly 96 it is still > 1 on all four prompts**,
which the preregistration names as a refutation.

| prompt | 48 | 72 | 86 | **96** | 120 | 144 |
|---|---|---|---|---|---|---|
| prose | 1.090 | 1.090 | 1.090 | **1.090** | 0.917 | 0.917 |
| code | 1.068 | 1.068 | 1.068 | **1.068** | 0.924 | 0.924 |
| math | 1.046 | 1.046 | 1.046 | **1.046** | 0.937 | 0.937 |
| dialogue | 1.071 | 1.071 | 1.071 | **1.071** | 0.923 | 0.923 |

Sweeping row by row, the real crossover is at **99, 99, 99 and 98 rows** —
`per_step + 2` or `+ 3`. The ratio then plateaus from **100 = per_step +
top_k** onward and does not improve again through 144.

Two things this is *not*:

* It is **not** an arena fraction. 98–99 rows is 6.8% of the 1440-row arena.
  The hypothesis `RESULTS-concentration.md` corrected — that the threshold is
  some fraction of the arena — is not resurrected by this; it predicts a
  crossover an order of magnitude away. The step-based rule gets the scale
  right to within 3%.
* It is **not** an artifact of the third model's instrumentation. Run the same
  way, the real cache crosses at *exactly* one step on both derivation models,
  8 traces of 8 — below one step every ratio > 1, at one step every ratio < 1.
  The boundary genuinely moves on Qwen and not on the others.

So the corrected statement is that `steps_held ≥ 1` is **necessary but not
sufficient**: a cache below one step never wins on any of the three models,
but a cache at exactly one step wins on two of three. What separates them is
not yet known. The obvious candidate is top-k — Qwen is the only top-4 trace,
and it is also the only model where the win is marginal at best (0.917 at its
best capacity, against 0.61–0.77 for OLMoE and 0.70–0.74 for Granite). One
model is not enough to attribute it to top-k, and this document is not going
to fit a rule to a single point; that is the mistake
[`RESULTS-concentration.md`](RESULTS-concentration.md) exists to record.

## P1b — confirmed, 12 of 12

Below 96 rows the cache takes **exactly zero** hits — every routed row-slot is
a transfer, on all four prompts and all three swept capacities.

That it is a *policy pathology* and not a capacity limit reproduces too. Run
across policies ([`crossover-qwen.json`](crossover-qwen.json),
[`score_crossover.py`](score_crossover.py)), below one step:

| | cells | LRU zero-hit | FIFO zero-hit | RANDOM zero-hit |
|---|---|---|---|---|
| Qwen, below one step | 12 | **12** | **12** | **0** |

Random retains plenty in exactly the regime where LRU and FIFO retain nothing —
the same signature as the two derivation models, at a third cycle length.

Note that on Qwen the zero-hit region extends *to and including* 96 rows. P1b
and the P1 miss are the same fact from two sides: the pathology switches off
two to three rows later than the cycle length, not at it.

## P2 — confirmed, zero false positives

48 cells, 12 capacities × 4 prompts ([`demand-qwen.json`](demand-qwen.json),
[`score_demand.py`](score_demand.py)). Demand-paging wins in 18; always-say-no
scores 0.625.

Both receipts cited here — `demand-qwen.json` and the registered
`RESULTS-third-model.json` — now derive every capacity from one shared
`capacity()` helper, so they agree on all 48 by construction. They did not
at first: the two harnesses used `int(arena * f)` and `int(round(arena * f))`,
and `1440 * 0.7` is `1007.9999999999999` in binary, so one said 1007 and the
other 1008 (Bugbot, #177). The rule is a floor and stays a floor; going
through `Fraction` on the decimal text removes the error. Across the three
arenas and twelve fractions it moves exactly one cell, no win flips, the
derivation receipt `demand.json` is byte-identical, and every number in this
document is unchanged.

| rule | accuracy | TP | FP | FN | TN |
|---|---|---|---|---|---|
| headroom ≤ 0.9 | 0.854 | 11 | 0 | 7 | 30 |
| **headroom ≤ 1.0** | **0.938** | **15** | **0** | **3** | **30** |
| headroom ≤ 1.1 | 0.938 | 16 | 1 | 2 | 29 |
| headroom ≤ 1.5 | 0.833 | 17 | 7 | 1 | 23 |

**Zero false positives at the mechanistic threshold**, which is what the
preregistration required and what the derivation set showed across 96 cells.
Spearman(headroom, wins) = **−0.792** on this model against −0.720 on the
derivation set.

The three misses are all above the threshold — demand-paging winning where the
rule does not claim it will, which the preregistration registered in advance as
expected and not evidence against the rule:

| prompt | frac | headroom | margin |
|---|---|---|---|
| prose | 0.90 | 1.08 | +3.3% |
| math | 0.15 | 1.81 | +89.8% |
| math | 0.20 | 1.35 | +97.3% |

The rule remains **sufficient, not necessary**, on a third model.

## A separate finding: the simulation is not the cache

P1 was scored against the real `DevRowCache`. The standalone LRU simulation in
[`score_crossover.py`](score_crossover.py) — which is what
[`RESULTS-crossover.md`](RESULTS-crossover.md) reports — disagrees with the
shipped cache at capacity == one step on **all 12 traces**, and the sign flips
by model:

| model | sim − real, at one step |
|---|---|
| OLMoE | +6,359 … +9,487 (sim pessimistic) |
| Granite | +14,004 … +15,259 (sim pessimistic) |
| Qwen | −3,659 … −5,927 (sim optimistic) |

The simulation is pure LRU over `(layer, expert)` keys; the cache resurrects
reclaimable rows and picks victims by LFU-then-LRU. On Qwen at 96 rows the
simulation records 3.7k–5.9k hits where the cache takes none — scoring P1
against the simulation would have recorded a **pass the cache does not earn**.

This does not overturn `RESULTS-crossover.md`: its threshold claim is about
eviction *policies*, it is stated as such, and the crossover location it
reports still matches the real cache on both derivation models. But its hit
*counts* describe a simulation, not the shipped cache, and that distinction was
not drawn in it. A banner has been added there.

## Exploratory, after the fact — not preregistered

Everything above is the registered test. What follows was measured after
seeing the result and is **hypothesis-generating, not confirmatory**; it is
here so the next preregistration has something to aim at. Harness:
[`reuse_overlap.py`](reuse_overlap.py), receipt
[`reuse-overlap.json`](reuse-overlap.json).

### A cache sized at exactly one step can only capture step-to-step reuse

Mean overlap between a step's routed set and the previous step's:

| model | top-k | E | raw overlap | wins at one step |
|---|---|---|---|---|
| OLMoE | 8 | 64 | 41.3% | yes |
| Granite | 8 | ≥40 | 43.6% | yes |
| **Qwen1.5-MoE** | **4** | **60** | **13.4%** | **no** |

A 3× gap that lines up perfectly with which models win — **and it is mostly an
artifact of not normalizing.** Under independent uniform routing the expected
overlap fraction is just `k/E`, which differs across these models by 3× on its
own (12.5% / 20.0% / 6.7%). Dividing it out:

| model | chance `k/E` | observed | observed ÷ chance |
|---|---|---|---|
| OLMoE | 12.5% | 41.3% | **3.30×** |
| Granite | 20.0% | 43.6% | **2.18×** |
| Qwen1.5-MoE | 6.7% | 13.4% | **2.01×** |

Granite's router did not expose an expert count (`n_experts` is null in its
trace metadata), so its E is the largest id actually routed — a lower bound.
If the true E is larger, Granite's chance rises and its normalized figure
falls *further* toward Qwen's, which only strengthens what follows.

Granite wins at one step and Qwen loses, and their chance-normalized
stickiness is **2.18× against 2.01×** — indistinguishable. Per prompt the two
groups **cross**: Granite's math trace is 1.99× and Qwen's prose is 2.38×, so
the "no overlap between the groups" reading does not survive normalization
either.

So routing is about equally sticky on all three models relative to chance, and
the raw-overlap gap is `k/E` arithmetic. That collapses the two candidate
explanations into one rather than separating them: what Qwen has less of is
**reusable rows per step**, and given E that is another way of saying top-k.

A fourth model can discriminate, because `k` and `k/E` are only confounded in
the models captured so far. **Mixtral-8x7B routes top-2 of 8** — the smallest
top-k of any candidate but `k/E` = 25%, the *largest*. If the one-step
crossover is governed by `k/E` it should win at one step; if by top-k it
should lose. That is worth preregistering before capturing it.

### One trace is a degenerate repetition loop

Qwen's **math** trace alternates with period 2 — overlap by lag
([`reuse_overlap.py`](reuse_overlap.py)):

| lag | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| qwen math | 10.5% | **84.6%** | 10.0% | **83.1%** | 10.0% | **82.0%** |

Even lags ~83%, odd lags ~10%. No other trace of the twelve shows this.
Capture is greedy (`argmax`, no sampling) over 512 steps, which falls into
repetition loops readily; the trace does not record tokens, so this is
inferred from the routing alone, not confirmed.

It is the likely cause of that trace's two outliers — the two largest P2
misses (`math` at frac 0.15 and 0.20, +89.8% and +97.3%) and random
beating LRU by 40% above the threshold, both of which are what a period-2
alternation does to a one-step cache.

**Every verdict above is robust to dropping it.** Rescored on the other three
prompts (`--prompts prose,code,dialogue`):

| | all four (registered) | without math |
|---|---|---|
| P1 | REFUTED, 4 violations | REFUTED, 3 violations |
| P1b | CONFIRMED, 12 of 12 | CONFIRMED, 9 of 9 |
| P2 | CONFIRMED, 0 FP of 48 | CONFIRMED, 0 FP of 36 |

Future captures should record the generated tokens so a degenerate loop is
visible without having to infer it, and should consider whether 512 greedy
steps is the right decode for a routing trace at all.

## What now holds, across three models and twelve traces

| claim | status |
|---|---|
| cache below one step never beats positional | holds, 3 models |
| cache at one step beats positional | **holds on 2 of 3** |
| LRU/FIFO take zero hits below one step, random does not | holds, 3 models |
| `headroom ≤ 1` ⇒ demand-paging wins, no false positives | holds, 3 models |
| the threshold is an arena fraction | refuted, 3 models |
