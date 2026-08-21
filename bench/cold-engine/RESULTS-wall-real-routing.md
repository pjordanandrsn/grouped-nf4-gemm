# The R2 wall null survives real routing — and one of my two predictions was a misreading of it

Receipts: [`wall_prose.json`](wall-real-routing-2026-08-21/wall_prose.json)
(100 arms), [`overlap.json`](wall-real-routing-2026-08-21/overlap.json),
four captured traces in the same directory. Harness:
[`run_wall_real_routing.py`](wall-real-routing-2026-08-21/run_wall_real_routing.py).
Preregistration: [`PREREG-wall-real-routing.md`](PREREG-wall-real-routing.md),
sha256 `3d903903…`, committed and pushed before the box was rented.
RTX 5090, one A6000-class session, **$2.12** including one wasted rental.

| prediction | outcome |
|---|---|
| **W1** — captured routing gives ≥1.5× the resurrections per routed invocation | **REFUTED** |
| **W2** — the `rows-k` vs `quarter` wall difference is under 2% | **REFUTED, and the prediction was mis-stated** |
| the published null itself, tested properly | **HOLDS on real routing** |

## What was actually being asked

[`RESULTS-r2-wall.md`](RESULTS-r2-wall.md) refuted R2 and drives the engine
with `routes()`, which draws fresh `torch.randn` logits every step. Routing is
independent across steps, so step-to-step reuse sits at chance. The null was
measured on the routing least favourable to the mechanism under test — so it
was worth re-measuring where the mechanism has something to work with.

**gpt-oss-20b's own routing**, captured from the model the arena was baked
from — 24 layers × 32 experts, top-4, no id remapping, four prompts × 512
decode steps. Whole-model step-to-step overlap:

| prompt | lag-1 overlap | ÷ chance (12.5%) |
|---|---|---|
| prose | 50.4% | **4.03×** |
| code | 43.8% | 3.51× |
| math | 42.3% | 3.38× |
| dialogue | 41.1% | 3.29× |

No trace shows a repetition loop. The wall harness drives **one layer**, and
layer 0 of the prose trace has **28.3%** overlap against the synthetic
sequence's 12.9% — a 2.2× ratio, not 4×. The single-layer number is the one
that governs this experiment and it is the one used below.

## W1 — refuted, because the premise had the sign backwards

| rows | protected | synthetic /routed | captured /routed | ratio | synthetic fills | captured fills |
|---|---|---|---|---|---|---|
| 12 | 8 | 13.18% | 15.04% | 1.14× | 597 | **426** |
| 16 | 12 | 12.70% | 14.75% | 1.16× | 469 | **286** |
| 24 | 20 | 9.47% | 7.23% | 0.76× | 258 | **91** |
| 32 | 28 | 4.39% | 1.27% | 0.29× | 72 | **20** |
| 48 | 44 | 0.00% | 0.00% | — | 5 | 9 |

Never 1.5×, and *falling* with capacity. I predicted that more reuse would
produce more resurrections. It produces **fewer**: a resurrection is a row
that was evicted and recovered, and routing that repeats itself gets its rows
back as ACTIVE HITS without ever losing them. More reuse means fewer
evictions, so there is less to resurrect.

The mechanism is not failing — it is delivering somewhere else. Real routing
cuts transfers by **21% to 72%** at matched capacity, which is the column on
the right and is exactly what a cache is for.

The preregistration says a W1 failure makes W2 uninformative *as registered*,
and it does. It also turns out W2 was not the right question anyway.

## W2 — refuted, and the fault is in the preregistration

Registered: *"the wall-time difference between `protected = rows-k` and
`protected = rows//4` at fixed rows is under 2%."* Measured, on captured
routing: **−19.7%, −39.4%, −52.9%, −53.6%, −38.4%** across the five capacities,
with repeat spread under 2%. Refuted at every point.

**That refutes nothing in `RESULTS-r2-wall.md`, because it never made that
claim.** Its own sweep reports wall from 0.376 ms to 1.823 ms across the
`protected` range — a 4.8× spread, on the page. Its claim is narrower and
different: wall is **transfer-bound** (wall vs transfers/step, r = +0.9748),
and **resurrection adds nothing once transfers are accounted for** (residual
r = −0.1778). I compressed "no wall effect attributable to resurrection" into
"no wall difference between the arms" when writing the prereg, and those are
not the same statement. The measurement is sound; the prediction was a
misreading of the paper it was testing.

What the number does show is worth keeping: a correctly-sized `protected`
is **19–54% faster in wall** than a thrashing quarter-sized budget, on both
sequences, reproducibly. That is the wall-clock counterpart of the offline
result that `protected == rows - k` is the only correct setting
([`routing-trace/RESULTS-third-model.md`](routing-trace/RESULTS-third-model.md)).

## The published null, tested the way it was actually stated

Its own analysis, re-run on each sequence. Cells are medians over 5 repeats,
so repeats do not inflate n.

| sequence | wall ~ transfers/step | wall ~ resurrection | **residual after transfers ~ resurrection** | implied fit |
|---|---|---|---|---|
| published (synthetic, other host) | +0.9748 | +0.8404 | **−0.1778** | 24.2 GB/s |
| synthetic (control here) | +0.9817 | +0.9583 | **+0.0609** | 46.5 GB/s |
| **captured (real routing)** | **+0.9872** | +0.8230 | **−0.3443** | 37.0 GB/s |

The control reproduces the published relationship on a different host. On real
routing with 2.2× the step-to-step reuse, wall is **still** transfer-bound —
slightly more tightly — and the residual attributable to resurrection is still
not positive. Residuals span −0.080..+0.053 ms against a 0.471–1.259 ms wall
range.

**R2's refutation stands, and now stands on routing that gave the mechanism
its best case.**

The two hosts differ in fitted bandwidth (46.5 and 37.0 GB/s here against
24.2 GB/s published), which is why wall is ~2.7× lower throughout. The
*relationships* replicate; the absolute constants are per-host and are not
compared.

## What this cost, and what it bought

$2.12, including $0.06 on a rental killed for a bad image tag. It bought a
replication of the one claim in this line of work that is **not** structurally
forced — [`routing-trace/structural_check.py`](routing-trace/structural_check.py)
shows the three offline claims move in 0 of 24, 0 of 24 and 0 of 27 synthetic
conditions, so no captured model can test them. Wall was the remaining
question, and it now has an out-of-distribution answer.
