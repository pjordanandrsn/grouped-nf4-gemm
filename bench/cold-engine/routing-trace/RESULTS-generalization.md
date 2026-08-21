# Two models, four prompts: one conclusion survives

Receipt: [`generalization.json`](generalization.json). Harness:
[`score_generalization.py`](score_generalization.py). Traces:
`{olmoe,granite}_{prose,code,math,dialogue}.jsonl`.

Every offline result in this campaign replayed **one** captured trace of one
model. Eight traces now exist across **two architectures**:

| model | shape | arena |
|---|---|---|
| OLMoE-1B-7B | 16 layers × 64 experts, top-8 | 1024 pairs |
| Granite-3.0-3B-A800M | 32 layers × 40 experts, top-8 | 1280 pairs |

Capacity is taken as a **fraction of the arena** (12.5%, 37.5%, 50%), so two
models with different expert counts are compared at the same pressure rather
than the same row count.

| conclusion | four prompts | + second model |
|---|---|---|
| **R4 refuted** — frequency beats short-window recurrence | holds 20/22 | **holds 18/18 — 38 of 40 overall** |
| **Device row cache beats the positional cache** | holds | **BREAKS** — 123–130% of positional at 12.5% |
| **Gate 3** — adaptive beats static | direction holds | **BREAKS** — neutral or worse on Granite math |
| **EWMA is the better policy** | already refuted | adaptive wins **13 of 24** overall |
| **Placement beats demand-paging when the tier is scarce** | already refuted | — |

**One conclusion of five survived both axes.**

## Holds everywhere: R4

Frequency beats short-window recurrence in **38 of 40** signal-bearing cells
across two models and four prompts — **unanimous, 18 of 18, on Granite**. Both
exceptions are OLMoE dialogue at 256 rows, by 0.018 and 0.056. This is the one
result that looks like a property of MoE routing rather than of a trace.

## Breaks: the device row cache

| capacity | olmoe (4 prompts) | granite (4 prompts) |
|---|---|---|
| 12.5% | 61–77% of positional | **123–130% — WORSE** |
| 37.5% | 9–47% | 48–54% |
| 50% | 6–33% | 28–36% |

**At 12.5% capacity on Granite the cache is worse than doing nothing new.**
This is the risk the cache was shipped with, stated in its own results
document:

> A miss now costs one extra device-side write (host→cache, then cache→slot)
> against a PCIe transfer it does not avoid. It pays only if re-routing to a
> new position is common.

On Granite it is not common enough at low capacity. Granite's scored working
set is **1091–1235 of 1280 pairs** — almost no concentration — so a 160-row
cache thrashes and the extra write per miss is paid on nearly every request,
while the positional cache at least skips same-position hits for free.

The documented failure mode was real; four prompts of one model could not
reach it, and one prompt of a second model did.

## Breaks: gate 3's premise

Adaptive beat static on all twelve OLMoE cells. On Granite the gains collapse
to **−0.3% to −4.4%**, and on Granite mathematics it is **+0.0%, +0.0%,
+0.2%** — *not better*, at every capacity.

So "adaptive re-placement never loses" is false. The premise that survives is
narrower: adaptive helps *when the routing distribution moves enough to be
worth tracking*, and on this model that is barely, or not at all.

## The variable, again, is concentration

| model | scored working set |
|---|---|
| olmoe math | **377 of 1024** (37%) |
| olmoe dialogue | 783 (76%) |
| olmoe prose | 899 (88%) |
| granite dialogue | 1091 of 1280 (85%) |
| granite code | **1235 (96%)** |

Every conclusion that broke, broke in the direction concentration predicts.
OLMoE mathematics concentrates hard, so capacity covers the working set and
demand-paging wins. Granite concentrates barely, so caches thrash and
placement has little to exploit. **The results were never really about
prompts or models — they were about how much of the arena a generation
actually touches**, and neither the prompt nor the architecture determines
that on its own.

## Limits

- Two models, four prompts each, 512 decode steps, one continuation.
- Reads and transfers counted, **not timed**.
- Both models are top-8. Nothing here tests a top-1 or top-2 router, where
  concentration would differ again.
- Granite's config reports `num_local_experts`, not `num_experts`; the
  capture records `n_experts: null` for it and the count (40) is recovered
  from the trace. Cosmetic, but it means that field cannot be trusted across
  architectures.
