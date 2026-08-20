# R2's premise, scored — reached at exactly one point, and that point is a thrashing cache

Registered (`PREREG-tribrid-stage3`, R2):

> VRAM resurrection is disproportionately valuable — **even 2–5% of routed
> invocations moves wall time** — refuted by *no measurable wall effect*.

**R2 is not scored here.** Its verdict is a wall measurement, and that half
needs an MXFP4 model this repo does not have (the device arena lives in
`Mxfp4NvmeResidency`; no gpt-oss/K3-lineage arena has been baked here).

What is scored is the **antecedent**: does the resurrection rate, measured
against *routed invocations*, reach the 2–5% R2 conditions its claim on?

Offline on `olmoe_routing_seq.jsonl` (512 decode steps × 16 layers × top-8
of 64 = 65,536 routed invocations) through `DevRowCache`/`VramSlots`.
Receipt `r2.json`, scorer `score_r2.py`. No GPU.

## It is reached once, out of twenty

| rows | protected | budget | resurrections | **per routed (R2)** | per resolved (R3) |
|---|---|---|---|---|---|
| 128 | 32 / 64 / 96 | ¼ / ½ / ¾ | 0 | **0.00%** | 0.0% |
| **128** | **120** | **rows−k** | **22198** | **33.87%** | 33.9% |
| 256 | 192 | ¾ | 458 | 0.70% | 1.1% |
| 256 | 248 | rows−k | 266 | 0.41% | 0.7% |
| 384 | 288 | ¾ | 853 | 1.30% | 2.5% |
| 384 | 376 | rows−k | 552 | 0.84% | 2.1% |
| 512 | 256 | ½ | 577 | 0.88% | 1.6% |
| 512 | 384 | ¾ | 683 | 1.04% | 2.6% |
| 512 | 504 | rows−k | 740 | 1.13% | 4.2% |
| 1024 | 256 | ¼ | 577 | 0.88% | 1.6% |
| 1024 | 512 | ½ | 782 | 1.19% | 4.6% |
| 1024 | 768 | ¾ | 134 | 0.20% | 5.1% |
| 1024 | 1016 | rows−k | 0 | 0.00% | — |

**One point of twenty clears 2%, and it clears it by 7×.** Every other
configuration lands between 0.00% and 1.30%.

## The point that clears it is a cache in free fall

At `rows=128, protected=120` the cache holds exactly one step's routed set
(16 layers × top-8 = 128) with only 8 slots reclaimable. Its fill count is
**43,338 against 65,536 routed — a 66.1% miss rate**. Resurrections and
fills together account for essentially every request; there are almost no
plain hits.

So the 33.87% is not a cache doing well. It is a cache thrashing so hard
that a third of its requests are rows it had just demoted and immediately
wanted back. The resurrections are real — each one avoids a refill — but
they are a symptom of the regime, not evidence of headroom in it.

Whether *that* moves wall time is exactly what R2 asks and exactly what
cannot be answered without the model. This document does not claim it does.

## A first version of this scorer got the opposite answer

The sweep originally ran `{¼, ½, ¾}` and **could not name `rows−k`** — which
is `DevRowCache`'s own default, and the budget `RESULTS-r3.md` had already
measured. On that omission this document concluded *"the 2–5% antecedent
never arrives"*, which is false: at the default budget it arrives at 33.87%.
Caught by Bugbot on gnf4#151.

Two things made it worse than a gap in a sweep. The budget was in the
`--budgets` default of `score_r3.py`, the file these conventions were copied
from, and it was read while writing this one. And the scorer caught only
`ValueError`, while `VramSlots` refuses a starving budget with a
`RuntimeError` — so a near-full budget on a small cache would have crashed
rather than been recorded as refused. Both are fixed; `rows−k` is now in the
default sweep and named in the help text as mandatory.

## Three predictions, one structural defect

R2 joins R3 and R9. Each names a rate without pinning the configuration that
decides it:

| | quantity | span across the sweep | verdict |
|---|---|---|---|
| R3 | resurrections / resolved evictions | 0.0 – 33.9% | undetermined (#144) |
| R9 | invocations with two valid copies | 0 – 57.3% | bounded, not scored (#150) |
| R2 | resurrections / routed invocations | 0.00 – 33.87% | antecedent reached at 1 of 20 |

The correction above *strengthens* this rather than weakening it: R2's rate
does not merely depend on configuration, it spans nearly the whole unit
interval across settings a reasonable person would call equivalent. A
prediction of the form *"rate X falls in band Y"* is only scoreable if the
configuration that sets X is fixed as part of the registration.

## What this does not establish

- **Nothing about R2's claim.** Whether a resurrection is disproportionately
  valuable *when it happens* is untouched.
- Twenty points on one trace and one geometry. `1024/1016` returns zero
  resurrections and zero overwrites — the cache holds the whole arena and
  nothing is ever evicted — so the sweep's ends are degenerate in both
  directions.
- A budget tuned to maximise resurrections-per-routed was not searched for.
