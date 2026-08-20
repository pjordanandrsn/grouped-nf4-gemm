# R2's premise, scored — the 2–5% it is conditioned on never arrives

Registered (`PREREG-tribrid-stage3`, R2):

> VRAM resurrection is disproportionately valuable — **even 2–5% of routed
> invocations moves wall time** — refuted by *no measurable wall effect*.

**R2 is not scored here.** Its verdict is a wall measurement, and the wall
half needs an MXFP4 model this program does not have: the device arena lives
in `Mxfp4NvmeResidency`, and no gpt-oss/K3-lineage arena has been baked in
this repo.

But the clause carries a quantity that can be checked without one. R2 is a
conditional — *given* resurrections reach 2–5% of routed invocations, wall
time moves. If the antecedent never holds on a workload, the conditional is
vacuous there whatever a chooser does.

Scored offline on `olmoe_routing_seq.jsonl` (512 decode steps × 16 layers ×
top-8 of 64 = 65,536 routed invocations) through `DevRowCache`/`VramSlots`.
Receipt `r2.json`, scorer `score_r2.py`. No GPU.

## It peaks at 1.30%

| rows | protected | resurrections | **per routed (R2)** | per resolved eviction (R3) |
|---|---|---|---|---|
| 128 | 32 / 64 / 96 | 0 | **0.00%** | 0.0% |
| 256 | 128 | 0 | **0.00%** | 0.0% |
| 256 | 192 | 458 | **0.70%** | 1.1% |
| 384 | 192 | 458 | **0.70%** | 1.1% |
| 384 | 288 | 853 | **1.30%** | 2.5% |
| 512 | 256 | 577 | **0.88%** | 1.6% |
| 512 | 384 | 683 | **1.04%** | 2.6% |
| 1024 | 512 | 782 | **1.19%** | 4.6% |
| 1024 | 768 | 134 | **0.20%** | 5.1% |

**The best configuration measured reaches 1.30%**, against a floor of 2%.
Most reach exactly zero. So on this workload R2's antecedent does not occur,
and no wall measurement — on any hardware — could score the prediction as
written.

## The two denominators, side by side

The last row is the cleanest warning in the table. At 1024 rows with 768
protected, resurrections are **5.1% of resolved logical evictions** and
**0.20% of routed invocations** — a factor of 25 between two numbers that
both get called "the resurrection rate".

`RESULTS-r3.md` measures the first and showed it swings 0.0–33.9% on a
budget R3 never pins. R2 names the second. A tier can resurrect nearly every
row it logically evicts while resurrections remain a rounding error against
the work the model actually does, and that is what happens here.

## Three predictions, one structural defect

R2 joins R3 and R9. Each names a rate without pinning the capacity or budget
that decides it:

| | quantity | what actually decides it | verdict |
|---|---|---|---|
| R3 | resurrections / resolved evictions | protected budget (0.0–33.9%) | undetermined (#144) |
| R9 | invocations with two valid copies | device-cache capacity (0–57.3%) | bounded, not scored (#150) |
| R2 | resurrections / routed invocations | rows **and** budget (0.00–1.30%) | premise not reached |

This is worth carrying into the next preregistration. A prediction of the
form *"rate X falls in band Y"* is only scoreable if the configuration that
sets X is fixed as part of the registration. All three of these were
registered as properties of the mechanism and turned out to be properties of
a knob.

## What this does not establish

- **Nothing about R2's actual claim.** Whether a resurrection is
  disproportionately valuable *when it happens* is untouched; only how often
  it happens is measured. A workload that did reach 2–5% could still confirm
  or refute the wall effect.
- One trace, one geometry, the DevRowCache/VramSlots path. A burstier
  routing pattern — which the prereg itself names as where gains should be
  largest — would plausibly push the rate up, and this does not measure that.
- The sweep is `rows × {¼, ½, ¾}` budgets. A budget tuned specifically to
  maximise resurrections-per-routed was not searched for; 1.30% is the best
  of fifteen points, not a proven ceiling.
