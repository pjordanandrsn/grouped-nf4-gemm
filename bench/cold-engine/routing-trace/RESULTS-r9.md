# R9's precondition, scored — the opportunity exists, and it is entirely a capacity question

Registered (`PREREG-tribrid-stage3`, R9):

> choosing between simultaneously-valid DRAM and VRAM copies by slack beats
> always taking the highest tier — **refuted if highest-tier-always ties or
> wins**.

**R9 is not scored here, and cannot be as registered.** Choosing "by slack"
needs a time-to-deadline estimate, and gate 2 established there is no
deadline estimator to supply one (`PREREG-tribrid-stage3-amendment1`).
Building a chooser in order to score a prediction about choosers would be
scoring the chooser.

What can be settled without one is whether the situation R9 describes ever
arises. The prediction is about invocations where **both** copies are valid;
if that set is empty, no policy has anything to decide, and R9 is bounded
before any chooser is written. That is the same shape as R1's uncontended
finding, where `P` collapsed to 1.000 because the event whose probability it
asked about never occurred.

Scored offline on `olmoe_routing_seq.jsonl` (512 decode steps × 16 layers ×
top-8 of 64 = 65,536 routed invocations). Receipt `r9.json`, scorer
`score_r9.py`. No GPU.

## Configuration, and why it is the only one that can produce two copies

A placement-tiered VRAM/DRAM split **cannot** produce simultaneously-valid
copies: placement makes the tiers disjoint by construction, so an expert
placed in VRAM is not in DRAM. The arrangement that does is gnf4#133's — a
`DevRowCache` in **front of** a `ColdTier`, where the device row and the
DRAM row it was filled from are two valid copies of one expert.

"Valid" is read from each side's own state, non-mutatingly, **before** the
step's requests are issued, so the counts describe what a chooser would have
seen rather than what its own request created. On the VRAM side that
includes `RECLAIMABLE` — a reclaimable row is exactly a valid copy nobody
owns, and excluding it would define R9's opportunity away.

## The opportunity is a pure function of device-cache capacity

DRAM tier fixed at 384 rows; device cache swept:

| vram rows | both valid | dram only | vram only | neither | **both rate** |
|---|---|---|---|---|---|
| 128 | 0 | 43646 | 0 | 21890 | **0.0%** |
| 224 | 0 | 43646 | 0 | 21890 | **0.0%** |
| 226 | 2653 | 40993 | 417 | 21473 | **4.0%** |
| 230 | 7667 | 35979 | 1367 | 20523 | **11.7%** |
| 232 | 10188 | 33458 | 1843 | 20047 | **15.5%** |
| 256 | 17994 | 25652 | 4204 | 17686 | **27.5%** |
| 384 | 18796 | 24850 | 4473 | 17417 | **28.7%** |
| 512 | 23560 | 20086 | 5683 | 16207 | **35.9%** |
| 1024 (full) | 37531 | 6115 | 11362 | 10528 | **57.3%** |

The four buckets sum to 65,536 at every point.

**Below ~224 rows the precondition is exactly empty.** The cache is shared
across all 16 layers, so a row placed while serving layer 0 must survive the
other 15 layers' claims — 120 further rows — before layer 0 comes round
again. Under that, nothing survives a cycle, the cache is pure churn, and
`vram only` is 0 as well: there is never a device copy at all.

Above it the rate climbs to 57.3% when the cache can hold the entire arena.

## A law I predicted and the data refuted

Seeing 0% at 224 and 27.5% at 256, I expected a clean threshold at
`2 × layers × top_k` = 256 — capacity for two full steps' routed sets, one
being filled while the previous survives. **There is no such law.** Sampling
between the two points shows a ramp beginning at 226 (4.0%) and rising
through 230 and 232, not a step. The tidy rule was an artifact of having
sampled 224 and 256 and nothing in between.

Recorded because it is exactly the kind of clean-sounding constant that
would have been repeated once written down.

## What this means for R9

The opportunity R9 depends on is **real but not intrinsic**. It is 0% or
57% on the same workload and the same trace, decided entirely by a capacity
the prediction never mentions. So R9 cannot be scored as a property of the
tiers; any answer is a property of the configuration it was measured in —
the same defect gnf4#144 found in R3, where "a budget it never pins decides
it".

Whoever builds the deadline estimator should score R9 across this axis
rather than at one capacity, and should report the capacity beside the
verdict. A single number would be meaningless.

## What this does not establish

- **Nothing about the chooser.** Whether slack beats highest-tier-always is
  untouched; only the size of the set it would choose over is measured.
- One trace, one geometry, one DRAM capacity (384). The DRAM side was not
  swept, so `dram only` and `neither` are not to be read as general.
- `protected` is held at half the cache throughout. The 240–257 region
  returns identical counts across differing capacities and protected
  budgets, which is unexplained and not relied on here.
