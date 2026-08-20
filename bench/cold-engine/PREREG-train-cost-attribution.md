# PREREG — training-step cost attribution

Filed **before the measurement**, and before any Stage-3 mechanism is ported
to the training path. Tool: `bench/cold-engine/train-attrib/attrib_train.py`
at the commit carrying this file.

## Why this runs before any porting

Stage 3's most useful result was not a speedup. It was
[`RESULTS-tribrid-gate1.md`](RESULTS-tribrid-gate1.md)'s addendum: at 1–10%
cold routing mass, **storage is 5–11% of what cold work costs** in decode.
That single number retired the hide-ratio clause (a perfect prefetcher could
not have moved the wall), explained why the prefetcher's 1% coverage was not
the binding constraint, and predicted gate 2's miss.

Porting reclaimable residency, the direct scatter and the deadline rule into
training without the equivalent number would repeat the mistake the whole
Stage-3 arc exists to have caught: optimizing a term before measuring what
fraction of the wall it is.

## The question

**What fraction of a training step is expert-weight movement?**

Everything the cold-path program optimizes — residency, landing, destination
— acts on that term and nothing else. If it is small, the program is mostly
irrelevant to training and should be said so.

## Claims under test

**T1 — the share.** Expert-weight movement is a **minority** of a training
step at ≤20% cold routing mass. Registered prediction: **<25%** of the step,
and below the 5–11% decode figure once compute-per-byte is accounted for,
because a training step does far more arithmetic per weight byte than a
decode step.

**T2 — the working set is the whole arena.** Measured in decode: a 64-token
prefill at top-8-of-64 routes **62–64 of 64 experts per layer**. A training
microbatch is larger, so registered prediction: **>95% of experts routed per
layer per step**, i.e. no cross-step locality to cache.

**T3 — forward→backward reuse is near-total.** The training path already
re-stages NVMe sub-stacks from the tier in backward ("the tier is the
recompute cache"). So a row fetched in forward should be *hit*, not re-read,
in backward. Registered prediction: **tier hit rate >60%** with the tier
sized to hold one step's routed set, against a decode baseline where hits
come from routing luck rather than algorithm.

If T3 holds, reclaimable residency has a **structural** reuse to exploit in
training rather than a statistical one, which is the strongest transfer
argument in the set — and if it fails, that argument collapses.

## Method

One model, one prompt, one placement, one box. A training step = forward +
backward + optimizer, gradient checkpointing on (the directive's default for
this path).

Arms, differing only in forced cold mass:

| arm | cold mass |
|---|---|
| control | 0% (VRAM + DRAM only) |
| cold-5 | 5% |
| cold-20 | 20% |

Attribution follows gate 1's method exactly, so the numbers are comparable:
disk time is `reads_in_window × row_bytes / B_nvme` at the box's **measured**
sequential ceiling, charged against `T_step(arm) − T_step(control)`. Using
the sequential ceiling makes the disk share a **lower bound**.

Reported: step wall and its spread, a self-pair, tier reads/hits/misses
inside the window, experts routed per layer, and the disk share of the
delta. Both raw halves; the ratio is derived.

## Stop conditions

If the hybrid training seam does not engage (patch count 0), that is
`not-engaged` and never a datapoint. If gradient checkpointing cannot be
enabled on this path, the run is reported as not measuring the intended
configuration rather than quietly measuring another one.

## What this will NOT establish

Not a speedup, not a port, not a placement recommendation. One box, one
model, one microbatch shape; instrument law 7 applies. The backward's CPU
kernel is `dgrad_nf4_grouped_cpu`, a different kernel from the forward GEMV,
so no cost constant fitted here transfers to the forward model or vice versa.
