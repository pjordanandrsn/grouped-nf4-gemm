# RESULTS — why the optimal cold destination flipped, and what actually moved

`RESULTS-tribrid-gate1.md`'s re-measurement recorded an unexplained
difference at 20% cold mass: cold-CPU beat cold-GPU, where the published run
had the reverse. It was filed as *"either run-to-run variation at the
noisiest point or prediction 4's territory (the optimal destination flips
with load) — recorded as an observation, not a scored result."*

**It is neither.** Nothing about the machine or the load flipped it. One of
our own optimizations did, and it only helps one destination.

Box: RTX 5090 + AMD EPYC 9655, 417.39 GB/s triad, G0 122.4% clean `proceed`.
Same model, arena and committed routing trace as gate 1. Receipts in
`dest-ordering-2026-08-20/`.

## Only one arm moved

| arm | published (original box) | measured now | change |
|---|---|---|---|
| cold-GPU | 64.22 ms | 62.45 / 64.05 ms | **unchanged** |
| cold-CPU | 69.55 ms | 57.66 / 57.57 ms | **−17%** |

The GPU path is exactly where it was. Calling this a "flip" was the wrong
frame: the ordering changed because one side got faster, not because the
trade-off moved.

## Two explanations tested and refuted

**The #112 self-ensure fix — refuted.** That bug lived in `ColdCpuView`,
which is the CPU destination's own path, so it was the obvious candidate.
Measured by running the pre-fix tree (`0a10eab`) and the fixed tree
**adjacently**, two rounds, so both see whatever the host's other tenants
are doing:

| tree | cold-CPU | cold-GPU |
|---|---|---|
| pre-#112 | 57.87 / 58.18 | 61.13 / 62.06 |
| fixed | 57.66 / 57.57 | 62.45 / 64.05 |

The CPU path is unchanged across the fix (0.7% apart), and cold-CPU wins on
**both** trees. #112 has nothing to do with it.

**The host — refuted.** `b_link` h2d_64mb is **28.47 GB/s on both boxes**,
identical to two decimals, so the GPU side of the trade is not what differs.
DRAM triad differs by 9.8% (380.1 → 417.39), nowhere near enough to turn a
7.7% GPU win into a 19% CPU win.

## What did it: the direct scatter, which only the CPU destination can use

`cold_dest="cpu"` defaults to `cold_direct=True`; the published gate-1 run
predates that landing being wired. Toggling it on the same box, 20% cold,
two rounds each:

| `cold_direct` | median | disk reads |
|---|---|---|
| false | 64.61 / 64.96 ms | 450 |
| **true** | **58.48 / 58.04 ms** | 450 |

**−10.1%**, with reads identical — a staging saving, not an I/O one.

**And with it off, the published ordering returns.** cold-CPU at 64.8 ms
against cold-GPU's 63.3 ms: cold-GPU wins, as published. The remaining
69.55 → 64.8 gap is box and intervening changes; the direct landing is the
part that reorders the destinations.

## Why this matters more than a corrected table

The direct scatter is **destination-asymmetric by construction**. An
external landing cannot serve `tier.row()`, and the GPU cold path reads
exactly that, so `enable_hybrid_tier` forces `cold_direct` off for any
destination that can route a row to the GPU — pure `"gpu"`, a threshold,
`"deadline"`, `"auto"`. Every one of those runs the copy path.

So the fastest cold path this engine has is **available only to the
destination the scheduler is least likely to pick under GPU pressure**, and
a deadline scheduler comparing the two is comparing an optimized CPU path
against an unoptimized GPU one. That is a scheduling bias baked into
plumbing, not a property of the hardware.

It also re-prices a follow-up already on record. e4b's
`gpu_stacks_via_view` (−5.9% at 5% cold, −12.1% at 20%) lets a GPU-destined
stack read the cold view instead of `tier.row()`. That removes the stated
reason for the restriction, which makes "direct scatter for GPU
destinations" the next thing worth measuring rather than a curiosity.

## Discarded, and why

A first attempt at the #112 comparison is **not** in this directory. Its
self-pair read 43.13 vs 48.99 ms — **13.6% disagreement** against the
0.4–1.0% every clean run here shows — because the host was carrying a load
average of 17.9 with none of our processes running. Read at face value it
looked like a clean refutation with cold-CPU at 57.75 ms. The number was
plausible and probably even directionally right; it is dropped because the
instrument disagreed with itself by more than the effect being measured.
The paired design above replaced it.

## Scope

One trace, one geometry, 20% cold mass, `order="tail"`. The −10.1% direct
figure is this box's; gnf4#122 measured −12.5% e2e at 20% cold on another,
which is the same claim at the same magnitude. Nothing here re-scores a
gate: gate 1's verdicts stand exactly as re-measured.
