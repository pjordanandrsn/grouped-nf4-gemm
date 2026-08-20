# RESULTS — lifting the direct-landing restriction does not pay, and #143 overstated the bias

`RESULTS-destination-ordering.md` (#143) found that the direct scatter is
what reordered the cold destinations, and closed with a claim about why that
matters:

> The fastest cold path this engine has is available only to the destination
> a scheduler is least likely to pick under GPU pressure, and a deadline
> scheduler comparing the two is comparing an optimized CPU path against an
> unoptimized GPU one.

The first half is true. **The second half is not, and this is the
measurement that says so.**

Box: RTX 5090 + AMD EPYC 9755, 442.18 GB/s triad, G0 112.4% clean
`proceed`. Same model, arena and committed trace. Receipts in
`direct-gpu-2026-08-20/`.

## The restriction, and why it could lift

A GPU-destined cold row used to read `tier.row()`, which an external-landing
tier refuses by name, so `cold_direct` was forced off for every destination
that can route to the GPU. e4b#184 made `_TieredStack` read the cold view
instead, which removes that reason: nothing on the path calls `row()` any
more (`_build_hot` reads through `_e4b_setup_tier`, and a direct view cannot
cast, so its stacks always carry the arena's own dtype and the view is never
declined).

The bar now lifts for pure-GPU **and** mixed (`deadline`, threshold,
`"auto"`) destinations. Both were measured.

## Both are null

| destination | cold | copy | direct | Δ | copy self-pair |
|---|---|---|---|---|---|
| pure GPU | 5% | 57.49 / 58.15 | 58.07 / 58.03 | **+0.4%** | 1.1% |
| pure GPU | 20% | 68.63 / 68.07 | 69.58 / 67.85 | **+0.5%** | 0.8% |
| mixed (4.0) | 5% | 57.95 / 59.89 | 58.91 / 58.25 | **−0.6%** | 3.3% |
| mixed (4.0) | 20% | 65.04 / 67.03 | 64.53 / 64.57 | **−2.3%** | 3.0% |

Every delta is inside the copy arm's own run-to-run spread. `e4b_path`
confirms `direct-scatter` was genuinely attached in all four direct arms,
and tokens match everywhere — this is a working feature that buys nothing,
not a feature that failed to engage. (That distinction is not free: the
first version of the view change *did* fail to engage, and its flat A/B read
exactly like this one. The `e4b_path` and `materializations` fields are here
so the two can be told apart.)

## Why: the view already removed the work direct would have saved

The direct landing accelerates **fills**. The view makes fills rare:

| destination | cold | materializations | view hits | hit rate |
|---|---|---|---|---|
| pure GPU | 5% | 237 | 15375 | 98.5% |
| pure GPU | 20% | 2899 | 37885 | 92.9% |
| mixed | 20% | 1277 | — | — |

At a 93–99% hit rate there is almost no fill left to accelerate, and what
remains is swamped by the GPU path's per-call gather and H2D — neither of
which direct touches. The pure-CPU path has no H2D, which is why the same
landing is worth −10.1% there and ~0% here.

## The correction to #143

The direct landing is **not** the reason a deadline scheduler compares an
optimized path against an unoptimized one. Give the GPU side the same
landing and the comparison does not move. The CPU destination's advantage at
20% cold is not plumbing it was denied — it is that the CPU path does not
pay an H2D per call.

#143's finding stands unchanged: the direct scatter is what reordered the
destinations, and `cold_direct=false` restores the published ordering
exactly. What is withdrawn is the inference drawn from it about scheduler
bias. A scheduler choosing between these destinations is choosing between
two honest measurements of two different cost structures.

## What shipped anyway, and why

The restriction is lifted in e4b (`gpu_stacks_via_view` gates it) even
though it pays nothing today. It is small, it is covered by an invariant
test, and it removes a documented limitation whose stated cause no longer
exists. It is recorded here as **null**, so nobody reads the capability as
a speedup.

A setup-time check refuses the combination rather than trusting the
argument: if a direct landing is attached for a GPU destination and any
segment's materialized dtype disagrees with the arena's, `enable_hybrid_tier`
raises with the reason, instead of failing on the first cold row.

## A skip that verified nothing

The invariant test for this originally **skipped** — the default test arena's
geometry is not scatterable, so `direct=True` was refused and the assertions
never ran while the suite read green. The same trap cost this program five
silently-skipped tests once before.

Fixed with an aligned fixture rather than a wider skip. The four per-expert
segment lengths are `INTER*H`, `INTER*H/8`, `INTER*H/2` and `INTER*H/16`
bytes, so all four are multiples of `align=4096` exactly when `INTER*H` is a
multiple of 65536 — which 256×256 is and the 64×128 default is not. The
test now fails rather than skips if direct is refused there, because that
fixture exists for no other purpose. 36 passed, 0 skipped.

## Scope

One trace, one geometry, two cold masses, threshold 4.0 as the mixed
destination. The null is specific to a regime where the view's hit rate is
high; a workload that churns its routed set hard enough to drive
materializations up would give direct more to do, and this does not measure
that.
