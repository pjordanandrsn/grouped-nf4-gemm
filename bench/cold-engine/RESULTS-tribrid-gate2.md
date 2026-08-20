# RESULTS — Stage 3, gate 2: does choosing by deadline beat choosing by threshold?

Registered in [`PREREG-tribrid-stage3.md`](PREREG-tribrid-stage3.md) as
amended by [amendment 1](PREREG-tribrid-stage3-amendment1.md) (stamped
`0d5f9dbe…`, filed before the estimator existed). Receipts in `gate2/`.

## Verdict: **MISS** — the deadline rule loses to its own baseline

| regime | gpu | cpu | threshold (baseline) | **deadline** | vs baseline | vs best fixed |
|---|---|---|---|---|---|---|
| gpu-loaded | 49.92 | **46.02** | 49.59 | 51.39 | **+3.65%** | **+11.68%** |
| cpu-loaded | 71.11 | **66.99** | 72.46 | 77.44 | **+6.87%** | **+15.60%** |

Median decode-step ms, 128 steps, RTX 5090 + EPYC 9655 (Zen 5), 20% forced
cold mass, `hot_rows=128`. Load asymmetry created by moving the VRAM/DRAM
split (614/21 vs 102/417 experts), not by adding synthetic work, so every
arm runs the same model on the same routing.

Amendment 1 made the threshold the **baseline**, not a competitor: a
deadline model has to beat the cheap rule. It does not beat it in either
regime, and it does not beat fixed-CPU either.

## Why, in two measurements rather than a theory

**1. The backlog term almost never changes a decision.** The whole claim of
a deadline estimate over a threshold is that it responds when an engine is
busy. It did not:

| regime | decisions | flips caused by backlog |
|---|---|---|
| gpu-loaded | 1975 | **0** |
| cpu-loaded | 1976 | **58** (2.9%) |

Zero flips in the regime built to provoke them. The predicted CPU and GPU
join times are far enough apart at these shapes that adding either backlog
does not cross them — the destination is decided by the group's shape, which
is exactly what the threshold already reads, and more cheaply.

**2. The decision cost is the wall difference.** Deadline routed
essentially identically to fixed-CPU — 12066/0 against 12066/0 in
gpu-loaded, 13985/58 against 14139/0 in cpu-loaded — and was still **11.7%
and 15.6% slower**. Same routing, worse wall: the gap is the rule's own
cost, which is the up-to-three device→host syncs per call that
`_cold_to_cpu_deadline` documents. That is the stall class the CPU router
exists to remove, reintroduced at the decision point.

So the estimator pays a measured price for a decision it almost never
changes.

## What this does and does not falsify

**Falsified here**: that a backlog-aware destination rule, in this form and
at these shapes, beats a rows-per-unique threshold. It does not, on either
axis of load asymmetry.

**Not falsified**: prediction 3 (a cold expert should sometimes execute on
the slower engine because the faster one is committed). The 58 flips in the
cpu-loaded regime are that phenomenon appearing — just far too rarely, and
too cheaply approximated by shape, to pay for the machinery. The directive's
"strongest receipt in the program" would need a regime where the join times
are close enough for backlog to be the deciding term; this workload's are
not.

**A cheaper form is not ruled out.** The costed decision is only worth
making where it can change the answer. Two obvious follow-ups, neither
attempted: hoist the counts onto the CPU router's existing host-side copy so
the syncs vanish, and skip the estimate entirely when the predicted margin
is wide (a cheap shape test first, the full estimate only near the
crossover).

## The honest headline

**Fixed `cold_dest="cpu"` won both regimes.** At 20% cold mass on this box
the best policy is no policy — always use the CPU. The threshold loses to it
by 7.8% and 8.2%; the deadline rule loses by more. A scheduler is only worth
its instrumentation where the right answer actually varies, and here it
barely does.

## What this run does not establish

- One box, one model, one cold mass (20%), one workload shape. Instrument
  law 7 applies.
- The **direct landing was off for every arm**: it is incompatible with a
  GPU-capable destination (the GPU cold path reads `tier.row()`, which an
  external-landing tier refuses), so all arms took the copy path. That is a
  fair comparison — the same for every arm — but it means these absolutes
  are not the fastest the engine can serve at `cold_dest="cpu"`.
- Prefetch off throughout.
- The load asymmetry is a placement change, so the regimes differ in
  resident-set composition as well as in which engine is busier. A pure
  load sweep at fixed placement would separate those.
