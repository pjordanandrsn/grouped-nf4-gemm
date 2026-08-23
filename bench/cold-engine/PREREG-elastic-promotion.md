# Preregistration — elastic GPU promotion, gate E: does the paying set exist, and is it identifiable online?

Registered before measurement. The reuse-count analysis below has not been
run on any trace, and no promotion microbenchmark exists.

## The design this gates

The elastic-saturation design: treat free VRAM as a fluid execution cache —
when the CPU tier is the long pole, opportunistically copy a DRAM-resident
expert up, execute it on GPU, and **retain it if expected reuse justifies the
copy**; contract when VRAM is worth more elsewhere. The target is
`min max(T_GPU, T_CPU, T_storage)`, with a slow-moving persistent residency
and an aggressive transient one.

That is a controller, and controllers are built last. This registers the two
questions everything else depends on, plus the microbenchmark that checks the
constants transfer.

## What is already measured, and what it implies

[`RESULTS-hybrid-phase0.md`](RESULTS-hybrid-phase0.md), Genoa 9654P + 2×5090:
DRAM grouped reads 264.3 GB/s, H2D 52.3–56.0 GB/s, VRAM 1572 GB/s. Its own
words: *"An expert computed in place beats an expert streamed to the GPU
whenever CPU kernel efficiency exceeds ~21% of the DRAM ceiling."*
[`RESULTS-hybrid-phase2.md`](RESULTS-hybrid-phase2.md) measured the CPU GEMV
at **134.0 GB/s** — 55.5%, comfortably above 21%. **Synchronous streaming
loses. Promotion can only pay through reuse or overlap**, which is exactly
the refinement the design proposes. The arithmetic, from those constants:

| CPU tier | n\* — total invocations for promote-and-retain to beat CPU-in-place |
|---|---|
| measured, 134.0 GB/s | **2.62 – 2.80** (best–worst measured link) |
| if the named G2 fix path lands 70%, 194.8 GB/s | 3.97 – 4.25 |
| H2D fully hidden under other work | **→ 1**, and GPU execution is 11.7× CPU per byte |

Link budget for the copies exists: at the wall harness's operating points the
residency engine uses 7.3–26.5 GB/s of the 52–56 GB/s link
([`RESULTS-wall-real-routing.md`](RESULTS-wall-real-routing.md) fills/step ×
13.22 MB rows), leaving **~26–45 GB/s of slack**.

And the identification signal is the one this program just measured:
[`RESULTS-router-rank.md`](RESULTS-router-rank.md) found rank predicts
**whether** an expert returns (monotone on all four models) and that eviction
cannot spend it, because eviction needs **when**. Promotion needs *whether and
how often* — the question rank actually answers. The 16 rank traces record
per-visit ranks for every selection, so both questions below are computable
offline from committed data.

## E1 — the paying set is non-trivial (offline)

> On **all four models**, at least **25%** of per-layer expert invocations
> are by experts that recur **≥ 3 more times within the following 32 steps**
> at that layer (n\* = 3, the measured-CPU break-even rounded up).

* Window swept at W ∈ {8, 32, 128}; the gate applies at **W = 32** and the
  others are reported. Sensitivity registered now: the fraction clearing
  **≥ 4** (the post-G2-fix break-even) is reported alongside, so the
  conclusion's dependence on the CPU fix landing is visible, not discovered.
* **Refuted** if any model falls under 25% at W = 32 — the paying set is too
  thin and elastic promotion dies here, whatever the controller design.

## E2 — the paying set is identifiable at selection time (offline)

Two identifiers, both registered now, both scored, both reported — not
whichever wins:

> **E2a (rank):** invocations whose expert was selected at **rank 1** clear
> E1's test at **≥ 1.25×** the all-invocations base rate, on all four models.
> **E2b (frequency):** invocations whose expert is in the top quartile by
> trailing 32-step selection count clear it at ≥ 1.25×, on all four models.

* E2 is **confirmed** if either identifier clears on all four; both results
  are reported regardless.
* **Refuted** if neither does — the paying set exists but cannot be found
  online, and promotion falls back to profile-time placement, which is gate
  3's territory, not a runtime controller's.

## E3 — the constants transfer (one box, phase-0 class)

> Measured directly on a Genoa-class + 5090 box — per-row T_CPU (the phase-2
> kernel), T_H2D (pinned), T_GPU (GEMV) at the 13.22 MB row size — the
> directly-computed break-even **n\*_direct lands in [2, 5]**.

* **E3b, reported and not gated:** hideability — the fraction of an H2D's
  wall hidden when issued on a side stream during concurrent CPU GEMV of
  other experts. This sets where between 2.6 and 1 the effective n\* sits,
  and is the number the controller design would be built on.
* E3 refuted ⇒ the model's constants do not transfer to the measured box and
  E1/E2's economic framing is re-derived before anything is built.

## Preconditions, unchanged from the last four preregistrations

**Falsifiability, checked before scoring:** a synthetic no-reuse trace must
fail E1 at 0%, and a rank-shuffled copy of a real trace must push E2a to
~1.0×. If either prediction cannot fail, it is reported as uninformative.

**Harness validation:** on a constructed trace with known reuse counts the
analysis must reproduce them exactly; windows must not cross the trace end
silently (the last W steps are excluded from the denominator, not counted as
non-recurring).

## What would count as a miss

* E1 fails ⇒ reported first and plainly: no controller, no phase 2.
* E1 passes, E2 fails ⇒ promotion is a placement-time story, not a runtime
  one; reported as such.
* E3 out of band ⇒ constants do not transfer; the offline verdicts stand as
  trace facts but the economics are re-derived.
* The box is destroyed when E3 ends; receipts are committed before it.

## Out of scope, registered so it cannot creep

The controller itself — the two residency populations, hysteresis, the
`min max` feedback loop, KV elasticity — and any e4b executor integration.
This slice decides whether they are worth building. The near-miss band stays
unused here too; it is a separate registered candidate.
