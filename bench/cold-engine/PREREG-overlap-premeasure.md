# PREREG — the routing-overlap pre-measurement (batch composition's go/no-go)

Registered before the statistic is computed. The e4b concentration closure
(`experts4bit-qlora/bench/hybrid-g9/concentration/`) proved placement
cannot reduce DRAM uniques further and named batch composition as the only
mechanism that changes the draw — gated on this offline pre-measurement.

## The question, made precise

Batch composition groups requests whose routed sets overlap, shrinking the
per-step DRAM-unique union that carries the 58 µs/unique decode bill. It
can only work if requests' routings are **correlated at the step level
beyond their own popularity profiles** — profile-level popularity is
already fully exploited by top-mass placement (the monotonicity proof).
So the null is *independence given per-request marginals*, and the signal
is the empirical union falling below it.

## Instrument

[routing-trace/overlap_premeasure.py](routing-trace/overlap_premeasure.py)
on the 16 committed rank traces (4 models × 4 prompts; each prompt = one
"request"). Per model:

* Pooled (layer, expert) mass over all four prompts ranks pairs; the top
  fraction f ∈ {0.50, 0.66, 0.75} is marked VRAM; the rest is the DRAM
  tail. (f = 0.66 mirrors the B=16 serving clamp: 4045 of 6144.)
* **Empirical union**: at each aligned decode step t and layer, the size
  of the union of the m requests' routed sets restricted to the DRAM
  tail; averaged over steps and layers. m ∈ {2 (all 6 pairs), 4}.
* **Independence baseline**: `Σ_{(l,e)∈DRAM} (1 − Π_i (1 − q_i(l,e)))`
  with q_i the *per-request* marginal — the same requests, the same
  profiles, correlation removed.
* The statistic: `R = empirical / independence` per (model, f, m), and
  the per-pair spread at m = 2 (does WHICH requests you group matter?).

## Registered decision bar

> Batch composition graduates to an architecture registration iff
> **R ≤ 0.90 at f = 0.66, m = 4 on at least 2 of 4 models** (≥ 10%
> unique-reduction beyond popularity). Otherwise this line closes:
> the tail draw is effectively independent and no scheduler can group
> what does not correlate.

Reported alongside, not gated: the m = 2 pair spread (composition
headroom between best and worst pairing), the f-sweep, and the µs value
of any reduction (Δuniques × 58 µs × layers per step). Same-request
temporal overlap (E1's regime) is printed for scale context only.

One computation; the bar is the bar; either verdict is the deliverable.
