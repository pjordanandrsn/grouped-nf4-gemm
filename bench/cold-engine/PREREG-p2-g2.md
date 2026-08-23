# PREREG — P2-G2: residency convergence of the reuse law, in replay

Registered before any measurement. The bars are those frozen in
[SPEC-elastic-phase2.md](SPEC-elastic-phase2.md) §11 (merged #211, before
this harness ran anywhere); this document fixes the replay parameters and
the exact aggregation. Offline on the 16 committed rank traces
(`routing-trace/*_{code,dialogue,math,prose}.jsonl`), cold start, trace
units — no box, so no box variance touches the gate.

## The instrument

[routing-trace/g2_replay.py](routing-trace/g2_replay.py) drives the **real**
`DevRowCache` (the score-the-shipped-thing rule): every routed set goes
through `want()` per layer exactly as the engine's flow does; the SMOOTH_CAP
budget uses the cache's own failed-fill API (`discard()`) for un-budgeted
misses, so throttling exercises shipped semantics. The controller's own
bookkeeping (resident ages via re-fill detection, per-PERIOD hit counters,
the persistent set) is external — that bookkeeping IS the law under test.

**Registered parameters:** total capacity sweep rows ∈ {128, 256, 512, 1024}
(per trace, capped at its distinct (layer, expert) pairs); persistent cap =
rows/4, transient = the rest; PERIOD = 64 and promote-age ≥ 128
(trace-scaled — the spec's ≈256 serving default cannot cross a boundary
inside a 512-step trace; I8 makes PERIOD calibratable, and this registration
is that calibration for 512-step traces); θ = zero hits for 2 consecutive
periods; η = 0.25; PROMO_FRAC arms: unthrottled, 1/16, 1/8, 1/4; EWMA
α = 1/16; plateau window steps 256–512; convergence = first step whose
trailing-32 mean fill rate ≤ 1.10 × plateau + 1.0 (the +1.0 guards
near-zero plateaus); eval window for fills: steps 128–512.

## Registered claims and exact aggregation

* **(a) Convergence** (unthrottled): a trace **passes** if it converges
  ≤ 64 steps at *every* swept capacity ≤ its pairs. Bar: **≥ 14 of 16**
  traces pass.
* **(b) Plateau quality** (unthrottled): total fills over steps 128–512 ≤
  **1.10 ×** ideal-LRU fills over the same steps at the same total
  capacity, at **every** (trace, capacity) arm.
* **(c) Equilibrium churn** (unthrottled, largest applicable capacity per
  trace): steady-state `EWMA(fills) ≤ (1 + η) · EWMA(novelty) + 1.0` on
  **every** trace. (Smaller capacities legitimately re-fill on capacity
  misses; §5's equilibrium is a capacity-adequate property, so it is gated
  only where capacity is adequate.)
* **(d) Throttle gracefulness**: at every (trace, capacity) arm and every
  PROMO_FRAC ∈ {1/16, 1/8, 1/4}: convergence ≤ 2 × max(unthrottled, 16)
  steps AND eval-window fills ≤ 1.05 × unthrottled.

Verdict: PASS iff (a) ∧ (b) ∧ (c) ∧ (d). Refuted at any clause ⇒ per-clause
report; the law's budgets/hysteresis are wrong and get fixed in spec, not
tuned against the traces.

## Falsifiability — both spoilers must fail, before the claims are read

1. **The I1 margin trap**: `protected = rows_t − 1` (margin 1 against
   routed sets of k = 8). Must blow the (c) bound (thrash — the measured
   6,144-fills-for-96-keys failure mode) or the (b) bound at its capacity.
2. **No-retention**: every fill counted then immediately `discard()`ed —
   residency never forms. Its plateau must sit at ≥ 0.90 × all-miss
   (m fills/step), demonstrating retention is what pays.

Spoilers run at each trace's largest applicable capacity, unthrottled.

## What would count as a miss

* Any clause refuted ⇒ REFUTED with per-clause, per-trace detail; no spec
  tuning against these traces — the fix is derived, re-registered, re-run.
* Either spoiler passing its own must-fail condition ⇒ UNINFORMATIVE.
* A trace whose pairs < 128 (sweep floor) is reported and excluded from
  (a)–(d) denominators; if more than 2 traces are excluded the run is
  UNINFORMATIVE (the sweep floor was mis-registered).
