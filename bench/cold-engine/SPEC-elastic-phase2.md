# SPEC — Elastic execution controller (phase 2)

A design document, not a preregistration: it commits to mechanisms and
interfaces, and defines acceptance gates whose *thresholds* are frozen in
per-gate preregistrations before each measurement. Every number below traces
to a merged receipt; every invariant traces to the measured trap that made it
necessary.

Revised on review (#203, both Bugbot findings): execution demotion when the
GPU leads — placement decoupled from residency — and the equilibrium predicate
unified with the deadband. The revisions are marked where they land.

Lineage: [`PREREG-elastic-promotion.md`](PREREG-elastic-promotion.md) →
[`RESULTS-elastic-promotion.md`](RESULTS-elastic-promotion.md) (gate E).
Authorised by gate E passing; the controller was explicitly out of scope
there and is the subject here.

## 1. Objective

Treat free VRAM as a fluid execution cache. When the CPU tier is the long
pole, execute some of its expert invocations on GPU — copy up, execute,
retain; when the GPU is the long pole or VRAM is wanted elsewhere, contract.
The target is

```
min  max(T_gpu, T_cpu, T_storage)        per decode step
```

**not** maximum GPU utilisation and **not** "always fill VRAM". VRAM capacity
becomes a performance dial rather than a binary requirement.

## 2. What gate E fixed about this design

Three planned mechanisms became unnecessary, and one number became the whole
argument:

| planned | measured outcome | consequence |
|---|---|---|
| benefit/byte selector per expert | 69–96% of invocations pay back a copy (E1); identifiers cannot beat the base rate (E2 ceiling) | **no selector — retain-on-execute** |
| tuned transient eviction policy | ordering rules sit in a ±6% band around LRU on these workloads | **eviction policy is a non-decision; use the machinery that exists** |
| reuse-probability estimation | effective n\* = 1 at measured 88% hide | **even a never-reused promotion is neutral-to-positive** |

Per 13.22 MB row, at the calibrated constants (Genoa 9J14 + 5090, matching
phase-0/2 to 0.3%):

| | µs/row |
|---|---|
| CPU grouped GEMV (134.4 GB/s) | 98.4 |
| H2D nominal (56.2 GB/s) | 235.2 |
| **H2D visible at 88% hide** | **28.2** |
| GPU GEMV (950.7 GB/s, bf16 proxy) | 13.9 |
| **promoted first use, visible** | **42.1 — 2.3× cheaper than CPU** |

Link headroom at the engine's operating points: 26–45 GB/s ≈ **2,000–3,400
promotions/s**, far above any plausible demand. **VRAM is the binding
resource, not the link** — the controller's hard problem is capacity
arbitration, not scheduling.

## 3. Architecture — two populations, one invariant

```
VRAM:  [ persistent pool ][ transient pool ][ KV / dense / reserve ]
DRAM:  every warm expert row (authoritative copy; CPU executes here)
NVMe:  cold tier (unchanged)
```

**Single source of truth: DRAM.** Promotion *copies*, never migrates; the
DRAM row remains valid. Contraction therefore comes in two forms, and only
one moves data:

* **execution contraction** — route a resident expert's invocation to CPU
  anyway. Free, immediate, per-step; no bytes move. This, not eviction, is
  how the controller rebalances when the GPU is the long pole (§5 — revised:
  the first draft only stopped promoting, and since retain-on-execute never
  evicts without insert pressure, an over-promoted pool would have kept
  feeding the GPU forever).
* **capacity contraction** — actually freeing rows: pressure-driven,
  event-gated, `shrink()` (§6). Idle resident rows cost capacity, never wall.

No write-back, no invalidation protocol in either direction.

| component | exists today | phase-2 change |
|---|---|---|
| persistent pool | `DevRowCache`/`VramSlots` retention + `hot_ids` placement | slow resize only (§5); LFU-style promotion from transient (the one policy lever that moves transfers: 43/44/22/2% of gap by model) |
| transient pool | — | a second `DevRowCache` instance; same states, same event gating, same `protected = rows_t − k` |
| copy path | pinned staging + side-stream H2D (the 88% hide *is* this path) | promotion enqueue; copy-event gate before first GPU use |
| CPU tier | phase-2 grouped GEMV kernel | executor may re-route an invocation to GPU after promotion lands |
| controller | — | **new, lands in e4b** (§10); gnf4 exposes counters + `shrink()` |

The transient pool reuses `DevRowCache` unchanged because the margin rule is
load-bearing everywhere: demotable margin `rows − protected` must equal `k` —
margin > k thrashes totally on cyclic patterns (measured: 4,096 fills for a
64-key static set), margin < k cannot serve an all-miss step. **One invariant,
both pools.** At transient's generous capacities the eviction policy is in
the measured don't-care band; no new allocator states are introduced.

## 4. The promotion path

1. Executor assigns the step's invocations: VRAM-resident → GPU **by
   default**, else CPU — and the controller may override the default,
   directing resident invocations to CPU when the GPU is the long pole.
   **Placement is the actuator; residency follows it** (revised — a hard
   residency→placement binding deadlocked the min-max loop in the
   GPU-long-pole direction).
2. Controller (§5) picks `p ≥ 0` CPU-assigned invocations to promote — any
   `p` of them; there is no ranking to compute (E2).
3. Each promoted row: pinned-staging H2D on the **side stream**; the row
   enters the transient pool ACTIVE under the standard event gate. The GPU
   may execute it only after the copy lands: the 88% hide was measured with
   an unsynchronised compute stream, and a premature sync destroys exactly
   the property the economics rest on. **Amended after G1
   ([RESULTS-p2-g1.md](RESULTS-p2-g1.md)) — dispatch is per BURST, not per
   row, and stream-ordered:** the step's `p` promotions are one `want()`
   call, per-row copies on the side stream, **one** event recorded after the
   last copy; slot ids go to the GPU **pre-staged device-side** (pinned int32
   staging, async copy — never a python list at launch: 81.5 µs vs 20.9 µs
   measured); and the burst's GPU execution is **enqueued, gated on the burst
   event, before the step's CPU execution begins**, so the host never waits
   between tiers. G1 measured the un-amended form losing its entire margin to
   serial dispatch (~100 µs clean, ~400+ µs under the CPU pool's spin
   window, against a 34.8 µs/row saving).
4. This step's instance executes wherever it lands first: if the copy event
   has not fired by the time the executor reaches that invocation, it runs on
   CPU as originally assigned and the row is simply resident for next time.
   **Promotion may never stall a step** — a stalled promotion inverts n\* = 1
   back into the synchronous-streaming regime phase 0 showed loses.
5. Retain-on-execute: the row stays until evicted by transient churn, a
   `shrink()` call, or promotion into the persistent pool (§5).

## 5. The controller

Observables (per step, EWMA α ≈ 1/16): `t_cpu`, `t_gpu` (executor timings),
`t_storage` (ColdTier read wall), `link_used` (rows moved × row bytes),
`vram_free` (allocator), transient hit counters (already in `stats()`).

Actuators: `p` (promotions this step), transient grow/shrink, persistent
resize (slow), and nothing else.

Moving one invocation between tiers changes **both** walls — CPU by
`t_cpu_row`, GPU by `t_gpu_row` — so both actuators solve the one-step linear
model for the **nearest band edge**. Revised wholesale in round 4: three
review rounds each found a drift between the branch guards, the pin
predicates and the calibration constraint, because the same quantity was
written in three places. Each is now defined **once** and used everywhere —
by the loop, by the equilibrium predicate, and by G2.

```
avail_up()   = min(cpu_queue, slack_rows(), GROW_CAP, slots_obtainable(),
                   LINK_CAP())
LINK_CAP()   = floor(cpu_queue · t_cpu_row / (t_cpu_row + t_link_row))
               # G1 amendment: a burst's copies must fit under the CPU work
               # that REMAINS after the burst is carved out — p rows of copies
               # hide under (cpu_queue − p) rows of CPU, which solves to this
               # cap. G1 measured the violation: at p = 8, 1354 µs of copies
               # under 371 µs of CPU.
avail_down() = min(resident_invocations, DEMOTE_CAP)

slots_obtainable() = transient slots free now, growable without breaching
                     RESERVE, or reclaimable-now occupied transient slots
                     (evict-on-insert; event-gated — a slot still under
                     readers does not count this step)

each step:
  if pressure():                     # KV growth or reserve breach
      shrink_transient(deficit)      # event-gated; completes this step (I3)
  if t_cpu > (1 + δ_hi) · t_gpu:                       # above band
      if avail_up() == 0:  hold                        # pin_up
      else:
          p* = (t_cpu − (1 + δ_hi)·t_gpu) / (t_cpu_row + (1 + δ_hi)·t_gpu_row)
          p  = min(ceil(p*), avail_up())               # ceil ≥ 1 out of band;
                                                       # min() cannot invert
          if p · (t_cpu_row − t_gpu_row) ≤ C_disp:     # G1 amendment: the
              hold                                     # dispatch floor — a
                                                       # burst below it loses
  elif t_cpu < (1 − δ_lo) · t_gpu:                     # below band
      if avail_down() == 0:  hold                      # pin_down
      else:
          q* = ((1 − δ_lo)·t_gpu − t_cpu) / (t_cpu_row + (1 − δ_lo)·t_gpu_row)
          q  = min(ceil(q*), avail_down())             # execution contraction:
                                                       # free, no bytes move
  else:  hold                                          # in band
```

Round-4 corrections folded in: `clamp(·, 1, avail)` inverted its bounds at
zero availability and the promote guard entered on VRAM alone while
`slack_rows() = 0` — availability is now checked **before** sizing, with one
`avail_*()` definition per direction, and a full transient pool no longer
reads as exhaustion (evict-on-insert churn is a legal, paying promotion; only
an empty pool with growth barred by RESERVE exhausts the direction).

every PERIOD (≈256) steps:
  persistent ← LFU-promote transient rows resident ≥ 2·PERIOD   (hysteresis)
  persistent shrink iff per-row hit rate < θ for 2 consecutive periods
```

Hysteresis is two-sided (`δ_hi > δ_lo`, promote-age ≥ 2·PERIOD, shrink needs
2 periods) because the persistent pool must move slowly — continuously
evicting and reloading it would destroy the retention benefit that is the one
measured win. The transient pool is the fast half by construction.

Equilibrium — **the same predicate as the hold band** (revised: the first
draft defined equilibrium at δ_lo while the controller held anywhere inside
δ_hi, so it could hold forever without ever "converging" by G2's measure):

```
in_band      ≜  (1 − δ_lo) · t_gpu ≤ t_cpu ≤ (1 + δ_hi) · t_gpu
pin_up       ≜  t_cpu above band  and  avail_up()  = 0
pin_down     ≜  t_cpu below band  and  avail_down() = 0
equilibrium  ≜  in_band  or  pin_up  or  pin_down
```

The pins reuse the loop's own `avail_*()` functions — the same expressions
that decide whether the controller acts decide whether G2 calls the state
converged, so the two cannot drift again (rounds 2–4 were three instances of
exactly that drift). `δ_lo`/`δ_hi` are the band's hysteresis edges, not a second, tighter target.

**The retention window is emergent, not configured.** A transient pool of
`rows_t` at promotion rate `r` holds a row for `W ≈ rows_t / r` steps; E1's
sweep maps W to payback capture (W=8: 20–71%, W=32: 69–96%, W=128: 98–99%).
Sizing follows from free VRAM; no timer exists to mistune.

## 6. VRAM accounting and the KV interface

* `RESERVE`: a standing low-water mark (initial: 512 MB) so allocation never
  blocks on reclaim.
* gnf4 exposes `shrink(bytes) → bytes_freed`, transient-first, event-gated —
  rows under in-flight readers retire through the existing RETIRING path, so
  freeing is safe by the same argument the residency engine already makes.
* Persistent shrink is only via the slow path; a pressure spike that exceeds
  the whole transient pool degrades to `p = 0` + persistent slow-shrink, and
  the engine is then exactly the shipped one (§8, OFF mode).
* e4b owns KV and calls `shrink()`; gnf4 never inspects KV.

## 7. Invariants

| # | invariant | measured trap it guards |
|---|---|---|
| I1 | both pools: `protected = rows − k`, margin exactly k | margin > k: total thrash (6,144 fills for a 96-key static set); margin < k: unservable all-miss step |
| I2 | copies on the side stream; compute never syncs it; first GPU use gated on the copy event | the 88% hide; premature sync reverts to synchronous streaming, which loses 2.6× |
| I3 | `shrink()` satisfiable from transient alone within one step | KV growth must never OOM against cached weights |
| I4 | promotion never stalls a step (CPU fallback always live) | a stalled promotion is synchronous streaming |
| I5 | DRAM row remains authoritative; VRAM rows are caches | free contraction; no write-back protocol to get wrong |
| I6 | controller OFF ⇒ bit-for-bit the shipped engine | A/B-ability; the `dev_cache=None` precedent |
| I7 | promotion changes *where* an invocation executes, never its result | token-identical A/B is the acceptance test (R10 precedent: 19/19 arms) |
| I8 | per-box calibration before enabling (§9); no baked-in constants | the environment drift result: traces did not reproduce across transformers versions; bandwidths will not either |
| I9 | promotions dispatch as one stream-ordered burst: single `want()`, one burst event, pre-staged device ids, GPU work enqueued before the CPU tier starts — the host never waits between tiers | G1: serial per-row dispatch (~100–500 µs/step) buried the 34.8 µs/row saving at every swept p |

## 8. Degraded modes

* **Low-reuse model (Qwen-class, 71% at W=32):** retain-on-execute captures
  less amortisation but n\* = 1 makes single-use promotion neutral-positive
  while hidden; worst case the transient pool is pure execution staging and
  still beats CPU 2.3× visible.
* **Hide collapses** (CPU queue empties — nothing to hide under): the balance
  condition self-limits, since promotion only runs while `t_cpu > t_gpu`.
* **Link saturated** (residency refills at tight capacity): `slack_rows()`
  goes to zero and promotion pauses; residency traffic has priority. Within
  a step, `LINK_CAP()` (§5) separately keeps a burst's own copies inside the
  CPU window they must hide under.
* **VRAM exhausted:** transient → 0, `p = 0`, engine ≡ shipped engine.
* **Controller misbehaving:** OFF switch (I6).

## 9. Calibration

`elastic_e3.py` is the startup probe, run once per box class: emits
`(B_cpu, B_link, B_gpu, hide)` with the thread sweep embedded, the derived
`n*` and per-row µs table, and — G1 amendment — **`C_disp`**: the measured
serial host cost of one amended burst (enqueue + one event + pre-staged-id
launch + sync, warm, idle GPU). `C_disp` feeds the dispatch floor (§5); the
feasible burst window on a box is `[p_min, LINK_CAP]` with
`p_min = min p : p·(t_cpu_row − t_gpu_row) > C_disp`, and a box whose window
is empty at the engine's `m` fails calibration for promotion exactly as an
out-of-range `n*` does. Controller thresholds (`δ_hi`, `δ_lo`,
`GROW_CAP`, `DEMOTE_CAP`, `RESERVE`) are computed from it, not baked in (I8) —
including the constraint that makes the integer actuators sound (corrected in
round 4 — the round-3 form used the raw quantum and missed that the far edge
itself moves): a step from just outside either edge must land inside, i.e.

```
demote:  (δ_lo + δ_hi) · t_gpu ≥ t_cpu_row + (1 + δ_hi) · t_gpu_row
promote: (δ_lo + δ_hi) · t_gpu ≥ t_cpu_row + (1 − δ_lo) · t_gpu_row
```

The demote inequality dominates and is the calibration requirement. It is
what lets a ceiling-sized step land inside the band rather than across it,
and what makes G2's ≤ 1-flip-per-32 bound meaningful rather than lucky. A box where
`n*_direct` falls outside [2, 5] un-hidden fails calibration and the
controller stays OFF — the E3 gate, kept as a runtime guard.

## 10. Repo split and sequencing

**gnf4** (this repo): transient pool instantiation; promotion copy path;
`shrink()`; counters; calibration probe; the P2-G1 A/B harness (extend the
`run_r2_wall` pattern — its measurement-boundary and counter-whitelist fixes
carry three Bugbot findings and must be imported, not copied).

**e4b** (the executor repo, runtime half): step timing, the controller
object, KV pressure calls, OFF switch plumbing.

Order: G1 mechanics (gnf4, one box) → controller in replay (e4b, offline,
driving captured routing) → G2/G3 closed-loop (one box) → G4 end-to-end.

## 11. Acceptance gates

Each gate gets its own preregistration freezing exact thresholds **before its
measurement**; the formulas below are the registered *shape*. All wall gates
are paired A/B on the same box, and every A/B must be token-identical (I7)
before any wall number is read.

* **P2-G1 (mechanics):** fixed `p`, CPU forced long-pole. Realised saving per
  promoted row ≥ **70%** of predicted
  `Δ = t_cpu_row − (1−hide)·t_link_row − t_gpu_row` from that box's
  calibration. Refuted ⇒ the promotion path has overheads the model missed;
  the controller is not built until the mechanics pay.
  **Outcome: REFUTED 2026-08-23** ([RESULTS-p2-g1.md](RESULTS-p2-g1.md)) —
  serial dispatch and the link budget, both now in the model. Superseded by
  G1b; the controller stays unbuilt until G1b passes.
* **P2-G1b (mechanics, amended):** the §4 burst mechanics on the same A/B
  design. Bar: realised step saving `wall_A − wall_B` ≥ **70%** of
  `p·Δ − C_disp` at **every p inside the calibrated feasible window**
  `[p_min, LINK_CAP]`; p outside the window is swept and reported (the floor
  and cap predict those lose — a win there indicts the model, not the gate).
  Two registered spoilers, both must fail their bar: synchronous copies
  (hide), and the **un-amended G1 mechanics verbatim** (dispatch) — the
  refuted configuration must stay refuted while the amended one pays.
  **Outcome 2026-08-23: window empty at m = 16**
  ([RESULTS-p2-g1b.md](RESULTS-p2-g1b.md)) — C_disp = 94.1 + 19.8·p µs
  against a 40.7 µs/row gain puts the dispatch floor (p ≥ 5) above
  LINK_CAP(16) = 3. No wall claimed; the boundary map prices the exits.
* **P2-G1c (mechanics, engine-regime m):** identical amended mechanics and
  bar, at the step-level queue the controller actually sees — `m = 128`
  (layers × k for a 16-layer top-8 shape), the G1b boundary map's own
  prediction of an open window `{5 … 30}` on the measured box class. Sweep
  spans the predicted window plus below-floor and at-cap points; same two
  spoilers. G1b's m = 16 finding stands — G1c registers the regime change
  from G1b's calibration BEFORE any new measurement, not after a miss inside
  a run.
  **Outcome 2026-08-23: REFUTED at every in-window p**
  ([RESULTS-p2-g1c.md](RESULTS-p2-g1c.md)) — not dispatch: the H2D copies
  and the CPU tier draw one host-DRAM budget (171.8 + 52.7 > 212 GB/s), so
  host traffic is p-invariant and no-reuse promotion cannot pay on a
  DRAM-bound tier at any m or glue. The E3b hide instrument sampled a
  lighter load (132.5 + 52.6 < 212) and honestly read hide = 1.0 there —
  hide is load-dependent. **Consequence for this spec:** the transient
  n* = 1 execution-staging leg is refuted; the promote actuator must be
  re-founded on reuse-based residency (E1's 69–96% paying set) with a
  DRAM-headroom budget (`B_cpu_load + B_link ≤ B_dram`, calibrated at the
  tier's operating load), and §5's balance law re-derived on that objective
  before G2 is registered. G2–G4 are NOT run against the refuted objective.
* **P2-G2 (convergence):** from cold start on captured routing, reach
  equilibrium (§5) within **64 steps**; after convergence, direction flips ≤
  1 per 32 steps. Refuted ⇒ the hysteresis is wrong, not tuned live.
* **P2-G3 (elasticity):** mid-run VRAM ballast injection: no OOM; transient
  shrinks within 2 steps; wall degrades monotonically (no cliff); recovery
  within 64 steps of release.
* **P2-G4 (the objective):** step wall ≤ **1.15 ×** `max(T_cpu, T_gpu,
  T_storage)` at equilibrium — overlap realised — against a
  sequential-sum baseline demonstrating the gap.

Falsifiability preconditions apply to every gate as in gates A–E: a
configuration must exist in which each prediction fails (e.g., G1 with the
side stream deliberately synced must miss its bar; G3 with shrink disabled
must OOM or cliff), demonstrated before the real measurement is read.

## 12. Out of scope

Multi-GPU (the phase-0 box has two 5090s; one pool, one link is phase 2 —
splitting populations across devices is phase 3). NVMe policy changes.
Training-path interaction. Cross-request KV eviction policy (e4b's, entirely).
The near-miss band (still unused, still a separate registered candidate).

## 13. Risks

| risk | why it is believed small | falsifier |
|---|---|---|
| DRAM contention between CPU GEMV and H2D staging | measured jointly: 7.8% completion inflation while absorbing a full copy stream | P2-G1 realised/predicted ratio |
| controller oscillation under bursty routing | deadband + rate cap + slow persistent half | P2-G2 flip count |
| transient pool churn evicting rows mid-read | same event-gated states as the shipped engine | I2/I3 + existing kernel tests |
| constants drift across boxes | calibration is mandatory and gating (I8) | E3 guard at startup |
| Qwen-class low reuse wastes VRAM | n\* = 1 makes waste ≈ 0 wall; capacity returns via shrink | G3 under a low-reuse trace |
