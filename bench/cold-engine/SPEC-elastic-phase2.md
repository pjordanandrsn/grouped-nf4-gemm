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
3. Each promoted row: pinned-staging H2D on the **side stream**; a copy event
   is recorded; the row enters the transient pool ACTIVE under the standard
   event gate. The GPU may execute it only after its event: the 88% hide was
   measured with an unsynchronised compute stream, and a premature sync
   destroys exactly the property the economics rest on.
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
avail_up()   = min(cpu_queue, slack_rows(), GROW_CAP, slots_obtainable())
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

## 8. Degraded modes

* **Low-reuse model (Qwen-class, 71% at W=32):** retain-on-execute captures
  less amortisation but n\* = 1 makes single-use promotion neutral-positive
  while hidden; worst case the transient pool is pure execution staging and
  still beats CPU 2.3× visible.
* **Hide collapses** (CPU queue empties — nothing to hide under): the balance
  condition self-limits, since promotion only runs while `t_cpu > t_gpu`.
* **Link saturated** (residency refills at tight capacity): `slack_rows()`
  goes to zero and promotion pauses; residency traffic has priority.
* **VRAM exhausted:** transient → 0, `p = 0`, engine ≡ shipped engine.
* **Controller misbehaving:** OFF switch (I6).

## 9. Calibration

`elastic_e3.py` is the startup probe, run once per box class: emits
`(B_cpu, B_link, B_gpu, hide)` with the thread sweep embedded, and the
derived `n*` and per-row µs table. Controller thresholds (`δ_hi`, `δ_lo`,
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
