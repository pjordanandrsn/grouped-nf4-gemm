# SPEC — Elastic execution controller (phase 2)

A design document, not a preregistration: it commits to mechanisms and
interfaces, and defines acceptance gates whose *thresholds* are frozen in
per-gate preregistrations before each measurement. Every number below traces
to a merged receipt; every invariant traces to the measured trap that made it
necessary.

Revised on review (#203, both Bugbot findings): execution demotion when the
GPU leads — placement decoupled from residency — and the equilibrium predicate
unified with the deadband. The revisions are marked where they land. Re-founded
after G1c (#210): the controller is a residency manager on reuse economics;
the δ-band law those reviews hardened is retired with its objective (§5).

Lineage: [`PREREG-elastic-promotion.md`](PREREG-elastic-promotion.md) →
[`RESULTS-elastic-promotion.md`](RESULTS-elastic-promotion.md) (gate E).
Authorised by gate E passing; the controller was explicitly out of scope
there and is the subject here.

## 1. Objective

Treat free VRAM as a fluid execution cache. When the CPU tier is the long
pole, execute some of its expert invocations on GPU — copy up, execute,
retain; when VRAM is wanted elsewhere, contract (KV wins, I3).
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
  how the controller behaves when the GPU stops paying (§5 — re-founded on
  reuse after G1c: a GPU long pole is a fault guard, not a balance point)
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
   default**, else CPU. (The old demote-override — routing resident
   invocations to CPU to balance walls — is retired with the δ-band law,
   G1c re-founding; residency itself is the actuator now, and the OFF
   switch covers the fault directions.)
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

## 5. The controller — the balance law, re-derived on reuse (G1c re-founding)

> Rewritten after G1c ([RESULTS-p2-g1c.md](RESULTS-p2-g1c.md)). The original
> law balanced *this step's* walls by moving work between tiers. G1c measured
> why that objective is unwinnable on a DRAM-bound CPU tier: host traffic is
> p-invariant — a promoted row's bytes cross the same DRAM the tier is
> saturating, with a mixing penalty on top. What promotion buys is never this
> step; it is **residency** — every future invocation served from VRAM
> removes a full row of host-DRAM traffic from a future step. The controller
> is therefore a residency manager, not a load balancer.

**Objective.** Minimize the steady-state cold mass per step —
`E[miss_t] · rb / B_eff`, the DRAM-bound wall — subject to VRAM capacity
(KV-coupled, §6) and to never visibly inflating any single step (I4's
spirit, extended: promotion spends only bandwidth and dispatch the step can
absorb).

**The reuse ledger** (per cold-executed invocation, from calibration):

```
cost(promote)  = (κ − 1) · rb/B_eff + C_disp/p    # the mix penalty: the
                 # row's bytes cross DRAM either way (CPU read vs DMA read);
                 # promotion adds only the interleave inefficiency. G1c
                 # measured κ ≈ 1.3 (mixed DMA+GEMV ran 128.5 GB/s vs 167.9
                 # pure), plus burst dispatch amortized over p.
payback(hit)   = rb/B_eff  per future resident invocation — a hit removes a
                 # whole row of host traffic from a future step.
break-even     : E[future reuses] > (κ − 1) + C_disp/(p · rb/B_eff) ≈ 0.3–0.5
```

E1 measured the reuse distribution: 69–96% of invocations recur ≥ 2 more
times within 32 steps. Expected payback exceeds promotion cost several times
over for the *population* — so, exactly as E2 found no selector is needed,
**the law promotes every cold-executed row, retain-on-execute, throttled
only by global budgets**. There is no per-row prediction to mistune; what
G1c killed was promoting *for the current step*, not promoting.

**The law** (each step):

```
avail()    = min(cold_t, SMOOTH_CAP, slack_rows(), slots_obtainable(),
                 LINK_CAP())
SMOOTH_CAP = ceil(PROMO_FRAC · m)   # bounds this step's visible inflation:
             # p·(κ−1)·rb/B_eff + C_disp ≤ ε · t_step, ε registered at
             # calibration (default 5%); PROMO_FRAC follows from it
LINK_CAP() unchanged (G1 amendment)  # copies must fit the step regardless

each step:
  if pressure():  shrink_transient(deficit)    # event-gated, I3 — KV wins
  cold_t = invocations executed on CPU this step (non-resident)
  p_t    = avail()                             # 0 is legal: no burst
  promote the p_t most recent of cold_t as ONE stream-ordered burst (I9);
  un-budgeted cold rows execute with no state change — the next recurrence
  re-qualifies them (retain-on-execute has no memory to corrupt)

every PERIOD (≈256) steps:
  persistent ← LFU-promote transient rows resident ≥ 2·PERIOD   (hysteresis)
  persistent shrink iff per-row hit rate < θ for 2 consecutive periods
```

Observables (per step, EWMA α ≈ 1/16): cold mass `miss_t`, novelty `new_t`
(first-ever (layer, expert) arrivals), fills, `vram_free`, transient hit
counters (`stats()`), and the calibrated constants. `t_cpu`/`t_gpu` walls
remain observed for the OFF switch and G4, but no longer drive an actuator:
**the δ band, its demote sizing, its pins, and the band-width calibration
constraint are retired** — they steered the objective G1c refuted. A GPU
long pole is a fault condition here (`t_gpu_row ≪ t_cpu_row`), handled by
`pressure()` and the OFF switch, not an operating point to balance around.

Hysteresis on the persistent pool is unchanged, for the same measured
reason as before: continuously evicting and reloading it would destroy the
retention benefit that is the one measured win. The transient pool is the
fast half by construction.

**Equilibrium** — residency has converged when promotions chase only
novelty and the split is stable:

```
converged  ≜  EWMA(fills) ≤ (1 + η) · EWMA(new_t)   # fills track first
                                                    # arrivals only
stable     ≜  no grow/shrink and no persistent-pool change this PERIOD
equilibrium ≜ converged and stable
```

η (default 0.25) absorbs legitimate re-fills of capacity-evicted rows; a
fill rate far above novelty is thrash (the I1 margin failure), and that is
a G2 spoiler, not a tuning knob.

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
* **CPU queue empties** (everything already resident): `cold_t = 0` so the
  law promotes nothing — the budget is demand-driven and self-limits.
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
launch + sync, warm, idle GPU). `C_disp` and — G1c additions — **`B_dram`** (the shared host ceiling, a
parallel triad) and **`κ`** (the DMA+GEMV mixing penalty: pure-GEMV
effective bandwidth over mixed-arm effective bandwidth; 1.3 measured) feed
SMOOTH_CAP's inflation budget (§5). **Measured-load rule (G1c):** any
hide/overlap constant is valid only at the CPU tier's operating thread
count and intensity — E3b's full hide at 132.5 + 52.6 < 212 GB/s was true
and useless for a 171.8 GB/s tier; probes must sample the operating point.
Controller thresholds (`PROMO_FRAC`/ε, `PERIOD`, θ, η, `GROW_CAP`,
`RESERVE`) are computed from calibration, not baked in (I8). The retired
δ-band constraints (rounds 3–4) live in git history only — they calibrated
the actuators of the refuted objective. A box where
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
* **P2-G2 (residency convergence, offline — re-registered on the reuse
  law):** controller-in-replay on the 16 committed rank traces, driving the
  **real** `DevRowCache` (the score-the-shipped-thing rule), cold start,
  scored in fill/miss trace units so no box is needed. Registered shape:
  (a) *convergence*: trailing-32 fill rate within 1.10× of the
  steps-256–512 plateau within **64 steps** on ≥ 14/16 traces;
  (b) *plateau quality* (unthrottled): total fills over steps 128–512 ≤
  **1.10×** same-capacity ideal-LRU on every trace; (c) *equilibrium*:
  post-convergence `EWMA(fills) ≤ (1 + η)·EWMA(novelty)`, η = 0.25 — §5's
  own predicate; (d) *throttle gracefulness*: PROMO_FRAC ∈ {1/16, 1/8, 1/4}
  degrades convergence ≤ 2× and plateau fills ≤ 1.05× vs unthrottled.
  Spoilers, both must fail: `protected = rows` (the I1 margin trap) must
  thrash past the (c) bound; a no-retention arm (evict-after-execute) must
  plateau at all-miss. Refuted ⇒ the law's budgets/hysteresis are wrong —
  fixed in spec, not tuned live.
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
| controller oscillation under bursty routing | demand-driven budget + slow persistent half | P2-G2 churn-vs-novelty bound |
| transient pool churn evicting rows mid-read | same event-gated states as the shipped engine | I2/I3 + existing kernel tests |
| constants drift across boxes | calibration is mandatory and gating (I8) | E3 guard at startup |
| Qwen-class low reuse wastes VRAM | n\* = 1 makes waste ≈ 0 wall; capacity returns via shrink | G3 under a low-reuse trace |
