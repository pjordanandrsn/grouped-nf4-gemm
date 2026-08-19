# PREREG — Stage 3: deadline-aware tribrid CPU/GPU/NVMe scheduling

Filed **before implementation**, per the directive's own requirement that
the tribrid work be preregistered. Predictions below are the directive's,
restated as falsifiable clauses with the thresholds and arms fixed here.
Architecture map: `docs/cold-engine/TRIBRID-ARCHITECTURE.md`.

Baseline this expands: Stage 2 at `experts4bit-qlora/bench/hybrid-g9/
HANDOFF.md` — G8 B=8 CLOSED (balance 0.978 reference host, 0.87–0.93 on
two other classes), G8 B=16 OPEN at 0.698, G9 OPEN at 45.4 tok/s.

## Claim under test

**Coldness is a scheduling property, not an execution penalty.** An
NVMe-resident routed expert can be read once, dispatched to whichever
compute engine has the better deadline path, and have most of that I/O
hidden underneath already-scheduled work — without changing any bytes.

## Primary metric (fixed here, so it cannot be chosen after the data)

    hide_ratio = 1 - (cold time exposed on critical path / cold isolated time)

*Exposed* = the increase in layer wall time attributable to cold work,
measured as `T_layer(arm) - T_layer(control)` at matched routing.
*Isolated* = the same cold work's issue→contribution span measured with
the layer otherwise idle. Both halves reported raw; the ratio is derived,
never measured directly.

Secondary: physical NVMe read count and bytes (NOT nominal placement
misses — see R8), exposed-stall attribution per stage, destination
decisions with their counterfactual estimates.

## First experimental gate — can cold mass be admitted at all?

**Deliberately not K3.** Use a model whose expert arena fits DRAM, then
constrain DRAM so a controlled subset is forced cold. Capacity is then
not a confound and a known-good hybrid baseline exists on the same box.

Arms, same model / prompt / routing trace / placement / box:

| arm | cold experts supplied to |
|---|---|
| **control** | nothing — VRAM + DRAM only (the Stage-2 hybrid) |
| **cold-GPU** | GPU only |
| **cold-CPU** | CPU only |
| **dynamic** | scheduler picks per invocation |

Forced cold routing mass swept at **1%, 5%, 10%, 20%** — swept, not
waited for.

Gate 1 passes when, at 5% forced cold mass, the dynamic arm:

- is **numerically equivalent** to the resident reference (bitwise per the
  equivalence table in the architecture notes — this clause cannot be
  traded against any timing clause);
- hides **≥70%** of isolated cold latency;
- beats **both** fixed arms on exposed wall;
- shows **≥1** destination flip across the run;
- costs **<5%** proportional slowdown vs the all-resident control.

A miss is reported exactly like a pass, per house rule.

## Second gate — does destination choice actually matter?

Constructed load asymmetries, with disk and PCIe pressure varied
independently: GPU-loaded, CPU-loaded, both.

The receipt this gate exists to produce:

    T_isolated_GPU(E) < T_isolated_CPU(E)   and   T_join_CPU(E) < T_join_GPU(E)

i.e. the GPU is intrinsically faster for expert E, and CPU execution still
delivers E's contribution earlier because GPU work is already on the
critical path. Absent that, the scheduler is a static placement rule with
extra steps, and Stage 3 should be re-scoped rather than argued for.

## Third gate — adaptive residency

Only after cold-path timing works. Distinguish one-shot cold experts,
short-lived warm bursts, repeatedly-routed DRAM-worthy experts, and
persistently hot VRAM-worthy experts, then feed observed cost/reuse back
into placement. The Stage-2A solver stays the starting point.

## Registered predictions — base tribrid

| # | prediction | falsified by |
|---|---|---|
| P1 | dynamic beats the better fixed policy by 10–30% exposed cold wall once both engines are meaningfully occupied | dynamic ≤ better fixed |
| P2 | at 1–5% cold mass, hide ratio >70% (possibly >90% on favourable shapes) | ~proportional wall growth with cold mass |
| P3 | CPU wins more cold assignments than intuition suggests, because GPU is the scarcer resource | CPU rarely chosen under GPU load |
| P4 | the optimal destination flips with batch — GPU-biased at B=1/4, CPU relatively more attractive by B=8/16, bounded by the row-scaling law | one universal winner across batch |
| P5 | a sharp cold-mass knee, likely 5–15%, strongly box-dependent | smooth degradation |
| P6 | raw NVMe GB/s predicts the knee poorly; `bytes / B_nvme` overpredicts exposed latency at low cold mass and underpredicts it under contention | the naive model tracks measurement |
| P7 | a bounded DRAM reuse cache matters more than an elaborate prefetch predictor | reuse cache shows no conspicuous drop in physical reads |
| P8 | transient residency beats rigid placement; three distinct timescales appear (permanent / burst / single-invocation) | no burst regime distinguishable |
| P9 | CPU-cache→GPU promotion beats rereading NVMe often enough to deserve a first-class path | promotion rarely cheaper |
| P10 | the best policy minimizes exposed stall, not total traffic — the fastest run may move more bytes | traffic minimization also wins wall |

**Quantitative preregistration.** Controlled model, DRAM could hold
everything, 5% of routed expert work forced cold. Dynamic arm: numerically
equivalent · hide ratio ≥70% · beats both fixed arms · ≥1 destination flip
· ≪5% proportional slowdown. Then force 20%: knee crossed, hide ratio
drops, storage/staging visible on the critical path, adaptive DRAM
retention becomes materially valuable.

**Prediction most worth falsifying:** that NVMe stops being the
interesting bottleneck quickly, and staging/copy orchestration or
destination slack becomes the wall.

## Registered predictions — reclaimable residency

Tested separately, because it is a distinct hypothesis: *the interval
between logical eviction and physical overwrite contains measurable
reusable information.* Note the architecture finding that in `ColdTier`
today that interval is ~zero by construction — creating it is part of the
mechanism under test, not a precondition assumed true.

| # | prediction | falsified by |
|---|---|---|
| R1 | 5–20% of logically evicted experts are referenced again before overwrite | consistently <1–2% |
| R2 | VRAM resurrection is disproportionately valuable — even 2–5% of routed invocations moves wall time | no measurable wall effect |
| R3 | DRAM resurrection rate (≈10–30%) exceeds VRAM (≈3–15%) | VRAM ≥ DRAM |
| R4 | short-window recurrence predicts resurrection better than long-run expert frequency | global frequency predicts as well or better |
| R5 | soft eviction ≤ hard eviction in cost almost everywhere | measurable regression not attributable to metadata/sync |
| R6 | largest gains at active working set ≈1.1–2× protected fast-tier capacity | gains flat across pressure |
| R7 | reclaimable residency moves the NVMe knee outward by 20–50% on workloads with temporal locality | knee unmoved |
| R8 | nominal placement miss rate becomes a poor I/O metric; physical refill rate is the operational one | the two stay close |
| R9 | choosing between simultaneously-valid DRAM and VRAM copies by slack beats always taking the highest tier | highest-tier-always ties or wins |
| R10 | reclaimable residency reduces promotion churn, NVMe rereads and H2D refills without reducing effective hit rate | churn unchanged or hit rate drops |

**Quantitative preregistration.** Identical protected VRAM and DRAM
budgets, arm A = hard eviction, arm B = reclaimable. Arm B predicted:
≥10% fewer physical NVMe reads · ≥5% fewer H2D expert refills · no
numerical difference · no increase in protected-memory usage · 5–15%
lower exposed cold-path wall where temporal locality is meaningful.
Larger on deliberately bursty routing.

**Prediction most worth falsifying:** the ghost working set — that a 10 GB
protected VRAM expert budget behaves like ~12–14 GB of effective burst
working set under favourable locality while never reserving more than
10 GB against the allocator. If it holds in both DRAM and VRAM, the
result generalizes past tiering: **capacity ownership and information
retention are not the same thing.**

The load-bearing probability, measured separately per tier:

    P(reuse before overwrite | logical eviction)

Cache hit rate alone is not sufficient and will not be reported alone.

## Instrumentation contract

Timestamps at: request issue · storage completion · host landing · H2D ·
CPU/GPU start · CPU/GPU end · contribution join. Reported: raw cold
latency separately from exposed stall · bytes read unnecessarily because
of eviction/promotion mistakes · destination decisions with
counterfactual estimates · per-tier completion traces · protected vs
resurrection hits per tier · logical-eviction-to-overwrite lifetime ·
bytes and critical-path time avoided by resurrection.

The scheduler stays falsifiable: every destination choice records what it
predicted for both paths, so a wrong prediction is visible as a wrong
prediction and not merely as a slow step.

## Explicit non-goals

Distributed storage · multi-node scheduling · remote object stores ·
arbitrary tensor-level paging · predictive routing models · learned cache
policy · fully dynamic placement across every layer · K3-scale execution
as a success condition.

Smallest useful new primitive, and the thing to prove first: **a cold
expert can be read once from NVMe, routed to whichever active compute
engine has the better deadline path, and have most of that I/O hidden
underneath already-scheduled work.**

## Instrument laws carried forward

The seven laws in `hybrid-g9/HANDOFF.md` apply unchanged. Two bind
especially hard here:

- **Law 5** — compare kernel numbers only at matched call shapes. A cold
  path changes call shapes; cross-shape extrapolation manufactured a
  phantom 2× executor loss last campaign.
- **Law 7** — a balance ratio moves with the GPU side. A faster GPU
  raises the bar for both the DRAM wall *and* the cold path. State the
  host class with every number.
