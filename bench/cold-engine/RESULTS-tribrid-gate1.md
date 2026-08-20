# RESULTS — Stage 3, gate 1: can cold routed work be admitted without
# proportional growth in wall time?

Registered in [`PREREG-tribrid-stage3.md`](PREREG-tribrid-stage3.md),
stamped `7bf5b2be87aef56dc514f67cf90ea219ba1289004aaf0901e5c3230503a52ef5`
before any measurement existed. Receipts in `gate1-5090-zen5/`; the
post-hoc reconciliation of the equivalence clause in `gate1-recon-a2000/`.

## Verdict: **MISS**

Reported exactly like a pass, per house rule. Every clause below is
attributable to a measured mechanism rather than an inference.

| gate-1 clause | result |
|---|---|
| numerically equivalent to the resident reference | **PASS on a matched reference** — cold-GPU vs a VRAM-sourced control is **bit-identical** (dmax 0.0000, identical tokens) at 1/5/10% cold. The original MISS came from a mis-specified comparison (see *Correction*); check (a) has now been run |
| hide ratio ≥ 70% | **MISS** — cold work is essentially unhidden; 5% cold mass costs 26–33% wall |
| beats both fixed arms | **MISS** — dynamic tracks the better fixed arm, never beats it |
| ≥1 destination flip | **PASS** — 225/1874 at 1%, 4005/5061 at 5%, 17973/9844 at 20% |
| < 5% proportional slowdown | **MISS** — 11% at the *cheapest* point (1% cold, cold-CPU) |

## Box and calibration

RTX 5090 (sm_120, 32 GB) + AMD EPYC 9655 (Zen 5, 48 cores, 1 NUMA node,
full AVX-512 incl. `vbmi`/`vnni`/`bf16`), 177 GB RAM, torch 2.8.0+cu129.
Calibrated on the box, never from spec sheets:

| ceiling | measured |
|---|---|
| `B_dram` triad | **380.1 GB/s** (48t, NT stores) |
| grouped scatter | **501.0 GB/s** → **G0 = 131.8% of triad, PROCEED** |
| `B_vram` | 1574.2 GB/s |
| PCIe | 28.47 / 28.25 GB/s (Gen5 x16) |
| `B_nvme` | 5.51 GB/s (O_DIRECT) |

G0 exceeding 100% was predicted in the PREREG and is legitimate: the
numerator is read-only traffic with no write-allocate, the denominator
carries a write stream.

Model: `allenai/OLMoE-1B-7B-0924`, NF4 arena (16 × 64 = 1024 rows, 3.6 GB
— fits DRAM 48× over, which is gate 1's precondition). Routing profile
from a real forward on real prose: 1010/1024 cells routed, 84,992 routed
slots, top 10% of cells carrying 30.5% of mass.

## The instrument

Control self-pair (same arm, twice, every point): **39.58–40.42 ms**, worst
disagreement **0.4%**. Every difference reported below is an order of
magnitude above that. Control does **0** disk reads inside the measured
window; cold arms do 105–3409, so NVMe is genuinely on the critical path.

## The sweep

Decode-shaped: 64-token prose prefill, then 128 greedy steps against the KV
cache. Median step wall, and growth over the control at the same point.

**prefetch off**

| cold mass | control | cold-GPU | cold-CPU | dynamic |
|---|---|---|---|---|
| 1%  | 39.7 | 48.05 (+21%) | **44.00 (+11%)** | 48.22 (+21%) |
| 5%  | 39.6 | 49.94 (+26%) | 50.63 (+28%) | 49.95 (+26%) |
| 10% | 39.7 | 56.32 (+42%) | 54.53 (+38%) | 56.81 (+43%) |
| 20% | 39.7 | 64.22 (+62%) | 69.55 (+75%) | 64.07 (+61%) |

**prefetch on**

| cold mass | control | cold-GPU | cold-CPU | dynamic |
|---|---|---|---|---|
| 1%  | 39.7 | 50.93 (+28%) | 46.16 (+16%) | 49.99 (+26%) |
| 5%  | 39.6 | 53.38 (+35%) | 52.60 (+33%) | 53.56 (+35%) |
| 10% | 40.0 | 60.93 (+52%) | 59.19 (+48%) | 61.11 (+53%) |
| 20% | 40.4 | 66.12 (+64%) | 71.87 (+78%) | 69.05 (+71%) |

Prefetch is **worse at every point**. That is not noise — see below.

## Why nothing is hidden: the prefetcher fires and does no useful work

The load-bearing measurement of this gate, because a prefetcher that never
fires and one that fires uselessly are indistinguishable in a median.

| config | wired | submitted | rows | spec_hits | spec_misses | demand_misses |
|---|---|---|---|---|---|---|
| off, gpu | 0/16 | 0 | 0 | 0 | 0 | **997** |
| off, cpu | 0/16 | 0 | 0 | 0 | 0 | **997** |
| on, gpu | 15/16 | 2097 | 1590 | **1555** | 35 | **988** |
| on, cpu | 15/16 | 2099 | 1629 | **1598** | 31 | **988** |

It is wired correctly (15/16 — the last layer has no L+1, by design) and
submits ~2,100 requests. But **1,555 of ~1,590 speculative rows were
already resident**: only ~35 were real reads. Demand misses fall
997 → 988, so it covers **under 1%** of the reads that actually land on the
critical path, while adding enough overhead to lose 5–9 points of wall.

It is prefetching rows that are already warm and missing the cold ones that
matter. This matches the architecture notes' own description of Phase 4 —
*"mechanism landed, explicitly untuned, nothing calls it"* — and G4's
requirement that the feature be **free** when NVMe mass is zero, which is
not the same as effective when it is not. Gate 1 is the first workload to
ask it to be effective.

**This is the named next lever**: a predictor that targets NVMe-placed rows
specifically rather than re-requesting resident ones. Until it exists, the
hide-ratio clause cannot be met by construction, and P2 (>70% hide at 1–5%
cold mass) is **not yet falsified** — it has not been given a working
hiding mechanism to be tested against. Stated plainly rather than scored.

## Correction: the equivalence clause was measured against the wrong reference

This document originally reported a cold→GPU **correctness bug**, filed as
[e4b#171](https://github.com/pjordanandrsn/experts4bit-qlora/issues/171).
That was wrong, and the issue is closed as not-a-defect. The finding is
retained here rather than deleted, because the mistake is instructive.

`force_cold_mass` defaults to `source="dram"`, so the experts moved to
`nvme` came **out of the DRAM tier — and a DRAM expert executes on the
CPU** (`_dram_contrib` → `cpu_grouped.gemv_nf4_grouped_cpu`, the native
fp32 locked tree). The control was therefore a *CPU-arithmetic* reference
for exactly the experts under test:

- control — moved experts in `dram` → CPU kernels
- `cold_dest="cpu"` — same experts in `nvme`, still CPU kernels → **matches
  by construction**
- `cold_dest="gpu"` — same experts on the fused GPU kernel → compute-dtype
  rounding

Cold-CPU matching was structurally guaranteed, not evidence. What was
measured is the **cross-placement rounding law this engine documents in its
own first docstring**, reached through a destination switch instead of a
tier move.

Every supporting observation is equally consistent with rounding —
determinism, independence from `hot_rows` (384/1024/2048), byte-identical
reproduction pre-PR, and growth with cold mass. Eviction was ruled out and
the Stage-3 PRs were ruled out; the documented rounding path never was.

**The matched-reference rule**: compare `"gpu"` against the same experts
**in VRAM**, and `"cpu"` against them **in DRAM**. This harness violated
it. A threshold arm is reproducible only against itself on the same routing
trace, since its destination is per-step.

Consequently the equivalence clause above is scored **NOT MEASURED** rather
than MISS. The timing results are unaffected — they never depended on the
reference — but no equivalence conclusion should be drawn from this run.

### Reconciliation with the e4b account (receipts in `gate1-recon-a2000/`)

The correction above was written from a pure-torch emulation of the two
destinations. It has since been **measured with the real engine and the real
kernels** — e4b
[#173](https://github.com/pjordanandrsn/experts4bit-qlora/pull/173),
`bench/hybrid-g9/issue171/` — and this section reconciles the two accounts.
Where they disagree, the measurement wins and the earlier figure is struck
rather than silently replaced.

Box: RTX A2000 12GB (sm_86), torch 2.8.0+cu128, triton 3.4.0, gnf4 native
AVX2 + OpenMP. One MoE layer at OLMoE-1B-7B geometry, four placements of the
SAME experts through the SAME engine, relative RMS on the layer output:

| pair | T=1 | T=8 | T=64 | bitwise |
|---|---|---|---|---|
| `control_dram` vs `cold_cpu` | 0.000e+00 | 0.000e+00 | 0.000e+00 | **yes** |
| `control_vram` vs `cold_gpu` | 0.000e+00 | 0.000e+00 | 0.000e+00 | **yes** |
| `control_dram` vs `cold_gpu` | 4.590e-03 | 4.622e-03 | 4.627e-03 | no |
| **`control_dram` vs `control_vram`** | **4.590e-03** | **4.622e-03** | **4.627e-03** | no |

**Three things change in the account above.**

1. *The figure.* `3.8e-3 relative RMS / 4.7e-3 relative max on the expert
   output` was the emulation. Measured, it is **4.59–4.63e-3 relative RMS on
   the layer output**, stable from decode (T=1, decode GEMV) to prefill
   (T=64, M-tile path) — so it is the epilogue and output rounding, not a
   launch-config artifact.

2. *The prediction is confirmed, at the layer level.* The last row is the
   whole finding: **DRAM against VRAM, with no cold path anywhere in it,
   reproduces the cold arm's divergence to the digit**, and both cold
   destinations are exact against their matched control. That is the
   `source="vram"` swap, measured directly rather than by re-running the
   sweep. What remains unrun is the **end-to-end token-sequence** swap on a
   5090-class box; nothing here bounds how a 16-layer greedy decode
   amplifies a 4.6e-3 layer perturbation.

3. *One attribution above is now wrong.* "cross-placement rounding
   accumulates past that even on the **correct CPU path** (0.0703)" does not
   survive: `control_dram` vs `cold_cpu` is **bitwise**, so the cold-CPU
   arm's 0.0703 is not cross-placement rounding on the moved experts. It is
   unexplained, and the next section names a candidate.

**A second way these arms are unmatched, and it is not the destination
switch.** `_HybridTier.dram_thin` is decided at enable time from the layer's
TOTAL DRAM population (`0 < n_dram <= offload_thin_uniq`), and
`force_cold_mass` shrinks that population. So a layer above the threshold in
the control can fall to or below it in a cold arm, and the DRAM experts that
**stayed** flip from the CPU tier to the GPU — a destination change on
experts the arm never touched. Measured (`thin_flip.py`, same box, one
layer, 3 of 7 DRAM experts moved, `cold_dest="cpu"` so every mover stays on
the CPU):

| `offload_thin_uniq` | control `n_dram`/thin | cold `n_dram`/thin | rel RMS | bitwise |
|---|---|---|---|---|
| `None` | 7 / False | 4 / False | 0.000e+00 | **yes** |
| `4` | 7 / False | 4 / **True** | 3.654e-03 | no |
| `8` | 7 / **True** | 4 / **True** | 3.362e-03 | no |

With the knob off the arms are bitwise; with it on they are not, and nothing
that moved changed engine. `offload_rows` reads per-step DRAM routing
statistics that `force_cold_mass` also perturbs, so it is the same class.

This is a **candidate** for the 0.0703 residual, not a proven cause, and one
observation cuts against it: the residual is *exactly* 0.0703125 at all four
cold masses, where a thin-flip should move as different layers cross the
threshold. It is recorded as the thing a re-run has to control for.

**Not recoverable from this run.** The receipt records the harness's own
arguments but **not the engine's `enable_hybrid_tier` kwargs**, and the
runner was a disposable box script. So whether `offload_thin_uniq` /
`offload_rows` were set here cannot be established after the fact. A
placement-arm receipt has to carry the engine configuration, not only the
sweep configuration — otherwise an arm can differ in a way no reader can
see.

**What the receipts already show and neither account stated.** At 1% cold
mass `cold-gpu` has `tokens_match: True` (0.9023 max abs logit); token
divergence starts at 5% (`first_divergence: 59`). The `dynamic` arm diverges
at 1% (20.5002, `first_divergence: 59`) — the same step index, a different
arm. Any restatement of "the cold path generates different text" has to name
the arm and the cold-mass point.

**Check (a): RUN, and the arms swap end to end.** Re-ran this sweep with
`force_cold_mass(source="vram")` on the same host class (RTX 5090 + EPYC
9655; this box `B_dram` 417.4 GB/s, G0 122.5% PROCEED), same model, arena,
routing prompt and workload; gnf4 `5b4463e` / e4b `2e88bd1`.

| control's experts sourced from | control executes on | cold-GPU | cold-CPU |
|---|---|---|---|
| `dram` (original run) | **CPU** | dmax 0.90–15.9, tokens ✗ | dmax 0.0703, tokens ✓ |
| `vram` (this run) | **GPU** | **dmax 0.0000, tokens ✓** | dmax 14.3–18.9, tokens ✗ |

Divergence tracks **which engine the control runs on**, not the cold path.

Stronger than a match: cold-GPU against a VRAM-sourced control is
**bit-identical** — `dmax = 0.0000` at 1%, 5% and 10% cold mass, with
identical greedy token sequences. A cold expert read from NVMe, gathered
through `_TieredStack` and executed on the fused GPU kernel produces
*exactly* the bytes that expert produces resident in VRAM. The cold path is
exonerated: no gather defect, no invalidation defect, no mis-indexed
weighting. e4b#171 was correctly closed.

Receipts: `gate1_vram.json`, `receipts-hybrid-calib-vram-arm.json`.

**Check (b): RUN — cold-CPU IS bitwise against its matched control, and
the `dram_thin` hypothesis is not the explanation.** Receipts
`check_b.json` / `check_b.py`, same box and trees as check (a).

    cold-CPU vs matched DRAM control: dmax = 0.000000, bitwise, tokens identical

Check (c) is folded in: the engine state is now recorded per module in both
arms, so this is answerable from the artifact rather than inferred.

    control   dram_thin 0/16 layers | n_dram 42..54 | offload_rows=None | fused_ffn=False
    cold-cpu  dram_thin 0/16 layers | n_dram 21..41 | offload_rows=None | fused_ffn=False

`force_cold_mass` does shrink the DRAM population exactly as the hypothesis
says (per-layer 45→34, 46→41, 45→34, 45→27, 42→21, 51→32 …), but **no layer
crosses a thin threshold in either arm, because `offload_thin_uniq` is
`None` and `dram_thin` is therefore False on all 16 modules in both.** The
flip it describes is real machinery and would bite a run that sets that
knob; it is not what these arms did.

**So both cold destinations are now bitwise against their matched control**
— cold-GPU vs VRAM-sourced (check a) and cold-CPU vs DRAM-sourced (check b).

What remains genuinely open: the original run's **0.0703 does not
reproduce** on current main (gnf4 `5b4463e` / e4b `2e88bd1`, which include
#172 and #173). Whether that residual was removed by one of those merges or
was specific to the original box instance is **not determined here**, and
this run cannot distinguish them — the original box is destroyed. Recorded
as unexplained rather than resolved.

## Addendum: the thin flip, exercised — and a control-design trap

Checks (a) and (b) settled the destination question on this box with the real
model, and nothing below revisits them. What they did **not** exercise is the
`offload_thin_uniq` flip: that run had the knob at `None`, so `dram_thin` was
False on all 16 modules in both arms, and the section above correctly says the
flip "would bite a run that sets that knob". This turns *would* into
*does, measured* — and bounds it.

`decode_arms.py` (receipts in `decode_arms.log`) is a portable version of the
gate-1 equivalence arms: a synthetic 8-layer MoE stack, greedy decode, arms
compared by generated tokens and max abs logit delta against a control whose
NVMe tier is empty. Synthetic weights on purpose — the mechanism is a property
of the engine, not of a checkpoint, and this re-fires in ~10 minutes on any
CUDA box instead of needing a rented one. It gives no OLMoE divergence indices
and is not offered as a substitute for the real run.

RTX A2000 12GB, torch 2.8.0+cu128, triton 3.4.0, gnf4 native AVX2 + OpenMP.
8 layers x 32 experts, top-4, 96 greedy steps, 4 movers per layer, ~23k cold
rows executed per arm. Max abs logit delta against the control:

| DRAM population | `offload_thin_uniq` | `source` | `cold_dest` | delta | tokens |
|---|---|---|---|---|---|
| 24 → 20 (never thin) | `None` | `dram` | `cpu` | **0.0000** | ✓ |
| 24 → 20 (never thin) | `4` | `dram` | `cpu` | **0.0000** | ✓ |
| **8 → 4 (straddles)** | `None` | `dram` | `cpu` | **0.0000** | ✓ |
| **8 → 4 (straddles)** | **`4`** | `dram` | `cpu` | **0.0709** | ✓ |
| 8 → 4 (straddles) | `None` | `vram` | `gpu` | **0.0000** | ✓ |
| 8 → 4 (straddles) | `4` | `vram` | `gpu` | **0.0000** | ✓ |

Read the fourth row against the third: **same arm, same movers, same
destination — only the knob differs, and a bitwise arm stops being bitwise.**
The experts that moved never changed engine; the four that STAYED did, because
the population fell to the threshold in the cold arm and not in the control.

The last two rows are the internal control. Under `source="vram"` the DRAM
population is untouched, so no layer crosses the threshold and the knob changes
nothing — the effect appears exactly where the population moves and nowhere
else. Rows 1–2 are the other bound: a population that never comes near the
threshold is equally immune. **The flip needs the population to straddle**,
which is why check (b) was right not to see it and why this is not a retraction
of that result.

It also bounds the size. 0.0709 here against the original run's 0.0703 is the
same order and the same shape — a constant, bf16-scale residual on an arm that
should have been exact. That is **suggestive and no more**: different model,
different geometry, different box, and the original run's knob setting remains
unrecoverable (which is what e4b#175 fixes going forward). It is recorded as
the first mechanism measured to produce a residual of that kind on a
destination-matched CPU arm, not as the cause of that one.

**A control-design trap, recorded because it cost a run.** The first version of
this probe left an NVMe population in the control. `cold_dest` applies to
*every* cold expert, so the cold-CPU arm switched those experts' destination
too, and the arm was unmatched for a reason unrelated to its movers — reading
0.1068 where the corrected control reads 0.0000. That is the same error class
as #171, one level up: an equivalence control has to hold the destination fixed
for **all** experts, not only the ones the arm moves. Run 1 is kept in
`decode_arms.log` rather than deleted.

## Addendum: the hide-ratio clause is aimed at 5–11% of the cost

Derived from the committed receipts (`gate1_v2.json` + the calibration
blob), no new measurement. Disk time is `win_reads × row_bytes / B_nvme`
using **this box's measured sequential ceiling (6.26 GB/s)** and the arena's
3.54 MB row, charged against the arm's own wall delta over 128 steps.

| cold | arm | Δ ms/step | win reads | disk ms/step | **disk share of Δ** |
|---|---|---|---|---|---|
| 1% | cold-GPU | 8.33 | 105 | 0.46 | **5.6%** |
| 1% | cold-CPU | 4.29 | 106 | 0.47 | **10.9%** |
| 5% | cold-GPU | 10.30 | 238 | 1.05 | **10.2%** |
| 5% | cold-CPU | 11.00 | 238 | 1.05 | **9.6%** |
| 10% | cold-GPU | 16.66 | 340 | 1.50 | **9.0%** |
| 10% | cold-CPU | 14.87 | 335 | 1.48 | **9.9%** |
| 20% | cold-GPU | 24.53 | 3400 | 15.02 | 61.2% |
| 20% | cold-CPU | 29.86 | 2025 | 8.94 | 29.9% |

**At 1–10% cold mass, storage is ~5–11% of what cold work costs.** The
other ~90% is the cold path's own software cost — tier `ensure` bookkeeping,
`ColdCpuView` materialization and `segment_into` host copies on the CPU
side, `_TieredStack` gather plus H2D on the GPU side. Only at 20%, where
the tier genuinely thrashes (3400 reads against 340 at 10%), does disk
become the dominant term.

Using the sequential ceiling makes these a **lower bound on the disk
share**: if the real achieved rate on 3.54 MB routed reads is below 6.26
GB/s, disk time is larger than shown. Even at half that rate it stays a
minority of Δ at 1–10%.

**This reframes the gate-1 hide-ratio MISS.** The clause asks whether NVMe
latency can be hidden underneath scheduled work. At the cold masses the
prereg targets for its strongest prediction (1–5%), NVMe latency is only
~10% of the exposure — so a *perfect* prefetcher, hiding 100% of disk,
could remove at most ~10% of the cost. **A 70% hide ratio is unreachable by
construction on this workload**, and the prefetcher's 1% coverage is not
the binding constraint it looked like.

It also explains the prefetch probe. 1,555 of ~1,590 speculative rows hit
because the predictor **already filters to `nvme_set`** (`want = sorted(ids
& nxt.nvme_set)`) and the tier is an effective cache: at 5% cold, 265 cold
rows fit a 384-slot pool, so after first touch they stay resident. The
prefetcher is not failing to hide disk; there is little disk left to hide.
The earlier note that it "prefetches warm rows and misses the cold ones"
was wrong about the mechanism — it prefetches cold rows that are already
cached.

**This is the directive's own most-worth-falsifying prediction, confirmed**:
*"NVMe itself will cease being the interesting bottleneck surprisingly
quickly. Once reads overlap and hot misses get retained, the next wall may
become staging/copy orchestration."* Retention did it, and staging/copy
orchestration is the wall.

The lever that follows is **not** a better prefetcher. It is the per-call
cost of the cold path itself, and the named follow-up already on record —
landing NVMe segments directly into the kernel-shaped stacks (the `preadv`
iovec scatter `ArenaExpertSource` already uses) instead of arena row →
`segment_into` → stack — attacks exactly that term.

## What this run does NOT establish

- **Prediction 3 (CPU wins more cold assignments than intuition suggests)**
  is *suggestive but unproven here*: cold-CPU wins at 1% and 10%, loses at
  5% and 20%. No clean story, and the 20% points sit where the tier is
  under real capacity pressure.
- **Reclaimable residency is inert in this configuration.** `protected_rows`
  defaults to `hot_rows`, so `resurrections = 0` throughout. R1–R10 are
  untested by this gate and need their own arm.
- **Gate 2 (destination choice under asymmetric load) was not run.** The
  "dynamic" arm here is a rows-per-unique-expert threshold, not a deadline
  estimate, and the receipt labels it as such.
- **One box, one model, one workload shape.** Instrument law 7 applies: a
  faster GPU raises the bar for the CPU side, and every number above is
  Zen 5 + 5090 specific.

## Methodological corrections made mid-run, recorded rather than quietly fixed

**A prefill-shaped harness was discarded.** The first version re-ran the
whole 64-token prompt every step. With top-8-of-64 routing that touches
62–64 of 64 experts in *every* layer — essentially the whole arena, every
step — so no tier size avoids thrash and "cold" stops being a controlled
fraction of routed work. Its numbers (cold arms reading 6,472–12,940 times
against the control's 1,024) measured tier thrash, not cold-path
scheduling, and are not reported as a result.

**The registered equivalence metric was applied to the wrong reference.**
The PREREG fixed a 5e-2 max-abs-logit tolerance, and every arm exceeded it.
The tolerance was **not widened after seeing the data**; generated-token
agreement was reported alongside it and the clause scored as registered.
Amending the metric requires its own justification and a new stamp — it is
not a post-hoc edit.

What this paragraph originally concluded from that — *"cross-placement
rounding accumulates past that even on the **correct** CPU path (0.0703)"* —
is **struck**. It is contradicted twice above and was left standing here only
because it sits in a different section: the *Reconciliation* shows
`control_dram` vs `cold_cpu` bitwise at the layer level, and check (b) shows
it bitwise end to end on this box. A matched CPU arm does not accumulate past
the tolerance; it does not accumulate at all. The 0.0703 belongs to the
original run and is recorded as unexplained, not as a property of the
metric.
