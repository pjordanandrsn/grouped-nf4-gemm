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

**Checks still open**: (b) re-run cold-CPU against the control with
`offload_thin_uniq=None` and see whether 0.0703 goes to 0.0000 — note this
arm now has a sharper framing, since cold-GPU *does* reproduce its matched
reference bitwise while cold-CPU does not reproduce its own to better than
0.0703; (c) record the engine kwargs in the receipt so (b) is answerable
from the artifact next time.

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

**The registered equivalence metric was the wrong one.** The PREREG fixed a
5e-2 max-abs-logit tolerance. Over 16 layers, cross-placement rounding
accumulates past that even on the *correct* CPU path (0.0703). The
tolerance was **not widened after seeing the data**; instead generated-token
agreement is reported alongside it, and the clause is scored as registered.
Amending the metric requires its own justification and a new stamp — it is
not a post-hoc edit.
