# RESULTS — Stage 3, gate 1: can cold routed work be admitted without
# proportional growth in wall time?

Registered in [`PREREG-tribrid-stage3.md`](PREREG-tribrid-stage3.md),
stamped `7bf5b2be87aef56dc514f67cf90ea219ba1289004aaf0901e5c3230503a52ef5`
before any measurement existed. Receipts in `gate1-5090-zen5/`.

## Verdict: **MISS**

Reported exactly like a pass, per house rule. Every clause below is
attributable to a measured mechanism rather than an inference.

| gate-1 clause | result |
|---|---|
| numerically equivalent to the resident reference | **MISS** — the dynamic arm inherits a **pre-existing** cold→GPU divergence ([e4b#171](https://github.com/pjordanandrsn/experts4bit-qlora/issues/171)). Cold→CPU alone **passes** at every point |
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

## The correctness finding

Cold experts routed to the **GPU** generate a different token sequence than
the resident reference — deterministically, from decode step 59. Cold→CPU
matches exactly at every sweep point (max abs logit diff flat at 0.0703 vs
0.90→15.9 growing with cold traffic on the GPU path).

Reproduces **byte-identically on `f62c119`** (before the Stage-3 PRs) and
`2ba26ff` (after), and at `hot_rows` 384 / 1024 / **2048** — with 2048 slots
for 265 cold rows there is no eviction pressure, ruling out the
address-vs-contents class. Filed as
[e4b#171](https://github.com/pjordanandrsn/experts4bit-qlora/issues/171).

The gate caught it because the equivalence clause was registered *before*
the run, comparing generated tokens rather than a tolerance chosen after
seeing data.

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
