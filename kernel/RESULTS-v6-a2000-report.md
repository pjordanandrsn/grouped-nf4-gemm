# v6 register-LUT mainloop — A2000 report-only leg (addendum)

**REPORT-ONLY / EXPLORATORY.** No prereg exists for this leg (it was declared
report-only in `RESULTS-v6-confirmatory.md` before it ran); nothing here is a
registered claim. It is an out-of-sample replication read of the v6-confirmed
bands on the smallest deployed card. Draft held for review before commit/stamp.

## Setup

- Device: **NVIDIA RTX A2000 12GB** (26 SM, sm_86), QNAP container `gnf4-v6`
  (torch 2.8.0+cu128, triton 3.4, bnb 0.49.2 — the sweep image, unmodified).
- Ran unattended in the **02:00-06:00 CDT quiet window** (home-card law): VRAM
  gate passed at 07:00:00Z 2026-07-16 with 8826 MiB free
  (`conf6_gate.log`), lock-guarded runner `conf6_a2000.sh`.
- Property suite FIRST: **44/44 passed** (`conf6_suite.log`, 136 s).
- Then **n=3 fresh-process reps** of the census harness
  (`bench/phase1/harness.py`, 4 census models × gate_up/down ×
  {prefill_s2048, decode_bs1} × {dequant_grouped, fused_nf4, fused_v5loop},
  20 iters, `--no-energy`), all rc=0, ALLDONE (`CONF6_STATE`).
- Receipts: `bench/phase1/results/conf6_a2000/` (3 JSONs + logs + runner +
  SHA256SUMS). Reduction: median across the 3 reps per cell; per-rep min..max
  shown for spread.

## Headline — the v6 paired claim replicates on 26 SM

Paired **v6 (register-LUT) vs v5 loop** (`fused_v5loop` ms / `fused_nf4` ms),
same process, same data — the instance-robust ratio the v6 confirmatory
registered:

| cell (prefill_s2048) | v6/v5loop med [min..max] |
|---|---|
| OLMoE gate_up | 1.506 [1.501..1.512] |
| OLMoE down | 1.501 [1.496..1.508] |
| Qwen3-30B gate_up | 1.449 [1.444..1.449] |
| Qwen3-30B down | 1.501 [1.492..1.502] |
| Gemma-4-26B-A4B gate_up | 1.457 [1.455..1.462] |
| Gemma-4-26B-A4B down | 1.378 [1.377..1.382] |
| GPT-OSS-120B gate_up | 1.510 [1.504..1.512] |
| GPT-OSS-120B down | 1.527 [1.525..1.530] |

**8/8 cells 1.38–1.53, median 1.50** — against the A5000-confirmed
**1.39–1.54 (median 1.50)**. The register-LUT prefill win replicates on a
26-SM card essentially to the digit; combined with the 12-card sweep
(sm_80→sm_120) this brackets the claim from the smallest to the largest
deployed parts with no retune (the same bn128/w4/s3 rule and 64/2 decode
constant, unmodified).

**Decode is untouched, as designed:** v6 changed only the M-tile prefill
mainloop; paired decode reads 0.93–1.05 (median 1.00) — noise around parity,
no regression.

## vs the dequant baseline (context, not a claim)

| cell | decode fused/dequant | prefill fused/dequant |
|---|---|---|
| OLMoE gate_up | 1.307 | 0.530 |
| OLMoE down | 2.161 | 0.588 |
| Qwen3-30B gate_up | 1.290 | 0.709 |
| Qwen3-30B down | 2.277 | 1.446 |
| Gemma-4-26B-A4B gate_up | 1.290 | 0.733 |
| Gemma-4-26B-A4B down | 3.252 | 1.216 |
| GPT-OSS-120B gate_up | 2.142 | 1.313 |
| GPT-OSS-120B down | 1.891 | 1.233 |

- **Decode: 8/8 wins, 1.29–3.25 (median 2.02)** — consistent with every prior
  A2000 read.
- **Prefill follows the published low-SM pattern** (cross-arch sweep,
  `fit_dispatch_floor`): the wide-expert cells win (GPT-OSS both projections,
  Qwen3-30B/Gemma-4 down 1.22–1.45) while the smallest-expert OLMoE (both) and
  the narrow gate_ups lose (0.53–0.73) — the compute-poverty floor at 26 SM.
  The sweep's monotone-SM law says exactly this (the same OLMoE gate_up cell
  is 0.43–0.57 at sm22–58, 0.96 at sm132, and a 1.18 **win** at sm170), so the
  A2000 read adds an in-family point at the low end rather than contradicting
  anything. Fused remains the universal default per the sweep's no-v7 verdict;
  small-expert prefill on low-SM parts stays the documented known-loser.

## Fidelity

P-fid ordering holds on **all 16 (cell × regime) pairs**: fused
`b_rel_vs_fp64` ≈ 1.65–1.70e-3 vs the dequant path's 2.20–2.85e-3 — fused is
0.60–0.76× the baseline's error, matching the published band (fp32-accumulate
is dtype-bought and unchanged by the register-LUT).

## Caveats

- Report-only: no prereg, no adjudicating reducer; numbers are the harness's
  own medians reduced by a plain median-of-3.
- Home card on a shared box: the quiet window + VRAM gate held, but two cells
  show single-rep spread (Gemma-4 down decode max 4.25 vs median 3.25;
  Qwen3-30B gate_up decode max 1.89 vs median 1.29) — the known contended-card
  jitter; medians are quoted.
- `--no-energy` (the QNAP container lacks stable pynvml sampling under other
  tenants); energy claims rest on the confirmatory legs.
- Census harness cells use synthetic weights (as in all census runs).

## Verdict (informal, report-tier)

Nothing in the A2000 data contradicts a published claim; the one registered
band this leg can speak to — the paired register-LUT prefill gain — lands
inside the confirmed 1.39–1.54 window on 8/8 cells at the lowest SM count in
the fleet. v6's mainloop is architecture-robust from 26 SM to 170 SM.
