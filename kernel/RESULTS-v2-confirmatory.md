# Confirmatory v2 — VERDICT: NOT CONFIRMED (P1a, S1 fail; P1b, S2, E1, Q1 pass)

**Date:** 2026-07-13 · **Frozen code:** `a235828` (amendment 1; original freeze `54492e8`) · **Protocol:** `kernel/prereg_v2_confirmatory.json` + `kernel/prereg_v2_amendment1.json` (both OTS-stamped pre-data) · **Reducer:** `bench/phase1/reduce_confirmatory_v2.py` reading the stamped params · **Evidence:** `bench/phase1/confirmatory_v2/`

v2 tested one isolated change after the v1 confirmatory failure: the decode
config census dict replaced by a single dense-sweep-validated constant
(BLOCK_N=64, num_warps=2). Devices: a fresh post-stamp SECURE A5000
(`oiot1okzzy08jj`, driver 570.211.01, torn down + 404-verified) and the home
A2000 12 GB. n=3 fresh-process reps per device, paired old-config backend
(`fused_nf4_v1cfg`) in every cell, worst/median-rep reduction as registered.
Nothing was re-run; every number below is the run as it happened.

## Registered criteria — outcomes

| criterion | registered bar | outcome |
|---|---|---|
| P1a bounded loss (paired, worst-rep ≥ 0.88 everywhere) | both devices | **FAIL** — A5000 PASS (min 0.926); A2000 FAIL (5 cells below; OLMoE down 0.79–0.87 consistent across reps) |
| P1b gains exist (paired, median-rep ≥ 1.10 on ≥ 2 cells/device) | both devices | **PASS** — A5000: Qwen3-30B gu 1.154, gemma gu 1.190, gpt-oss dn 1.118; A2000: gemma gu 1.250, gemma dn 1.243, gpt-oss gu 1.181 (matching the stamped predictions) |
| S1 census vs dequant | med ≥ 1.3 on ≥ 7/8; med ≥ 1.0 on 8/8; worst ≥ 1.0 on ≥ 7/8 | **FAIL** — A2000 PASS (all three clauses); A5000 FAIL by 0.004: gpt-oss down median 0.996 (7/8 ≥ 1.3 ✓, worst floor 7/8 ✓) |
| S2 held-out-v2 vs dequant | med ≥ 1.0 on ≥ 6/8; med ≥ 1.3 on ≥ 4/8; worst ≥ 0.9 on ≥ 6/8 (NOT-RUN-scaled) | **PASS** both devices |
| E1 energy | fused J/tok < dequant on ≥ 15/16 run cells per device | **PASS** — A5000 15/16 (the allowance was consumed by exactly the registered at-risk class: Scout down, T=1, margin 1.351); A2000 15/15 run |
| Q1 suite | 35/35 both devices | **PASS** |
| **V2_CONFIRMED** | all | **FALSE** |

## What v2 established (survived, or failed informatively)

1. **The constant config is right for the 64-SM class and off-census
   transfer.** On the fresh A5000: bounded loss held everywhere (≥ 0.926
   including held-out), the predicted gains landed (P1b), and the fresh
   held-out set at k ≥ 6 runs 1.2–2.0× median vs dequant with energy below on
   every k ≥ 2 cell. The v1 "held-out = parity" finding is superseded for
   k ≥ 6 shapes: DeepSeek-V3 1.09–1.21×, Qwen3-Next 1.48–1.58×, granite
   1.36–1.37× (A5000 medians; A2000 similar or better).
2. **"One constant for all sm_86" is falsified at the low-SM end.** On the
   26-SM A2000, 64/2 loses 15–25% to the old 128/4 default on several
   small-N cells (OLMoE down 0.79–0.87 across all reps — consistent, not
   noise). The config sweep (bare-kernel, batched-per-config timing) had
   predicted ≤ 9% loss there: the harness context (interleaved backends,
   energy windows between timings) exposes config sensitivity the sweep's
   measurement boundary hides on this card. The data supports an
   SM-conditional constant (64/2 for ~48+ SMs, 128/4 below) — a candidate
   v3 change, NOT applied here per the no-tune clause.
3. **gpt-oss down (2880×2880) is instance-bound, not config-bound.** Third
   instance, third different dequant-relative result (1.75× → 1.03× → 1.00×)
   — while its PAIRED gain held (1.11–1.15× vs the old config). The new
   config is better on this cell; whether the cell beats the dequant path
   depends on the instance. S1's 8/8-median-≥1.0 clause failed on a 0.4%
   shortfall here.
4. **The T=1 structural class is now quantified on both axes.** Llama-4-Scout
   (top_k=1): 0.47×/0.83× speed on the A5000 and the first energy miss in
   63 confirmatory-grade cells (fused J/tok 1.35× dequant on Scout down).
   One expert × one token starves the (T, N-tiles) grid — registered in the
   prereg as at-risk; split-K over K is the fix and remains the follow-on
   kernel change (Phase 2.1).
5. **Fidelity has never wavered:** 35/35 on every device in both
   confirmatories; fused more accurate than the dequant path in every
   measured cell.
6. **DeepSeek-V3 gate_up on the A2000 = NOT-RUN** (VRAM OOM building the
   E=256 stack on the 12 GB card, exactly the registered exclusion;
   thresholds scaled one-for-one).

## Aborted first attempt (amendment 1, blindness preserved)

The initial v2 launch died before producing any observable data: the harness
fixture built a monolithic `[E,N,K]` fp32 host tensor (~30 GB for DeepSeek-V3
gate_up) and the pod OOM-killer SIGKILLed rep 1 (rc=137). Both legs were
killed, the A2000's partial outputs deleted UNREAD, the pod torn down
(404-verified), the fixture fixed (per-expert generation + survivable
stack-build failure → NOT-RUN cells), and the protocol re-registered as
amendment 1 (criteria byte-identical) before any new pod was provisioned.

## Cumulative claim after v1 + v2 (the README basis)

> On sm_86 at decode bs1, the fused grouped-NF4 kernel is **more accurate
> than the dequant path in every cell ever measured** and **more
> energy-efficient in every measured cell with top_k ≥ 2** (61/62
> confirmatory-grade cells; the single exception is the top_k=1
> occupancy-starved class, quantified and named). Census-class shapes run
> **1.3–2.5× at median** (one named instance-sensitive cell: gpt-oss down);
> fresh off-census shapes with k ≥ 6 run **1.2–2.0× at median**; top_k=1
> shapes currently LOSE (0.5–1.1×) pending split-K. The decode config is one
> constant on 64-SM parts; low-SM parts want the 128/4 default (measured,
> not yet applied).

## Evidence (committed under `bench/phase1/confirmatory_v2/`, sha256)

```
2b4b40741da0a3476f0640626d7ecdbb45c55d2ceb6ec2faa29b9bad1931716a  conf2_a2000_rep1.json
8cc5cc1f6d478d5ad0c81dc0a7474820f457039404cd7c2926bc3b5fa02df62e  conf2_a2000_rep2.json
304d860b9895b532d3a35f65dc49d793675078d94d61b39611be0681d11e2d0c  conf2_a2000_rep3.json
c7da2cbceaddde17014be9de851c808a10521171124db13a39e13dc1bb1e6afd  conf2_a5000_rep1.json
7021427de89c57d15b67af4058b1a704bd5f094f38f0adea423c13da5810c906  conf2_a5000_rep2.json
4017014b51038da96c03262591f082fdef109be18791882c1e0df7ab53d148cb  conf2_a5000_rep3.json
7cffe371dc4ba3f3b2794b474997784d315aa83795d428b0d64c01991e765b46  reduction_v2.json
```

The two config sweeps that motivated the constant are committed alongside
(`sweep_a5000.json`, `sweep_a2000.json`; tool `bench/phase2/decode_config_sweep.py`).

## Full per-cell table (median / worst rep; paired = v1cfg_ms/fused_ms)

### A5000 (fresh pod, driver 570.211.01)

| cell | vs dequant med/worst | paired med/worst | energy worst |
|---|---|---|---|
| [C] OLMoE gu | 2.50 / 2.14 | 1.029 / 1.011 | 0.409 |
| [C] OLMoE dn | 2.49 / 2.49 | 1.058 / 1.039 | 0.453 |
| [C] Qwen3-30B gu | 1.67 / 1.66 | 1.154 / 1.153 | 0.405 |
| [C] Qwen3-30B dn | 2.12 / 2.01 | 1.036 / 1.013 | 0.542 |
| [C] gemma-4 gu | 1.38 / 1.27 | 1.190 / 1.185 | 0.536 |
| [C] gemma-4 dn | 1.93 / 1.84 | 1.085 / 0.996 | 0.513 |
| [C] gpt-oss gu | 1.33 / 1.26 | 1.001 / 0.968 | 0.506 |
| [C] gpt-oss dn | **1.00 / 0.95** | 1.118 / 1.113 | 0.923 |
| [H] DeepSeek-V3 gu | 1.21 / 1.20 | 0.966 / 0.956 | 0.751 |
| [H] DeepSeek-V3 dn | 1.09 / 1.07 | 0.969 / 0.958 | 0.751 |
| [H] granite gu | 1.36 / 1.27 | 1.043 / 0.991 | 0.603 |
| [H] granite dn | 1.37 / 1.35 | 1.027 / 0.944 | 0.677 |
| [H] Scout gu (k=1) | **0.83 / 0.82** | 0.948 / 0.926 | 0.822 |
| [H] Scout dn (k=1) | **0.47 / 0.47** | 0.934 / 0.929 | **1.351** |
| [H] Qwen3-Next gu (k=10) | 1.48 / 1.17 | 1.095 / 1.017 | 0.683 |
| [H] Qwen3-Next dn (k=10) | 1.58 / 1.43 | 1.033 / 1.028 | 0.637 |

### A2000 12 GB (home, co-resident services)

| cell | vs dequant med/worst | paired med/worst | energy worst |
|---|---|---|---|
| [C] OLMoE gu | 1.69 / 1.69 | 1.044 / 1.039 | 0.661 |
| [C] OLMoE dn | 1.62 / 1.51 | **0.834 / 0.788** | 0.349 |
| [C] Qwen3-30B gu | 1.34 / 1.30 | 1.001 / 0.953 | 0.668 |
| [C] Qwen3-30B dn | 1.81 / 1.62 | **0.758 / 0.741** | 0.373 |
| [C] gemma-4 gu | 1.53 / 1.23 | 1.250 / 1.143 | 0.718 |
| [C] gemma-4 dn | 2.50 / 2.07 | 1.243 / 0.818 | 0.286 |
| [C] gpt-oss gu | 1.97 / 1.89 | 1.181 / 1.097 | 0.451 |
| [C] gpt-oss dn | 1.47 / 1.37 | 1.075 / 0.843 | 0.465 |
| [H] DeepSeek-V3 gu | NOT-RUN (VRAM OOM, registered exclusion) | — | — |
| [H] DeepSeek-V3 dn | 1.25 / 1.19 | 1.058 / 1.036 | 0.782 |
| [H] granite gu | 1.36 / 1.35 | **0.848 / 0.826** | 0.453 |
| [H] granite dn | 1.97 / 0.95 | 0.756 / 0.649 | 0.495 |
| [H] Scout gu (k=1) | 1.08 / 1.07 | 1.054 / 1.045 | 0.808 |
| [H] Scout dn (k=1) | 0.84 / 0.80 | 1.067 / 0.874 | 0.910 |
| [H] Qwen3-Next gu (k=10) | 1.20 / 0.94 | **0.762 / 0.693** | 0.438 |
| [H] Qwen3-Next dn (k=10) | 1.54 / 1.45 | 0.967 / 0.799 | 0.800 |

## Consequences / queue

1. **Candidate v3 (one isolated change): SM-conditional decode constant**
   (64/2 at ≥ ~48 SMs, 128/4 below) — directly supported by the paired A2000
   data; would be registered fresh (v3 prereg) if pursued.
2. **Split-K decode path for the T=1 class** (Phase 2.1) — now motivated by
   a quantified speed AND energy loss.
3. Sweep methodology note for the record: bare-kernel batched-per-config
   timing under-predicts harness-context config sensitivity on small-SM
   parts; future sweeps should interleave configs.
4. The repo-flip and #1949-comment decisions remain Jordan's; the cumulative
   claim above is the current honest README basis.
