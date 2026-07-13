# Gate-2 blind confirmatory — VERDICT: NOT CONFIRMED as registered; claim narrowed

**Date:** 2026-07-13 · **Frozen code:** `ad2bef0` · **Protocol:** `kernel/prereg_gate2_confirmatory.json` (OTS-stamped before any confirmatory data existed) · **Reduction:** `bench/phase1/reduce_confirmatory.py` (mechanical; worst-rep) · **Evidence:** `bench/phase1/confirmatory/`

The confirmatory exists so the final verdict is not the run that was iterated
on. It worked: the exploratory Gate-2 result (8/8 census speed ≥ 1.3×) did
**not** fully survive blind replication, and the held-out shapes exposed real
census-overfit in the speed claim. The energy and fidelity claims survived
everywhere. Per the registered no-tune clause, nothing was re-run and nothing
below was measured twice: **worst rep of n=3, reported as-is.**

## Registered criteria — outcomes

| criterion | registered bar | outcome |
|---|---|---|
| C1 census speed | worst-rep ≥ 1.3× on all 8, both devices | **FAIL** — 13/16 device-cells (A5000 6/8, A2000 7/8) |
| C2 census energy | fused J/token < dequant on all 8, both devices | **PASS** — 16/16, margins 0.29–0.75 |
| C3 held-out speed | ≥ 1.3× on ≥ 6/8 and ≥ 1.0× on 8/8, per device | **FAIL** — A5000 0/8 ≥ bar (4 cells < 1.0); A2000 1/8 (2 cells < 1.0) |
| C4 held-out energy | fused J/token < dequant on ≥ 6/8 per device | **PASS** — 16/16, margins 0.44–0.95 |
| C5 property suite | 35/35 both devices | **PASS** — 35/35 (A5000), 35/35 (A2000) |
| **GATE2_CONFIRMED** | all of the above | **FALSE** |

## What survived blind replication (the confirmed claims)

1. **Energy: fused J/token strictly below the dequant path in all 32 of 32
   device-cells** — census *and* held-out, both devices, worst rep. Range
   0.29–0.95× the dequant path's J/token. This is now the headline claim.
2. **Fidelity: fused is more accurate than the dequant path everywhere** —
   in-bench fused/dequant error ratio ≤ 0.755 in all 96 measured cells
   (both devices × 3 reps × 16 shapes), on top of the 35/35 property suite
   per device.
3. **Census speed, narrowed:** median-rep 1.25–2.97× (A5000) and 1.37–2.56×
   (A2000); never slower at median on any census cell; the worst-rep ≥ 1.3×
   floor held on 13/16 device-cells.

## What did not survive (reported at full volume)

- **The universal ≥ 1.3× worst-rep census floor.** Three misses:
  - `gpt-oss-120b down` (N=2880 K=2880, default config) on the fresh A5000:
    1.01× worst / **1.03× median** — persistent across all 3 reps, not a
    transient. The tuning-session pod measured 1.75× on this cell with the
    same frozen code and the same GPU model. Per-instance variation
    (driver 580 vs 575, clocks, host CPU) moves this shape across the bar.
  - `gemma-4 gate_up` (N=1408 K=2816, tuned config) on the A5000: 1.22×
    worst / 1.25× median (tuning pod: 1.89×). Same story, milder.
  - `gemma-4 gate_up` on the A2000: 0.74× worst — a single-rep two-sided
    transient (fused 0.68→1.02 ms while dequant 0.94→0.75 ms in that rep;
    other reps 1.39×/1.37×) consistent with co-resident home-lab services
    contending for the card. The registered reduction counts it, so it
    counts.
- **Off-census speed transfer.** Held-out shapes (all on the default decode
  config) sit at parity, not ≥ 1.3×: worst-rep 0.77–1.25× (A5000) and
  0.83–1.65× (A2000). Reproducible regression shape: `Phi-3.5-MoE down`
  (N=4096 K=6400) is ~0.8× on **both** devices. The decode speed win is
  census-tuned and does not generalize by default — the cost-model follow-on
  is now motivated by data, not hypothesis.

## The narrowed Gate-2 claim (what the README may say)

> On sm_86, for the 8-cell MoE census at decode bs1, the fused grouped-NF4
> kernel is **more energy-efficient than the dequant path in every measured
> cell (32/32 blind, worst-rep)** and **more accurate in every measured cell**,
> with **median speedups of 1.25–2.97×** that clear 1.3× on most cells but are
> instance- and shape-sensitive; off-census shapes run at **parity speed with
> the same energy and fidelity advantage**. A per-shape cost model (planned)
> is required before any general speed claim.

## Devices, deviations, evidence

- **A5000 leg:** RunPod SECURE pod `8gx8ukgb08z87n` (provisioned 2026-07-13
  **after** the prereg stamp, never used in tuning), driver 580.159.04,
  torch 2.8.0+cu128. First provisioning attempt earlier the same hour
  returned no-stock; no pod existed. Pod torn down after evidence pull
  (DELETE → 404-verified; account at 0 pods).
- **A2000 leg:** home RTX A2000 12 GB (driver 575.64.05), docker
  `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime` on the QNAP. **Aborted
  first attempt:** the `-runtime` image ships no C compiler, so triton could
  not build its launcher stubs — the suite reported 34/35 errored and the
  fused backend never produced a timing. `gcc` was installed and the leg
  restarted from scratch; no fused timings existed before the restart, and
  no timing from the aborted attempt is used anywhere. (REPRO.md now lists
  the compiler requirement.)
- All 16 shapes ran on both devices — the 12 GB card fit every held-out
  stack, so the registered A2000 OOM/NOT-RUN exclusion was never invoked.
- **Evidence** (committed under `bench/phase1/confirmatory/`, sha256):

```
12c7e36a37abfd878a378a4250b519c9aaa5437a01e4ee4de8fc150b103017e8  conf_a2000_rep1.json
e0f09cff5de168d8088ac1a16e508c645c6038c3cc35fa2b3309d8cbb06434fb  conf_a2000_rep2.json
39d721ca13320d7f21e30e0b63d07ee596e54ff7e0e6688574992683b8c3b64b  conf_a2000_rep3.json
70c76ce62c8f942237a843f12c0d0bd47692e00107cea7f6228c2e2bc4b55a19  conf_a5000_rep1.json
ce4116601d9217d4de4137e7db74c65786d0b99ebde00546ac7e755a5294d266  conf_a5000_rep2.json
e546003ec0813cac3dc5eb14c91dbd7124881b54de0fed3c2e5c7831573f7668  conf_a5000_rep3.json
18affc9f363f99dcb0be75d0f710cd5b051a9b37297896a94724f53d1faa9e30  reduction.json
```

## Full per-cell worst-rep table

Ratio = dequant_ms / fused_ms (min over 3 reps); energy = fused_J / dequant_J
(max over 3 reps). [C] census (tuned table where marked †), [H] held-out
(default config).

### A5000 (fresh SECURE pod, driver 580.159.04)

| cell | shape (N×K) | worst speed | reps | worst energy |
|---|---|---|---|---|
| [C] OLMoE gate_up | 2048×2048 | 2.14× | 2.14/2.59/2.20 | 0.425 |
| [C] OLMoE down | 2048×1024 | 2.51× | 2.51/2.71/2.83 | 0.375 |
| [C] Qwen3-30B gate_up † | 1536×2048 | 1.47× | 1.47/1.54/1.63 | 0.474 |
| [C] Qwen3-30B down | 2048×768 | 2.66× | 3.00/2.97/2.66 | 0.432 |
| [C] gemma-4 gate_up † | 1408×2816 | **1.22×** | 1.22/1.25/1.53 | 0.574 |
| [C] gemma-4 down † | 2816×704 | 2.12× | 2.12/2.44/2.44 | 0.417 |
| [C] gpt-oss gate_up | 5760×2880 | 1.44× | 1.44/1.53/1.50 | 0.464 |
| [C] gpt-oss down | 2880×2880 | **1.01×** | 1.03/1.01/1.03 | 0.718 |
| [H] DeepSeek-V2-Lite gate_up | 2816×2048 | 0.84× | 0.84/0.89/0.96 | 0.825 |
| [H] DeepSeek-V2-Lite down | 2048×1408 | 1.25× | 1.25/1.33/1.28 | 0.613 |
| [H] Qwen2-57B gate_up | 5120×3584 | 1.18× | 1.18/1.22/1.24 | 0.738 |
| [H] Qwen2-57B down | 3584×2560 | 0.96× | 1.00/0.96/1.00 | 0.839 |
| [H] Phi-3.5-MoE gate_up | 12800×4096 | 0.87× | 0.90/0.87/0.87 | 0.952 |
| [H] Phi-3.5-MoE down | 4096×6400 | 0.77× | 0.81/0.77/0.81 | 0.906 |
| [H] Mixtral gate_up | 28672×4096 | 1.16× | 1.19/1.16/1.19 | 0.796 |
| [H] Mixtral down | 4096×14336 | 0.91× | 0.98/0.91/0.96 | 0.889 |

### A2000 12 GB (home, co-resident services, driver 575.64.05)

| cell | shape (N×K) | worst speed | reps | worst energy |
|---|---|---|---|---|
| [C] OLMoE gate_up | 2048×2048 | 1.46× | 1.58/1.54/1.46 | 0.668 |
| [C] OLMoE down | 2048×1024 | 1.43× | 1.43/1.64/1.76 | 0.336 |
| [C] Qwen3-30B gate_up † | 1536×2048 | 1.33× | 1.38/1.33/1.44 | 0.657 |
| [C] Qwen3-30B down | 2048×768 | 1.68× | 1.68/1.68/2.71 | 0.371 |
| [C] gemma-4 gate_up † | 1408×2816 | **0.74×** | 1.39/0.74/1.37 | 0.748 |
| [C] gemma-4 down † | 2816×704 | 1.77× | 2.56/2.62/1.77 | 0.287 |
| [C] gpt-oss gate_up | 5760×2880 | 1.75× | 1.83/1.75/1.95 | 0.487 |
| [C] gpt-oss down | 2880×2880 | 1.35× | 1.50/1.35/1.53 | 0.497 |
| [H] DeepSeek-V2-Lite gate_up | 2816×2048 | 1.00× | 1.12/1.00/1.05 | 0.834 |
| [H] DeepSeek-V2-Lite down | 2048×1408 | 1.65× | 1.65/2.10/2.27 | 0.443 |
| [H] Qwen2-57B gate_up | 5120×3584 | 1.13× | 1.17/1.13/1.15 | 0.816 |
| [H] Qwen2-57B down | 3584×2560 | 1.05× | 1.09/1.05/1.06 | 0.836 |
| [H] Phi-3.5-MoE gate_up | 12800×4096 | 1.12× | 1.15/1.12/1.12 | 0.820 |
| [H] Phi-3.5-MoE down | 4096×6400 | 0.83× | 0.83/0.89/0.92 | 0.892 |
| [H] Mixtral gate_up | 28672×4096 | 1.08× | 1.08/1.09/1.09 | 0.886 |
| [H] Mixtral down | 4096×14336 | 1.05× | 1.07/1.06/1.05 | 0.864 |

## Consequences

- The **repo-public flip** and the **#1949 coordination comment** remain
  Jordan's calls; if flipped, the README leads with the narrowed claim above
  (energy + fidelity everywhere; speed census-median with named misses).
- **Next engineering, in order of what this data motivates:** (1) decode
  cost model / cached runtime autotune — fixes both the off-census parity
  and the cross-instance census misses; (2) profile `Phi-3.5-MoE down`
  (4096×6400) and `gpt-oss down` (2880×2880) as the named regression
  shapes; (3) prefill parity (unchanged).
