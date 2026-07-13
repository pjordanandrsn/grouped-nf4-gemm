# Phase-1 tail: J/token over measured routing + per-layer occupancy sweep

**2026-07-13** · A5000 SECURE pod (sm_86, torch 2.8+cu128, bnb 0.49.2,
grouped_gemm 0.3.0) · receipts `p1t_{olmoe,qwen}_{energy,perlayer}.json` ·
harness `8c32c18` · replayed the saved routing histograms (no model reload) ·
pod 404-verified. Closes the two measurement gaps the previous routing pass
flagged.

## J/token — the energy correction tracks the time correction

Energy sampled (NVML) on the representative-layer measured-vs-uniform prefill,
grouped path. Same 16384 token-rows both ways.

| model | proj | backend | uniform µJ/tok (W) | measured µJ/tok (W) | J ratio |
|---|---|---|---:|---:|---:|
| OLMoE | gate_up | dequant_grouped | 42.6 (183) | 48.6 (195) | **1.14** |
| OLMoE | gate_up | unsloth | 158.5 (229) | 161.4 (227) | 1.02 |
| OLMoE | down | dequant_grouped | 34.7 (175) | 40.7 (194) | 1.17 |
| OLMoE | down | unsloth | 69.8 (222) | 75.1 (222) | 1.08 |
| Qwen3-30B | gate_up | dequant_grouped | 84.6 (217) | 70.2 (207) | **0.83** |
| Qwen3-30B | gate_up | unsloth | 220.1 (229) | 188.3 (229) | 0.86 |
| Qwen3-30B | down | dequant_grouped | 68.3 (176) | 61.0 (183) | 0.89 |
| Qwen3-30B | down | unsloth | 116.5 (225) | 91.7 (223) | 0.79 |

The J/token correction is the same shape and roughly the same magnitude as the
time correction: uniform routing **understates OLMoE energy 2–17%** (occupancy
~1, skew makes it slower and slightly hotter) and **overstates Qwen3-30B energy
11–21%** (occupancy 0.80, the 27 empty groups it never has to launch). Watts
are stable (175–229 W) across the pair, so J/token ≈ time here — the energy
story adds no independent axis for these grouped prefill cells, it confirms the
timing one. The fused kernel's registered "J/token strictly below the dequant
path" bar should be evaluated against the *measured*-routing dequant number
(70.2 µJ/tok on Qwen gate_up, not the uniform 84.6).

## Per-layer sweep — occupancy drives grouped cost near-linearly, and the spread is real

`prefill_measured --routing-layer all`, every histogram layer (timing only,
grouped_gemm path, gate_up shown):

| model | layers | occupancy range | ms/step range | spread | driver |
|---|---:|---|---|---:|---|
| OLMoE (E64) | 16 | 0.98–1.00 | 11.51 → 11.80 | **1.03×** | n_groups 63→64 |
| Qwen3-30B (E128) | 48 | 0.68–0.99 | 11.96 → 16.56 | **1.38×** | n_groups 87→127 |

- **OLMoE is layer-invariant** (1.03× spread): it hits ~all 64 experts in every
  layer, so the representative-layer choice barely matters — the previous
  single-layer number is robust for OLMoE.
- **Qwen3-30B varies 1.38× across its layers**, and the cost tracks `n_groups`
  (= occupancy × E) almost linearly: the sparsest layer (occ 0.68, 87 non-empty
  groups) runs 11.96 ms, the densest (occ 0.99, 127 groups) runs 16.56 ms. The
  median-occupancy representative layer (0.79, 101 groups, ~13.6 ms) sits
  mid-range and is a fair central estimate, but a real full-model prefill is the
  *sum over all 48 layers at their own occupancies* — so the per-layer receipts,
  not a single number, are the honest object.

**This confirms and bounds the previous caveat:** the ~15% "uniform overstates
Qwen" figure is a layer-average; per layer the overstatement ranges from ~0 (the
occ-0.99 layers, where uniform is nearly right) to ~30% (the occ-0.68 layers).
n_groups is the cost knob on the grouped path — which is exactly why the fused
single-launch design, whose cost is meant to be far less n_groups-sensitive, is
the thing to measure against this spread in Phase 2.

## Phase-1 status: complete (with two noted deferrals)

Measured on two cards (A5000 sm_86 + A2000), five backends, all four census
models, both regimes, with J/token and now per-layer routing robustness. The
baseline field the fused kernel must beat is fully characterized:

- decode bs1: beat gemv_4bit (~0.27 ms, but 2× fidelity err) and clear marlin's
  fidelity; the grouped path is the floor to avoid (2.5–2.9× slower).
- prefill: beat the *measured*-routing dequant number (occupancy-corrected,
  ~15% under uniform for wide MoEs); grouped path stays the floor and its cost
  scales with n_groups (1.38× layer spread on Qwen3-30B).
- energy: J/token tracks time; the bar is the measured-routing dequant J/token.

**Deferred, noted not faked:** measured routing for Gemma-4 (gated) and
GPT-OSS-120B (57 GB 4-bit → 80 GB card; k=4 ≠ the measured k=8 models, so no
clean proxy) — rides a future 80 GB box. Their GEMM cells are measured on
uniform routing.

Next is Phase 2 (the Triton fused kernel, 2-week time-box), dropped into the
same backend registry so its cells land beside these baselines and the
registered thresholds.
