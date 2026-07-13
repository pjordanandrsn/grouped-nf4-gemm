# Phase 1 — baseline harness

Measures the baselines the fused kernel is registered against
(`gemm_predictions.json`): census shapes × regimes × backends, with CUDA-event
medians, J/token power receipts, and the per-cell fp64-reference fidelity that
`TOLERANCE_CONTRACT.md` makes the comparator (the dequant path's `b_rel` is the
2× bound's denominator).

```
python bench/phase1/harness.py --smoke              # tiny shapes, sanity (GPU)
python bench/phase1/harness.py --models OLMoE       # one model's cells
python bench/phase1/harness.py                      # full census sweep
```

Backends in v0: `dequant_grouped` (the e4b product path: per-active-expert
`dequantize_4bit` → bf16 mm), `gemv_4bit` (bnb's in-kernel-dequant reference,
bs1 only). `unsloth` and `marlin` are registered but import-guarded — they
record as `skipped` with the reason until their wiring lands (their absence is
visible in the receipts, not silent).

Receipts: `phase1_<gpu>.json` — one cell per (model, proj, regime, backend)
with `ms_median`, `tok_per_s`, `j_per_token` (+ sampling method/rate),
`b_rel_vs_fp64`, and the env pin (GPU, capability, driver, torch, bnb). The
Phase-2 kernel drops into the same registry so its cells land beside the
baselines it must beat: ≥1.3× tok/s at decode bs1 on sm_86, ≥1.0× train fwd,
J/token strictly below, fidelity-ordered per the contract.

Regime notes: `decode_bs1` = top_k experts × one token (the skinny extreme the
roofline says is memory-bound, ceiling ~8.1× vs the two-pass dequant path);
`prefill_s2048` = uniform-routing analytic M per expert (census note; replaced
by measured routing histograms later in Phase 1).
