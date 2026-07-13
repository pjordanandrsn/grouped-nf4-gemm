# Phase-1 first pass — sm_86 (RTX A5000), OLMoE + Qwen3-30B cells

**2026-07-13** · RunPod SECURE A5000 (capability 8.6, driver 570.211.01, torch
2.11.0+cu128, bnb 0.49.1) · receipts `phase1_A5000_sm86.json` (32 cells) ·
harness `bench/phase1/harness.py @ 1ed90d6` · pod torn down, 404-verified;
piggybacked on the axolotl #3797 gate pod (~$0 marginal).

## decode bs1 (the registered ≥1.3× target regime)

| shape | dequant_grouped | gemv_4bit | gemv vs dequant |
|---|---:|---:|---:|
| OLMoE gate_up (N2048 K2048 E64 k8) | 0.565 ms | 0.346 ms | **1.63×** |
| OLMoE down (N2048 K1024) | 0.420 ms | 0.280 ms | 1.50× |
| Qwen3-30B gate_up (N1536 K2048 E128 k8) | 0.421 ms | 0.274 ms | 1.54× |
| Qwen3-30B down (N2048 K768) | 0.445 ms | 0.274 ms | 1.62× |

bnb's own in-kernel-dequant gemv — a **per-expert Python loop of k launches** —
already beats the dequant+mm product path 1.5–1.6× at bs1. The fused grouped
kernel's ≥1.3× registered threshold is therefore conservative: the kernel must
beat the *dequant path* by 1.3×, and the existing kernel-dequant reference
clears that with launch overhead to spare (roofline ceiling ~8.1×). The
per-call floor (~0.27–0.35 ms for k=8 launches) also shows what a single
grouped launch has to amortize.

## Two findings that sharpen the registered claims

1. **Fidelity ordering does NOT hold for bnb's gemv_4bit**: its error vs the
   fp64 reference is ~2× the dequant path's (3.3e-3 vs 1.7e-3 rel Frobenius,
   every bs1 cell) — right at the registered B-rel bound. So "dequantize inside
   the kernel" does not automatically buy fidelity; the P-fid claim rests on
   fp32 accumulation specifically, and the fused kernel must beat *both* paths.
   (The dequant path's 1.66e-3 across all cells is the bf16-GEMM rounding floor
   and is the comparator denominator for the 2× bound.)
2. **J/token at bs1 is parity, not a freebie**: gemv is 1.6× faster but draws
   more sustained power (185 W vs 109 W — it keeps the SMs busier), so J/token
   lands ≈ equal (0.0072 vs 0.0070 J/tok, OLMoE gate_up). The "strictly below
   in ALL cells" energy threshold is a real bar the fused kernel has to earn
   through fewer launches + fewer bytes, not something speed alone delivers.
   At prefill the dequant path's J/token is ~5e-5 (compute-bound, amortized) —
   the energy claim there is about deleting the dequant round-trip.

## Coverage + known v0 limitations

- Cells: OLMoE + Qwen3-30B, both projections, decode_bs1 + prefill_s2048;
  unsloth/marlin recorded as skipped-with-reason (not installed on this pod).
- Gemma-4-26B and GPT-OSS-120B cells not run: the harness keeps the fp64
  reference stack resident (`w_ref64`), which for GPT-OSS is ~17 GB fp64 —
  needs chunked reference computation before those cells fit a 24 GB card.
- prefill routing is the uniform analytic M (census note); measured routing
  histograms are the remaining Phase-1 item alongside the Unsloth/Marlin
  wiring and an A2000 (QNAP) replication point.
