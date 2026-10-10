# Calibration of the GGUF backward bars, before the path under test was scored

[`DESIGN.md`](DESIGN.md) fixes the oracle, the bars and the faults for dX = dY · W through a frozen k-quant weight.
This note measures, before any path under test was scored, where those bars sit:
- the **floor**: the error of a correct bf16 path against the fp64 oracle;
- the **null**: the error each deliberate fault produces.

A bar is only meaningful if it lies between the two. Everything here runs on CPU, with no timings. Outputs are in
[`results-2026-10-10/`](results-2026-10-10/), produced by [`calibrate.py`](calibrate.py) (synthetic blocks) and
[`gate_real_bytes.py`](gate_real_bytes.py) (released bytes).

## Synthetic blocks (48 cases)

The cases cover Q8_0, Q4_K, Q5_K and Q6_K. Shapes (M, N, K) are (17, 37, 256), (128, 256, 256), (3, 129, 2816) and
(64, 2048, 512), each with 3 seeds. The blocks are random packed bytes with finite scale fields.
- **Magnitude:** DESIGN.md §5 meant the scales to give W about 1/√K. With random sub-scales and quants, W's std
  is about 1x that for Q8_0 and Q2_K, but 3x, 15x, 31x and 90x for Q3_K, Q4_K, Q5_K and Q6_K.
- **Why it does not matter here:** every error in this note is relative, so it does not depend on scale. The released
  bytes below carry real magnitudes.

| type | floor (bf16 out), max | fp32-out variant | adjoint identity, max | closest fault, min |
|---|---|---|---|---|
| Q8_0 | 2.41e-3 | 1.72e-3 | 3.5e-16 | 1.00 (zero dX); every mutation ≥ 1.36 |
| Q4_K | 2.49e-3 | 1.85e-3 | 1.3e-15 | 5.4e-2 (dmin zeroed) |
| Q5_K | 2.52e-3 | 1.89e-3 | 1.6e-15 | 2.6e-2 (dmin zeroed) |
| Q6_K | 2.39e-3 | 1.71e-3 | 4.0e-15 | 1.00 (zero dX) |

The realized bound min(2 · e_ctrl, 1e-2) is about 4-5e-3, so the closest synthetic fault sits about 5x above it. That
fault's size depends on dmin relative to d, and these blocks set dmin = d/2. Hence the gate below.

## Released bytes (the gate)

Every tensor is sha256-pinned; fetch them with `scripts/fetch_gguf_fixtures.py` (the attention set) and `--ffn` (the
expert / FFN set).

- **Attention projections:** 8 tensors, Q2_K-Q6_K, from two Muse-Glimmer-30B GGUFs.
- **Expert matrices:** Qwen3-30B-A3B (MoE) experts 0 and 77 of `ffn_{gate,up,down}_exps`, in Q3_K, Q4_K, Q5_K, Q6_K
  and Q8_0.
- **Dense FFN:** Muse-Glimmer-30B row prefixes, in Q2_K, Q3_K and Q8_0.

The cases are M ∈ {17, 128} × 3 seeds per tensor; the transpose fault runs on the leading square block.

- **Released files ship far larger mins than the synthetic blocks.** The median dmin/d is about 8 for Q4_K and about
  17 for Q5_K, in every tensor class, against 0.5 synthetic. Zeroing dmin scores 1.42-2.14 on real bytes, not 2.6e-2.
- **Floor:** 2.24-2.38e-3 for every type and class. The realized bound is ≤ 4.77e-3 everywhere.
- **Faults:** each scores at least **130x** its case's bound. The smallest margin is the Q4_K / Q5_K scale-spill fault;
  every other fault is ≥ 280x.
- **One profile covers attention, expert and dense-FFN tensors.** Q8_0 is calibrated on released bytes: 6 expert
  matrices and 3 FFN prefixes, every fault ≥ 298x.

## Super-block scales must stay fp16/fp32 (a kernel constraint)

Rounding the fp16 scales d and dmin to bf16 is a plausible kernel shortcut. On released bytes, across 60 draws of dY
per tensor (M ∈ {1, 17, 128}):

- **Cost:** it costs a median 2.1-2.9x the floor on Q2_K, Q4_K and Q5_K, and as little as 1.3x on some draws. Q3_K,
  Q6_K and Q8_0 stay within the bound.
- **Detection by the error bar:** the per-case bar catches it in most draws, but not every time. Over the 60 draws its
  cost spans 1.3x to 2.9x the floor, against a bound of 2x. An earlier 6-case check (M ∈ {17, 128} x 3 seeds) put every
  case above the bound; that was the sample, not a guarantee.
- **Synthetic blocks hide it:** there it costs only 1.2-1.4x. In released blocks the two terms of
  w = d · sc · q − dmin · m largely cancel, which amplifies a scale error.
  - In a released Q4_K expert, ‖d · sc · q‖ is 2.35x ‖w‖ and ‖dmin · m‖ is 2.14x.
  - In the random blocks the min term is only 0.06x ‖w‖, so nothing cancels.
- **Not universal:** two released Q6_K attention tensors carry scales that are already exact in bf16, so rounding
  changes nothing there.

A statistical bar that catches a fault most of the time is not a pin. So the constraint is pinned by a **decode
probe**, not by the error bar, and any training path that refuses a kernel for rounding the scales keys on this probe. With dY = I, the backward returns W itself,
which must equal the reference decode bit for bit. Rounded scales change 16-25% of the weights on synthetic blocks of
every type, and 15-26% on every released tensor whose scales are not already bf16-exact, so the
probe catches them every time, on synthetic and released bytes alike (`test_kquant_backward.py`).

## Model level (`kernel/test_kquant_gradcheck.py`)

A tiny 2-layer model has a norm, a LoRA-adapted square projection, a gated FFN and a LoRA-adapted down projection, all
over frozen k-quant bytes.
- **The bf16 path against an fp64 model:** its input and adapter gradients sit 2e-3 to 1.7e-2 away (Q4_K, Q5_K, Q6_K,
  Q8_0), measured before the tests were written. A backward off by 0.1% falls inside that spread, so there is **no
  statistical model-level bar**.
- **The checks are exact instead:**
  - `torch.autograd.gradcheck` runs the same Function code with `compute_dtype=torch.float64`;
  - the bf16 path's gradients must equal, bit for bit, those of the same model over the dense bf16 weight.
- **Fast-mode gradcheck is not enough.** It passed a backward that used Wᵀ for the square projection; the full
  Jacobian (slow mode) caught it. Slow mode is therefore required, run w.r.t. one input token to keep it to seconds.
  Gradcheck catches a backward scaled by 1.001, a zeroed one and the transposed one.

## What this settles

- **The bars are registered as designed:** e_test ≤ min(2 · e_ctrl, 1e-2), a live control, and the adjoint identity
  to 1e-12. Every fault must fail a bar or the decode probe with `AssertionError`.
- **The calibrated types are Q8_0, Q4_K, Q5_K and Q6_K,** plus Q2_K and Q3_K on the same evidence.
- **The i-quants stay out of scope** (DESIGN.md §5).
