### K34 read (RTX 5090): NONE — the shipped K16 plan stays at the 32- and 64-row tiles; one cell (`o`, 64 rows) runs at 0.794× with BLOCK_N 32, about 1 % of the served step

Register: `gnf4.kernel.k34-k16-wide-plan-census.5090.2026-10-09`; `kernel/RESULTS-k34-k16-wide-plan-census.md`.
- **The reading** (`k34-5090-2`, exploratory). 48 plans of `gemm_int4_b32_smallm` were timed at Qwen3-30B-A3B's fused
  `qkv` (5120 × 2048) and `o` (2048 × 4096) projections, 48 layers in CUDA graphs, at the 32- and 64-row tiles.
  - At `qkv` the shipped plan (64, 128, 4, 4, 2) is the fastest of 48 at both tiles.
  - At `o`, 64 rows, BLOCK_N 32 with KC 256 reads 0.794×. The shipped plan launches 128 programs on 170 SMs there, so the
    gain is mostly grid fill.
  - Every tested layer-0 plan passed max absolute error ≤ 2^-7 times the max absolute fp32 reference output (sanity
    gate), and the instrument read 0.998–1.000.
- **The predictions.** The instrument held. The 64-row gain held for `o` and missed for `qkv`. The guessed winner shape
  (8 warps, a smaller split-K) and the predicted CANDIDATE missed.
- **What follows (registered).** Nothing moves. `o`/64's cell is about 1 % of the served 64-row step, so it gets no
  confirmatory read on its own; a per-shape plan in `plan_smallm` would ride with a larger lever.
- **Cost:** $0.074.
