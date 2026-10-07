### K33 read (RTX 5090): LEVER — the bandwidth-targeted NF4 decode GEMV runs Qwen3-30B-A3B's single-row expert projections at 0.341× the served route

Register: `gnf4.kernel.k33-nf4-decode-gemv-bw.5090.2026-10-07`; `kernel/RESULTS-k33-nf4-decode-gemv-bw.md`.
- **The reading** (`k33-5090-1`). The run covered every layer at each family's served shape: 8 rows, the top-8 experts,
  one CUDA graph per projection.
  - On Qwen3's pair, `GNF4_GEMV_BW=1` (`prmt32`) takes 0.961 ms against dot-pad's 2.818 (gate_up 0.332, down 0.357),
    at 0.74 / 0.63 of the copy floor. This is with K33's swept per-shape plans and `GNF4_PDL=0`; the default plan
    read about 0.37× on the pair, and PDL (on by default for 8-row calls) made the down projection 1.17× slower.
  - `prmt32` is bitwise the exact tree decode, and every decode meets the tolerance contract.
  - Granite and OLMoE read 0.22–0.27× their scalar GEMV.
- **The predictions.** Numerics, the faster decode, the plan shape (BLOCK_N 16, split-K 1), the family ratios, the
  instrument and the verdict held.
  - The Qwen3 ratios came in faster than their bands.
  - The int4-b32 comparator was slower than predicted: int4 / bw 1.03–1.22.
  - PDL slowed Qwen3's and OLMoE's down projections by about 17 %.
- **What follows (registered).** `GNF4_GEMV_BW` stays opt-in here. experts4bit-qlora registers the served lane P116 (W1
  and W16, P110's quality bar). On its DEFAULT read this repository fills `_BW_SHAPES` and makes `auto` the default.
- **Cost:** $0.037.
