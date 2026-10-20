### `swiglu_in`: the bandwidth decode GEMV computes the MoE swiglu in its loads (experts4bit-qlora#1313)

- `gemm_4bit_grouped(..., swiglu_in=True)` takes bf16 `[T, 2K]` gate-then-up rows (a MoE layer's gate_up output)
  and returns the down projection of `silu(gate) * up`. Where the bandwidth-targeted single-row GEMV engages, each
  activation load computes `bf16(silu(gate) * up)`: `int4_b32.swiglu_rows`' expression in its order, then its one bf16
  rounding (`_bw_x`). The separate swiglu launch and its `[T, K]` intermediate go away. Every other route runs
  `swiglu_rows` first. Either way the output is bitwise `gemm_4bit_grouped(swiglu_rows(a_cat), ...)`.
- Opt-in by the call: the default (`swiglu_in=False`) leaves every route exactly as it was. New tally key `bw_swiglu`.
- `kernel/test_nf4_gemv_bw.py` (item 6) holds the bitwise check on a card at the down shapes of Qwen3-30B-A3B,
  Granite-3.1-3b-a800m and OLMoE-1B-7B, for both decodes, one pass and split-K; the fallbacks bitwise; malformed rows
  refused. On an RTX A2000 the six mutants (no bf16 rounding, a reassociated product, gate and up swapped, the K row
  stride, a fallback without the swiglu, the flag not passed) each fail it.
