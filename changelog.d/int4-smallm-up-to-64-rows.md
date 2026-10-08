### `gemm_int4_b32_smallm` serves up to 64 rows: a 32- or 64-row M tile above 16 (`block_m`; e4b#846)

The K16 small-M int4 GEMM refused more than 16 rows, so experts4bit-qlora's attention projections fall back to a cached
bf16 copy and cuBLAS once a batched decode step passes 16 rows. On Qwen3-30B-A3B's 64-row step that copy is 1.81 GB a
step, against 510 MB on the int4 grid it is dequantised from.

The kernel was already generic in its M tile, so only the wrapper changes:
- `block_m=None` (the default) launches a 16-row tile up to 16 rows, a 32-row tile up to 32 and a 64-row tile up to
  64 (`smallm_block_m`). More than 64 rows is still refused.
- Up to 16 rows the launch is K16's exactly: the same grid, tile, plan, warps and stages, and the same bits as 0.43.0.
  A test records the launch and compares outputs with `torch.equal`, compiled on a GPU as well as in the interpreter.
- An explicit `block_m` (16, 32 or 64) must hold every row. A workspace built for a tile serves that tile and every
  smaller one.

Above 16 rows the outputs keep K16's contract: within one bf16 ulp of the dequant-then-GEMM reference, deterministic
for a config, and within one ulp across split-K and across tiles. Nothing in this package routes to it on its own. Its
speed is experts4bit-qlora's to read (lane P124).
