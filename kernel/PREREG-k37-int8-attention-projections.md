# K37 — int8-weight projections at decode, one row and 2 to 16 rows (`dense_i8.linear_i8`): P149's prototype kernels, shipped verbatim under a tolerance contract fixed before any timing (registered 2026-10-19)

Lane number reserved on the bus and acknowledged by the maintainer (2026-10-19). Issue: experts4bit-qlora#1313
(single-stream decode on the default NF4 server); experts4bit-qlora lane P149. No rental: the contract runs interpreted
in CI and compiled on an RTX A2000 (correctness only).

## Why

- **The lever.** experts4bit-qlora's P147 ruled I8ATTN GO: rounding Mixtral-8x7B's 128 attention projections
  symmetrically per output row to int8, with an fp32 row scale, keeps the W8A16 split premise's quality gate at both
  gates. At W1 the upper bound was 0.002325 against 0.011560, and argmax agreement was 0.9813. The int8 copies halve the
  bytes a decode step reads for those projections.
- **The premise.** P149's box K (`p149k-5090-8`, one RTX 5090 at a 575 W power limit) timed two prototype Triton
  kernels over those int8 weights against cuBLAS on the bf16 weights, in a CUDA-graph microbench at Mixtral's shapes.
  It read **K_GO**: the projected in-situ saving at one row was 0.983796 ms against the registered bar of 0.1625 ms
  (2 % of P145's 8.125 ms period).
  - That is a microbench premise, never a gain.
  - Per shape at one row: q and o (4096 x 4096) read 1.485 ms for cuBLAS against 0.759 ms for the kernel; k and v
    (1024 x 4096) read 0.580 against 0.306.
  - At 16 rows the small-M kernel read 0.790 ms against cuBLAS's 1.555 at q and o, and 0.522 against 0.569 at k and v.
- **P149's contract held on the box.** At both shapes and at both row counts, every configuration's error against its
  own dequantized weights stayed within 1.05 x cuBLAS's error against the bf16 weights. For example, at q and o with
  one row it was 1.670e-3 against cuBLAS's 1.676e-3.

## What ships (`kernel/dense_i8.py`, a new module beside `dense_gemv`)

- **The two kernels are P149's VERBATIM**, compared with experts4bit-qlora `bench/p149/p149_kernels.py` at `ffb1632c`
  (that file's sha256 is `8aa099d8…7354f1`). Each source runs from `def` to its last line, LF:
  - `_gemv_i8_kernel` (one row; P148's GEMV with K33's split-K): sha256
    `b96c4fd3e5c89cac00028546d945a18e2e572bf0d0763b9d67cf5ca1376acc3b`;
  - `_gemm_i8_small_kernel` (2 to 16 rows; an int8 tile dequantized to bf16 in registers, exact since |q| <= 127, then
    `tl.dot` with fp32 accumulation and the scale applied once): sha256
    `77be4e84ee769d5f2ce5eb50e85f94cae450d719ec59766d01f234e51d40b52c`.

  `kernel/test_dense_i8_interp.py` pins both hashes. P149's `DOT_F32` switch comes with them: under Triton's
  interpreter the small-M kernel's dot runs in fp32, because the interpreter runs `tl.dot` on bf16 operands' raw bits.
  Compiled code always runs the bf16 dot.
- **The quantizer is K36's, re-exported, not a second function.** `dense_i8.quantize_rows_int8` is
  `dense_gemv.quantize_rows_int8`, the same object. P149's own prototype quantizer differs from it only in source text:
  it adds `.detach()`, which is moot for these inference kernels, and `.contiguous()` on `q`. K36's `q` keeps a
  strided (for example transposed) input's layout, and `linear_i8` refuses non-contiguous `q` rows rather than inheriting
  them silently, so a caller passes `q.contiguous()`. The arithmetic is the same: `s = absmax / 127`, round to nearest, clamp to [-127, 127], and `s = 1` for an
  all-zero row. The interpreter test runs P149's prototype text (pinned by sha256) beside K36's, and checks bitwise-equal
  `(q, s)` on random, all-zero, single-spike and extreme rows and on transposed (non-contiguous) weights, and that
  `linear_i8` refuses the strided `q` and reads `q.contiguous()` exactly as P149's.
- **`linear_i8(x, q, s, out=None)`** is the launch wrapper, the only new code.
  - **One row** (`x` a bf16 `[K]` or `[1, K]`) launches the GEMV exactly as P149's `gemv_i8` does with one split: grid
    `(cdiv(N, BLOCK_N), 1)`, `SPLITK=False`.
  - **2 to 16 rows** (`x` a bf16 `[M, K]`) launch the small-M kernel exactly as P149's `gemm_i8_small` does with one
    split.
  - **Refused:** M > 16 (a prefill-sized call keeps the caller's own path), M == 0, any dtype other than bf16 `x`,
    int8 `q` and fp32 `s`, a K mismatch, `q` rows or `x` rows that aren't contiguous, an `s` that isn't a contiguous
    `[N]`, and a wrong `out`.
  - `q` is an int8 `[N, K]` with contiguous rows, and `s` a contiguous fp32 `[N]`. `out` may be preallocated (a
    CUDA-graph capture must not allocate). It quantizes nothing: the caller makes `(q, s)` once.
- **The plans are frozen at box K's choices** (`p149k-5090-8`), as (BLOCK_N, KC or BK, num_warps):
  - one row: `PLANS_GEMV = {(4096, 4096): (4, 512, 4), (1024, 4096): (4, 2048, 8)}`;
  - 2 to 16 rows: `PLANS_SMALL = {(4096, 4096): (32, 256, 4), (1024, 4096): (32, 256, 4)}`.

  Any other shape runs `PLAN_GEMV_DEFAULT = (4, 512, 4)` or `PLAN_SMALL_DEFAULT = (32, 256, 4)` **untimed**, and a new
  shape joins only after a read of its own. No plan splits K, so the kernels' `SPLITK` branches are never taken.
- **Nothing routes,** and nothing quantizes a model. No model call reaches these kernels unless a consumer calls them.

**Box K timed the shipped code.** The kernel bodies are verbatim, the plans are box K's, and only the launch wrapper is
new, so box K's microbench timed these kernels at these plans. Any later edit to a body leaves the consumer's served
read as the only timing of record, and a change to a body must say so.

## The contract (fixed before any timing)

**On the card** (`kernel/test_dense_i8.py`, `TRITON_INTERPRET=0`; an RTX A2000 for this registration, sm_86):
1. **Against cuBLAS, at Mixtral's two projection shapes,** at M = 1 and M = 16: P149's box K contract, unchanged.
   - Take one bf16 weight `W` (`randn / sqrt(K)`), its int8 copy `(q, s) = quantize_rows_int8(W)`, and four
     activation draws.
   - The kernel's worst relative RMS against the fp64 reference of its own dequantized weights (`q * s`) must be at
     most **1.05 x** cuBLAS's worst against fp64 on the bf16 weights (`torch.nn.functional.linear(x, W)`).
2. **One-hot readback is exact:** a one-hot row reads `bf16(s * q[:, j])` bit for bit, on both paths.
3. **Bitwise run to run:** the same call repeats bit for bit (no atomics).
4. **Rounding:** every output lies within bf16's rounding of the fp64 reference of the dequantized weights, at
   Mixtral's shapes and at ragged shapes (1031 x 2050, 37 x 2560, 5 x 1000), for M in 1, 2, 7 and 16.

**What the A2000 can't see.** sm_86 runs the same Triton source, but not sm_120's code generation. The RTX 5090 path
was exercised only by P149's box K (`p149k-5090-8`: the 135-test suite compiled on the card, and this contract at every
configuration).

**Interpreted in CI** (`kernel/test_dense_i8_interp.py`, `TRITON_INTERPRET=1`):
- both verbatim hashes;
- the frozen plans and defaults, and that the wrapper never takes a split branch;
- the quantizer re-exported, and P149's to the bit;
- every output within one bf16 step of the dequantized fp64 reference, at Mixtral's shapes and ragged ones, for M in 1,
  2, 7 and 16, for `[K]` and `[M, K]` rows and a preallocated `out`;
- one-hot readback, exact as this mode stores it;
- bitwise run to run;
- the refusals.

The interpreter truncates fp32 to bf16 where the compiled store rounds to nearest, so the 1.05 x contract is checked
compiled only.

## What comes after

- **The consumer:** experts4bit-qlora's `Int8Linear` and the `E4B_SERVE_ATTN_INT8` knob. They route Mixtral's
  attention projections at decode (one row and 2 to 16 rows) through `linear_i8`, over a `(q, s)` made once at load,
  with engagement counters.
- **Timing of record:** P149's box Q+S, the quality rule at decode and then the served read, under P149's registration
  and its amendments. No default changes before that read.
- **The release:** the maintainer's.
- **Review:** one non-consumer seat, then the maintainer's final pass.
