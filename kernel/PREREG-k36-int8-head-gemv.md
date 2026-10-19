# K36 — an int8-weight single-row decode GEMV for an int8 lm_head (`dense_gemv.gemv_i8_bw`): P148's int8 prototype kernel and quantizer, shipped verbatim under a tolerance contract fixed before any timing (registered 2026-10-19)

Lane number reserved on the bus and acknowledged by the maintainer (2026-10-19). Issue: experts4bit-qlora#1313
(single-stream decode on the default NF4 server). No rental: the contract runs interpreted in CI and compiled on an RTX
A2000 (correctness only).

## Why

- **The lever.** experts4bit-qlora's P148 read lever D2, an int8 lm_head, GO in a CUDA-graph microbench on an RTX 5090.
  The head is 151936 x 2048 on Qwen3-30B-A3B, rounded symmetrically per row to int8 with an fp32 row scale, which halves
  its bytes. P148 timed a prototype Triton GEMV over that head against cuBLAS on the bf16 head.
  - Projected onto P142's in-situ head time, it saves 0.189546 ms a step: 4.84 % of the unprofiled 3.913 ms
    one-request step (P142's profiled 4.380 ms is context only).
  - That is a microbench premise, never a gain.
  - All 18 configurations read between 0.188 and 0.192 ms, near the card's bandwidth floor. The winner was 4-row
    blocks over 1024-wide K chunks, 4 warps.
- **P148's contract.** The kernel's error against its own dequantized weights read 1.660e-3, where cuBLAS's against
  the bf16 head read 1.665e-3.
- **Quality came first.** experts4bit-qlora P150 ruled on quality before any kernel shipped. Rounding only the lm_head
  to this quantizer's grid keeps P125's gate at both of its gates, on held-out windows (GO):
  - t1: upper bound 0.001225 against 0.005205;
  - k16: 0.001130 against 0.005659.

## What ships (`kernel/dense_gemv.py`, beside K35)

- **The kernel and the quantizer are P148's VERBATIM**, compared with experts4bit-qlora `bench/p148/p148_kernels.py`
  at `caf5a28c` (that file's sha256 is `9a7133fd…a06ac91`). Each source runs from `def` to its last line, LF:
  - `_gemv_i8_kernel`: sha256 `729955a718f9d0887cfabc5dca8e8644726d093703579ea1eb86e703f7cb59f7`;
  - `quantize_rows_int8`: sha256 `8ea6dd9ee1b52594419f40339272059e413956aaed2a2f02d382450f854e8c6c`.

  The quantizer is symmetric per-row absmax: `s = absmax / 127`, with `s = 1` for an all-zero row.
  `kernel/test_dense_gemv_i8_interp.py` pins both hashes.
- **`gemv_i8_bw(x, q, s, out=None)`** is the launch wrapper, the only new code. It launches exactly as P148's `gemv_i8`
  does: grid `(cdiv(N, BLOCK_N),)`.
  - It serves **one row**: `x` is a bf16 `[K]` or `[1, K]`, and M > 1 is refused.
  - `q` is an int8 `[N, K]` with contiguous rows, and `s` a contiguous fp32 `[N]`.
  - `out` may be preallocated (a CUDA-graph capture must not allocate).
  - It quantizes nothing: the caller makes `(q, s)` once.
- **The plan is frozen:** `PLANS_I8 = {(151936, 2048): (4, 1024, 4)}`, P148's measured winner. `PLAN_I8_DEFAULT =
  (4, 1024, 4)` covers any other shape, untimed, and a new shape joins only after a read of its own.
- **The module now imports torch at load,** because the verbatim quantizer reads it as a module global. torch is the
  package's declared dependency.
- **Nothing routes,** and nothing quantizes a model. No model call reaches this kernel unless a consumer calls it.

**P148 timed the shipped code.** The kernel body is verbatim and only the launch wrapper is new, so P148's microbench
timed this kernel. Any later edit to the body leaves the consumer's served read as the only timing of record, and a
change to the body must say so.

## The contract (fixed before any timing)

**On the card** (`kernel/test_dense_gemv_i8.py`, `TRITON_INTERPRET=0`, an RTX A2000 for this registration):
1. **Against cuBLAS, at the served head:** P148's D2 contract, unchanged. Take one 151936 x 2048 bf16 head `W`, its int8
   copy `(q, s) = quantize_rows_int8(W)`, and 16 rows. For each row:
   - the kernel's error is its relative RMS against the fp64 reference of its own dequantized weights (`q * s`);
   - cuBLAS's error is `torch.nn.functional.linear(x, W)`'s against fp64 `W @ x`;
   - the kernel's must be at most **1.05 x** cuBLAS's.
2. **One-hot readback is exact:** a one-hot row reads `bf16(s * q[:, j])` bit for bit.
3. **Bitwise run to run:** the same call repeats bit for bit (no atomics).
4. **Rounding:** every output lies within bf16's rounding of the fp64 reference of the dequantized weights. This holds
   at the head and at ragged shapes (1031 x 2050, 37 x 2560, 5 x 1000).

**Why the 1.05 x bound is held at the head only.** It compares the output roundings of two different weights (`W` and
`q * s`), so its noise falls with N. At 151936 outputs per row that noise is a fraction of a percent; at 1031 it would be
a few percent. The ragged shapes take the per-element bound instead, which is stricter for each output.

**Interpreted in CI** (`kernel/test_dense_gemv_i8_interp.py`, `TRITON_INTERPRET=1`):
- both verbatim hashes;
- the frozen plan;
- the quantizer's grid:
  - int8 in [-127, 127], and `s = absmax / 127` in fp32;
  - each row's absmax lands at +-127;
  - every weight lies within half a step of its dequantized value;
  - an all-zero row reads `s = 1`, `q = 0`;
- every output within one bf16 step of the dequantized fp64 reference:
  - at the head's K under the head's plan (N = 4096; the full head runs compiled);
  - at ragged shapes;
  - for `[K]` and `[1, K]` rows and a preallocated `out`;
- one-hot readback, exact as this mode stores it;
- bitwise run to run;
- the refusals.

The interpreter truncates fp32 to bf16 where the compiled store rounds to nearest, so the 1.05 x contract is checked
compiled only. Eight mutants of the module each fail the interpreter suite:
- the kernel drops the scale;
- the quantizer's grid moves;
- the plan, or the default plan, drifts;
- M > 1 is not refused;
- the scale's shape, or `q`'s dtype, goes unchecked;
- the grid is halved.

## What comes after

- **The consumer:** an experts4bit-qlora knob that routes the lm_head's one-row decode GEMV through `gemv_i8_bw`, over a
  `(q, s)` made once at load.
- **Timing of record:** D2's served read, in P143's form under P151's terms: the 575 W floor, and the unprofiled step as
  the denominator. It is registered on its own, and no default changes before that read.
- **The release:** the maintainer's.
- **Review:** one non-consumer seat, then the maintainer's final pass.
