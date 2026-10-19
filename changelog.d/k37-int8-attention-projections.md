### `dense_i8.linear_i8` (K37): int8-weight projections at decode, one row and 2 to 16 rows, shipped verbatim under a tolerance contract

A new module, `dense_i8`, adds `linear_i8(x, q, s, out=None)`. It computes `x @ (q * s[:, None]).T` for 1 to 16 bf16
activation rows over an int8 `[N, K]` weight with an fp32 per-row scale, with fp32 accumulation and a bf16 output. That
is W8A16 for Mixtral-8x7B's attention projections, whose int8 copies have half the bf16 bytes.
- **Provenance:** both kernels, a one-row GEMV and a 2-to-16-row small-M kernel, are experts4bit-qlora's P149
  prototypes verbatim. Tests pin both sources' sha256.
  - P149's box K timed them against cuBLAS on an RTX 5090 at Mixtral's shapes and read K_GO, a microbench premise.
  - experts4bit-qlora P147 ruled the int8 grid GO on quality.
- **The grid** is K36's `dense_gemv.quantize_rows_int8`, re-exported rather than redefined. It is P149's quantizer to
  the bit.
- **The plans** are frozen to box K's choices at the two shapes (q/o 4096 x 4096, k/v 1024 x 4096). Any other shape runs
  the defaults, untimed, and no plan splits K.
- **Refusals:** more than 16 rows (a prefill-sized call keeps the caller's path), zero rows, a dtype other than bf16 `x`
  / int8 `q` / fp32 `s`, a K mismatch, and non-contiguous weight rows, activation rows or scales.
- **No speed is claimed:** nothing routes a model call; a consumer's served read is the timing of record.
