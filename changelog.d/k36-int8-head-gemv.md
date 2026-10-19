### `dense_gemv.gemv_i8_bw` (K36): an int8-weight single-row decode GEMV for an int8 lm_head, shipped verbatim under a tolerance contract

`dense_gemv` adds `gemv_i8_bw(x, q, s, out=None)` and `quantize_rows_int8(w)`. Together they make a GEMV for one bf16
activation row over an int8 `[N, K]` weight with an fp32 per-row scale: `out[n] = s[n] * sum_k q[n, k] * x[k]`, with
fp32 accumulation and a bf16 output. It suits an int8 lm_head, which has half the bf16 bytes.
- **Provenance:** the kernel and the quantizer (symmetric per-row absmax, `s = absmax / 127`) are experts4bit-qlora's
  P148 prototypes verbatim, and tests pin both sources' sha256.
  - P148 timed that kernel against cuBLAS on an RTX 5090, at Qwen3-30B-A3B's 151936 x 2048 head.
  - experts4bit-qlora P150 ruled the quantizer's grid GO on quality.
- **The plan** is frozen to P148's winner at the head. Any other shape runs on `PLAN_I8_DEFAULT` (4, 1024, 4), untimed.
- **Refusals:** a multi-row call, a dtype other than bf16 `x` / int8 `q` / fp32 `s`, and non-contiguous weight rows or
  scales.
- **The tolerance contract** is checked compiled on a CUDA card, as P148's was:
  - at the head, relative RMS against the dequantized weights at most 1.05 x cuBLAS's on the bf16 head;
  - one-hot readback exact, and bitwise run to run;
  - every output within bf16's rounding.

  The interpreter suite runs in CI.

Nothing routes through it and nothing is quantized by default. No speed is claimed here: the consumer's served read is
the timing of record (`kernel/PREREG-k36-int8-head-gemv.md`).
