### `dense_gemv.gemv_bf16_bw` (K35): a dense bf16 single-row decode GEMV, shipped verbatim under a tolerance contract

A new module, `dense_gemv`, ships `gemv_bf16_bw(x, w, out=None)`. It is a bf16 GEMV for one activation row (a decode
step at T == 1), with fp32 accumulation, a bf16 output and no split-K.
- Its kernel body is experts4bit-qlora's P148 prototype verbatim; a test pins the source's sha256. That prototype was
  timed against cuBLAS on an RTX 5090 at Qwen3-30B-A3B's attention-projection shapes.
- The launch plan is frozen to P148's measured winners. Any other shape runs on `PLAN_DEFAULT` (2, 1024, 4), untimed.
- A multi-row call is refused (the caller keeps cuBLAS).
- The tolerance contract is checked compiled on a CUDA card: relative RMS at most 1.05 x cuBLAS's own, one-hot
  readback exact, bitwise run to run. The interpreter suite runs in CI.

Nothing routes through it by default, and no speed is claimed here: the consumer's served read is the timing of record
(`kernel/PREREG-k35-dense-bf16-decode-gemv.md`).
