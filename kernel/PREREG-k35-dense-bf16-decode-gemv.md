# K35 — a dense bf16 single-row decode GEMV (`dense_gemv.gemv_bf16_bw`): P148's prototype kernel, shipped verbatim under a tolerance contract fixed before any timing (registered 2026-10-19)

Lane number reserved on the bus and acknowledged by the maintainer (2026-10-19T06:57Z). Issue: experts4bit-qlora#1313
(single-stream decode on the default NF4 server). No rental: the contract runs interpreted in CI and compiled on an RTX
A2000 (correctness only).

## Why

experts4bit-qlora's P142 per-name census put the dense bf16 GEMVs (cuBLAS `kernel`) at the top of the removable time in
Qwen3-30B-A3B's one-request step on an RTX 5090. Fused q/k/v (5120 x 2048) and o (2048 x 4096) ran at 79 % and 75 % of
the card's DRAM rate. P148 (experts4bit-qlora `bench/p148`) timed a prototype Triton GEMV against cuBLAS in one
CUDA-graph harness on an RTX 5090. Its lever D1 read GO, projected onto P142's in-situ times at 0.131 ms a step, 2.99 %
of it. That is a microbench premise, never a gain. The winning configuration was the simplest one:
- 2-row blocks over full-K chunks of 1024, 4 warps;
- **no split-K**, which lost at every configuration by about 20 %;
- the same configuration at both shapes.

## What ships (`kernel/dense_gemv.py`)

- **`_gemv_bf16_kernel` is P148's kernel body VERBATIM.** Its source, from `def` to its last line, has sha256
  `f104918f516bb7b3d8f922b3f7b4b26db1117e1c12606429306dbb838e432bf0`, equal to `_gemv_bf16_kernel` in
  experts4bit-qlora `bench/p148/p148_kernels.py` at `caf5a28c` (that file's sha256 `9a7133fd…a06ac91`).
  `kernel/test_dense_gemv_interp.py` pins the hash. The kernel's `SPLITK` branch stays in the body but is never taken.
- **`gemv_bf16_bw(x, w, out=None)`** is the launch wrapper, the only new code. It launches exactly as P148's split_k == 1
  path does: grid `(cdiv(N, BLOCK_N), 1)`, `KSPAN` rounded up to whole `KC` chunks, `SPLITK=False`.
  - It serves **one row**: `x` is `[K]` or `[1, K]`, and M > 1 is refused (the caller keeps cuBLAS).
  - It takes bf16 only, with contiguous weight rows, and an optional preallocated `out` (a CUDA-graph capture must not
    allocate).
- **The plan is frozen:** `PLANS = {(5120, 2048): (2, 1024, 4), (2048, 4096): (2, 1024, 4)}`, P148's measured winners.
  `PLAN_DEFAULT = (2, 1024, 4)` covers any other shape, and a new shape joins only after a read of its own.
- **Nothing routes.** No model call reaches this kernel unless a consumer calls it.

**K timed the shipped code.** Because the kernel body is verbatim and only the launch wrapper is new, P148's
microbench timed this kernel. Any later edit to the body leaves the consumer's served read as the only timing of record,
and a change to the body must say so.

## The contract (fixed before any timing)

**On the card** (`kernel/test_dense_gemv.py`, `TRITON_INTERPRET=0`, an RTX A2000 for this registration):
1. **Against cuBLAS:** relative RMS against an fp64 reference at most **1.05 x** cuBLAS's own
   (`torch.nn.functional.linear` on the same bf16 row and weight). The served shapes are checked over 48 independent
   weights each (one model's layers' worth), the ragged 1031 x 2050 over 8.
2. **One-hot readback is exact:** a one-hot row reads one weight column bit for bit.
3. **Bitwise run to run:** the same call repeats bit for bit (no atomics).
4. **Rounding:** every output lies within bf16's rounding of the fp64 reference.

**Interpreted in CI** (`kernel/test_dense_gemv_interp.py`, `TRITON_INTERPRET=1`):
- the verbatim hash;
- the frozen plan;
- every output within one bf16 step of fp64 at both served shapes and at ragged ones (N off the 2-row block, K off the
  1024 chunk, K below one chunk), for `[K]` and `[1, K]` and a preallocated `out`;
- one-hot readback, exact;
- bitwise run to run;
- the refusals.

The interpreter truncates fp32 to bf16 where the compiled store rounds to nearest, so the 1.05 x contract against
cuBLAS is checked compiled only.

## What comes after

- **Timing of record:** experts4bit-qlora registers a served read in P143's form, after this lane ships in a
  grouped-nf4-gemm release. An `E4B_*` knob routes fused q/k/v and o at T == 1 through `gemv_bf16_bw`, read as ABBA at
  one and 16 requests, with a NOISY rung and the card's power limit recorded. No default changes before that read.
- **The int8 head** (P148's D2) waits on experts4bit-qlora P150's quality ruling. If GO, its kernel takes a lane number
  of its own, reserved on the bus.
- **Review:** one non-consumer seat, then the maintainer's final pass. The release is the maintainer's.
