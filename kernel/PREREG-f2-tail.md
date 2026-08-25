# PREREG — F2: the graph-step tail (attention combine + QKV launches)

Registered 2026-08-25, before any on-box measurement. Basis: the
BV2-cycle census (graph anchor 7.35 ms; eager confounds disclosed
there). The graph step's identifiable tail: attn-proj cuBLAS GEMVs
1.92 ms across ~241 launches/step, attention split+combine 0.53 ms
(the combine is 48 extra launches/step), router 0.31, memcpy 0.15.

## Treatments

- **T1 (gnf4): fuse the combine into the f32 decode split.**
  `FUSE_COMBINE` + arrival counters already exist in this file for
  BOTH the packed and the fp8-compute kernels; the default f32 split
  path is the only one that ignores the `fuse_combine=True` kwarg and
  always launches `_fp8_combine` separately. Port the existing
  mechanism (last-arriving CTA per (seq, kv-head) combines in-kernel).
  Gate: **bitwise** equality vs the two-kernel path on randomized
  pools, shapes, and n_split — the ported epilogue reduces partials in
  the same fixed 0..n_split order as the standalone kernel, so the
  arithmetic is order-identical by construction; only the launch
  structure changes.
- **T2 (e4b): fuse Q/K/V projections at load.** Stock HF runs three
  cuBLAS GEMVs per layer over the same input; one fused
  `[Hq·D + 2·Hkv·D, hidden]` weight computes all three in one launch,
  split by views. **Bar derivation note (pre-registration, from the
  CPU gate itself):** the first draft claimed bitwise row-identity
  ("rows are independent dots"). FALSE as a mechanism guarantee — BLAS
  kernel selection depends on the output dim, so fusing can change a
  row's accumulation ORDER. Measured while writing the CPU gate:
  rel 2.9e-7 at a small sgemm shape (fp32-reorder class), bitwise at
  the real shape — a property of that box's kernel table, not of the
  mechanism. The honest equivalence class is reorder noise. Gate:
  per-projection `max|Δ| ≤ max|ref| · 2⁻⁷` (the K6 relative frame,
  `max_abs_ref` recorded), plus the end-to-end identity gate below.

## Identity gate (end-to-end, per arm)

Greedy continuations vs the OFF/OFF arm, same seed and prompts.
- T1-only: **exact token identity required** (bitwise treatment; any
  divergence is a bug, REFUSE).
- T2 and T1+T2: **first divergence step ≥ 32** (the K6-B
  MIN_DIVERGE_STEP frame — reorder noise may legitimately flip a
  near-tie argmax late; a mechanism bug diverges early). Full identity
  is the expected outcome and is reported either way, with the
  divergence step and both token streams in the receipt. The
  check-traces degeneracy law applies to every stream.

## Arms (one anchor-compliant box; the hunt's pre-gate applies)

OFF/OFF A/A → T1 only → T2 only → T1+T2, all on the graph loop,
tokens captured. Per treatment and combined: step time + identity.

## Bars (gain frame; anchor class 7.35 ms)

- **PASS** — combined step cut ≥ 0.35 ms (ratio ≤ 0.952) with the
  identity gate green on every arm; ships both as defaults.
- **PARTIAL** — cut ≥ 0.15 ms; ships whichever single treatment
  carries a real gain under A/A.
- **REFUTED** — < 0.15 ms; the tail is not addressable at this cost.
- REFUSE: T1 token divergence (no numerics excuse), T2/T1+T2
  divergence before step 32, T2 projection check outside the 2⁻⁷
  frame, A/A wider than half the PASS margin, anchor non-compliance.

## Calculator

`f2_verdict.py`, self-tested; receipts in `kernel/receipts-f2/`.
