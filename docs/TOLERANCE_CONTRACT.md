# Tolerance contract — "not bit-exact, here is why, here is the measured bound"

*Phase 0.4. Said first, by us. This is the brand item.*

## Why the output is NOT bit-identical to dequantize+linear

The per-element dequantization itself is **exact and identical**: `codebook[nibble] × absmax`,
the same arithmetic `dequantize_4bit` performs. All divergence comes from **GEMM reduction
order** — the fused kernel accumulates in fp32 down its own K-tile schedule, while the
dequant path hands a materialized bf16 weight to cuBLAS with its own split-K order. This is
the same property our own gemm_4bit-routing commit documented (§9a of METHODOLOGY pinned
bit-exactness only because both paths there shared one reduction; a fused MMA mainloop does
not). Floating-point addition is not associative; different orders → different last bits.

## The registered fidelity ORDERING (falsifiable, and the stronger claim)

The dequant path rounds every weight to **bf16 in global memory** before the GEMM. The fused
kernel dequantizes into **fp32 registers** and accumulates fp32 — strictly less intermediate
rounding. We therefore register, in advance:

> **P-fid:** against an fp64 reference on identical inputs, the fused kernel's error is
> **≤ the dequant+linear path's error** (median over the census shapes, both projections,
> all three regimes).

If P-fid fails, that is a kernel bug (not an excuse) — find it or report it.

## The bound (comparative, self-calibrating — the S3-parity lesson applied)

A fixed epsilon invites both false alarms and quiet laxity. The gate is comparative:

- **B-rel:** per-shape relative Frobenius error vs fp64 reference
  `‖out − out_fp64‖_F / ‖out_fp64‖_F` must be **≤ 2× the dequant+linear bf16 path's** error
  on the same inputs, per census shape, per regime.
- **B-abs (backstop):** median relative error ≤ 1e-2 (bf16-regime sanity bound; catches a
  catastrophically wrong kernel even if the baseline is also broken).

## Property tests that enforce it (Phase 2.4 suite, specified now)

1. Census-shape sweep: every (model, proj) × {M=1, M=p50, M=p95} × both layouts.
2. **Adversarial absmax**: per-block absmax spanning {tiny (1e-30), huge (1e30), mixed
   alternating, denormal-adjacent} — the LUT×absmax multiply must not flush or overflow
   differently than `dequantize_4bit`.
3. **Expert-boundary cases**: empty groups (expert with 0 tokens), single-token groups,
   all-tokens-one-expert, group count G < E, non-contiguous expert_ids, K == blocksize
   (single-block rows).
4. Nibble-order exhaustiveness: a [1×K] probe row per codebook entry position — catches
   packed-layout/endianness mistakes that average away in random tests.
5. P-fid and B-rel asserted on every run; bounds recorded per shape into the receipts JSONL.

## What we will publish

The measured per-shape bound table, alongside the claim — never "bit-exact", always
"within the registered bound, fidelity-ordered above the dequant path, here are the tests."
