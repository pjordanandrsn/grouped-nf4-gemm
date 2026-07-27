# PREREG — is the decode GEMV under-occupied, or structurally limited?

**Tier: CONFIRMATORY. Status: STAMPED before the pod was created.**
gnf4 @ `0668ba3`, e4b @ `3510248`. Both local, unpushed.

## A correction first

#40 concluded that 56.8% of the 235B step is grouped-GEMM tiling, and I attributed
it to the expert GEMM being *"an M=1 GEMV padded to `tl.dot`'s M≥16 minimum,
wasting ~94% of every tensor-core tile."*

**That is wrong.** `gemm_4bit_grouped` already dispatches `_gemv_nf4_grouped`
whenever `max(sizes) == 1`, with a tuned `_decode_plan` and a split-K variant —
*"decode: every group is one token; the reduction path skips the M-tile."* The
99.2 GB/s measured in #39 **was** the GEMV path. There is no `tl.dot` at decode
and nothing to write; I nearly spent a pod rebuilding what exists.

The measured gap is real and unexplained: the GEMV moves the same 56.62 MB that
the flat decode kernel moves at **487 GB/s**, and gets **99.2**.

## The hypothesis this replaces it with

`_decode_plan` returns the constant `(BLOCK_N=64, num_warps=2)`, its docstring
recording that as "the universal constant — median regret 1.000" from dense
sweeps on two grids. For N=3072, K=4096, T=8 on an 84-SM A40 that yields
`8 × ceil(3072/64) = 384` programs at 2 warps each ≈ **9 warps/SM against a
48-warp capacity, ~19% occupancy**, and split-K does not engage because
384 ≥ 2×84.

Low occupancy starves memory-level parallelism, which is a standard way to sit at
~20% of bandwidth while doing nothing wrong arithmetically.

`gemm_4bit_grouped` already exposes `decode_config=(BLOCK_N, num_warps)` and
`split_k` "for benchmark/ablation support only". This uses them.

## Predictions

- **G1a — the default is not optimal here.** Some swept config beats the default's
  99.2 GB/s by ≥ **1.5×**. *Falsified below 1.2×*, which would mean the GEMV is
  structurally limited and **tuning is abandoned** — the next move would have to
  be a different kernel structure, not a config.
- **G1b — and occupancy is why.** The winning config has **more warps than 2, or
  split_k > 1, or both**. *Falsified if the winner is (64, 2, 1)* — the current
  default — which would make G1a's win come from somewhere I have not identified.
- **G1c — the ceiling is the flat decode.** Best config reaches ≥ **40%** of
  487 GB/s (≥195 GB/s). *Reported, not gated*; the decode kernel does no
  reduction and no output write, so it is a bound, not a target.
- **G1d — GATE.** Every config returns the same answer as the default to within
  **2e-3** relative. Split-K reduces in fp32 on the host, so a wrong reduction
  would be silent.

## Pre-committed decisions

- **G1a confirmed** → `_decode_plan`'s universal constant is wrong for at least
  this (shape, card), and the fix is a plan change — cheap, no new kernel.
- **G1a falsified** → the GEMV is structurally limited, config tuning is
  abandoned, and #40's 2.19× prize needs a different mechanism than any of the
  three I have now proposed for it.

## Confounds

1. One shape (N=3072, K=4096, T=8) on one card. `_decode_plan`'s constant was
   chosen across dense sweeps on two other grids; finding it suboptimal here does
   **not** mean it is wrong there, and a plan change must not regress those.
2. The 487 GB/s ceiling is a flat streaming read with no reduction and no
   output — genuinely unreachable, useful only as a bound.
