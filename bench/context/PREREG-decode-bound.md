# PREREG — is every NF4 kernel decode-bound? and re-decompose on the right KV

**Tier: CONFIRMATORY. Status: STAMPED before either pod was created.**
gnf4 @ `90780fb`, e4b @ `3510248`. Both local, unpushed.

## Two measurements, one of which should have come first

**(A) Decode isolation.** #38 abandoned the NF4 attention kernel after a
byte-count profile diagnosed it wrong: it sat at 2.2% of bandwidth, and removing
69% of its traffic made it *slower*. The binding term was the inner loop.

The grouped NF4 GEMM — now **46%** of the 235B step — has the **same inner-loop
primitives** (nibble unpack, LUT gather) and sits at **1.3%** of A100 HBM
(27.3 GB/s of 2039), independently 3.9% on a 4090 (#23). The roofline says 75× of
headroom. #38 says that exact argument misled me an hour ago.

So: time the decode **in isolation** rather than inferring it. This is the
experiment that would have saved #38 had it been run first.

**(B) Re-decompose on bf16 KV.** #36's split (experts 46.4% / attention 44.5%)
was measured with `residence="host"` NF4 KV — the setting #37 then showed is
wrong for these contexts. Attention's share is inflated by a config choice, so
the number driving all of this planning is not trustworthy.

## Predictions — (A) decode isolation

Three Triton kernels over the same packed bytes: `read` (load and sum bytes),
`decode` (load, unpack nibbles, LUT gather, scale, sum), `gemm` (the shipped
grouped path).

- **A1a — decode dominates.** `decode` ≥ **50%** of `gemm` time. *Falsified below
  25%*, which would mean the GEMM's cost is elsewhere and the roofline gap is a
  tiling problem after all.
- **A1b — the read is nearly free.** `read` ≤ **15%** of `gemm`. *Falsified above
  35%* — that would make it genuinely memory-bound and #38's lesson would not
  generalize.
- **A1c — decode is far off its own roofline.** `decode` achieves < **20%** of the
  card's bandwidth on the bytes it reads. *Reported, not gated.*

## Predictions — (B) re-decomposition on bf16 KV

- **B1a — attention's share collapses.** attention < **20%** of the step, from
  44.5%. *Falsified above 35%*, which would mean the KV dial was not what made
  attention expensive and #37's correction was too strong.
- **B1b — experts become clearly dominant.** experts > **60%** of the step.
  *Falsified below 45%.*
- **B1c — still GPU-saturated.** busy > **90%**, so the CUDA-graphs verdict from
  #36 survives the config change. *Falsified below 75%.*

## Pre-committed decisions

- **A1a confirmed** → the target is the **decode primitive itself**, shared by
  every NF4 kernel in the project, and that becomes the next build. Note the
  prior: a select-tree reconstruction replacing the LUT gather was tried during
  the fused-dequant work and reverted as *slower*, so "just drop the LUT" is
  already known not to be the answer.
- **A1a falsified** → the grouped GEMM has real memory headroom and the target is
  its tiling and vectorization, not the decode.
- **B1a falsified** → #37's correction is over-stated and the KV dial goes back
  under review.

## Confounds

1. (A) runs on a small card and (B) on the 235B box; the *ratios* are what
   transfer, not the absolute times — a rule this session has broken before and
   is stating up front.
2. (B) shares a load across cells, so ordering effects land on later cells.
