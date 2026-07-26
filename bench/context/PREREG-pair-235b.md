# PREREG — the pair on the 235B: routed staging × grouped kernel, full matrix

**Tier: CONFIRMATORY. Status: STAMPED before the pod was created.**
Code: gnf4 @ `ea9fa71`, e4b @ `c67c421` (routed staging 6/6 incl. composition).
Both local, unpushed.

## Why a matrix and not one arm

Three findings each measured one half of the step, and each was misread once:

- **#21** measured the grouped kernel at **1.6%** and called it "a correctness
  fix, not a speedup" — true of a step where transfer outweighed compute 15:1.
- **#22** measured routed staging at **5.95×** and left a 0.577 s residual.
- **#23** found that residual is the **per-expert Python loop** (0.414 s/token on
  a 4090), which the grouped kernel cuts **2.03×** — so #21's reading was an
  artifact of masking, not a property of the kernel.

The pair has **never been run together on the real model**. All four cells are
measured here so the interaction is observed rather than inferred, and so #21's
1.6% and #23's 2.03× can be reconciled on one box with one load.

## Fixture

Qwen3-235B-A22B, 2×A100-SXM-80GB, NF4 experts pinned, KV NF4 host-resident,
`prefetch=False` (routed staging refuses prefetch-linked handles), greedy, 12 new
tokens, median of 2, **one process** so all cells share a load and a link.

| cell | weights | experts |
|---|---|---|
| `bulk+ref` | bulk staging | reference loop | *the shipped default*
| `bulk+grouped` | bulk staging | grouped kernel | *#21's configuration*
| `routed+ref` | routed staging | reference loop | *#22's configuration*
| **`routed+grouped`** | routed staging | grouped kernel | *the pair*
| `routed+grouped` @32768 | | | *context under the fast step*

## Predictions

- **S1a — the pair.** `routed+grouped` / `routed+ref` ∈ **[0.72, 0.92]**. #23
  measured the loop at 4.403→2.174 ms/layer on a 4090 (0.210 s/token saved); an
  A100's host-side Python cost is similar while its device work is faster, so the
  saving should be of that order against a 0.936 s step. *Falsified outside
  [0.60, 1.00]* — above 1.00 would mean the kernel costs time here.
- **S1b — masking, quantified.** The grouped kernel's benefit is **larger under
  routed staging than under bulk**:
  `(bulk+ref)/(bulk+grouped)` < `(routed+ref)/(routed+grouped)`, by ≥ 0.10.
  This is #23's claim stated as an interaction and is the point of the matrix.
  *Falsified if the gap is < 0.03 or inverts.*
- **S1c — GATE.** Greedy ids for all four cells are **identical**. Routed staging
  is bit-identical by construction and the grouped kernel is within kernel
  rounding, so the token stream must not diverge at this length. *Any divergence
  voids S1a/S1b.*
- **S1d — end to end.** `bulk+ref` / `routed+grouped` ≥ **6.0×**. *Falsified
  below 4.0.*
- **S1e — context, restated.** At 32768 the pair costs ≤ **1.25×** its own 512
  step. #22 measured context at 5% of a 0.936 s step; on a faster step the same
  absolute cost is a larger share, and this records where it lands rather than
  predicting it precisely.

## Pre-committed decisions

- **S1b confirmed** → #21's "1.6%, a correctness fix not a speedup" is amended in
  place with the interaction, and the two optimizations are documented as a
  **pair** rather than alternatives.
- **S1b falsified** → #23's 4090 decomposition does not transfer to the A100, and
  the 2.03× stays a 4090 result with an explicit scope note.
- **S1a confirmed and S1d ≥ 6.0** → routed staging + `enable_fast` becomes the
  documented configuration for streamed MoE inference, with this matrix as its
  receipt.
- **Throughput stays suppressed in `plan.py` regardless.** The byte model was
  10.7× wrong on bulk and 2.1× wrong on routed; one more arm does not refit it.

## Confounds, stated in advance

1. All cells share one process and one load, which removes load-to-load variance
   but means an ordering effect (allocator state, thermal) would hit later cells.
   Order is fixed as listed, worst-case for the pair (it runs last).
2. #23's loop cost was measured on a 4090 with resident experts; the A100 figure
   is genuinely unknown, which is why S1a is a band and not a point.
3. One model, one box, 12 new tokens, median of 2.
