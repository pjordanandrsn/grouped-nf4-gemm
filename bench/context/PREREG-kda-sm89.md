# PREREG — KDA on SM89: does the consumer story survive its attention tier?

**Tier: GATE (S3). Status: DRAFT until stamped; stamp precedes the pod.**

K3 runs 69 of its 93 layers on Kimi Delta Attention. Moonshot's FlashKDA
kernels require SM90+; a 4090 is SM89, so consumer K3 rides fla-core's Triton
path — present (desk-audited: no arch gate) but unmeasured. The bars are
derived from the consumer decode budget, not from taste, and are fixed here so
"eh, that's probably fine" is not available after the number exists.

## Fixture

Community/secure RTX 4090 (SM89), torch cu128-class, `fla-core` current.
K3's KDA dims exactly: B=1, H=96, d_k=d_v=128 (hidden 7168 for the projection
GEMMs), bf16, gate params per the released config (lower_bound −5). Synthetic
tensors; this is a PERF gate — outputs checked finite, not correct.
Median-of-N with warmup; per-layer figures = kernel + the layer's projection
GEMMs.

## Bars (both sides pre-committed)

- **B1 — decode:** full KDA layer step (recurrent kernel + projections) ≤
  **3.0 ms/layer**. Rationale: consumer byte floor ≈ 32.1 GB / 23 GB/s ≈
  1.4 s/token; attention's budget is ≤15% of floor ≈ 210 ms; 210/69 ≈ 3.0.
  *Fail ⇒ consumer DECODE narrows to 5090/SM90+ and the roadmap + launch
  section say so explicitly.*
- **B2 — prefill:** KDA layer chunk time at T=32K ≤ **0.63 s/layer**.
  Rationale: the expert stream costs ~58 s per 32K chunk (1.45 TB / 25 GB/s);
  attention hides behind it iff per-layer KDA prefill ≤ 58/92 ≈ 0.63 s.
  *Fail ⇒ consumer PREFILL is attention-bound, and the S1b extraction
  economics + agentic-prefill story must be re-derived at the measured rate
  (cloud-prefill/KV-import hybrid becomes a roadmap item).*
- **B3 — verify widths (spec-dec):** T=4 and T=8 steps cost ≤ 2× the T=1
  step. Rationale: sub-linear verification is the premise the union economics
  ride on; their report claims it for their kernels, ours must be measured.

## Decision rule

B1∧B2 hold → the consumer story stands as written. Any bar fails → the
specific narrowing above is recorded in ROADMAP + LAUNCH-K3-SECTION before
launch, not after. No bar is reinterpreted post-hoc; a Triton-version caveat
may be recorded but does not flip a verdict.

## Cost

One 4090, ≤ 1 h, ≤ $1. Teardown + evidence discipline as standing.
