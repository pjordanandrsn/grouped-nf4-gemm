# PREREG — what does the grouped kernel cost in perplexity?

**Tier: CONFIRMATORY. Status: STAMPED before the harness was written.**
Code: gnf4 @ `d478ffc`, e4b @ `a9bba8d`. Both local, unpushed.

## Why

#25 measured the grouped kernel shifting model logits by **12.9%** while leaving
greedy ids identical — the ids were absorbing the shift because the prompt was
peaked (top-1 0.911). The speed claim (the pair at **7.88×**, `enable_fast` worth
**1.32×** under routed staging) therefore stands with **no fidelity number beside
it**, which is exactly the incompleteness #17 forced onto the KV cache's
"3.56× memory for ~2.1% perplexity".

Perplexity is the instrument that priced the KV cache. This applies it to the
kernel.

## Fixture

OLMoE-1B-7B, NF4 experts, offloaded, **bulk staging held constant** so the only
variable is the kernel. Wikitext-2 test, **independent 2048-token chunks**, mean
NLL over identical chunks in both arms.

Method matters and is fixed in advance: sliding-window and independent-chunk
perplexities are **not comparable** (a prior cross-scoring artifact in this
project put two numbers 18.7% apart for that reason alone). Both arms use
independent chunks; only the kernel changes.

Arms: `reference` (per-expert loop) · `grouped` (`enable_fast`) ·
`routed+grouped` (gate, see Q1c).

## Predictions

- **Q1a — magnitude.** `|ppl_fast − ppl_ref| / ppl_ref` ≤ **5%**.
  *Falsified above 15%.* 12.9% on logits need not become 12.9% on perplexity:
  softmax is contractive on a peaked distribution, and the KV cache's much
  larger per-token perturbation cost only ~2.1%.
- **Q1b — DIRECTION, and it may be a gain.** `ppl_fast` ≤ `ppl_ref`.
  `fast.py` states the fused path accumulates in **fp32** where the reference
  materializes **bf16**, and that it "measured *more* accurate than the
  reference on every cell of the kernel's stamped property suite". If that holds
  at model scale, the kernel is not an accuracy *cost* at all and #25's framing
  ("unquantified accuracy cost") is wrong in sign. *Falsified if `ppl_fast`
  exceeds `ppl_ref` by more than 1%.*
- **Q1c — GATE.** `ppl(routed+grouped)` **exactly equals** `ppl(bulk+grouped)`.
  Routed staging is bit-identical, so this must hold to the last digit; any
  difference means the perplexity harness is nondeterministic and Q1a/Q1b are
  not interpretable. *Any inequality voids them.*

## Pre-committed decisions

- **Q1b confirmed (`ppl_fast` ≤ `ppl_ref`)** → #25's "unquantified accuracy cost"
  is **corrected in place**: the kernel is a speedup *and* a fidelity improvement,
  and the 12.9% logit shift is movement toward the fp32 answer rather than away
  from it. The pair becomes recommendable on both axes.
- **Q1b falsified** → `enable_fast` carries a real, now-quantified perplexity cost,
  which travels with the 1.32× everywhere the speed is quoted — the same
  treatment #17 imposed on the KV cache.
- **Q1a falsified above 15%** → the kernel is not fit for depth regardless of its
  speed, and the pair is not recommended at 94 layers on this evidence.

## Confounds, stated in advance

1. **OLMoE is 16 layers; the flagship is 94.** Compounding grows with depth, so
   whatever is measured here is a *lower bound* on the 235B's error. This
   prereg does not claim otherwise, and a depth-scaling claim needs its own run.
2. Both arms are NF4-quantized, so this prices the **kernel against the reference
   path**, not against bf16. That is the decision a user actually faces.
3. One model, one corpus, one chunking scheme.
