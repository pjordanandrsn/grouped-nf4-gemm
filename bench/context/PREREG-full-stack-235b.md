# PREREG — all three, composed, on the 235B

**Tier: CONFIRMATORY. Status: STAMPED before the pod was created.**
gnf4 @ `3e3ecbb`, e4b @ `7a5840a`. Both local, unpushed.

## What has never been measured together

Three optimizations, each verified alone, never composed on one box:

| | measured | where |
|---|---|---|
| routed staging | 5.95× | 235B, #22 |
| + grouped kernel | 6.97× | 235B, #31 |
| + speculative d=2 k=8 | 1.330× | **30B, 48 layers**, #33 |

The stack figure in #33 (~9.3×) multiplies a 48-layer result by a 94-layer one
across different boxes and is labelled there as **not a measured number**. This
measures the whole ladder in **one process, one box, one link**.

## Fixture

Qwen3-235B-A22B, 2×A100-80GB, NF4 experts pinned, KV NF4 host-resident,
`prefetch=False`, natural prompt, greedy, one load. Four rungs in order:
`bulk+ref` → `routed+ref` → `routed+grouped` → `routed+grouped+speculative`,
each dumping results and gates immediately (the harness that lost everything to
end-of-script scoring was fixed in #31).

## Predictions

- **F1a — speculative pays at 94 layers as it did at 48.**
  `all-three / routed+grouped` ∈ **[0.70, 0.85]** (a 1.18–1.43× gain). At
  ~22 GB/s the 235B moves 3.86 ms/layer against 2.17 ms/layer of compute, so it
  is transfer-bound and speculation can hide only the compute: `max(3.86, 2.17)`
  plus ~15% miss ≈ 4.44 against 6.03 serial, i.e. **1.36×**. *Falsified outside
  [0.60, 1.00]* — above 1.00 means it costs time at this depth.
- **F1b — GATE.** All-three is **bit-identical** to `routed+grouped`:
  `max|Δlogit| = 0`. Verified at 48 layers (#33); 94 layers is 2× the
  opportunity for a stale or unconsumed buffer. *Any nonzero difference voids
  F1a and F1c.*
- **F1c — the end-to-end number.** `bulk+ref / all-three` ≥ **8.0×**.
  *Falsified below 6.0.* #31 measured 6.97× for the first two rungs on this
  model; F1a's midpoint would put the third at ~9.3×.
- **F1d — the prediction transfers across depth.** In-situ hit rate ∈
  **[0.75, 0.92]**. The 30B gave 0.8536 at 48 layers and #32's offline
  measurement 0.8471. *Falsified outside [0.65, 0.95]* — a 235B whose routing is
  much less predictable would make the whole scheme model-specific.

## Pre-committed decisions

- **F1b holds and F1a confirms** → the three compose, the ladder is a measured
  end-to-end result rather than a multiplication, and `enable_speculative_staging`
  joins routed staging as documented policy for streamed inference.
- **F1b fails** → speculative staging is withdrawn from the 235B path regardless
  of speed, and the 48-layer verification is scoped to models of that depth.
- **F1a falsified above 1.00** → speculation costs time at flagship depth; #33's
  1.330× is scoped to 48 layers and the composed recommendation stops at two
  rungs.

## Confounds

1. One prompt, one box, greedy. The rungs share a load, so an ordering effect
   would land on the last rung — which is the one under test. Order is fixed
   worst-case for it, as in #23.
2. #33's 1.330× was measured where transfer/compute was **1.78:1**; the 235B at
   this link is nearer **1.78:1** as well, but the layer count differs 2×. The
   per-layer arithmetic should transfer; the miss *latency* is what may not.
