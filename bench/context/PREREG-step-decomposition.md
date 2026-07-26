# PREREG — what is the other 67% of the step?

**Tier: CONFIRMATORY. Status: STAMPED before the pod was created.**
gnf4 @ `aaefd78`, e4b @ `3510248`. Both local, unpushed.

## Why

Every optimization since #21 attacked expert bytes, and #34's speculation
finished that job: the 235B step is **0.6263 s** against **0.3708 s** of expert
transfer, so transfer no longer binds. Expert *compute* accounts for 0.2044 s.
**0.4219 s — 67% of the step — has never been measured.**

There is already a tell. Per layer, the 30B costs 5.04 ms and the 235B 6.66 ms —
**1.32×** for a model with 4× the experts and 2× the hidden size. Per-layer cost
scaling that far below model size is the signature of a **fixed** term, not
arithmetic.

## Method

CUDA-event ranges on disjoint categories (attention, router, experts, norms, LM
head), accumulated across many decode steps and reduced with a **single**
synchronize at the end — the amortized pattern from K1, where per-call syncs were
found to be ~50% of a kernel microbenchmark.

The decisive quantity is **wall − GPU-busy**. Kernels that run back to back leave
no gap; a large gap is launch and Python overhead, which is what CUDA graphs
would remove.

## Predictions

- **D1a — the step is launch-bound.** GPU-busy summed over all categories is
  **< 60%** of wall-clock, i.e. **>40% of the step is gaps**. *Falsified above
  80%* — that would mean the GPU is nearly saturated and CUDA graphs cannot help.
- **D1b — experts are the largest busy category** but not a majority: expert
  compute ∈ **[25%, 50%]** of GPU-busy. *Falsified outside [15%, 65%].*
- **D1c — the LM head is small.** < **10%** of the step, despite a 151k vocab,
  because it is one matmul against 94 layers of everything else. *Falsified above
  20%.*
- **D1d — attention is not the story.** attention + KV < **20%** of the step at
  this 48-token context. *Falsified above 35%*, which would send the next move
  back to the KV tier instead of to graphs.

## Pre-committed decisions

- **D1a confirmed** → **CUDA graphs is the next build**, and the decomposition
  says how much is on the table (the gap fraction is its ceiling).
- **D1a falsified** → the GPU is busy and the next move is arithmetic, not
  scheduling; graphs are dropped without being built.
- **D1c or D1d falsified high** → that component is the target instead, and
  graphs wait.

## Confounds

1. Instrumentation perturbs: CUDA events are cheap but not free, and the hooks
   add Python calls on the very path whose Python overhead is under test. The
   measured gap is therefore an **upper bound** on the true one, and the
   uninstrumented wall time is recorded alongside for comparison.
2. 48-token context. Attention and KV grow with context, so D1d is scoped to
   short prompts and says nothing about 32K.
