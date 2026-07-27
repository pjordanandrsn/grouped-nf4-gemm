# PREREG — the ladder on the KV setting the project actually recommends

**Tier: CONFIRMATORY. Status: STAMPED before the pod was created.**
gnf4 @ `704c568`, e4b @ `3510248`. Both local, unpushed.

## Why re-run a ladder that already has a number

#34 measured bulk → routed → +grouped → +speculative at **9.09×** on the 235B.
Every rung ran `residence="host"` NF4 KV. #37 then measured bf16 KV **1.27–1.42×
faster** at these contexts, and #40 confirmed that under `nf4_host` attention was
**44.5%** of the step against **15.9%** on bf16.

So the ladder was measured inside a configuration the project now tells users not
to run. The **ratio** is sound — within-pod, common baseline — but no ladder has
ever been run on the recommended setting, and the absolute numbers are not the
best the configuration can do.

There is a second-order reason to expect the ratio itself to move: the KV
improvement helps the **fast** rung proportionally more than the slow one. At
`bulk+ref` the step is dominated by 16× surplus expert transfer and KV is noise;
at `routed+grouped+speculative` attention was nearly half the step. Making
attention cheaper should therefore *widen* the ladder.

## Fixture

Qwen3-235B-A22B, 2×A100-80GB, one process, one load, natural prompt, greedy,
median of 3. Four rungs on **bf16 KV**, plus the final rung repeated on
`nf4_host` so the KV comparison is within-pod rather than across pods — a
distinction this session has already been burned by.

## Predictions

- **L1a — the ladder widens.** End-to-end `bulk+ref / routed+grouped+spec`, both
  on bf16, ≥ **10.0×** (up from 9.09× on `nf4_host`). *Falsified below 8.5×* —
  which would mean the KV setting does not interact with the ladder the way #40's
  decomposition implies.
- **L1b — bf16 wins within-pod at the fast rung.** `nf4_host / bf16` at
  `routed+grouped+speculative` ≥ **1.15×**. *Falsified below 1.0×*, which would
  contradict #37 on the same box.
- **L1c — GATE.** Within a KV setting, every rung is bit-identical to
  `routed+grouped`: `max|Δlogit| = 0`. Verified at 94 layers in #31 and #34; this
  re-checks it on a cache type those runs never used.
- **L1d — the per-rung steps reproduce.** routed/bulk ≥ 4.0×, grouped step
  ∈ [1.0, 1.3], speculative step ∈ [1.3, 1.9]. *Reported per rung*; a rung far
  outside its #34 value on a different KV setting is itself the finding.

## Pre-committed decisions

- **L1a confirmed** → this becomes *the* headline ladder, replacing #34's, and
  the README carries it with the KV setting stated. #34 is kept as the
  `nf4_host` measurement, not deleted.
- **L1a falsified** → #34's 9.09× stands as the headline and the KV setting is
  documented as not materially affecting the ladder.

## Confounds

1. A different pod, so absolute times are **not** comparable to #34's — only the
   within-run ratios are. The final-rung KV pair exists precisely so the bf16-vs-
   NF4 claim does not have to cross pods.
2. Rungs share a load; ordering effects land on later rungs, and the order runs
   worst-case for the rung under test.
