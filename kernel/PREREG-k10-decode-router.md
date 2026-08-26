# PREREG — K10: the decode router row, attributed then treated

Registered 2026-08-26, before measurement. Replaces K9 (VOID: its
mechanism is not on the T=1 path). The measured fact that survives
both K9 amendments:

SV2's knob-ON 6.46 ms census carries **311.4 us/step** under `router`,
in two kernels at **exactly 48 calls each** — one per layer:

| kernel | us/step | per call |
|---|---|---|
| `sbtopk::gatherTopK` | 174.0 | 3.6 us |
| `bitonicSortKVInPlace` | 137.4 | 2.86 us |

Selecting 8 of 128 experts for ONE token costs 6.5 us per layer.

## Stage A — attribution FIRST, and it gates everything

K9 died twice from designing a treatment before proving who owned the
cost. Stage A therefore produces an owner, not a plan:

1. **Identify the call site** of each kernel in a profiled captured
   replay — via a stack-attributed profile, or by ablation
   (a run with the candidate call replaced by a pre-computed
   equivalent must make the kernel's count go to zero).
2. **Working hypothesis, explicitly labelled as such and NOT
   assumed**: both kernels are one `torch.topk(..., k=8, sorted=True)`
   per layer — ATen's single-block top-k selects, then
   `sortKeyValueInplace` sorts the k results because `sorted` defaults
   True. The identical 48/48 call counts are consistent with one call
   site, which is evidence, not proof.
3. **REFUSE conditions**: if the two kernels do not resolve to a
   single owner, or the owner is not the router, Stage A reports what
   it found and NO Stage B is run under this prereg. A treatment for
   an unidentified cost is what K9 was.

Stage A cannot fail a bar. It publishes the owner and the measured
per-call cost.

## Stage B — only if Stage A identifies the router, and it is staged

**B1, the near-free probe (run first).** If the owner is
`topk(sorted=True)` and the consumer does not depend on the order of
the selected experts, then `sorted=False` should remove
`bitonicSortKVInPlace` outright — **137.4 us/step for one kwarg**.

This is NOT free of numerics: the selected SET is unchanged, but the
order changes, so `w / w.sum()` sums the same eight floats in a
different order and fp addition is not associative. It is therefore
gated exactly like the other numerics-changing lanes:

- **B1-Q (BAR)**: held-out perplexity through the DECODE path
  (`--ppl-steps`, the K8 instrument and its amendment) must be
  **<= sorted-True + 0.05** — the epsilon TR2 and K8 both used.
- **B1-C (REFUSE)**: the selected expert SET per layer must be
  identical between arms. A changed set is a different model, not a
  reordering, and refuses regardless of perplexity.
- **B1-S (BAR)**: `bitonicSortKVInPlace` count must fall to **0** in
  the profiled replay. If the kernel is still there, `sorted=False`
  did not do what this stage claims and the measured delta is
  something else.
- PASS: all three, with step delta recorded. REFUTED otherwise.

**B2, the fused router (only if B1 is refuted or insufficient).** A
single-CTA top-8-of-128 kernel for the T=1 shape, bitwise-identical
selection, judged as a fraction of Stage A's measured 174.0 us
`gatherTopK` share. Registered here so B2's bars are fixed before B1's
outcome is known: PASS >= 60% of that share, PARTIAL >= 30%.

## REFUSE gates (both stages)

- A/A spread <= 2% on paired step measurements, tokens identical.
- Anchor: knob-ON step within +/-5% of the certified 6.476 ms.
- Same box for all arms in a comparison.
- Stage A's owner-identification gate above.

## Frame note (before measurement)

The whole row is 311.4 us. Even deleting it entirely leaves the
measured pool ~1.7 ms against SV2's 2.48 ms bar, so K10 does not
close 250 and is not registered as if it could. It is registered
because it is measured, bounded, and — unlike K9 — will have a
verified owner before anything is built.

## Receipts

`kernel/receipts-k10/` — Stage A attribution (profile or ablation),
B1's three gate outputs, paired step receipts, box_meta with anchor.
`k10_verdict.py` (self-tested) is committed BEFORE the box cycle.
