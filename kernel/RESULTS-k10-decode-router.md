# RESULTS — K10: Stage A proved the owner. B1 REFUSED on the gate
# that mattered, with perplexity comfortably inside its bar.

Measured 2026-08-26 under PREREG-k10-decode-router. Receipts in
`receipts-k10/` (RTX 5090, anchor 7.25 ms, all arms knob-ON;
instance destroyed, vast zero by-id and list).

```
K10 VERDICT: REFUSE
  B1-C: selected expert SETS differ between arms -- that is a
  different model, not a reordering
```

## Stage A: the attribution K9 failed twice to establish

| measurement | result |
|---|---|
| `torch.topk` calls/step | **48.0** — exactly one per layer |
| shapes seen | `{(1, 128) k=8: 48}` — *only* the router shape |
| `sbtopk::gatherTopK` /step | 48 |
| `bitonicSortKVInPlace` /step | 48 |
| **ablation: same, `sorted=False`** | **0** |

The census `router` row is fully explained: **one
`torch.topk(k=8, sorted=True)` per layer**, which ATen splits into a
select (`gatherTopK`, 174.0 us/step) and a sort of the 8 results
(`bitonicSortKVInPlace`, 137.4 us/step). Nothing else in the decode
step calls `topk`.

This was **proved by ablation, not inferred**: forcing `sorted=False`
drove the sort kernel to exactly zero. K9 died twice from reading a
plausible owner out of the source; Stage A's gate is what turns that
reading into a fact ([[attribute-from-the-profile]]).

## B1: the speed is real, the quality bar passes, and it still refuses

| arm | result |
|---|---|
| step, `sorted=True` (A/A) | 6.493 / 6.494 ms |
| step, `sorted=False` (A/A) | 6.359 / 6.365 ms |
| **delta** | **0.132 ms** |
| perplexity (1024 teacher-forced decode tokens) | 4.9425 → 4.9717 = **+0.0292** vs eps 0.05 |
| greedy token streams, base vs sorted=False | **identical** |
| **selected expert SETS** | **DIFFER** (`041c27bd…` vs `f6040ada…`) |

So the cheap signals all said ship it. The step really is 0.132 ms
faster. Perplexity moved 58% of the way to its bar and stopped. The
generated text is character-for-character identical. And the
structural gate refused anyway, because somewhere across 48 layers ×
1024 scored steps the model routed to a **different set of experts**.

That gate was registered before measurement precisely so this
decision would not be made after seeing the numbers: *"A changed set
is a different model, not a reordering, and refuses regardless of
perplexity."* It is honoured. B1 does not ship.

## What is NOT established, and why it decides the next step

These receipts prove the sets diverge; they do **not** localise it.
Two candidates, with different consequences:

- **Cascade (benign-ish).** `sorted=False` permutes the k weights, so
  `w / w.sum()` sums the same floats in a different order. That
  perturbs the layer's output, which perturbs the next layer's router
  logits, which flips an 8th-place near-tie somewhere downstream. The
  treatment is then inherently trajectory-changing — not a bug, but
  correctly refused by a set-equality frame.
- **Kernel-level (serious).** ATen may select a *different set* under
  `sorted=False` (e.g. a different tie-break between the radix and
  sort-based selection paths). 50 randomised CPU cases showed
  identical sets, but that is CPU, and it is not proof for the CUDA
  kernels.

The digest in these receipts is aggregate over all calls, so it
cannot separate them. A per-layer / first-call digest would: if
layer 0 agrees and later layers diverge, it is cascade; if layer 0
already differs on identical inputs, it is kernel-level and
`sorted=False` is unsafe for anyone, not just for this frame.

**That question is worth answering before B2**, because if it is
kernel-level it also constrains what a fused replacement may do.

## The lane's disposition

- **B1: refused.** The 137.4 us the sort costs is genuinely
  removable — the ablation proves the mechanism works — but not by
  this route under this frame.
- **B2 is now the live path** and its bars were fixed in the prereg
  before B1's outcome was known: a fused single-CTA top-8-of-128 for
  the T=1 shape with **bitwise-identical selection**, judged against
  Stage A's measured 174.0 us `gatherTopK` share (PASS ≥60%,
  PARTIAL ≥30%). A fused kernel that keeps the set exactly has no
  B1-C problem by construction.
- Frame position is unchanged: the whole row is 311.4 us, so K10
  never could and still cannot close 250.

## Anchor finding (recorded, not acted on)

This was the **third** box. Attempts 1 and 2 both probed 7.12 ms and
were destroyed by the ±3% gate on 7.39 — **for being faster than the
reference.** Session probes: 7.23 / 7.28 / 7.26 / 7.12 / 7.12 / 7.25,
mean 7.21. The constant sits ~2.5% above the population it gates, so
the window [7.17, 7.61] admits boxes 5.7% slower than the mean while
refusing ones 0.5% faster. Two provisioning cycles were spent on it.

The constant was deliberately **not** adjusted mid-cycle — moving a
gate to make a run pass is the failure the gate exists to prevent.

**Correction (added with PREREG-m2):** 7.39 is not the campaign's
certified constant at all. The certified class is **7.35 ms**
(`RESULTS-k6b`, `PREREG-f2-tail`); 7.39 lived only in the scratchpad
hunt harness and appears in no RESULTS document. So the two rejected
boxes were screened against an uncertified number. Against 7.35 the
picture is the same in direction and slightly smaller in size (window
[7.13, 7.57], centre +1.9% above the sample mean, 7.12 still just
outside). Both defects are the subject of M2.
