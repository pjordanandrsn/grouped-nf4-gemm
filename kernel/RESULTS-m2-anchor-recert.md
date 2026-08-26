# RESULTS — M2: the anchor is unchanged. The CLASS is not a class.

## 7.369 ms ±4.2%, and the ±3% gate was never describable.

Measured 2026-08-26 under PREREG-m2-anchor-recert. Receipts in
`receipts-m2/` (three RTX 5090 boxes, rented with **no** anchor gate;
all destroyed, verified zero).

```
M2 ANCHOR CERTIFIED: 7.369 ms +/-4.2% (n=3, inter-box spread 8.5%)
  vs prior 7.35: +0.26% (A/A noise 0.16%)
  NOTE: population spread 8.5% exceeds 6%: the window is widened to
  +/-4.2% and the class is too dispersed for a 3% gate
```

## The boxes

| box | A/A pair | pair | A/A spread | gen64 probe | driver |
|---|---|---|---|---|---|
| 1 | 7.757 / 7.745 | 7.751 | 0.16% | — | **580.105.08** |
| 2 | 7.151 / 7.143 | 7.147 | 0.11% | 7.129 (−0.3%) | 610.43.02 |
| 3 | 7.374 / 7.365 | 7.369 | 0.12% | 7.344 (−0.3%) | 610.43.02 |

Every box measures **itself** to 0.16% or better. The disagreement is
entirely *between* boxes: **8.5%**.

## The constant was right. The window was never possible.

The prior constant 7.35 was **correct to 0.26%** — the median of
three fresh A/A pairs is 7.369. The re-certification found no drift
worth the name.

What it found instead is that **a ±3% window cannot contain this
population.** The boxes span 7.147–7.751; the certified window
[7.13, 7.57] excludes the slowest of three boxes outright. No choice
of centre fixes that, because the spread (8.5%) exceeds the window
(6% wide). The gate was not miscentred — **it was narrower than the
thing it was gating.**

That is why K10 burned two provisioning cycles: those boxes were not
anomalous, they were the population.

## Two hypotheses I published and the receipts refuted

Both were mine, both were stated confidently, both are wrong:

1. **"The constant sits ~2.5% above the population."** It sits 0.26%
   off the median of a properly-measured sample. The earlier estimate
   came from six SINGLE-SHOT probes, which is not the same instrument
   as an A/A pair and was never a population estimate.
2. **"The gate is structurally biased by a window mismatch"** — the
   hunt probes `--gen-tokens 64` while 7.35 came from `n_steps=127`,
   so I argued KV growth made probes read systematically low.
   **Measured: −0.3% on both boxes that ran the pair.** Negligible.
   The addendum that measured it was added mid-cycle precisely
   because the claim deserved a number rather than an argument.

## The rule fired, and the rule is defective

The registered decision rule obliges a ladder correction when the
shift exceeds A/A noise: 0.26% > 0.16%, so it fires. **It should not
be honoured as written**, and the defect is worth recording:

A/A noise measures how well one box repeats *itself* (0.16%). The
uncertainty on a median across boxes is set by the *between*-box
spread — with 8.5% dispersion at n=3 that is roughly 4.9%. Comparing
a between-box statistic against a within-box noise floor is a
category error, and it mechanically demands a "correction" of 0.26%
that the data cannot support.

**Action taken:** the published entry is restated as a **range, not a
point** — which is the change the data does support — and the
comparison rule is recorded as needing the between-box scale. No
0.26% point correction is made, because publishing it would imply a
precision the receipts refute.

## A SECOND rule defect, found while committing the constant

The widening rule says window = ±(spread/2) when spread > 6%. Applied
here: ±4.25% on a median of 7.369 gives **[7.060, 7.678]** — which
**excludes box 1 at 7.751**, one of the three boxes the constant was
computed from.

A half-spread window centred on the MEDIAN only spans the population
when the median is also the midrange. Here the median is 7.369 and
the midrange is 7.449, so the window misses the top of its own
sample. The rule was written as if those coincide; they generally do
not.

**Recording it was not enough, and that is its own lesson.** The
first version of this document described the defect and then still
published ±4.2% as the harness constant — which would have destroyed
box 1, a normal box, for being normal: the exact failure this cycle
was run to diagnose. Documenting a defect is not fixing it (Bugbot,
gnf4#278).

**The gate is therefore no longer a centre±window at all.** It is an
explicit bounds pair covering the MEASURED population with a 2%
margin each side:

```
GATE = [7.004, 7.906] ms      (population [7.147, 7.751] + 2%)
```

It is 12.6% wide, and that width **is** the finding: against 8.5%
inter-box dispersion, a rental gate can only exclude gross outliers —
it cannot certify that two boxes inside it are comparable. The
certified anchor (7.369 ms) remains the central estimate for
reporting; it is no longer doing double duty as a gate centre.
`test_decode_anchor.py` asserts that every box M2 measured passes,
that a grossly broken box still fails, and that the retired ±4.2%
window would have rejected box 1.

## Restated ladder entry

| configuration | ms/step | tok/s |
|---|---|---|
| **certified default** | **7.37 ms ±4.2%** | **130–142** (point 135.7) |
| `GNF4_GEMV_DOTPAD=1` | 6.476 | 154.4 |
| + `GNF4_ATTN_COMPUTE=fp8` | 6.281 | 159.2 |

The two knob rows are **same-box** measurements and are unaffected:
their ratios were never against the anchor. Only the absolute default
row carried the dispersion, and it now carries it visibly.

## Driver correlation: suggestive, not established

The one box on driver **580.105.08** is the slow one (7.751); both on
**610.43.02** are faster (7.147, 7.369). But those two differ by
3.1% from each other, so driver version is at most part of it. At
n=3 this is an observation to test, not a finding — and it is the
obvious next question if the dispersion matters enough to model.

## Harness fixes (both required by this cycle)

1. The hunt gate reads `kernel/decode_anchor.py` — a committed
   source, not a scratchpad literal — which is the defect that let
   7.39 (a number in no RESULTS document) gate every cycle in this
   campaign. The gate is `GATE_LO_MS`/`GATE_HI_MS`, **not** a
   centre±window; see the second rule defect above.
2. Given 8.5% dispersion, an anchor gate should be understood as
   *excluding outliers*, not *certifying a class*. Every cycle's own
   G2 check against its same-box denominator is what actually
   protects a verdict, and that was true throughout.
