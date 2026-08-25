# RESULTS — K6 Stage A: REFUSE as adjudicated; the correctness bar was
# mis-derived for the registered mechanism

Measured 2026-08-25 under PREREG-k6-bespoke-gemv. Receipts in
`receipts-k6-bespoke/stageA/` (box 10, gnf4 `73652c2`; instance
destroyed, vast zero).

```
K6 VERDICT: REFUSE
  gate_up: no config passed its correctness gate -- the census pair
  is INCOMPLETE and a single cell must not be judged as the pair
```

The REFUSE stands: under the registered gate (max|Δ| ≤ 1e-2), zero of
72 configs pass, and the calculator correctly refused to read any
timing as a verdict.

## What the rejections actually show

Every one of 72 rejections carries **max|Δ| = 0.500 exactly** with
argmax agreement 4070–4078 / 4096 (99.4%), uniformly across every
tile shape. That is not 72 broken kernels — it is one systematic,
small divergence: the dot-pad mechanism rounds BOTH operands to bf16
before the MMA (bf16 inputs, fp32 accumulate), a 2⁻⁸-relative input
rounding that the fp32-scalar reference chain does not perform. On
outputs of this magnitude that budget is ~0.4–0.5 — the observed
delta IS the mechanism's inherent arithmetic, and the 1e-2 bar was
**physically unachievable for the design the prereg itself
registered**. The bar was mis-derived (borrowed from K1's fidelity
style without computing the bf16-MMA error budget); the mechanism was
not mis-built.

## AMENDMENT (registered here, before any re-adjudication)

- The Stage A correctness gate for bf16-input MMA becomes
  RELATIVE: `max|Δ| ≤ max|ref| · 2⁻⁷` (twice the single-rounding
  budget), with `max|ref|` recorded in the receipt, plus argmax
  agreement ≥ 99% recorded (not gating alone). End-to-end token
  identity remains K6-B's gate, as always registered.
- The TIMING decision bars (36/58 µs, set before measurement) are
  NOT moved. For disclosure only: the fastest configs timed at
  27.4 µs (gate_up) + 19.0 µs (down) = **46.4 µs/pair** vs the 69.5 µs
  same-box baseline — inside the pre-set PARTIAL band (36, 58]. No
  verdict is taken from this number until a re-gate under the amended
  bar passes on-box; it is disclosed because hiding an observed
  number the bars already band would be worse.
- Re-adjudication requires one kernel-only box cycle (~10 min):
  re-run the gate at the amended bar recording max|ref|, then the
  calculator on the same decision bars.

## Standing

The kernel lane's status is REFUSE-pending-amendment, not PARTIAL —
the verdict only changes if the amended gate passes on receipts. The
composed-ceiling arithmetic in e4b RESULTS-s3 cites the 46.4 µs
number only as a disclosed trajectory (~177 tok/s) with exactly this
caveat.
