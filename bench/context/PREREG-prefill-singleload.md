# PREREG — the same single-load fix on the prefill path

`_gemm_nf4_grouped` (`kernel/nf4_grouped.py`) carries the **identical**
duplicated-byte load that #43 fixed in the decode GEMV:

```python
bytes_ = tl.load(b_base + (kk[None, :] // 2), ...)   # kk over consecutive k
```

#43 deliberately left it alone because this path feeds `tl.dot`. Registering
before touching it, and registering a **pessimistic** expectation.

## Why this should help much less than it did in the GEMV

In the decode GEMV, `M == 1`: every packed byte loaded serves exactly **one**
output row, so halving the loads halves the dominant cost — measured 4.08× on
A100. Here the loaded `w[BLOCK_N, BLOCK_K]` feeds `tl.dot` against
`a[BLOCK_M, BLOCK_K]`, so each weight element is reused **`BLOCK_M` times**
(64 by default). Arithmetic intensity is ~64×, and #39's roofline already
registered prefill as **compute-bound**, with the standing claim there being
"parity + energy, not speedup".

**If the MMA dominates, removing half the load instructions changes nothing.**

## Two candidate transforms

**P-A — interleave back to the original layout.** Load `BLOCK_K/2` bytes once
into `w_hi`/`w_lo`, then `tl.join(w_hi, w_lo)` → `[BN, BK/2, 2]` and reshape to
`[BN, BK]`, which reproduces exactly `w[:, 2i]=hi, w[:, 2i+1]=lo`. The A operand
and the single `tl.dot(a, trans(w))` are **untouched**. Preferred: only the
weight path changes.

**P-B — split the contraction.** Two dots of `K=32`
(`dot(a_even, w_hi) + dot(a_odd, w_lo)`). Avoids `tl.join`, but makes the A
loads stride-2 and halves each MMA's k-depth. Fallback if `tl.join` is
unavailable or wrong across Triton majors.

## Pre-committed predictions

Measured through the real `gemm_4bit_grouped` API in **prefill mode**
(`sizes > 1`), census shapes, against the shipped kernel on the same card.

| arm | prediction |
|---|---|
| P-A (interleave) | **1.00–1.20×** |
| P-B (split dot) | **0.85–1.10×** — the k-32 MMAs may cost more than the loads save |

**I expect this to be a null result.** Registering it because #43's success
creates exactly the wrong prior — the same defect in a different regime is not
the same opportunity.

## Decision rules, fixed now

- **≥ 1.15× on BOTH cards** → land it.
- **1.00–1.15×** → do **not** land. A sub-15% win that must hold across two
  architectures and two Triton majors is not worth the risk on a `tl.dot`
  mainloop that four registered result-sets already depend on. Record as a
  measured null.
- **< 1.00× on either card** → falsified, record, leave the prefill path alone
  permanently and say so in #43.

## The rule this run exists to obey

#43's composed kernel passed 44/44 tests and was **2.2× slower** on the second
card. So:

1. **Nothing ships on one card's evidence.** A2000 first because it is free;
   the A100 check is required before landing, not optional.
2. **If the A2000 shows < 1.15×, stop — do not rent.** The decision rule already
   says that outcome does not ship, so the A100 run would buy nothing.
3. Correctness gate is unchanged: agreement with the shipped kernel at the bf16
   floor, and `kernel/test_nf4_grouped.py` **44/44** against its own control.

## Confounds recorded in advance

1. The A2000 is a **shared production GPU**; #43 saw the same kernel/shape vary
   0.687 → 1.519 ms across runs. Prefill arms must be interleaved in-process and
   reported as medians, and a <15% difference there is **inside the noise** —
   which is itself part of why the landing bar is 1.15×.
2. `VARIANT` 0/1/3 are three different mainloops. The fix must be measured on
   the **live default** (variant 1, register-LUT) and must not silently change
   which variant runs.
3. `GROUPS == 2` (BLOCK_K 128) selects a `tl.where` scale path and is documented
   dead on sm_86. Not in scope; must still compile.
