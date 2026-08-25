# PREREG — K6: the bespoke GEMV against the ~59 µs compute wall

Registered 2026-08-25, before any measurement. This is the lane K5's
STRUCTURE-REFUTED verdict left open under exactly one condition: "a
bespoke `tl.dot` GEMV (registered only with a fresh prereg and this
probe's receipts as its baseline)." It is also, after S2's refutation
of speculative decoding for this architecture, the campaign's LIMIT
lane: the certified step is 7.41 ms of which the NF4 GEMV owns ~4.84,
and the loads floors (K4 receipts, wide words) are **9.98 µs
(gate_up) / 10.04 µs (down)** against K1-winner kernels at **44.5 /
27.9 µs** — the gap is ~59 µs/pair of COMPUTE (nibble shifts + LUT
gather + scalar `tl.sum` reduce), i.e. ~2.9 ms/step.

## What K5 refuted and what it did not

K5 refuted the EXISTING M-tile kernel at M=1 (ratio 1.303): a kernel
whose loads are per-element and whose LUT lives in a `tl.gather` per
K-step, padded to a 16-row MMA of which 15 rows are waste. It did NOT
test a kernel that keeps K4's WIDE UINT32 LOADS (certified at the
roofline) and changes only the compute structure. K6 builds exactly
that, twice:

- **V1 `dot-pad`** — wide uint32 loads (K4's certified structure);
  dequant in registers exactly as the wide SCALAR kernel does it
  (shift + `tl.gather` LUT + absmax scale — the gather lowering
  exists on the floor stack, K5 receipts); then the mul-reduce runs
  as `tl.dot(A[16, BK], W[BK, BN])` where A carries x in row 0 and
  zeros elsewhere. A design honesty note, registered: a GEMV has no
  free M dimension — any K-split across M rows would need a
  different W per row, which one dot cannot express — so the M slot
  IS 15/16 waste. The bet is not "no waste"; it is that MMA
  throughput is so far above scalar-FMA throughput that a wasted
  15/16 still beats `tl.sum`. Swept over (BN, BK ∈ {64, 128},
  warps, stages).
- **V0 `wide-scalar`** — the existing `GNF4_GEMV_WIDE_LOADS=1`
  kernel (K4's 69.3 µs pair), re-timed same box as the
  decomposition arm: V0 and V1 share loads and dequant EXACTLY, so
  V1 − V0 isolates the reduce structure — the specific question K4
  left and K5 could not answer for existing kernels.

Both bench-local in `bench/k6_bespoke_bench.py` — no product change
in this cycle (a win registers K6-B: productization with P-fid +
token gates, exactly K1's ladder).

## Gates (fixed before measurement)

- **Correctness gate first, on-box**: each variant's output vs the
  production GEMV at the same census cells, `atol=0` `rtol=0` NOT
  required — `tl.dot` accumulates TF32/fp32 in different order, so the
  gate is the K1-class property gate: max |Δ| ≤ 1e-2 on bf16 outputs
  AND exact argmax agreement over 4096 random rows. A variant failing
  its gate is excluded from timing (not the cycle).
- **Timing**: chunked-median, same estimator as K1–K5, at the two
  census cells with the K1-winner GEMV as baseline (re-timed same
  box, same session; 5% drift gate).

## Decision map

Best surviving variant's census-pair time P (baseline ~72.4 µs,
floor 20.0 µs):

- **PASS** — P ≤ 36 µs (≥ 2× over baseline; more than half the gap
  to the floor closed) ⇒ register K6-B productization. Projected
  step: 7.41 − 4.84·(1 − P/72.4) ≈ ≤ 5.0 ms ⇒ **≥ 200 tok/s**, with
  the floor trajectory (~294) still open.
- **PARTIAL** — 36 < P ≤ 58 µs (20–50% gain) ⇒ K6-B proceeds only if
  the on-box A/A spread of the pair is < half the gain.
- **REFUTED** — P > 58 µs ⇒ the compute wall stands against tensor
  cores with floor loads too; the kernel lane CLOSES for this
  hardware generation and the campaign ceiling is the fusion tail
  (~150 tok/s single-stream).

## Verdict calculator

`k6_verdict.py`, self-tested both directions; receipts in
`kernel/receipts-k6-bespoke/`. Box: kernel-only provisioning (no
model), floor stack, driver ≥ 580, triad advisory.
