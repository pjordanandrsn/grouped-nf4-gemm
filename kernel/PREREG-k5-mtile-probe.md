# PREREG — K5: the M-tile tensor-core probe for M=1 decode

Registered 2026-08-25, before any measurement. K4's receipts moved the
wall: loads are at roofline under wide words (9.98 µs floor vs 8.0),
and **~59 µs of COMPUTE** (nibble shifts + LUT gather + the scalar
`tl.sum` mul-reduce) owns the GEMV. The codebase already contains the
alternative compute structure: the M-tile kernel
(`_gemm_nf4_grouped`) runs the same math through **tensor-core
`tl.dot` with the VARIANT-1 register-LUT** — both absent from the
GEMV reduction. This cycle PROBES that existing path at the decode
shapes before any new kernel is written.

## Stage A — the probe (existing kernels, one bench, no product change)

`bench/k5_mtile_probe.py` times, at the two census cells with
`sizes=[1]*8`:
1. the production GEMV at the K1 winners (baseline);
2. the M-TILE path at exactly `sizes=[1]*8`: the bench launches the
   PRODUCT kernel (`_gemm_nf4_grouped`) directly with
   `build_group_tiles([1]*8, block_m)` — one 1-row tile per group,
   handled by the kernel's own `m_mask` — swept over `BLOCK_N ∈
   {64, 128, 256}` × warps ∈ {4, 8} × stages ∈ {2, 3} × GROUPS ∈
   {1, 2} (BLOCK_K 64/128, both product-supported) at
   `prefill_variant=1` (register-LUT) with `block_m=16` — the sweep
   contains the product's native VARIANT-1 config (128/4/2). No wrapper
   change, no replica kernel: the launch section is replicated in the
   bench, the kernel is the product's own, so there is nothing to
   drift.

Registered probe decision:
- **M-tile best cell-sum ≤ 0.6× the GEMV winners' (≤ ~44 µs)** ⇒
  register K5-B: a `decode_via_mtile` routing knob in the wrapper
  (product change, its own cycle: P-fid gates — `tl.dot` TF32
  accumulation is NOT bitwise vs the GEMV's scalar fp32 chain, so
  the fidelity story is K1-class: property suite + token agreement
  with disclosure, never a pretended bitwise bar).
- **Between 0.6× and 0.9×** ⇒ marginal: the lane pauses and the
  elementwise-fusion lane takes priority (a <1.7 ms/step total prize
  does not outrank a 2.5 ms one).
- **≥ 0.9×** ⇒ the compute-structure theory is refuted for existing
  kernels; the lane's remaining option is a bespoke `tl.dot` GEMV
  (registered only with a fresh prereg and this probe's receipts as
  its baseline).

Verdict maps (fixed in `k5_verdict.py`): ratio ≤ 0.6 → MTILE-WINS;
0.6 < ratio < 0.9 → INCONCLUSIVE-PAUSE; ratio ≥ 0.9 →
STRUCTURE-REFUTED. Boundary ties: 0.6 wins, 0.9 refutes. REFUSE on:
noise gate failure at either cell, any cell with zero successful
M-tile configs, or a missing/non-positive ratio.

## Gates

Noise gate 5% on the GEMV baseline re-timed at start/end; the probe
is timing-only (outputs discarded; correctness is NOT claimed — the
M-tile path's numerics are certified separately by its own suite and
a K5-B cycle would carry the fidelity gates).

## Verdict calculator

`k5_verdict.py`, self-tested both directions; receipts in
`kernel/receipts-k5-probe/`.
