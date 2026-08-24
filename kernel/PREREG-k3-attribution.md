# PREREG — K3: attribution of the M=1 GEMV's roofline gap

Registered 2026-08-25, before any measurement. **A measurement-only
cycle**: no kernel-path change ships from it. RESULTS-k2 leaves the
lane honest and stuck — at the K1 winner configs the census pair runs
**72.8 µs against a ~13.5 µs roofline floor (~18% of 1573.9 GB/s)**,
and the one mechanism theory tested so far (duplicate byte loads, K2)
is REFUTED with bitwise receipts. Per the program's law (an
elimination is not an account), this cycle ACCOUNTS for the 5.4× gap
component-by-component and emits the branch decision for the next
kernel registration.

## Cells

The two census (shape, winner-config) cells — gate_up
`(1536, 2048, T=8) @ (64, 2, sk16)` and down
`(2048, 768, T=8) @ (32, 2, sk1)` — plus gate_up at `sk=1` (isolates
split-K's contribution against the same shape).

## Instruments (two, registered in preference order)

1. **NCU** (preferred): `ncu --set full` on single launches per cell,
   CSV export, parsed locally for: `dram__bytes_read.sum` (read
   amplification vs the logical bytes), achieved occupancy, LSU/L1
   sector efficiency, and the warp-stall breakdown. Known risk,
   registered: rented containers frequently block GPU performance
   counters (`ERR_NVGPUCTRPERM`); if NCU cannot run, instrument 2 is
   the cycle's evidence and the RESULTS say so plainly.
2. **The peel battery** (counter-free fallback, and run REGARDLESS as
   cross-check): bench-local REPLICA kernels of the GEMV mainloop —
   never the product kernel — each with one component removed:
   full → no-LUT-gather (nib used directly) → no-absmax → 
   no-activation-load → loads-only (raw byte streaming, the measured
   floor for this access pattern). Timed with the registered
   chunked-median estimator; attribution by subtraction, with the
   residual REPORTED (peels are not perfectly additive — the account
   must show its arithmetic, not hide it).

## Gates

- **G-N (noise)**: the full-replica cell re-timed at battery start
  and end within 5%, and the replica's full-kernel time within 15% of
  the PRODUCT kernel's time at the same cell (the replica must be
  measuring the same thing, or the battery is NO-VERDICT).
- **G-C (coverage)**: named components + loads-only floor + residual
  must sum to the full time by construction; the account is judged on
  whether a SINGLE component ≥ 25% emerges (as in T5b's H-A).

## The preregistered branch map (decision, not optimization)

- **LUT-gather share ≥ 25%** ⇒ register the register-LUT GEMV variant
  (the M-tile kernel's VARIANT-1 treatment, never ported to decode —
  `tl.gather` codebook-in-registers).
- **loads-only floor itself ≥ 3× the roofline floor** ⇒ the ACCESS
  PATTERN is the wall (small per-CTA reads / sector waste that L2
  hides from K2's variant but not from DRAM) ⇒ register the packing-
  layout question.
- **split-K delta (gate_up sk16 vs sk1) ≥ 20% of the cell** ⇒
  register the reduction restructure.
- **NCU stall profile: long-scoreboard dominant + occupancy < 40%**
  ⇒ register pipelining/num_stages + per-CTA work-size line.
- **Diffuse (nothing ≥ 25%)** ⇒ the kernel lane PARKS at 74.3 tok/s
  and the elementwise-fusion lane takes priority — the ladder does
  not stall on a diffuse account.
- Multiple triggers compose, largest first.

## Verdict calculator

`k3_verdict.py`, self-tested both directions (including the
replica-mismatch NO-VERDICT and the diffuse park), before the box.
Receipts in `kernel/receipts-k3-attribution/`.
