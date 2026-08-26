# PREREG — K7: the round-2 decode GEMV (occupancy, not new math)

Registered 2026-08-26, before any implementation is measured. Follows
from RESULTS-sv2-device-census (e4b): the dot-pad GEMV runs the
census pair at **3.8× its streaming floor** (2.469 ms/step to move
1.019 GB the box streams in 0.649 ms), while the neighboring cuBLAS
gemvx runs at 1.14×. SV2's composition frame registered this slice as
the lane to 250 tok/s single-stream, target ≥1.5 ms off the slice.

## Diagnosis being tested

The K6-B kernel is latency/issue-bound, not bandwidth-bound: grid
(8, N/16) × 2 warps ≈ 9 warps/SM on a 170-SM part, each warp running
a long dependent chain (u32 load → shift/mask → LUT gather → absmax
mul → bf16 trans → tl.dot) with stages=2. The treatment is
**parallelism and pipelining on the SAME dequant math**:

1. **Split-K across CTAs**: grid gains a K-split dimension; each CTA
   reduces its K-chunk; partials land in a fixed-slot fp32 buffer and
   a second pass (or fixed-order single-CTA sum) folds them.
   **Determinism is a design REQUIREMENT, not an option**: fp32
   atomicAdd reduction is banned — order-nondeterministic sums would
   break the bitwise A/A and token-identity gates every downstream
   certification depends on (BV3b, SV2 G1).
2. **Retune at census shapes under the new grid**: BLOCK_N / warps /
   stages / BLOCK_K sweep, same regate protocol as K6.

Explicitly OUT of scope: fp8-MMA activations (different numerics
class; needs its own P-fid frame), any dequant-chain change (the
bf16-MMA rounding mechanism and absmax pre-fold stay IDENTICAL to
the certified K6-B chain).

## Census cells

Same pair as K6: gate_up (N=1536, K=2048) and down (N=2048, K=768),
E=8 routed experts per call, decode T=1. Timing basis: graph-replayed
chunked-median per the K6 harness, A/A noise gate carried in the
receipt (`noise_gate_pass`).

## Gates (all REFUSE)

- **G1 correctness, mechanism-derived (inherited from AMENDMENT
  RESULTS-k6-stageA)**: vs the fp32-scalar reference,
  max|Δ| ≤ max|ref|·2⁻⁷ and argmax agreement ≥ 0.99 per cell. The
  mechanism is unchanged bf16-MMA rounding; split-K only reorders the
  fp32 partial summation, which stays inside the same band. Timing an
  incorrect config certifies nothing.
- **G2 determinism**: two identical invocations of the candidate on
  identical inputs must produce BITWISE-identical outputs
  (`torch.equal`). An atomics-ordered sum fails here by construction.
- **G3 A/A**: the harness noise gate must pass (same protocol as K6);
  pair times are chunked medians.
- **G4 baseline presence**: the SAME box must measure the current
  dot-pad pair (`GNF4_GEMV_DOTPAD=1` path) as the ratio denominator;
  cross-box ratios are not adjudicated.

## Bars (bars-follow-the-claim; ratio vs the K6-B dot-pad pair)

- **PASS: best gated candidate pair ≤ 0.39×** the same-box dot-pad
  pair — the SV2 frame number (slice 2.469 → ≤ 0.97 ms ≈ 1.5× its
  streaming floor, the ≥1.5 ms cut the 250 frame needs from this
  slice).
- **PARTIAL: ≤ 0.60×** — ≥0.99 ms off the slice; ships as a knob
  worth composing (projects a ~5.5 ms step ≈ 183 tok/s class) but
  does NOT by itself keep 250-by-composition alive; the RESULTS must
  recompute the SV2 frame arithmetic with the measured ratio.
- **REFUTED: > 0.60×** — occupancy was not the binding constraint;
  the RESULTS must say the 3.8×-floor gap is NOT addressable by
  parallelism alone and the composition route loses its dominant
  slice (250-by-composition falls unless a different mechanism is
  registered).

## Ship gate

The winner ships behind its own env (`GNF4_GEMV_SPLITK=1`),
OFF-by-default, inheriting the K6-B posture: numerics differ from the
scalar chain (same 2⁻⁷ class), so composed default-ON requires the
e4b-side step cert with token-identity receipts (the SV1/SV2
protocol) on a fresh box before any default flips. Kernel-level PASS
here licenses registering that composed cert, not the flip itself.

## Receipts

`kernel/receipts-k7/` — harness report JSON (cells + summary),
sweep table, box_meta with anchor probe. The bench harness
(`k7_bench.py`) and verdict calculator (`k7_verdict.py`, self-tested
refusal directions) are committed BEFORE the box cycle
([[commit-the-instrument-not-just-receipts]]).
