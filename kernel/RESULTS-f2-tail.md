# RESULTS — F2: the graph-step tail (PREREG-f2-tail)

Adjudicated 2026-08-25 by `f2_verdict.py` (self-test green) on
`receipts-f2/f2_report.json`. One anchor-compliant box (RTX 5090 +
EPYC 9B14, instance 48680711; anchor probe 7.53 ms, OFF base 7.548 ms
— inside the registered 7.35 ± 3% window). Arms: OFF/OFF A/A → T1
(`GNF4_F32_FUSE_COMBINE=1`) → T2 (`--fuse-qkv`) → T1+T2, all on the
b1d graph loop, 127 greedy steps each, unit gates run on-box first.

## VERDICT: PARTIAL — both treatments ship

```
combined cut 0.197 ms (ratio 0.974; 7.351 vs 7.548)
T1 alone +0.041, T2 alone +0.120, A/A spread 0.001
identity t1:identical, t2:identical, both:identical
```

- Cut 0.197 ms lands in the registered PARTIAL band [0.15, 0.35).
- The registered PARTIAL line ships "whichever single treatment
  carries a real gain under A/A". With the A/A spread at 0.001 ms,
  BOTH qualify (T1 +0.041, T2 +0.120), so both ship: T1's f32
  fused-combine default flips ON (`GNF4_F32_FUSE_COMBINE=0` is the
  rollback), and e4b's fused QKV defaults ON at load
  (`--no-fuse-qkv` is the rollback).
- Combined vs sum-of-singles (0.197 vs 0.161) is a small
  superadditivity, unresolved at this spread and not claimed.

## Identity and numerics

- All three treatment arms produced token streams IDENTICAL to OFF
  over all 127 steps (T1's exact-equality gate; T2/T1+T2 never needed
  their divergence-step allowance).
- T2 projection receipt (`f2_t2_proj.json`, all 48 layers, bf16, worst
  layer by delta/ref ratio): q 0.0 (bitwise), k 1.27e-4, v 3.40e-3 —
  all inside the registered `max|ref|·2⁻⁷ = 7.8e-3` frame. The prereg's
  re-derivation note stands: bitwise was falsified as a *guarantee*
  (BLAS kernel choice depends on output dim), and this box happened to
  be bitwise on q only.

## What this buys

On the anchor class (7.35 ms), ratio 0.974 puts the shipped default
step at ~7.16 ms ≈ **139–140 tok/s single-stream** (was ~136).
Class-relative, not composed with the K6-B dot-pad knob — that
composition has no receipt.

## Bar notes (recorded in the prereg before measurement)

- The binding PASS bar is the 0.35 ms CUT; the "(ratio ≤ 0.952)"
  parenthetical is a rounded restatement, reported never enforced.
- T2's gate is the K6 relative frame + divergence-step ≥ 32, after its
  first-draft bitwise claim was falsified by its own CPU gate.
- Bugbot (#260) closed a real hole pre-merge: length-mismatched
  receipts now refuse everywhere; T1 identity is exact list equality.
