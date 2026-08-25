# RESULTS — K6-B: PARTIAL on two boxes; knob ships available, not default

Measured 2026-08-25 under PREREG-k6b-productize. Receipts in
`receipts-k6b/` (box 12: driver 580.119; box 13: driver 610.43,
anchor-compliant at 7.34 ms; both instances destroyed, vast zero).

```
K6-B VERDICT: PARTIAL  (both boxes)
  box 13: step ratio 0.896 (6.59 vs 7.35 ms; A/A 0.01)
  box 12: step ratio 0.901 (7.14 vs 7.92 ms)
```

## The two findings

1. **The step gain is real, stable, and smaller than the kernel's µs
   win predicted.** Dot-pad cuts the census pair 33–41% in isolation,
   which projected a step ratio ~0.80–0.84; the measured end-to-end
   ratio is 0.896–0.901 on two independent boxes. ~10% of the step,
   not ~16–20% — the difference is un-modeled overlap: part of the
   GEMV time the kernel removes was already hidden behind other work
   in the captured step ([[hide-is-load-dependent]] class). The
   PARTIAL band adjudicates exactly this outcome, and per the
   registered map the knob ships **OFF-but-available**.
2. **Token identity was PERFECT on both boxes: first divergence NONE,
   127/127 greedy tokens identical**, against a registered floor of
   step ≥ 32. The campaign's first numerics-changing kernel produced
   bitwise-identical continuations end-to-end in every certification
   run. The P-fid concern registered in the prereg did not
   materialize at this window length; longer-horizon divergence
   remains undetermined and the disclosure stands.

## Ladder statement (single-stream)

- Certified default: **7.35 ms class ≈ 136 tok/s** (the 133.4–135.4
  band across boxes).
- With `GNF4_GEMV_DOTPAD=1` (certified available, PARTIAL): **6.59 ms
  ≈ 151.7 tok/s** — the highest single-stream number the campaign has
  measured, reproducible, one env var away.

The remaining registered single-stream program: F2 on the fresh
census (attn-proj cuBLAS 1.92 ms is now the largest single non-NF4
item; fp8 combine/decode + router + memcpy ≈ 1.0 ms of tail), with
the composed realistic ceiling unchanged (~200–290 requires further
kernel wins that are not yet registered beyond F2).
