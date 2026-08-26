# RESULTS — K7: split-K REFUTED. Occupancy was never the constraint.

## And the 9.8% that did appear is not the registered mechanism.

Measured 2026-08-26 under PREREG-k7-gemv-round2. Receipts in
`receipts-k7/` (RTX 5090, gnf4 `a54445d`, anchor 7.28 ms; instance
destroyed, vast zero by-id and list).

```
K7 VERDICT: REFUTED  (ratio 0.903 = 41.6 / 46.1 us; PASS <= 0.39, PARTIAL <= 0.6)
  gate_up: 27.2 -> 26.9 us
  down:    18.8 -> 14.8 us
  occupancy was not the binding constraint; 250-by-composition loses
  its dominant slice (PREREG-k7)
```

## Every gate passed before a timing was read

| gate | gate_up | down |
|---|---|---|
| G1 correctness (max\|Δ\| vs budget) | 0.644 / 1.386 | 0.350 / 0.963 |
| G1 argmax agreement | 8/8 | 8/8 |
| G2 bitwise determinism | True | True |
| G3 A/A noise | 0.0000 | 0.0000 |
| G4 same-box dot-pad denominator | 27.2 us | 18.8 us |

40 configs timed per cell, 0 gated out, 0 errored.

## Split-K did not work, and the sweep says so twice over

Best time at each split factor:

| cell | sk=1 | sk=2 | sk=4 | sk=8 | sk=16 |
|---|---|---|---|---|---|
| gate_up | 27.30 | 27.10 | 27.05 | **26.86** | 28.30 |
| down | **14.76** | 16.83 | 16.78 | 16.82 | 20.91 |

- **gate_up is FLAT in sk.** 27.30 → 26.86 across a 16× range of
  splits (1.6%), and sk=16 regresses. Adding concurrent CTAs to this
  kernel does approximately nothing.
- **On `down`, splitting actively HURTS.** The unsplit kernel at
  14.76 us beats every split configuration by ~14%. The partial-write
  plus reduction pass costs more than any latency it hides.

The registered diagnosis — 9 warps/SM on long dependent dequant
chains, therefore occupancy-starved — is **wrong**. In hindsight the
grid said so: 8 groups × 96 n-tiles = 768 programs on a 170-SM part
is 4.5 programs/SM before any split, which is not a starved grid by
`_decode_plan`'s own rule (it engages split-K only below
`2 × SM`). The bottleneck lives inside the mainloop, not in how many
mainloops run at once.

## The win that did appear belongs to the OTHER treatment

The pair's 9.8% comes almost entirely from `down`, and its winning
configuration is **`bn=32, warps=2, stages=3, sk=1`** — a pure
config retune with split-K OFF. That is the prereg's *second*
registered treatment, not the headline one.

**This attribution exists only because the instrument was fixed
before the box ran.** The sweep grid originally carried `sk>=2` rows
only; the A2000 pre-flight self-test surfaced it. Under that grid
`down`'s best would have been 16.78 us (sk=4), the pair ratio 0.947,
and the write-up would have read "split-K bought 5%" — crediting the
refuted mechanism for a win it did not produce. A sweep that cannot
express the null hypothesis will always confirm the alternative.

## What this does to the 250 frame

RESULTS-sv2 priced this lane as the dominant term of the composition
route: the MoE GEMV slice measured 2.469 ms/step at **3.8× its
1.019 GB streaming floor**, with an "addressable ceiling" of 1.82 ms
and a realistic estimate well above 1 ms. Measured, the best gated
configuration cuts the census pair 9.8%, which scales to
**~0.24 ms/step** — roughly **7× short** of the framed estimate.

Recomputing SV2's pool with measurement replacing estimate:

| lane | SV2 framed | now |
|---|---|---|
| MoE GEMV round 2 | ~1.5–1.8 ms | **0.24 ms (measured)** |
| attn-proj GEMV | 0.22 ms | 0.22 ms (already 1.14× floor) |
| fp8-COMPUTE attention | 0.15–0.2 ms | unmeasured (K8 registered) |
| fusion/norm/router residue | 0.7–0.8 ms | unmeasured, unregistered |
| **total** | **~2.5–2.7 ms** | **~1.3–1.4 ms** |

Against the registered 2.48 ms bar, **250-by-composition does not
clear on current evidence**. Per the prereg's REFUTED branch it falls
"unless a different mechanism is registered" — and the two remaining
lanes would have to roughly DOUBLE their estimates to close the gap,
which nothing in these receipts suggests.

**What is NOT claimed:** that the 3.8×-floor gap is unreachable. K7
tested one hypothesis about its cause — insufficient parallelism —
and refuted it cleanly. The gap is still there and still real; a
different mechanism (a different tiling, a weight-stationary or
lower-precision compute path, a dequant chain change with its own
numerics frame) may yet reach it. What is closed is that you cannot
get it by running more copies of this kernel at once.

## Disposition

- `_gemv_nf4_dotpad_splitk` ships behind `GNF4_GEMV_SPLITK`,
  **OFF by default and staying off** — an unset env is byte-identical
  to the certified path. It is kept, rather than reverted, because it
  is the evidence for this refutation and costs nothing dormant.
- The `down` retune (`bn=32, w=2, s=3`) is a real 21% cell win under
  a passing correctness gate, but baking it is a `_DOTPAD_CONFIGS`
  change to the certified default path and is NOT licensed by this
  prereg (whose bars are pair ratios, and whose PARTIAL band it does
  not reach). It is registered as a follow-on, with the caveat that
  a synthetic-fixture win must be confirmed on the composed step
  before it becomes a default.
