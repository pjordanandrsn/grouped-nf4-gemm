# RESULTS — the expert-major coalescer

Binds `PREREG-expert-major-coalescer.md` and
`PREREG-expert-major-coalescer-amendment-1.md` (sha256
`0afceab4aa17917008106bc97e91d2641367f206c87c0f62bb3aa4bdf19f31d3`, stamped
pre-data).

## Verdict: DEAD. The premise was false.

The coalescer was built, is correct, and is **not worth landing**. The 235B
fixture was **not rented**, on the amendment's pre-registered rule.

## The correctness half — PASSED

Measured on the QNAP A2000 (correctness testbed, per policy), CUDA, bnb 0.50.0:

| gate | required | measured |
|---|---|---|
| staged rows vs name-major control | bit-identical | **identical, `max\|Δ\| = 0.000e+00`** |
| copies per stage | must fall | **12 → 6** (E=16, 3 routed experts, 4 tensors) |
| copies at 235B shapes | 8–16 | **16** (8 experts × 2 dtypes) |
| default behaviour | unchanged | `_arena_enabled()` returns `False` when unset |

The copy-count control had to be instrumented before this meant anything. The
first version of that assertion read `expert < name or name == 0`; the control
was uninstrumented, reported **0**, and the `or` clause passed the test
vacuously — the exact no-op-fast-path failure the test exists to catch.

## The timing half — the bound is not reachable

`copy_granularity.py`, two rented cards, same total bytes moved as N copies of
(total/N), pinned → device:

| card | 2.7 MB (today) | 5.4 MB (coalesced) | best observed | **gain** | self-pair |
|---|---|---|---|---|---|
| RTX A5000 | 22.82 GB/s | 23.06 GB/s | 24.57 GB/s | **1.010×** | 1.002× |
| RTX A6000 | 19.07 GB/s | 19.27 GB/s | 19.78 GB/s | **1.010×** | 1.001× |

Routed staging's current per-copy size is already at **92.9 %** (A5000) and
**96.4 %** (A6000) of the best bandwidth either card reaches at *any* granularity.
The knee is between 0.25 and 2 MB; by 2 MB both cards are within ~3 % of
asymptotic. **2.7 MB is past the knee, and 5.4 MB is further past it.**

Amendment rule: `gain < 1.02` → DEAD. Both cards read 1.010×, an order of
magnitude below the 1.06 × landing bar — and that is measured *at the copy
layer*, the single most favourable place the effect could appear, before the
step dilutes it by the non-transfer fraction (#40: experts are 71.3 % of a step).

## What #52's 0.59× actually was

Not per-copy inefficiency. `routed_gbps` divides bytes by **wall step time**
(corrected in e4b `fix/routed-gbps-wall`), so 8.66 / 14.68 counts every
microsecond the link is idle during attention, the norms, the router and the
expert GEMM. **A duty cycle and a per-copy efficiency are different quantities,
and only the second is addressable by copy granularity.** The transfer is
running at ~93–96 % of peak while it runs; the missing 41 % is not copy overhead
to be recovered.

This was worth catching before the rental, not after: the correction cost
**$0.13 and about 25 minutes** and it retired a prediction of 1.06–1.18 ×.

## Disposition

- **Do not merge to `main`.** The prereg's 1.00–1.06 × rule already said a
  sub-6 % win does not pay for touching the pinned-memory layout every offload
  path depends on. At a measured **1.010 × ceiling** that reasoning is stronger,
  not weaker.
- **Keep the branch** (`feat/expert-major-coalescer`, e4b `6d91e1e`). It is the
  reproduction. The layout is correct and bit-identity is proven; if a future
  workload ever has a per-copy size *below* the knee — small-expert MoEs, many
  more tensors per expert, or a host with a slower link — this is the fix,
  already written and already verified.
- **The routed-transfer lane closes.** R4 → R5 was the last branch
  `PREREG-routed-residual` pre-committed to, and it terminates negative.

## What is NOT claimed

- Nothing about the *duty cycle* itself. Whether the 41 % idle is recoverable by
  overlapping the stage with compute is a different question with a different
  mechanism, and no measurement here bears on it.
- Nothing about small-copy regimes. Below ~2 MB the sweep shows a real knee
  (0.25 MB is 68 % of peak on the A5000); coalescing *there* would pay. Routed
  staging at 235B shapes is simply not there.
- Nothing about the gnf4 kernel. #50/#51/#53 closed that line independently.
