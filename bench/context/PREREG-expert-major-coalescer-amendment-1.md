# AMENDMENT 1 — PREREG-expert-major-coalescer

**Written PRE-DATA.** The original prereg is not edited; its stamp still binds
its original bytes, so the two can be diffed by anyone.

## Why

The original registers `1.06–1.18×` from cutting the routed stage's copy count
32 → 16 per layer. That number rests on one unmeasured premise, stated there in
a single clause:

> ~32 copies per layer of ~2.7 MB each — **below where an H2D reaches asymptotic
> bandwidth**

If 2.7 MB is *already* at asymptotic bandwidth, merging pairs of copies cannot
produce bandwidth that was not missing, and the coalescer's ceiling is ~1.00×
regardless of how correct the implementation is.

The premise is also weaker than it looked when it was written, because of a
change that landed after it: `routed_gbps` was corrected to divide bytes by
**wall step time** (e4b `fix/routed-gbps-wall`). The 8.66 / 14.68 = 0.59× in #52
is therefore *bytes moved ÷ the entire step*, which includes every microsecond
the link sat idle during attention, the norms, the router and the expert GEMM.
**A duty cycle and a per-copy efficiency are not the same quantity, and only the
second is addressable by copy granularity.** #40 measured experts at 71.3 % of
an e4b step, so a substantial idle fraction is expected and is not a defect.

This is the `feedback-benchmark-testbed-policy` corollary applied to itself:
*before optimising toward a bound, check the bound is reachable.* The gnf4 arc
spent months aiming at a "4.9× headroom" that was distance to a coalesced read
the kernel could not perform.

## What is added

A gating step **before** the 235B fixture: `bench/context/copy_granularity.py`.
Same total bytes, same pinned source buffer, moved as N copies of (total/N),
sweeping per-copy size across 0.25–32 MB. Paired, with a self-pair that must
read ~1.000× or the run is void.

It measures achieved bandwidth **inside the copy window only**. It does not
measure duty cycle and does not attempt to.

## Decision rules, fixed now

Let `gain = GB/s at 5.4 MB ÷ GB/s at 2.7 MB` — the exact granularity change the
coalescer performs.

- **`gain < 1.02`** → **DEAD.** Copy granularity is not the constraint. The
  coalescer's ceiling is ~1.00×, #52's 0.59× is duty cycle, and the A100 run is
  **not rented**. Record the negative and close the transfer lane.
- **`1.02 ≤ gain < 1.06`** → **MARGINAL, do not proceed.** Below the landing bar
  *at the copy layer*, before the step dilutes it by the non-transfer fraction.
  A win that is already too small in the most favourable place it could appear
  cannot become large once diluted.
- **`gain ≥ 1.06`** → **LIVE.** The premise holds; run the 235B fixture on two
  cards under the original prereg's rules.

## This amendment cannot manufacture a positive

It can only *stop* the expensive run. `gain ≥ 1.06` does not land the coalescer
— it merely licenses the measurement that decides. Every gate in the original
prereg (bit-identity, copy-count-down, two cards, ≥1.06× on both) stands
unchanged.

## Status of what is already measured

The correctness half is **done and passing** on the QNAP (correctness-only, per
policy): under `E4B_OFFLOAD_ARENA=expert`, staged rows are bit-identical to the
name-major control, `max|Δ| = 0.000e+00`, and instrumented copy count falls
12 → 6 at the test fixture (E=16, 3 routed experts, 4 tensors), which is
32 → 16 at 235B shapes and inside the registered 8–16 band.

That result stands on its own regardless of this amendment's verdict. It is a
correctness result, and the QNAP is a valid testbed for exactly that.
