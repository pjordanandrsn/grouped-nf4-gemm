# RESULTS — P4: REFUTED — the activations were never the traffic

Registered in [PREREG-p4-rowblock.md](PREREG-p4-rowblock.md) (#229). Run
2026-08-23 on an EPYC 9V74 (32 effective cores, triad 347 GB/s), A/B/A
order (e4b instrument law 6), same staged harness for every arm, clean
rebuilds between arms, bit-exact test passing on-box before any timing.
Receipts: [p4-receipts/](p4-receipts/).

**Scored verdict: REFUTED** — C1 = **1.037×** at rows = 128 (bar ≥ 1.25);
C3 best new ratio **0.743** (bar ≥ 0.85; the confirm run read 0.575,
*worse* than old's 0.644–0.657); C2 = 0.996× (the swap does no harm);
bit-exactness held on both formats. Run-to-run spread on this shared host
was ±13% — and the registered bar sat far above it, so the miss is
decisive, not noisy.

| arm | rows=64 GB/s | rows=128 GB/s | ratio |
|---|---|---|---|
| old A | 88.2 | 58.0 | 0.657 |
| new B | 80.9 | 60.1 | 0.743 |
| new B2 | 91.2 | 52.5 | 0.575 |
| old A2 | 84.7 | 54.6 | 0.644 |

## What the refutation teaches — the premise was wrong at serving shapes

The L3-restream theory computed activation traffic as
`N × rows × K × 4 ≈ 1.6 GB/call`, silently assuming every column re-reads
**all** rows. But the grouped serving call has **~4.4 rows per unique
expert** (128 rows / 29 uniques): each group's activation slab is
`4.4 × K × 4 ≈ 36 KB` — **L1-resident all along**, under either loop
order. The old column-outer order was never streaming activations from
L3 at these shapes; the swap changed cache behavior that was already
fine, and measured exactly that: nothing.

The rows-penalty (0.65 ratio here, 0.67 in e4b's `b16close`) therefore
lives in per-row work that scales with T at *fixed* weight traffic —
the surviving candidates, for a separate registration if pursued:

1. **Partial register-cells**: `NF4_CELL_ROWS = 8` accumulator blocking
   against 2–5 rows/group leaves half-empty cells; per-byte decode work
   does not amortize across rows the way full cells assume.
2. **Work-item granularity**: `(group × 32-column tile)` items with
   unequal `sizes[g]` — imbalance grows with rows variance.
3. The per-call fixed floor (`a ≈ 183 µs` in e4b's own fit) interacting
   with both.

## Disposition

The loop swap is **reverted** in this PR — measured neutral (C2 0.996×)
is not a reason to carry unproven churn in the hottest kernel; the
Phase-8 order returns verbatim. The bit-exact multi-chunk × multi-tile
test **stays** as the guard for any future retiling, and the
`p4_rowscale.py` bench stays as the registered instrument for the next
hypothesis. Per the prereg's hard stop: one box, one A/B, no bar
corrections — the next kernel hypothesis (partial-cell amortization) gets
its own registration or none.

e4b sequencing consequence: with P1 immaterial and P4's first mechanism
refuted, the honest state of G8 B=16 is that the ~3.5–3.8× executor-
structure loss remains characterized but unexplained at the mechanism
level; partial-cell amortization is the leading suspect with a concrete
falsifiable shape (the penalty should track rows-per-unique crossing the
cell size, testable offline against existing receipts before any box).

Box destroyed; zero instances; ~$0.16 this A/B; program total ~$1.61.
