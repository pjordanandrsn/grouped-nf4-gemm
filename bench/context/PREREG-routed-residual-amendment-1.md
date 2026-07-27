# PREREG-routed-residual — AMENDMENT 1

**Amends:** `PREREG-routed-residual.md`, OTS-stamped sha256
`ba7a29ceb025a3ddbcd4b06fdf238898f84298b0b78699104c61e52517d2dc47`.
**Status: PRE-DATA.** No pod has been created, no arm has run, no timing exists.
The registration window is still open, which is the only reason this may be
written at all. It must be committed and OTS-stamped before the pod exists.

**The original is NOT edited.** Its stamp stands and its bytes are unchanged, so
anyone can still read exactly what was registered first and diff it against
this. An amended prereg whose original was rewritten is not a registration.

**Nothing here weakens a prediction.** R1–R6, every bar, every falsification
criterion, and R5's decision rule are carried over verbatim. So are the arms
(C / T1), the model (Qwen3-235B-A22B), the host class (2×A100-SXM-80GB), and the
requirement that speculative staging and the expert cache stay OFF in every arm.

## A1 — Fixture: `reps` 2 → **6** (the reason for this amendment)

The original fixture says "median of 2". It is now **median of 6**, 24 runs
across the four arms.

**Evidence, from three OLMoE smoke runs** (A2000, gen3 ×8, 6.23 GB/s —
correctness only, no timing claim is made or implied from them):

- A median of 2 is the arithmetic mean of 2, so one draw carries the ratio.
- Smoke run 3 was quiet (self-pair spread 0.010) and still incoherent: `T1s` and
  `T1c` each measured **faster** than control (0.989, 0.990) while combined `T1`
  measured **slower** (1.047) — though T1 is exactly T1s + T1c. The cause is
  visible in the raw positions: `T1` drew 0.1963 at one position and 0.1792 at
  the other. Two samples cannot resolve a ±5% band.

`reps` **must be even**, or the ordering in A2 cannot balance.

## A2 — Ordering: ABBA, stated explicitly

The original requires arms "interleaved rather than blocked". That is now
specified as **reversal on alternate reps (ABBA)**, which equalises each arm's
sum of run positions (69/69/69/69 at reps=6) and cancels linear drift exactly.

This is a specification of the original clause, not a replacement for it — but
it is stated here because plain repetition also satisfies "interleaved" on a
loose reading, and plain repetition is what smoke run 1 showed to be wrong: C
was pinned to first position, and all three other arms read above it
(1.018–1.037) in one direction with no scatter against a 0.030 spread.

## A3 — R6 adjudication: no verdict when the run cannot resolve the band

**This is registration-relevant and is the reason it appears here rather than in
a commit message.** It is a pre-committed rule about when R6 yields a verdict at
all, and such a rule is worthless if written after seeing the data.

**Rule: if the self-pair spread ≥ R6's band width (0.05), R6 returns no verdict.**
It reports `UNDERPOWERED`, `pass = None`, and the nominal ratio for inspection.
Not a pass, not a fail.

Smoke run 2 is why: balanced ordering, but the box drifted 7.4% inside a single
two-minute run, giving a spread of 0.071 — wider than the band R6 is trying to
resolve — while R6 still printed "REGRESSION" from noise alone. A band the run
cannot resolve must not produce a registered finding in either direction.

R6's band itself is unchanged: **[0.95, 1.00]**, below-band still a miss to
explain rather than a win to claim.

## A4 — Harness gates that are NOT registered predictions

Three checks were added to the harness after the original was stamped. They are
**gates and diagnostics, not predictions**, and may not be quoted as
confirmatory results. Registered here so their status is fixed in advance rather
than argued afterwards:

- **`logit_identity`** — max |Δlogit| across arms, expected exactly 0. This is
  *stronger* than R1's greedy-id gate and matches the standard finding #24
  established (gate on logits, natural prompt). **R1 as stamped remains the
  registered gate**; this is reported beside it. It does not replace R1 and R1 is
  not retroactively strengthened.
- **`arm_fidelity`** — that each arm ran its own copy loop. Both plans write
  identical bytes, so an arm leak is invisible to every other check; without
  this, arm C could silently run the treatment's copy loop and R6 would measure
  half of T1 while reporting it as T1.
- **`position_balance`** — per-arm position sums in the receipt, so A2 is
  auditable rather than trusted.

Any of these failing invalidates the run. None of them, passing, is a result.

## Unchanged and restated

- **R4 is still the load-bearing prediction**: routed-policy implied GB/s ≤ 0.70 ×
  the probed pinned ceiling. **R5's decision rule stands in both directions**,
  including the negative — R4 falsified means *do not build the coalescer*, and
  that outcome is registered, not a disappointment to be reinterpreted.
- **Not claimed**: nothing about K3; nothing about tok/s as a product number;
  nothing about speculative staging or the expert cache. The A2000 smoke numbers
  are plumbing verification and are not results — its link negotiated gen3 ×8 at
  6.23 GB/s against the A100's 22.21, and no fraction measured there transfers
  across host classes.
