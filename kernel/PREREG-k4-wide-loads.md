# PREREG — K4: wide-word loads for the M=1 GEMV

Registered 2026-08-25, before any measurement. The branch K3's account
fired (`RESULTS-k3-attribution.md`): the loads-only floor is 86.8% /
83.3% of the two census cells at 6.73× / 5.34× roofline — the wall is
per-element load-instruction ISSUE RATE. K2's refutation showed that
halving the COUNT of same-width loads is the wrong axis; K4 changes
the WIDTH.

## The registered edit (bitwise-provable construction)

The packed rows are contiguous per (n, k-strip); no repacking. The
wrapper passes `B.view(torch.int32)` (legal: K/2 % 4 == 0 at every
census K) and both decode GEMV kernels gain a `WIDE_LOADS` constexpr
branch: load `[BLOCK_N, BLOCK_K/8]` uint32 words (8× fewer load
instructions), unpack 8 nibbles per word by shift arithmetic. With
little-endian byte order, word bits map deterministically: element
`8m + e` has shift `(e//2)*8 + (4 if e%2==0 else 0)` — built as a
`[BK/8, 8]` shift broadcast and reshaped to `[BK]`. Same nibbles,
same lanes, same LUT gather ⇒ **bitwise-equal outputs at any fixed
config** (the K2-class hard gate, this time aimed by an account).
Opt-in via `GNF4_GEMV_WIDE_LOADS=1` until the cert; flips default in
the RESULTS PR if certified (#220 lesson).

## Instruments

- **The mechanism gate lives in the peel battery**: `k3_attr_bench`
  gains a `loads_only_wide` peel — the wide-load streaming floor. K3
  measured the scalar floor at 56.5 / 23.5 µs; if width addresses the
  named wall, the floor ITSELF must drop.
- The K1 sweep re-run UNDER wide loads (env set): the optimal
  (bn, warps, sk) may shift with 8× fewer load instructions; the
  wide-winners are what H-K evaluates and what a cert bakes.
- On-box CUDA bitwise (bench, both kernels, legacy-winner configs and
  wide-winner configs), e2e graph pair via the b1d harness, B=16
  sanity, full props suite. Interp subprocess tripwire extended to
  the wide path.

## Bars (before any number)

- **G-B (hard)**: bitwise legacy-vs-wide at fixed configs, both
  kernels, on CUDA; e2e tokens identical between paths. ANY mismatch
  ⇒ REFUTED regardless of speed.
- **H-M (mechanism)**: wide loads-only floor ≤ **0.5×** the scalar
  floor per cell. FAIL ⇒ the width theory is wrong even if wall time
  moves — REFUTED-FOR-MECHANISM, escalate to k-strip-major repacking
  (the named next), no partial credit.
- **H-K (kernel)**: wide-winner pair-time ≤ **40 µs** (from 72.8;
  1.8×). PARTIAL (40, 55]; > 55 ⇒ REFUTED-FOR-WIDTH ⇒ repacking is
  the lane.
- **H-E (e2e)**: graph step ≤ **12.0 ms** (≥ 83 tok/s).
  KERNEL-WIN-NOT-AT-WALL if H-K passes without it.
- **GS**: B=16 in the certified band.
- Consequences: all pass ⇒ CERTIFIED — wide becomes the default and
  the wide-winner configs bake into `_decode_plan`, both in the
  RESULTS PR.

## Verdict calculator

`k4_verdict.py`, self-tested both directions; receipts in
`kernel/receipts-k4-wide/`.
