# RESULTS — K3 attribution: the streaming floor IS the wall

Run 2026-08-25 against `PREREG-k3-attribution.md` (#244), EPYC 9755 +
RTX 5090 (kernel-only provisioning; one SSH-dead box auto-refused
first; both destroyed + verified zero). Receipts in
`receipts-k3-attribution/`; decision by `k3_verdict.py` (self-tested,
6 branches). NCU could not run on the rental (the registered risk —
its log is in the receipts); the peel battery carried the cycle, with
its replica within **1.4% / 3.2%** of the product kernel (G-N held).

## The account (shares of each cell's full time)

| component | gate_up (65.1 µs @ sk1) | down (28.2 µs) |
|---|---|---|
| **loads-only floor** | **86.8%** (56.5 µs = **6.73×** roofline) | **83.3%** (23.5 µs = **5.34×**) |
| LUT gather | 3.6% | 8.7% |
| absmax | ~0% | 8.7% |
| activation | ~0% | 1.4% |
| residual | 9.5% | 0.0% |
| split-K delta (sk16 vs sk1) | **−43.1%** (sk16 faster) | — |

## What it means

- **The per-element byte loads themselves are ~85% of the kernel** —
  with ALL arithmetic removed, streaming the same access pattern
  still costs 5-7× the roofline. The wall is load-instruction issue
  rate, not decode arithmetic, not the LUT, not DRAM bandwidth.
- **This retroactively explains K2 exactly**: halving the COUNT of
  scalar byte loads (and paying interleave shuffles) attacked the
  wrong axis — the loads were still per-element uint8. The fix the
  account licenses is WIDTH, not count-of-the-same-width.
- Split-K's negative delta confirms K1's sk16 winner from the other
  direction; the reduction-restructure branch does not fire.

## The branch decision (preregistered map, `access_pattern` fired on both cells)

**Register K4: wide-word loads.** The packed rows are already
contiguous per (n, k-strip) — no repacking is required to load them
as `uint32` (8× fewer load instructions, 8 nibbles unpacked per word
by shift arithmetic). Element order under little-endian byte layout
is deterministic, so the construction is bitwise-provable — the same
hard G-B gate class as K2, now aimed by an account instead of an
intuition. If width alone under-delivers, the true repacking question
(k-strip-major layout for fully coalesced wide loads) is the named
escalation, in that order.

The kernel lane's arithmetic after K3: floor ≈ 8.0+4.0 µs against
today's 44.5+27.9 (winner configs) — the account says most of the
~60 µs gap is load-issue, which is exactly what width addresses.
