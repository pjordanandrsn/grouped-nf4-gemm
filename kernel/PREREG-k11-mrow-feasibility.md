# PREREG — K11: is the M-row waste addressable at T=1 at all?

Registered 2026-08-26, before measurement. This is the lane
RESULTS-250-closing named as decisive: SV2 registered **"MoE GEMV
round 2 — tensor-core mapping beyond dot-pad's 15/16 M-row waste"**
and it must supply **57–63% of remaining headroom** for
250-by-composition to close. Nothing else in the frame moves the
verdict.

K7 refuted a *different* hypothesis about the same slice (occupancy)
and explicitly scoped itself out of this one. So this lane has never
been tested — and before building a kernel for it, this cycle asks
whether a qualifying mapping **exists**.

## Why feasibility comes first

K9 was registered, written, and voided because its subject was never
called. The cost was writing time only because the premise was
checked before a box. The same discipline applies here, and harder:
a tensor-core remap is a substantially larger build than split-K was.

## The arithmetic the lane must beat

A GEMV at T=1 has `K·N` useful MACs. `tl.dot` computes `M·K·N`. At
the shipped `BLOCK_K=64, BLOCK_N=16, M=16`:

```
useful MACs per K-block   1024
MMA computes             16384
waste                     93.8%
```

The waste is not a coding defect — **`M` has nothing in it** because
a single-token GEMV has no second right-hand side.

## Qualifying criterion (fixed before any candidate is judged)

A candidate mapping qualifies iff it **strictly increases useful MACs
per MMA issued** *without* increasing **weight bytes read per
output**. Both halves are required: a mapping that fills `M` by
re-reading weights trades a compute win for a bandwidth loss on a
kernel already at 3.8× its streaming floor.

## Candidates enumerated, with their disposition

| candidate | qualifies? | why |
|---|---|---|
| more tokens in `M` (batch) | **no** — out of frame | that is B>1 serving (BV3, already certified); this lane is single-stream |
| the 8 routed experts in `M` | **no** | `tl.dot` shares ONE `B` across all `M` rows; distinct experts need distinct weights |
| gate/up projections in `M` | **no** | already concatenated along `N`; not available to `M` |
| speculative draft rows in `M` | **no** | speculation refuted twice by measurement (S3, SV2) |
| K-chunks scattered across `M` | **no — TESTED** | correct (max abs err 1.2e-6 vs the reference) but redistributes the SAME `BLOCK_K` nonzeros; MMA volume is `M·K·N` either way, ratio unchanged at 6.2% |
| structured (2:4) sparse MMA | **open** | hardware supports 2:4 on Ampere+; our `A` is 1:16-dense, and Triton's `tl.dot` exposes no sparse path. Named so it is refused explicitly, not overlooked |
| a smaller MMA tile (`M=8`) | **open** | would halve the waste; Triton's `tl.dot` requires `M>=16`, so this needs a non-`tl.dot` path |

## Stage A — adjudicate feasibility (no box)

Publish the table above with each disposition **argued to the
criterion**, and resolve the two OPEN rows against the installed
Triton/hardware rather than from memory:

1. Does this Triton build expose any sparse-MMA or sub-16 `M` path?
   Answer from the installed package and the generated PTX, not from
   documentation.
2. If either exists, it becomes a Stage B candidate with its own
   bars. If neither does, the lane is **REFUTED-INFEASIBLE** on this
   toolchain, and the RESULTS must say the constraint is the
   toolchain's, not the hardware's.

## Stage B — measure the argument (one box, only if useful)

The K-scatter row is settled by arithmetic, and arithmetic has been
wrong in this campaign before. If Stage A resolves both OPEN rows to
"no", Stage B implements K-scatter anyway and measures it: a
**correct** kernel that should show **no** step change. That converts
"I argued no gain" into a receipt.

- **REFUSE** if K-scatter is not bitwise-equivalent to the certified
  path within the K6-B `2^-7` band — a wrong kernel measures nothing.
- **The lane is REOPENED** if K-scatter measurably beats the shipped
  dot-pad by more than the A/A noise floor. That would falsify the
  MAC-ratio argument this whole prereg rests on, and the RESULTS
  would have to say so.

## What a REFUTED-INFEASIBLE verdict means for 250

It closes 250-by-composition — not by arithmetic on estimates, but by
establishing that the lane carrying 57–63% of the remaining need has
no mechanism at T=1 on this toolchain. The RESULTS must then state
plainly that **single-stream 250 is out of reach for this design**,
and that the honest paths left are B>1 aggregate throughput (already
certified at 419 tok/s) or a different quantisation/compute format —
neither of which is this target.

## Receipts

`kernel/receipts-k11/` — the toolchain probe output, the K-scatter
equivalence check, and (if run) its paired step measurement.
`k11_verdict.py` (self-tested) is committed BEFORE any box.
