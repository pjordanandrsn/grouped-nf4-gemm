# PREREG — a REACHABLE target for the decode GEMV

Every "headroom" figure in this project (#39's **4.9×**, #46's **~2.1×
remaining**) is distance to a number the project's own prereg calls
*"genuinely unreachable, useful only as a bound"*:

> "The 487 GB/s ceiling is a flat streaming read with no reduction and no
> output — genuinely unreachable, useful only as a bound."
> — `PREREG-gemv-occupancy.md`, confound 2

Optimising toward an acknowledged-unreachable number is how a project talks
itself into work with no payoff. This run replaces it with a bound that prices
what a GEMV **actually has to do**.

## What 487 GB/s already includes, and what it omits

#39's ladder was: `read` 685.5 GB/s → `decode` (unpack + LUT + scale) **487.0**
→ shipped gemm 99.2. So the 487 figure **already carries** nibble unpack, the
16-entry codebook gather, and the absmax scale.

It omits three things a GEMV cannot skip:

1. **the activation multiply** — `w * a[None, :]`
2. **the K-reduction** — a cross-lane tree reduction, the term most likely to
   dominate and the one no streaming read pays
3. **the store** — `[T, N]` bf16 out

## Design: one term at a time, same bytes, same shape

Five kernels over identical packed input (E=8, N=3072, K=4096), each adding
exactly one term to the one above it. Paired timing with a self-pair check
(#46's lesson: an unpaired sweep once called the default 1.283× faster than
itself).

| rung | adds | what it isolates |
|---|---|---|
| R1 `read` | — | pure byte streaming |
| R2 `decode` | unpack + LUT + scale | reproduces #39's 487 |
| R3 `mul` | × activation | is the multiply free? |
| R4 `reduce` | K-reduction | **the suspected dominant term** |
| R5 `store` | write `[T,N]` | = a complete GEMV |

**R5 is the reachable target.** It does everything the shipped kernel does and
nothing it doesn't, with no tiling or dispatch overhead. `shipped / R5` is the
honest remaining headroom.

## Pre-committed predictions

| quantity | prediction |
|---|---|
| R2 / R1 | **1.3–1.6×** (≈ #39's 685.5→487 = 1.41×) |
| R3 / R2 | **1.00–1.15×** — activations are `K` elements, L1-resident |
| **R4 / R3** | **1.5–3.0×** — the reduction is the big omitted term |
| R5 / R4 | **1.00–1.10×** — output is `T×N` bf16, tiny beside the input |
| **shipped / R5** | **1.2–1.8×** |

**The headline prediction is that the real headroom is ~1.5×, not 4.9× or
2.1×** — i.e. most of the apparent gap is work a GEMV is obliged to do.

## Decision rules, fixed now

- **shipped / R5 < 1.3×** → the kernel is already near its reachable limit.
  **The structural line CLOSES** as a measured negative; do not write the
  hand-rolled CUDA GEMV, and retire the 4.9×/2.1× figures from the docs.
- **1.3–2.0×** → real but modest. Record the corrected target; a hand-written
  GEMV is only justified if something else motivates it.
- **> 2.0×** → a genuine gap survives pricing the reduction and the store. The
  hand-written CUDA GEMV becomes worth its cost, and R5 tells it what to beat.

## Rails

- **Rented compute, not the QNAP** — this is timing, and the QNAP is a
  correctness-only testbed (shared production box, 2.2× run-to-run drift).
  One cheap single GPU; a kernel microbench does not need two.
- Paired timing, self-pair must read ≈1.000×, or the run is void.
- Every rung must produce a **numerically checkable** result where meaningful
  (R5 against the shipped kernel at the bf16 floor); a faster rung that computes
  less is not a bound, it is a mistake.
- **Two cards before any conclusion that changes the docs** — #43 shipped a 2.2×
  regression and #44 a wrong-answer kernel on one card's evidence.
