# PREREG — the GEMV is instruction-issue bound (hypotheses 4, 5, 6)

Stamped before writing any kernel. Registered against finding #41, which
falsified three mechanisms for the same gap and pre-committed the next step to
"a real profiler, not a fourth guess". This *is* a fourth guess — but it comes
from reading the kernel source, which none of the first three did, and it makes
a falsifiable prediction the previous three did not.

## The gap being attacked

`_gemv_nf4_grouped` (`kernel/nf4_grouped.py`) moves 56.62 MB at **99–111 GB/s**
where a flat decode of the same bytes reaches **487**. #41 established it is:

- not memory-bound (14% of A40 HBM peak)
- not compute-bound (0.95% of peak FLOP/s)
- not decode-bound (the decode primitive is 20.4% and runs at 85.3% of achievable)
- not occupancy-bound (**8× the warps/SM bought 16%**)

**Reframing:** those four negatives are jointly the signature of a kernel bound
by *instruction issue and loop-carried latency*, not by any throughput resource.
Low achieved bandwidth is the symptom, not the cause. Two issue-side defects are
visible in the source.

## Hypotheses

**H4 — every packed byte is loaded twice.**
```python
bytes_ = tl.load(b_base + (kk[None, :] // 2), ...)
```
`kk` spans `BLOCK_K` *consecutive* k. Lanes 0,1 address byte 0; lanes 2,3 byte 1.
A 32-lane warp therefore covers **16 distinct bytes**, so each 32 B sector is
half wasted and the kernel issues **2× the load instructions** it needs. Fix:
load `BLOCK_K/2` bytes once, extract both nibbles.

**H5 — a cross-lane reduction runs every iteration.**
```python
acc += tl.sum(w * a[None, :], axis=1) * am
```
`tl.sum(axis=1)` is a tree reduction over `BLOCK_K` lanes, executed `K/BLOCK_K`
times (**64×** at K=4096). Fix: accumulate a 2-D tile scaled per k-block
(`acc2 += w * a[None,:] * am[:,None]`) and reduce **once** after the loop.

**H6 — `BLOCK_K` is a correctness constraint, which is why #41's sweep found
nothing.** The absmax index `(k0 // BLOCK_K)` assumes exactly one absmax per
`BLOCK_K` elements, pinning `BLOCK_K` to the NF4 blocksize (64). #41 swept
`BLOCK_N × warps × split_k` and **structurally could not reach the dominant
knob** — the loop trip count. Fix: index absmax by `k // blocksize`
independently of `BLOCK_K`, so `BLOCK_K` becomes tunable.

## Pre-committed predictions

Measured on the flagship shape (8 experts, N=3072, K=4096, 56.62 MB) on an
RTX A2000 12 GB, median of 7, CUDA-event timed, versus the shipped kernel **on
the same card in the same process** (ratios transfer between cards; absolute
times do not — the project's own law).

| arm | prediction |
|---|---|
| H4 alone (single-load nibbles) | **1.15–1.6×** |
| H5 alone (one reduction at the end) | **1.25–1.8×** |
| H6 alone (BLOCK_K decoupled, retuned) | **1.1–1.5×** |
| **all three composed** | **≥ 2.0×** |

**Diagnostic that discriminates issue-bound from anything else:** sweep K at
fixed bytes-per-expert. If the kernel is issue/latency bound on loop trip count,
time is **super-linear in trip count** at constant bytes; if it is bandwidth
bound, time tracks bytes and is flat in K. Registered prediction: the shipped
kernel is **super-linear**, and the fixed kernel is **flatter**.

## Decision rules, fixed now

- **Composed ≥ 2.0×** → hypothesis class confirmed; land the kernel, re-measure
  the 235B step, and report the step-level number *measured*, not extrapolated
  from #40's 2.19× arithmetic.
- **Composed 1.3–2.0×** → partial. Land it (it is free throughput) but record
  the gap as **still open**, and do not claim the mechanism is understood.
- **Composed < 1.3×** → **falsified, and I stop guessing.** Three plus three is
  enough. Escalate to Nsight Compute per #41's original branch. No seventh
  hypothesis without a profiler trace.

## Correctness gate — non-negotiable

Every arm must hold **max rel err ≤ 5e-05** against the reference path on the
same inputs (#41's sweep measured 4.57e-05 across 64 configs; this is that bar).
A faster kernel that misses this is **discarded, not reported with a caveat.**
Composition with `enable_fast`'s scatter-combine must stay bit-stable.

## Confounds recorded in advance

1. **The A2000 is a shared production GPU** (voice-tts, SDXL, vLLM serve the
   house). Contention inflates variance. Mitigation: interleave arms within one
   process, report medians, and record GPU utilization alongside. If run-to-run
   spread on the *shipped* kernel exceeds 10%, the box is too noisy and the
   comparison is void — re-run in a quiet window rather than reporting it.
2. **One shape, one card.** A2000 (sm_86, 12 GB, ~288 GB/s) is not the A40 of
   #39 nor the A100 of #40. Only the **ratio** is being claimed to transfer.
3. **#40's 2.19× is arithmetic, not a measurement.** It assumes the GEMM reaches
   its decode floor. Any step-level claim must be measured on the 235B, and that
   is a separate rented run — not something this A2000 arm can deliver.
4. `_decode_plan`'s existing constants were tuned on other grids; a plan change
   that helps here must not regress those.
