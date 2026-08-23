# RESULTS — P2-G1: do the promotion mechanics pay? (REFUTED)

Registered in [PREREG-p2-g1.md](PREREG-p2-g1.md) (#204). Run 2026-08-23 on a
rented EPYC 9J14 (48 effective cores, NPS1×2) + RTX 5090 (driver 610.57),
`pytorch/pytorch:2.9.0-cuda12.8-cudnn9-devel`, repo at `38b7005`.

**Verdict: REFUTED at every swept p, exactly as the registered rule scores
it.** The realised per-promotion saving `(wall_A − wall_B)/p` is negative at
every `p ∈ {1,2,4,8}` against a bar of `≥ 0.70 × Δ = 24.3 µs/row`. The
promotion mechanics, as the engine currently dispatches them, do not pay in
the no-reuse regime — "the controller is not built until the mechanics pay"
(the prereg's own words). This is the gate doing its job: two structural
costs the phase-2 model never charged are now measured and named.

## The scored table

| p | wall_A (ms) | wall_B (ms) | save/row (µs) | realized/pred | arm |
|---|---|---|---|---|---|
| 1 | 0.775 | 1.303 | **−528.1** | −15.19 | side-stream |
| 2 | 0.755 | 1.362 | **−303.5** | −8.73 | side-stream |
| 4 | 0.756 | 1.522 | **−191.7** | −5.51 | side-stream |
| 8 | 0.754 | 2.168 | **−176.8** | −5.09 | side-stream |
| 1 | 0.781 | 1.422 | −641.3 | −18.45 | spoiler |
| 2 | 0.756 | 1.675 | −459.2 | −13.21 | spoiler |
| 4 | 0.753 | 2.020 | −316.8 | −9.11 | spoiler |
| 8 | 0.801 | 2.841 | −255.1 | −7.34 | spoiler |

Bar: ≥ +24.3 µs/row (0.70 × Δ, Δ = 34.8 µs from this box's calibration).
Medians over 5 repeats × 32 steps, paired A/B per step. Arm A sits within 2%
of its calibrated prediction (16 × 46.4 µs = 742 µs) at every p.

Correctness (all registered checks pass — the walls are scored, not voided):
CPU bit-exact vs `ref_gemv_grouped`; promoted device bytes identical to the
DRAM source; GPU rel-max within the committed 2e-2 (measured 2.3e-3);
retention with zero H2D on re-invocation. `p = 0` validation: walls within
6.6%, zero copies. Counter accounting exact (`p × steps`) at every p.

**Falsifiability held**: the registered spoiler (synchronous default-stream
copies) fails the bar *worse* than the side-stream arm at every p — the
instrument distinguishes hiding from not-hiding, so the refutation is a
measurement, not noise.

## Why it loses — two unmodeled structural costs

**1. Serial host dispatch (~100–500 µs/step depending on CPU contention).**
The step's promote/launch/sync host path runs on the same thread that drives
the CPU tier, serial with it. Measured clean on an idle box: 19.8 µs to
enqueue both copies + event, 81.5 µs for a warm decode launch with list
expert-ids (20.9 µs with pre-staged device ids), 4 µs sync. Measured inside
the harness step under the GEMV pool's spin window: 215 µs enqueue, 211 µs
launch. At `p = 1` the modeled saving is 34.8 µs — the dispatch alone buries
it, and dividing a per-step cost by `p` is why larger `p` looks less bad.

**2. The link budget breaks the prereg's own premise at p ≥ 4.** The prereg
asserted "the CPU is the long pole by construction at every swept p" by
comparing the copies against all 16 CPU invocations (~2 ms). But arm B's CPU
work is the **remaining 16 − p** rows. At this box's calibrated rates
(t_link_row = 169 µs, t_cpu_row = 46.4 µs): p = 4 puts 677 µs of copies
under 557 µs of CPU; p = 8 puts 1354 µs under 371 µs. The copies cannot
hide under work that shrinks as p grows — a constraint the phase-2
controller must encode (promotions per step bounded by the link budget:
`p · t_link_row ≤ (m − p) · t_cpu_row`), independent of VRAM or reuse.

## What broke on the way (all fixed, all merged)

* **Latent int32 kernel bug, found live** ([#205](../../pull/205)): the first
  scored attempt crashed with `cudaErrorIllegalAddress` at `p = 8`, step ~30
  — transient-pool slot ids reach 244 and `eid × stride_be` passes 2^31.
  `nf4_grouped` had measured this exact boundary and casts eid to int64; the
  MXFP4 port dropped the cast. Fixed in both MXFP4 kernels; regression tests
  now live on the compiled path in `test_offsets_2gib.py` (boundary sampling
  both sides of 2^31 + the as_strided pool shape that found it). That first
  run is **void** as a verdict; its surviving numbers matched the scored run
  in sign and size at p ∈ {1,2,4}.
* **Scrub contention artifact** (disclosed pre-data): the 1 GiB cache scrub
  between paired arms initially ran on torch's default intra-op pool (256
  threads here); its spin-wait inflated the promote path ~10×. The scored
  run scrubs single-threaded. The dispatch cost it was inflating is still
  real (see cost 1) — the fix removed the amplifier, not the cost.
* CPU tier run at `--threads 64`, the thread count the box's own calibration
  found best (B_cpu = 189.9 GB/s at 64 of 48 physical — SMT), keeping the
  bar and the measurement on the same configuration.

## Box provenance (the calibration gate earned its keep)

Three hosts were refused before one qualified, ~$0.60 total: an EPYC 9645
host whose SSH key plumbing never came up; a Bergamo 9754 slice measuring
n* = 1.54 (B_cpu 75 GB/s on Zen4c, GPU proxy 540 GB/s — throttled); a
9654 "Emb" host at n* = 1.64 whose NPS4 topology puts a first-touch arena on
one of 8 NUMA nodes (raw triad 315 GB/s, kernel-path 83 GB/s; the container
blocks `set_mempolicy`, so no interleave fix exists inside it). The 9J14
that qualified: n* = 4.87 ∈ [2,5], B_cpu 189.9, B_link 52.0, B_gpu 756.6,
hidden_frac 1.0 (`elastic-2026-08-23-e3.json`).

## What this buys the program

G1 refutes the *mechanics as dispatched*, not the economics: the calibrated
per-row saving (34.8 µs) is real, but it is smaller than the per-step serial
dispatch cost and the link budget at exactly the p values a controller would
want. The phase-2 spec's controller design therefore inherits two hard
constraints before G2 is worth running: **batch and amortize** (one dispatch
per promotion burst, pre-staged device ids, launches issued off the CPU
tier's critical thread) and **cap p by the link budget** per step. Receipts:
`elastic-2026-08-23-e3.json`, `p2-g1-2026-08-23-scored.json` (+ the void
first run's log in `p2-g1-2026-08-23-crashed.log`).
