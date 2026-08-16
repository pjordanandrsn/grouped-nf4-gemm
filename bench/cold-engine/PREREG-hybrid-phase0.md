# PREREG — Hybrid tier Phase 0: calibration + gate G0

Filed before the target-box run. Tool: `bench/calibrate.py` +
`bench/hybrid_calib.c` at the commit carrying this file.

## Claim under test

The G0 gate of the hybrid CPU/GPU tier program: **grouped scattered
per-expert DRAM reads sustain ≥70% of the same box's STREAM triad
bandwidth.** This is the premise that DRAM can serve as a *compute* tier —
per-token expert reads are MB-scale contiguous blocks chosen by routing, and
if scattering them across an arena costs little against sequential ceilings,
CPU kernels can run at DRAM speed on routed experts.

## Gate rule (fixed in advance)

- scatter_best / triad_best ≥ 70% → **proceed** to Phase 1/2.
- 50–70% → proceed, but re-solve expected hybrid speedups with the measured
  number and report before Phase 2.
- < 50% → **stop and report**; CPU-tier economics need redesign (e.g.
  expert-block interleaving). Human decision.

Note recorded in advance: the ratio can legitimately exceed 100% — the
numerator is read-only traffic (no write-allocate/RFO), the denominator
carries a write stream. The gate compares the two as defined; both raw GB/s
numbers are reported alongside the ratio.

## Fixed configuration

- Routing trace: xorshift64* seed **20260816**, k=8 distinct experts per
  fetch, 400 fetches, E = arena/block (capped 4096, floor 128 per the gate's
  "k=8 over 128+ experts").
- Block sizes swept: 2/4/8/16/32 MiB (brackets real per-expert row sizes:
  Qwen3-30B ≈ 2.5 MiB, gpt-oss-120b ≈ 3.5 MiB, K3 ≈ 17.6 MiB).
- Arena 8 GiB, THP requested, parallel-spread first-touch.
- Triad: 3 × double arrays, 2 GiB total, regular AND non-temporal stores,
  thread ladder = powers of two + all-cores + one-L3-domain, compact vs
  spread pinning; median of 5 timed reps after a warmup rep; first-touch by
  the timing partition, arrays reallocated per config.
- Serial discipline: one bench at a time; cgroup CPU quota, governor, THP
  state, load average recorded in the blob.

## Boxes

1. **Target class:** rented whole-machine single-socket Zen 4/5 EPYC or
   Threadripper PRO (AVX-512 + VBMI + VNNI, many DDR5 channels) with a
   Blackwell-class GPU on Gen4/5 x16 — the box class the hybrid tier is
   designed for. B_vram / B_link / B_nvme recorded on the same box.
2. Secondary (already banked, smoke): AVX2 2-channel DDR4 box
   (`receipts-hybrid-calib-gpu-dev-smoke.json`, quick mode) — sanity only,
   not a citable calibration.

## What will be reported

Full sweep tables (every thread/pin/block point, not just bests), best
configs, the gate ratio and verdict, hardware fingerprint (without
rented-host identifiers), and the calibration blob hash. A miss is reported
exactly like a pass.
