# RESULTS — Hybrid tier Phase 0: calibration + gate G0

Pre-registered: `PREREG-hybrid-phase0.md` (+ `.ots`, stamped before the run).
Tool: `bench/calibrate.py` + `bench/hybrid_calib.c` at the commit carrying
this file. Blob: `receipts-hybrid-calib-genoa-9654p-2x5090.json`
(sha256 `f0b480245c67a814…`).

## Verdict

**GATE G0: PASS — grouped scatter = 95.0% of STREAM triad (threshold ≥70%).**
Phase 1 and Phase 2 may proceed.

## Target box (rented whole machine, ~25 min)

Single-socket AMD EPYC 9654P (96c/192t, Zen 4, 12 L3 domains), AVX-512
F/BW/VL/**VBMI**/**VNNI** all present and used (`compiled_simd: avx512`),
251 GiB DDR5, no cgroup CPU quota. 2× RTX 5090 32 GiB (575 W limit) on
Gen5 x16. NVMe: single consumer Gen4 drive, O_DIRECT honored. Governor
`schedutil` (host root unavailable; recorded, not changed). THP
`madvise` honored.

## Measured ceilings (achieved, never spec)

| bus | number | config |
|---|---|---|
| **B_dram** triad | **278.3 GB/s** | 32t, spread-across-CCDs, NT stores |
| **B_dram grouped scatter (the gate)** | **264.3 GB/s = 95.0% of triad** | 32t, 8 MiB blocks, E=1024, k=8, fixed-seed trace |
| B_vram triad (each 5090) | 1571.9 / 1572.5 GB/s | fp32, 12 B/elem convention |
| B_link H2D / D2H (64 MiB pinned) | 52.3–56.0 / 56.1 GB/s | both GPUs, Gen5 x16 |
| B_link 8 KiB pinned latency | 3.3 µs H2D / 2.9–3.0 µs D2H | per copy, evented |
| B_nvme seq / rand (1 MiB, O_DIRECT) | 7.32 / 7.34 GB/s | QD16 best |

Sweep shape (full tables in the blob):

- Scatter is **flat across block sizes**: 2/4/8/16/32 MiB → 264.0 / 264.3 /
  264.3 / 264.0 / 263.7 GB/s. MB-scale contiguous blocks amortize routing
  randomness completely; there is no small-expert penalty down to 2 MiB.
- Triad thread ladder (NT, spread): 45 GB/s at 1t → 241 at 8t → 278 at 32t,
  flat-to-slightly-down beyond. Bandwidth saturates at ~1/6 of the threads;
  Phase 2 kernels should budget ~32 cores for bandwidth and leave the rest
  alone.

## What the ratios say for the tier model

- **DRAM-compute vs streaming: grouped DRAM reads are 4.7× the link**
  (264.3 vs ~56). An expert computed in place beats an expert streamed to
  the GPU whenever CPU kernel efficiency exceeds ~21% of the DRAM ceiling —
  and gate G2 demands ≥70%. This is the economics the directive's G3 (≥4×
  end-to-end) rests on, now grounded in this box's own numbers.
- VRAM : DRAM : NVMe = 1572 : 264 : 7.3 ≈ 215 : 36 : 1 — three genuinely
  distinct tiers; the placement solver's bandwidth inputs are these, not
  spec sheets.
- 8 KiB round-trip raw latency ≈ 6.3 µs (3.3 + 3.0). Phase 1's ≤35 µs p50
  per-layer budget is therefore spent on synchronization/launch design, not
  wire time — the same conclusion the #105/#108 campaign reached from the
  other side.

## Honest caveats

- Triad at 278 GB/s is ~60% of this platform's theoretical 12-channel peak.
  Likely NUMA-interleave (NPS1-style) and population effects on a rented
  box; not investigated further because the gate is a *ratio* on the same
  memory and the tier economics use the achieved number. If a future box
  measures materially higher B_dram, re-run the blob there — every phase's
  targets are stated against the blob, so they move with it.
- Governor was `schedutil`, not `performance` (container without host
  root). Recorded in the blob per protocol; NT-store triad and scatter run
  long enough that DVFS settling is amortized.
- O_DIRECT worked on this box; the fallback (page-cache + fadvise drop)
  shipped but did not engage. First box where it engages must say so in its
  receipt.
- `wall_seconds: 40.7` for the CPU section — the box is fast; sizes were
  the full defaults (2 GiB triad arrays, 8 GiB arena, 400 fetches, 5 reps),
  not quick mode.

## Secondary smoke (AVX2 box, quick mode — sanity only, not citable)

Dev box (Comet Lake 6c/12t, 2ch DDR4, AVX2): triad 26–27 GB/s (NT), scatter
28–29 GB/s = **~105–108% of triad** — the ratio exceeds 100% because the
numerator is read-only while triad carries a write stream; recorded in the
PREREG as an anticipated property, not an anomaly. Note the older
torch-based `phase0_ddr_bench.py` triad on the same box reads ~15 GB/s:
different tool, regular stores (RFO traffic uncounted), no pinning — the
receipts name their tool per house rule; both are banked.

## What this unlocks

- **Phase 1** (CPU router) — proceed; latency budget confirmed feasible.
- **Phase 2** (AVX-512 grouped GEMV) — target on this box class:
  ≥70% × 278 GB/s ≈ **195 GB/s** sustained from packed bytes at decode
  shapes, bit-exact vs `dequant_ref`.
- **Phase 3** solver constants: `B_vram_effective : B_dram_grouped ≈ 6 : 1`
  on this class; link carries activations (µs-scale) + cold NVMe stream
  (7.3 GB/s) without contending with either compute tier.
