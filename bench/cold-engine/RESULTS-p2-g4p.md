# RESULTS — P2-G4': REFUTED, scored — the tiers share one memory system, so max() was never the wall

Registered in [PREREG-p2-g4p.md](PREREG-p2-g4p.md) (#225, screen
corrections #226/#227 — both disclosed, both conservative). Run 2026-08-23
on a clock-healthy 9J14 + RTX 5090 (n* = 4.21, B_gpu 950.7, burst gate:
matmul20 173 ms, gap 36.8 µs ≤ bound 56.6, keep-warm mode), repo at
`f5353ed`. Receipts:
[p2-g4p-2026-08-23-registered.json](p2-g4p-2026-08-23-registered.json),
[elastic-2026-08-23-e3-g4p.json](elastic-2026-08-23-e3-g4p.json).

**Scored verdict: REFUTED on both gates, with the spoiler failing as
required** — the instrument distinguishes, so this is a measurement, and
under the hard stop it closes phase 2's gate program with a scored
objective.

## The numbers

Clocks finally honest: t_gpu_row **19.9 µs** (the program's expected
class; G4's collapsed host read 162). Correctness 3/3. Steady state
(medians, steps 32–63):

| quantity | value |
|---|---|
| VRAM hit rows / step | **94.5 of 96** (the residency law at work) |
| CPU rows / step | 1.5 |
| NVMe traffic / step | 13.2 MB — exactly one cold row landing |
| T_storage alone | 2.88 ms | 
| T_gpu alone | 1.88 ms |
| T_cpu alone | 0.08 ms |
| max(alones) → G4a bar | 2.88 → 3.31 ms |
| sum(alones) | 4.83 ms |
| **sequential arm** | **4.41 ms** (> 3.31: spoiler fails ✓) |
| **overlap arm** | **5.22 ms** (> 3.31: G4a ✗; > 0.80×4.41: G4b ✗) |

**The overlap arm is slower than the sequential arm** — and slower than
the serial *sum* of the alone walls. Concurrency costs 18% here, it does
not pay.

## The mechanism — the program's oldest law, third appearance

The objective `wall ≤ 1.15 × max(T_cpu, T_gpu, T_storage)` presumes the
tiers are independent resources. They are not: the NVMe landing writes
DRAM, the view materialization memcpys read and write DRAM, the VRAM
fills DMA-read DRAM, and the CPU tier reads DRAM — one memory system
under all of it (G1c measured the identity for H2D vs GEMV; G3 measured
the crossover it implies; this run closes the triangle for storage).
Running the storage pipeline concurrently with compute adds interleave
inefficiency (the κ-mixing penalty class) and thread-convoy overhead on
top of a wall that byte-conservation already pins near the sum. On
shared-memory-system hardware, **max() was never achievable; the tribrid's
true wall model is bytes-through-DRAM plus mixing**, and the engine-design
consequence is the same one every gate has been converging on: *don't
overlap DRAM-crossing work with DRAM-bound work — schedule it into DRAM
headroom* (idle gaps, or steps whose compute is genuinely
non-memory-bound).

What held: the residency law delivered a 98% VRAM hit rate at 0.7×
capacity; the storage tier landed rows correctly at 4.6 GB/s O_DIRECT
with byte-identity; every kernel met its committed contract; and the
falsifiability arm failed exactly as registered, on the first run whose
clocks were verified.

## Phase 2's gate program: closed, fully scored

G1c: no-reuse promotion REFUTED (the DRAM budget). G2'': the reuse law's
substance at 100%. G3': the elasticity mechanism green twice. G4':
**the independent-tier overlap objective REFUTED, scored, spoiler-valid**
— superseding G4's UNINFORMATIVE with the same instrument on verified
clocks. The spec's objective (§1) survives in revised form: minimize
bytes-through-DRAM per step (residency, the measured crossover), spend
headroom on landings — not overlap for its own sake.

Box destroyed; zero instances; program total ~$1.45.
