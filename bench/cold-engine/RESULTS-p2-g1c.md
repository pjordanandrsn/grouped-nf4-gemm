# RESULTS — P2-G1c: REFUTED — promotion and the CPU tier share one DRAM budget

Registered in [PREREG-p2-g1c.md](PREREG-p2-g1c.md) (#209). Run 2026-08-23 on
a rented EPYC 9654 + RTX 5090 (driver 590.48; NUMA pre-gate: 2 nodes, triad
212 GB/s), repo at `ded63c0`. Box calibration in-gate: n* = 4.75, window
{8, 16, 24, 30} at m = 128 — non-empty, exactly as G1b's boundary map
predicted. The arms ran, and the verdict is scored.

**REFUTED at every in-window p** — and the failure is not dispatch this
time. It is the deepest layer yet: **the H2D copies and the CPU tier draw
from the same host-DRAM bandwidth.** Promotion in the no-reuse regime cannot
pay on a DRAM-bound CPU tier, on any box, at any m, with any dispatch glue —
the promoted row's bytes still cross the same DRAM the CPU tier is
saturating.

## The scored table (burst arm; spoilers below)

| p | wall_A (ms) | wall_B (ms) | save/step (µs) | bar (µs) | in-window |
|---|---|---|---|---|---|
| 2 | 6.718 | 7.167 | −448.5 | −52.4 | no |
| 4 | 6.710 | 7.450 | −739.7 | −32.2 | no |
| 8 | 6.722 | 7.652 | **−929.4** | +7.7 | yes |
| 16 | 6.691 | 8.028 | **−1337.0** | +92.7 | yes |
| 24 | 6.713 | 8.439 | **−1726.5** | +205.3 | yes |
| 30 | 6.718 | 8.778 | **−2060.4** | +280.6 | yes |

Correctness all-pass (CPU bit-exact; bytes identical; GPU rel-max 2.1e-3;
retention zero-H2D); p=0 walls 4.8% apart, zero copies; counters exact.
Spoilers fail worse everywhere in-window (sync: −1977 … −5882; serial:
−1042 … −2770), so the instrument distinguishes the arms and the refutation
is a measurement. Arm A sits within 2.3% of its calibrated prediction
(128 × 51.3 µs = 6.57 ms) at every p.

## The decomposition — why this is DRAM, not dispatch

Two signatures in the walls, both pointing at one mechanism:

1. **Excess per promoted row starts at t_link_row.** Charge arm B its CPU
   reduction and look at what each promoted row actually cost:
   `e(p) = (wall_B − (wall_A − p·t_cpu_row)) / p` = **167 µs at p = 8**
   (t_link_row = 167.1), settling to 135/123/**120 µs** at 16/24/30. The
   copies are not hidden at all — they are paid nearly in full.
2. **wall_B grows at t_cpu_row per promoted row** (+51.2 µs/row from p = 8
   to 30; t_cpu_row = 51.3). Each row moved to the GPU gives back its CPU
   time and takes it again in DMA-displaced bandwidth — one for one.

The budget arithmetic says why. This box's shared DRAM ceiling is ~212 GB/s
(triad). The CPU tier alone runs at 171.8 GB/s (64 threads). Adding the
link's 52.7 GB/s of host-DRAM *reads* demands **224.5 GB/s — over budget**
— so the DMA and the GEMV throttle each other and the copy time surfaces in
the wall. And host-DRAM traffic is **p-invariant**: arm B moves
`(m − p)·rowbytes` for the CPU tier plus `p·rowbytes` for the DMA =
`m·rowbytes` regardless of p. In the no-reuse regime there is nothing to
save — only GPU-side tail and dispatch to add.

**Why calibration said hide = 1.0:** the E3b hide instrument's CPU side runs
at 132.5 GB/s (32 threads, its own receipt) — 132.5 + 52.6 = 185 GB/s,
*under* the 212 budget, so its measured full hide was honestly true **at
that operating point**. The real tier's 64-thread 171.8 GB/s crosses the
ceiling. Hide is load-dependent, and the instrument sampled the wrong load.
This also retro-explains part of G1/G1b's losses beyond their dispatch
accounting: the same coupling was active at m = 16.

## What survives, and what is dead

* **Dead: the n\* = 1 transient execution-staging leg** of the elastic
  design — promotion to relieve *this step's* CPU wall — on any
  DRAM-bandwidth-bound CPU tier. The G1b glue exits (F-trim, fused row copy)
  are **moot**: no dispatch engineering recovers bytes that cross the same
  DRAM either way.
* **Alive: reuse economics.** A promoted row that is *resident* serves later
  invocations from VRAM with zero host-DRAM traffic — the row crosses once
  and amortises. Gate E measured exactly this population: 69–96% of
  invocations recur ≥ 2 more times in 32 steps (E1), retain-on-execute with
  no selector (E2). Promotion also still pays when the tier has **DRAM
  headroom** (CPU queue light, or `B_cpu_load + B_link ≤ B_dram`).
* The controller's promote actuator must therefore be re-founded on
  **reuse-based residency with a DRAM-headroom budget**, not same-step
  relief. That is a spec §5 objective change, recorded there; G2 is not run
  against the refuted objective.

Receipts: [elastic-2026-08-23-e3-g1c.json](elastic-2026-08-23-e3-g1c.json),
[p2-g1c-2026-08-23-scored.json](p2-g1c-2026-08-23-scored.json). Box
destroyed; zero instances; ~$0.15 this run, ~$0.77 for the G1→G1c arc.
