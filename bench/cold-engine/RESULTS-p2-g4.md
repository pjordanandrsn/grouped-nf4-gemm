# RESULTS — P2-G4: UNINFORMATIVE — the box's GPU never left idle clocks, and the gates never checked

Registered in [PREREG-p2-g4.md](PREREG-p2-g4.md) (#222, harness contiguity
fix #223). Run 2026-08-23 on an in-gate 9B14 + RTX 5090 (n* = 3.81,
triad-class CPU 158.5 GB/s, NVMe O_DIRECT probe **5.70 GB/s**). Repo at
`c9bdce8`. Receipts:
[p2-g4-2026-08-23-registered.json](p2-g4-2026-08-23-registered.json),
[elastic-2026-08-23-e3-g4.json](elastic-2026-08-23-e3-g4.json).

**Scored verdict: UNINFORMATIVE** — the sequential arm did not fail the
objective bound, so the instrument could not distinguish overlap from
serialization on this box, and per the registered definition no wall is
scored. Under the prereg's one-run hard stop, phase 2's gate program closes
here.

## What the run demonstrated before the verdict collapsed

* **The tribrid executed end to end**: a 10 GB arena baked to NVMe (row
  layout contract-tested), `ColdTier` landing rows at 5.7 GB/s O_DIRECT,
  `ColdCpuView` materializing pinned kernel-shaped stacks, the VRAM pool
  filling over the side stream, all three kernels on the same bytes — and
  **correctness 3/3**: tier rows byte-identical to the snapshot ground,
  CPU bit-exact, GPU within the committed 2e-2.
* Steady walls: overlap 16.73 ms, sequential 17.49 ms — a real but small
  gap (4.4%), both under the (inflated) objective bound.

## The anatomy — clock-ramp collapse, measured

The solo rates expose it: **t_gpu_row = 161.9 µs**, ~10× every prior 5090
measurement in this program (the G1c-class decode executes in ~12–16 µs).
Post-run diagnostics on the same box: **SM clock 180 MHz of 3,090 max,
91 W of 575 W**, and the 20×4096³ matmul took 340 ms (healthy boxes: 158–
177). This host's power policy never ramps the GPU for the decode pattern —
short launches separated by host work and per-step syncs — so the GPU tier
ran ~17× below boost throughout, T_gpu dominated the alones (15.3 of the
16.7 ms wall), and the 1.15× bar floated above both arms. e3's own
B_gpu proxy read 693.7 GB/s because its sustained stream ramps and stays —
which is exactly why the registered box gates (n*, triad, NVMe) missed it:
**none of them measures burst-pattern GPU clocks.**

Two findings leave the run:

1. **The box-gate omission**: wall gates involving the GPU tier need a
   burst-clock probe (interleaved launch-sync at decode granularity,
   compared against the sustained rate) — a ramped-streams calibration says
   nothing about a launch-gap workload, the same instrument lesson as
   E3b's load-dependent hide.
2. **The phenomenon itself is serving-relevant**: decode serving IS short
   GPU bursts between host work. On hosts with lazy clock policies the
   GPU tier's effective rate is not a property of the silicon but of the
   launch pattern and the driver's ramp policy — an engine-level concern
   (persistence mode, locked clocks, or batching to sustain boost)
   recorded for any future registration.

## Phase 2's gate program, closed

G1→G1c: promotion mechanics REFUTED — the DRAM budget; the reuse law
derived. G2→G2'': the law's substance at 100% (73/73 LRU-class, exact
novelty equilibrium); gate closed on the throttle bound, admission law
measured. G3→G3': elasticity mechanism green twice (real-VRAM shrink,
4/4 spoiler OOMs, crossover replicated); gate closed on the spike bar.
G4: UNINFORMATIVE on clock-collapse. Every verdict from a prereg frozen
before its measurement; every closing bar honored at its registered price.

Box destroyed; zero instances; program total ~$1.25.
