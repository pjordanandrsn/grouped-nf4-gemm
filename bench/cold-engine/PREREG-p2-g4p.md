# PREREG — P2-G4': the objective, with the GPU actually at speed

Registered before any G4'-scored run, from G4's anatomy alone
([RESULTS-p2-g4.md](RESULTS-p2-g4.md)): the instrument, gates, bars,
traces, arena, correctness rules, and hard-stop structure are **identical
to G4** (#222/#223) — the only additions manage and *verify* GPU clocks,
closing the box-gate omission G4 measured (SM pinned at 180 MHz for the
decode launch pattern while every registered gate passed).

## Additions, in full

1. **Clock management ladder** (`--clock-mode auto`, applied before any
   timed region and held for the whole run):
   a. `nvidia-smi -pm 1` and `-lgc <max SM clock>` — the locked-clock
      path. Rented containers often lack the privilege; a failure falls
      through, disclosed in the receipt.
   b. **Keep-warm stream**: a 64×64 matmul on its own CUDA stream every
      ~2 ms from a daemon thread — ~0.1% occupancy, no privileges, defeats
      the down-ramp. This is also the engine-relevant mitigation: an
      engine serving decode on a lazy-ramp host must do exactly this (or
      batch to sustain boost). Runs during BOTH arms, the solo rates, and
      the spoiler — one uniform environment.
2. **Burst-clock box gate**, after the ladder, before anything timed:
   * the 20×4096³ matmul must complete ≤ **220 ms** (the program's healthy
     5090 range is 158–177; G4's lazy-ramp host took 340);
   * a decode-pattern probe — 100 launches of an **8.8 MB device-to-device
     copy** (the decode gemv's memory-bound profile, ~12 µs at boost) with
     a sync and a 2 ms host sleep between each, compared against the same
     probe with no gaps: pass iff **gap ≤ max(3 × no-gap, no-gap +
     45 µs)**. *(Twice corrected pre-measurement, both disclosed: the
     first bar (absolute 50 µs on a tiny matmul) measured sync-wake
     latency and rejected a healthy host at 50.5 µs; the second (pure
     ratio ≤ 3.0) still did — wake appears only in the gap arm and does
     NOT cancel, measured +34.6 µs on a host whose matmul20 was 168 ms
     with the keep-warm firing every 2 ms so clocks could not have
     dropped. The exec-dominated workload separates the signals: a lazy
     host's copy collapses ~13× (≫ both bound terms); a healthy
     interrupt-wait host adds bounded wake and passes. The scored gates
     G4a/G4b are untouched by all of this — the screen has only ever
     erred conservative, rejecting healthy hosts. No G4'-scored
     measurement has been made.)*
   A box failing both rungs of the ladder and the gate is destroyed and
   re-hunted, never waived (eleven-box precedent).
3. The matmul health check moves to the **pre-gate** (run before e3, with
   the NUMA/triad probe — G4's box would have been caught there for
   $0.02).

Everything else — G4a `median wall ≤ 1.15 × max(alones)`, G4b `≤ 0.80 ×
sequential`, the sequential arm as the must-fail spoiler, correctness
voids, solo-rate alones, phase medians, the full dry-step warm-up, n* and
NVMe gates — is unchanged and re-frozen by reference.

**Hard stop, unchanged in kind**: one G4'-scored run. PASS or REFUTED
closes the phase-2 gate program with a scored objective; a second
UNINFORMATIVE (spoiler unable to fail *despite* verified clocks) closes it
UNINFORMATIVE with the anatomy — no third attempt.
