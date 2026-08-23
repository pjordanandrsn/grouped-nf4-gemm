# PREREG — P2-G4: the objective — overlap realised across all three tiers

Registered before any measurement. Spec §11's G4 shape ("step wall ≤ 1.15 ×
max(T_cpu, T_gpu, T_storage) at equilibrium — overlap realised — against a
sequential-sum baseline demonstrating the gap"), frozen here with every
operational definition, against the real tribrid: the baked NVMe arena
(`nvme_arena.bake`, the layout contract-tested against the engine's in
`test_mxfp4_arena_layout`), `ColdTier(pinned=True)` as the DRAM hot set,
the CPU kernel reading the tier's flat pinned landing buffer directly via
`as_strided` stacks (the O_DIRECT scatter path needs 4096-multiple segment
lengths, which the scales segment is not; a strided read has no such
constraint), `SegmentedRowPool` (#217)
as the VRAM pool, the G1b burst copy path, and the MXFP4 kernels. One box.

## Design

[p2_g4_tribrid.py](p2_g4_tribrid.py) on a calibration-gated box (n* ∈
[2, 5]; triad ≥ 100 GB/s; **NVMe probe**: O_DIRECT sequential read ≥ 1 GB/s
on a freshly written file, else the box has no real storage tier and is
rejected).

* **Arena**: synthetic gpt-oss-shaped snapshot at the G1 gate/up shape
  (N1 = 5760, K1 = 2880; down projection baked too — cold reads fetch whole
  expert rows as the engine does), L = 24 layers × E = 32 experts = 768
  rows, baked to the box NVMe with align = 4096. Trace: `gptoss_code`
  (768-pair space, 732 seen).
* **Tiers**: ColdTier hot_rows = ⌈0.75 × 768⌉ = 576 pinned DRAM rows (the
  tier's own LFU residency; capacity misses keep the storage tier active at
  steady state); VRAM pool as G3 (0.7 × pairs, 64-row segments,
  PROMO_FRAC = 1/4). Execution per §4.4: VRAM hits on GPU (per-segment
  gemms), everything else on CPU from the view's stacks; fills copy from
  the view's pinned rows on the side stream (I9 burst).
* **Prefetch = the overlap under test**: at each step's start, a background
  thread `ensure()`s step t+1's routed set (oracle prefetch — registered
  simplification: E2's null makes speculative prediction unpromising, so
  G4 measures the mechanism's **ceiling** under known-next routing; ctypes
  and O_DIRECT release the GIL, so the reads genuinely overlap the CPU
  tier).
* **Arms**: A = overlapped (prefetch on); B = sequential (prefetch off,
  every cold read lands inline inside its step). B is both the registered
  baseline and the spoiler.
* **Alone walls = solo rate × measured work**, per steady step from arm A's
  own counters, then medians: T_cpu_alone(t) = cpu_rows(t) × t_cpu_row_solo
  (a 64-row GEMV from the view's stacks, median of 50, measured on-box);
  T_gpu_alone(t) = gpu_rows(t) × t_gpu_row_solo (measured likewise);
  T_storage_alone(t) = nvme_bytes(t) / B_nvme_solo (the O_DIRECT probe rate
  on the arena's own file). Solo rates are what "alone" means — each tier
  running with the machine to itself.
* **Tier concurrency**: the prefetch thread is the ONLY ColdTier user
  between its spawn and the next step's join (the tier is single-controller,
  like every allocator in this repo): per step — join previous prefetch →
  `ensure(t)` on the main thread (warm hits) → spawn prefetch(t+1) →
  compute overlaps the reads.
* **Blip-robust scoring (G3' inheritance)**: all gates on **phase medians**;
  one full untimed dry step (NVMe read + fills + gemms + CPU call) before
  any timed region — the complete warm-up inventory; steps 0–1 excluded
  from every statistic.

## Registered claims (steady phase = steps 32–63, medians)

* **G4a (the objective)**: median wall_A ≤ **1.15 × max(T_cpu_alone,
  T_gpu_alone, T_storage_alone)**.
* **G4b (the demonstrated gap)**: median wall_A ≤ **0.80 ×** median wall_B.
* **Spoiler**: arm B must **fail** G4a's bound (median wall_B > 1.15 × max
  of the alones) — serialization must be visible to the instrument, or the
  run is UNINFORMATIVE.
* **Correctness voids walls**: CPU outputs bit-exact vs `ref_gemv_grouped`
  on the view's own bytes (sampled); GPU within the committed 2e-2
  (bf16-fed reference, sampled); a tier-resident row byte-identical to its
  on-disk bytes (sampled) — the `test_nvme_residency` contract, re-checked
  live.

The G1c budget prediction `(bytes_cpu + bytes_h2d + bytes_nvme)/B_triad`
is computed and **reported** beside G4a — if the shared-DRAM budget makes
1.15 × max unachievable, the refutation arrives with its mechanism attached.

PASS iff G4a ∧ G4b ∧ spoiler-fails ∧ correctness. **Hard stop**: one run;
a refutation closes phase 2's gate program REFUTED on the objective, with
the budget arithmetic as the finding — no bar corrections.

## What would count as a miss

G4a fails ⇒ the tiers do not overlap on this hardware as the spec's
objective demands (expected mechanism: the G1c shared-DRAM budget — CPU
reads, H2D fills, and NVMe landings cross one memory system). G4b fails ⇒
prefetch buys < 20% over serial — the overlap exists but does not pay.
Spoiler passing G4a ⇒ UNINFORMATIVE. Box gate failures ⇒ destroy and
re-hunt, never waived (six-box precedent).
