# PREREG — P2-G3: elasticity under real VRAM pressure

Registered before any measurement. Spec §11's G3 shape ("mid-run VRAM
ballast injection: no OOM; transient shrinks within 2 steps; wall degrades
monotonically (no cliff); recovery within 64 steps of release"), frozen
here as exact thresholds against the real machinery: `SegmentedRowPool`
(#217 — the §10 gnf4 deliverable whose `shrink()` frees real VRAM),
`DevRowCache` segments, the G1b burst copy path, the real MXFP4 CPU/GPU
kernels, real `torch` ballast on a real allocator. One box.

## Design

[p2_g3_elastic.py](p2_g3_elastic.py) on a calibration-gated box
(`elastic_e3.py` first; `n* ∈ [2, 5]` or the box is rejected; NUMA
pre-gate ≤ 2 nodes, triad ≥ 100 GB/s before any spend).

* **Traces** (registered): `gptoss_code` (pairs 732, m = 96, k = 4) and
  `qwen_code` (pairs 1439, m = 96, k = 4) — smallest and largest working
  sets of the committed rank set. Arena: the G1 builder at the G1 shape
  (rowbytes 8,812,800), `E = pairs`, one arena row per (layer, expert).
* **Pool**: `SegmentedRowPool`, `seg_rows = 64`,
  `segments = ceil(0.7 · pairs / 64)`, `routed = k`. The law: retain-on-
  execute with `SMOOTH_CAP = ceil(m/4)` (PROMO_FRAC = 1/4, in-window per
  the G2'' admission law). Per §4.4, a row filled this step executes on
  **CPU** this step (promotion never stalls a step); GPU executes resident
  hits only, per-segment gemms; copies ride the side stream under one
  event per layer-burst.
* **Schedule**: steps 0–63 converge; **step 64**: the pressure event —
  the law shrinks `ceil(S/2)` segments, then ballast of
  `(free_measured − RESERVE) + ceil(S/2) · seg_bytes` is allocated
  (RESERVE = 2 GiB) — sized so it CANNOT fit without the shrink; steps
  64–127 hold; **step 128**: ballast freed, pool grows back; steps
  128–191 recovery.
* **Baseline**: 16 all-CPU steps (pool bypassed) before the run →
  `wall_nocache` (median).

## Registered claims

1. **No OOM**: the ballast allocation succeeds after the law's shrink, and
   every subsequent step completes.
2. **Shrink latency** (I3): the `shrink()` call's wall ≤ 2 × the median
   pre-pressure step wall.
3. **No cliff**: every step wall in all phases ≤ **1.10 × wall_nocache** —
   the elastic engine never does worse than having no cache at all.
4. **Recovery**: within 64 steps of release, trailing-16 mean wall ≤
   **1.10 ×** the pre-pressure steady wall (median of steps 32–63) AND
   capacity restored to 100% of its pre-pressure segments.

PASS iff 1 ∧ 2 ∧ 3 ∧ 4 on **both** traces. Correctness voids walls
(registered, as G1b): sampled resident-row GPU outputs within the committed
2e-2 vs the bf16-fed reference; sampled promoted bytes identical.

## Falsifiability — the registered spoiler, run after the main arms

**Shrink disabled**: the identical schedule with `shrink()` a no-op. The
ballast allocation MUST raise a CUDA out-of-memory error (caught; recorded;
no wall scored past it). If ballast fits without shrinking, the instrument
cannot distinguish elasticity from slack VRAM and the run is UNINFORMATIVE.

## What would count as a miss

Any clause failing on either trace ⇒ REFUTED per-clause; correctness
failure ⇒ walls void; spoiler not OOMing ⇒ UNINFORMATIVE; a box failing
the n* or NUMA pre-gates is destroyed before any G3 measurement (three
gate-refused boxes preceded G1's run — the pattern stands).
