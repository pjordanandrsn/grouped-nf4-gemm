# RESULTS — leg 3: interleaved pairing

**Grades `kernel/prereg_dequant_forward_interleaved.json`** (OTS-stamped
pre-data at `13f7bdd`, stamp `74140de`). **Read
[`ERRATUM-amendment2-overstated.md`](ERRATUM-amendment2-overstated.md) first** —
this prereg's stated motivation is overgeneralised and the erratum says how.

## VERDICT: NOT CONFIRMED — but the 4090 passed, for the first time in the program

| | H100 (sm_90) | RTX 4090 (sm_89) |
|---|---|---|
| live | 15 / 32 | **30 / 32** |
| F1 | not adjudicable (2 live cells) | **PASS — 7/7 at bar, median 1.539, band HIT** |
| Q1 / Q2 / Q4 | FAIL / pass / pass | **pass / pass / pass** |
| **device** | **VOID** | **CONFIRMED** |

Four device-runs across three legs have now each had exactly one clean device,
and the identity of the clean one keeps changing. That is the finding.

> **MEASUREMENT CLASS — backfilled 2026-08-14**
> (`kernel/prereg_gpu_busy_labelling.json`)
>
> **F1, this leg's primary criterion, is a STEP RATIO, not a kernel
> measurement.** It is graded on the decode band, where both arms run far below
> 50% GPU-busy — fused 9–33% (H100) / 13–52% (4090), baseline 4.4–11% / 4–55% —
> so roughly 90% of each step is host time and F1 compares one kernel launch
> against a per-expert Python loop. Its median of **1.539 is a real wall-clock
> step ratio and it replicates well** (1.588 / 1.522 / 1.539 across three
> pairing schemes and two devices). It is not a claim about kernels, and the
> PASS above should be read as a step-ratio PASS.
>
> The numbers in this document are unchanged. Measured in
> [`host_bound/`](host_bound/) and reported in
> [`FINDING-host-bound-small-batch.md`](FINDING-host-bound-small-batch.md); the
> instrument now runs in every leg beside the self-pair, so no later leg
> acquires its label after grading.

## THREE DEFECTS IN THIS LEG, ALL MINE, ALL FLATTERING

**1. The self-pair gate became near-vacuous.** It is not degenerate — per-pair
ratios scatter with IQR 0.0250 and a p05–p95 spread of 0.0701. But the median
is taken over ~500 pairs, so its standard error is ≈ IQR/(1.35·√n) ≈ **0.0008**,
and the registered ±3% band is therefore **~37 standard errors wide**. It fired
on **0 of 64** cells across both devices. The band was calibrated for the block
estimator, which was ~25× noisier. So "30/32 live" is partly a bar that stopped
biting, not an instrument that started passing. **The band is NOT retroactively
tightened** — that is the forbidden move; it is registered pre-data for any
future interleaved leg, set from the estimator's measured standard error.
Every void on both devices came from the halves gate.

**2. `P1` does not measure what it was registered to measure.** It computes the
block statistic as `median(tb)/median(ta)` **from the interleaved collection**,
where A and B are already interleaved in time and therefore already
drift-immune. So it compares two *reductions* of one drift-immune dataset, not
two *collection strategies*. It reads a near-null 0.9858 (4090) and 1.0125
(H100) for exactly that reason. Measuring the real dividend needs a genuine
block collection — all A, then all B — run alongside. **P1's numbers here
should be ignored.**

**3. The no-sync timing is contaminated on CPU-bound cells.**
⚠️ **THE 3.76× IN THIS SECTION IS RETRACTED** — see
[`FINDING-host-bound-small-batch.md`](FINDING-host-bound-small-batch.md). It
compared leg 2's figures from one pod against leg 3's from a different pod, a
cross-run comparison this program's own rule forbids. Re-run on the same cell in
the same process the two instruments agree, median 1.00× on both devices. The
section is kept as written, with this marker, because deleting it would hide the
error rather than record it. Per-call device time, leg 3 over leg 2 (which
synchronised every iteration):

| device | small cells | 5–13 ms cells |
|---|---:|---:|
| H100 | **3.76×** ⚠️ RETRACTED (0.44 → 1.68 ms) | 1.11–1.25× |
| RTX 4090 | **0.60×** ⚠️ RETRACTED (0.92 → 0.41 ms) | 0.93–0.97× |

On the 4090, dropping the per-call sync does what it should: spans get shorter
and cleaner, because the sync stall is gone and the GPU is the bottleneck. On
the H100 they get 3.8× **longer**, because a no-sync span absorbs CPU wait — and
at decode-band sizes on that device the step **is CPU-bound**. That is what
drove the H100's 17 halves failures, together with a pilot that gave it a median
of **103 pairs against the 4090's 514** (`_pairs_for` measures wall time, so a
CPU-bound cell looks expensive and gets fewer pairs — backwards).

## THE FINDING THAT IS WORTH MORE THAN THE VERDICT

**At small batch on an H100, this training step is CPU-bound: the GPU finishes
in 0.44 ms while the whole step takes ~1.68 ms.** Leg 2 measured GPU-only time
(sync per call); leg 3 measured the whole step. Both are real and they answer
different questions, which is why leg 3's H100 F1 reads 1.853 against leg 2's
1.588 — not a disagreement, a different quantity. On the 4090 the GPU is the
bottleneck and the two agree.

Any small-batch claim on a fast datacenter card is therefore substantially a
claim about Python, not about either kernel. That is not fixable by pairing.

## WHAT SURVIVES, ACROSS THREE INSTRUMENTS

The measurement itself is stable where it can be compared. `decode_m8` median
`D_base/G_base`:

| | |
|---|---:|
| leg 2 run 1, H100, block pairing | 1.588 |
| leg 2 run 2, H100, long blocks | 1.522 |
| **leg 3, 4090, interleaved** | **1.539** |

Three pairing schemes, two devices, all inside leg 1's registered 1.3–3.0 band.
Fidelity is unchanged again (`b_rel` G/D ≈ 0.76).

## What this licenses

Nothing new. One device confirmed and one void is not the two-device
conjunction, and this leg's three defects mean its own instrument claims are
weaker than leg 2's, not stronger. What it adds is the CPU-bound finding above
and one more replication of a number that has now survived three instruments.
