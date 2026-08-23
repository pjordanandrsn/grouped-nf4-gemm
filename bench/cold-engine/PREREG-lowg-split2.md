# PREREG — the low-G split, re-certified under the A/A gate

Registered before any measurement. The first certification
([RESULTS-lowg-split.md](RESULTS-lowg-split.md)) was REFUTED as scored on
a host whose same-binary A/A spread (median 15%, worst 55%) made ±10%
bars unscoreable; the change was reverted unproven with one likely-real
signal (1.8× at the N=256 starved cell). This re-certification applies
the instrument law that failure bought, with every revision disclosed:

1. **The A/A gate** (new): arm A (old kernel, parent commit) runs
   TWICE before anything else. Per scored cell,
   `noise = |A1 − A2| / mean`. The box is accepted iff noise ≤ 10% at
   both B1 cells AND median noise ≤ 5% across the B2 set — else destroy
   and re-hunt, up to **3 hosts**, then report UNRUNNABLE. B never runs
   on a box that failed the gate.
2. **Noise-derived bars** (new): each B2 cell's allowance is
   `max(10% of mean(A), 3 × |A1 − A2|)` — three times its own measured
   noise or the round bar, whichever is larger. B1's improvement must
   exceed `3 × noise` at its cell as well as the fixed floor.
3. **Kernel-policy revision** (disclosed): the split now targets
   `items × rsplit ≥ 2 × threads` (was ≥ 1×) — the first cert measured
   the N=768/G=1 cell stuck at 48 items over 32 threads (1.5 ragged
   waves, ratio 1.18) while the 4×-items N=256 cell delivered 1.8×.
   The starvation guard is unchanged: well-fed calls keep `rsplit = 1`
   bit-for-bit.

## Scored cells (the full grid is not re-run — exposure shrinks to the claims)

* **B1 (the win)**, both required: (G=1, rows=128, N=256) and
  (G=1, rows=128, N=768): `(mean(A) − B)/mean(A) ≥ max(0.30, 3 × noise)`.
* **B2 (no harm)**, every cell: well-fed sentinels (16,128,256),
  (32,64,256), (8,128,768), (16,128,768), (32,128,768), (64,128,768) and
  the serving shapes (29,64,768), (29,128,768):
  `|B − mean(A)| ≤ max(0.10 × mean(A), 3 × |A1 − A2|)`.
* **B3**: the multi-row bit-exact test on-box before any timing (voids).

Protocol: A1 → A2 → gate → B → A3 (confirm; if A3 deviates from
mean(A1,A2) by more than the gate bounds at any scored cell, the run is
VOID — the box drifted mid-experiment). Medians of 50 per cell, staged
harness, clean rebuilds, 32 threads.

**Hard stop**: up to 3 hosts for the gate; ONE scored B arm total.
REFUTED ⇒ revert again and the line closes — a third certification would
need a fundamentally different instrument (perf-counter isolation on
bare metal), not another bar.
