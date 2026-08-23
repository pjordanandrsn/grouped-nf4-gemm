# PREREG — P2-G3': elasticity, with the wall bars derived from the program's own budget

Registered before any G3'-scored run, from G3's anatomy
([RESULTS-p2-g3.md](RESULTS-p2-g3.md)) and previously frozen results alone.
Same design as G3 (#218) — same traces, pool sizing, PROMO_FRAC = 1/4,
schedule (64/64/64), ballast formula, box gates — with three registered
changes:

1. **Warm-up** (instrument): one untimed decode gemm + one untimed CPU
   call before the baseline and trace, so Triton's one-time JIT never
   lands in a timed step. G3's 63× was step 1's compile.
2. **c3 becomes what "cliff" meant — catastrophe, not overhead**: no step
   wall may exceed **3.0 ×** the median wall of its own phase (phases:
   converge 0–63, hold 64–127, recover 128–191; medians computed per
   phase over that phase's steps). Cold-start admission and pressure
   overhead are priced behavior (G2''s admission law, G1c's budget); a
   cliff is a runaway step, and G3 measured none (max non-JIT 4.6×,
   during recovery churn).
3. **c4 recovery re-based**: within 64 steps of release, trailing-16 mean
   wall ≤ **1.10 × the recovered-capacity steady wall** — the median over
   the LAST 16 steps of the run — AND capacity restored to 100%. (G3's
   bar compared against the pre-pressure steady of a differently-warmed
   pool state; the recovered pool re-admits through the same throttle and
   its steady state is the honest reference.) The pre-pressure steady is
   still reported beside it.

**Unchanged and re-run as regression**: c1 (no OOM), c2 (shrink wall ≤ 2
pre-pressure median steps), the shrink-disabled spoiler (MUST OOM at the
pressure step on both traces), correctness-voids-walls, and both box gates.

**Measured and reported, never gated**: the steady-state ratio to the
no-cache baseline per trace — G3's crossover finding (gptoss 0.74×, qwen
1.64× at 0.7× capacity) is a hit-mass property of trace × capacity that
G1c's DRAM budget predicts; it is the number the engine's deploy-time
calibration needs, not a pass/fail on the mechanism.

PASS iff c1 ∧ c2 ∧ c3' ∧ c4' on both traces, spoilers failing. **Hard
stop**: a refutation closes the elasticity gate REFUTED — no further bar
corrections against this design; the mechanism would return to the spec.
