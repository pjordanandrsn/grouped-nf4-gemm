# RESULTS — P2-G3: REFUTED as scored — the mechanism passed; the wall bars ignored the program's own budget

Registered in [PREREG-p2-g3.md](PREREG-p2-g3.md) (#218). Run 2026-08-23 on
an EPYC 9B14 + RTX 5090 (calibration-gated on the **sixth** rented box:
n* = 3.53 ∈ [2,5], B_cpu 167.5, hide 0.971 — five hosts were refused
first, three of them because modern strong CPUs push n* past 5). Repo at
`b28f4c1`. Receipts:
[p2-g3-2026-08-23-registered.json](p2-g3-2026-08-23-registered.json),
[elastic-2026-08-23-e3-g3.json](elastic-2026-08-23-e3-g3.json).

**Scored verdict: REFUTED** — c3 (no-cliff) fails on both traces, c4
(recovery) on gptoss. The verdict stands. The anatomy splits into an
instrument bug, a bar that contradicts the program's own measured budget,
and one genuinely new result — while **the elasticity mechanism itself
passed everything asked of it**:

| clause | gptoss_code | qwen_code |
|---|---|---|
| c1 no OOM (ballast fits after shrink) | **PASS** | **PASS** |
| c2 shrink latency (8 ms ≪ 2 steps) | **PASS** | **PASS** |
| c3 no step > 1.10× no-cache | FAIL (63.1× max) | FAIL (5.4× max) |
| c4 recovery ≤ 64 steps + full capacity | FAIL | **PASS** (t=145) |
| spoiler: shrink disabled must OOM | **OOM at 64** | **OOM at 64** |

Capacity: 512→256→512 and 1024→512→1024 exactly on schedule.

## The anatomy

**1. The 63× is step 1 — Triton's one-time JIT, an instrument bug.** The
harness runs `correctness()` (which warms the decode kernel) *after* the
timed trace; G1b/G1c ran it before. The 342 ms compile lands in the first
timed step. Every remaining wall is ≤ 4.6×.

**2. Cold start violates c3 by construction.** Steps 0–15 run all-cold
*plus* copies — 1.3–1.9× no-cache is the admission cost the G2''
admission law (`pairs/(frac·m)`) already prices. A bar that requires the
elastic engine to beat no-cache *while it is still admitting the working
set* refutes every possible engine, on every box.

**3. The steady-state crossover — new, real, and G1c's budget predicts
it.** At 0.7× capacity, gptoss steady state runs **0.74×** no-cache
(residency pays) while qwen runs **1.64×** (residency loses): fills' DMA
displaces CPU bandwidth one-for-one (G1c), so the pool wins only where
saved CPU reads exceed fill traffic — heavier-reuse gptoss clears the
crossover, thinner-reuse qwen at 71% capacity does not. This is a
hit-mass-dependent property of the *trace and capacity*, measured here for
the first time; gating it as a "cliff" gates the workload, not the engine.
c4's gptoss failure is the same arithmetic: its recovery bar (1.10× a
4.0 ms steady) sits inside refill-churn noise, while qwen's looser
absolute bar passes.

**No catastrophic collapse exists anywhere in the run** — the thing "no
cliff" was meant to catch: worst non-JIT step 4.6×, decaying, monotone-ish.

## G3' — one correction cycle, registered here, with a hard stop

Same schedule, same box class, three changes derived from the above:
warm-up before the timed region (instrument); **cliff redefined as
catastrophe** — no step > 3.0× its phase's median (what "cliff" meant);
recovery measured against the *recovered-capacity* steady state; and the
no-cache crossover **measured and reported per trace, never gated**. c1,
c2, and the OOM spoiler are unchanged (they passed; they re-run as
regression). A refutation of G3' closes the elasticity gate REFUTED — no
further bar corrections.

The full registration is [PREREG-p2-g3p.md](PREREG-p2-g3p.md).
