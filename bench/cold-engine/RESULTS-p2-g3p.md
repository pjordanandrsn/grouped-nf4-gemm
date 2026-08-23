# RESULTS — P2-G3': the elasticity gate closes REFUTED on the spike bar — with the mechanism green twice over

Registered in [PREREG-p2-g3p.md](PREREG-p2-g3p.md) (#220). Run 2026-08-23
on the same in-gate 9B14 + RTX 5090 as G3 (n* = 3.53), repo at `77f4ee8`.
Receipt: [p2-g3p-2026-08-23-registered.json](p2-g3p-2026-08-23-registered.json).

**Scored verdict: REFUTED — and per this prereg's hard stop, the
elasticity gate closes here; no further bar corrections.** c3'
(no step > 3.0 × its phase median) fails on both traces: gptoss at
**7.99×**, qwen at **3.008×** — over the bar by 0.008; the bar is the bar.

Everything else, across BOTH G3 runs now:

* **c1 no OOM** — the unfittable-without-shrink ballast fit, twice per
  trace, four times total. **c2** — `shrink()` freed real VRAM in ~8 ms.
* **c4' recovery** — both traces recovered to their recovered-capacity
  steady within the window (gptoss t = 177, qwen t = 145), capacity
  restored 512→256→512 and 1024→512→1024 on schedule.
* **The spoiler OOM'd at step 64 on all four spoiler runs** across G3+G3'.
* **The warm-up fix worked**: G3's 63× JIT step is gone
  (max-over-nocache 4.83× both traces).
* **The crossover replicated**: gptoss steady state **0.56×** no-cache
  (residency pays, better than G3's 0.74 with the JIT removed), qwen
  **1.44×** (capacity-bound loss, vs 1.64) — the G1c budget's hit-mass
  dependence, now measured twice.

## What the closing failures are made of

The receipt locates every phase maximum: gptoss converge max is **step 1
again** (34.9 ms — ~30 ms of residual one-time initialization the
single-gemm warm-up did not cover), and every other phase max on both
traces is a **~19–29 ms blip appearing about once per phase** (hold step
86: 25.9 ms; recover step 165: 21.5 ms; qwen converge step 9: 28.9 ms …)
— the same magnitude regardless of phase, trace, or pool state, on a
shared 24-NUMA-node cloud host. A 3.0× ratio over 4–10 ms phase medians
is, in practice, a ~25 ms-blip detector. No spike grows with time,
correlates with pressure state, or resembles engine runaway.

Per the hard stop, that observation changes nothing about the verdict —
it defines what returns to the spec: (1) the residual step-1 init needs a
full warm-up inventory at engine level; (2) single-step wall bars on
shared cloud boxes need blip-robust formulations (a quantile over steps,
or a repeat-median per step) — G4's registration inherits both, recorded
in spec §11. The elasticity *mechanism* — event-gated real-VRAM shrink,
pressure survival, throttled recovery, falsifiable OOM — passed every
clause aimed at it, twice.

Box destroyed; zero instances; the G3 arc spent ~$0.31 across six rentals
(five gate-refused), program total ~$1.08.
