# Confirmatory v3 — VERDICT: NOT CONFIRMED (X1, X2, X3, X4 fail; X5 passes)

**Date:** 2026-07-13 · **Frozen code:** `591020d` · **Protocol:** `kernel/prereg_v3_confirmatory.json` (OTS pre-data; exploratory Scout calibrations committed pre-stamp) · **Reducer:** `bench/phase1/reduce_confirmatory_v3.py` · **Evidence:** `bench/phase1/confirmatory_v3/`

v3 tested the two post-v2 changes — the SM-conditional decode constant and
split-K for occupancy-starved grids — on 26 shapes (8 census + 8 v2-continuity
+ 10 fresh blind: Grok-1, Mixtral-8x22B, Hunyuan-Large, Arctic,
Switch-Base-128) × 4 backends × n=3 × two devices. It failed more criteria
than v2, and the failures teach more than a pass would have.

## Registered criteria — outcomes

| criterion | outcome |
|---|---|
| X1a config recovery (A2000, ≥1.10 paired on ≥3 of 4 named v2-loss cells) | **FAIL** — 0/4 (measured 0.93–1.07). **The v2 A2000 paired losses did not reproduce**: v2 said 128/4 beats 64/2 by 17–24% on these cells; v3 measured the two configs within noise of each other. The SM-conditional constant's premise was substantially run-context noise on a busy home card. |
| X1b config trade bounded (A2000) | **FAIL** — gpt-oss down paired 0.793 vs the 0.80 floor; median-of-medians < 1.0. Same conclusion: on the 26-SM card, config differences are noise-dominated in harness context. |
| X2 split-K | **FAIL** — A5000: Scout-down paired 1.461 ✓ (≥1.30) and Hunyuan-down 1.177 ✓, but Switch gu 0.655 / Switch dn 1.039 → 1 of 3 blind cells ≥1.15 (needed 2). A2000: Hunyuan-down 0.9497 vs the 0.95 floor (boundary miss; Switch cells 1.28/1.59). Split-K's paired gain is REAL on genuinely-starved moderate cells and harmful on tiny-K cells — the plan splits on a starvation test alone and needed a minimum-work floor too. |
| X3 Scout-down ≥1.0 vs dequant, both devices | **FAIL** — 0.61 (A5000), 0.79 (A2000) median. Against: 0.47 (v2), 1.11/1.12 (pre-stamp exploratory, one of them the SAME A2000). Scout-down vs dequant has now measured {0.47, 0.61, 0.79, 0.84, 1.11, 1.12} across six contexts — the latency-bound T=1 class is not a stable claim in EITHER direction. |
| X4 energy (≥24/26 below dequant + Scout-down < 1.0) | **FAIL on A5000** — 21/26: Switch-Base cells cost **4–7× MORE energy** than the dequant path (5.07/6.76), Scout down 1.47, Arctic down 1.010, gpt-oss down 1.005. A2000 passed (scaled bar). |
| X5 suite 41/41 both devices | **PASS** (35 v1 tests + 6 split-K tests) |
| **V3_CONFIRMED** | **FALSE** |

## What v3 actually established

1. **The kernel's replicated domain is now sharply drawn: bandwidth-bound
   cells.** Census k≥4 shapes ran 1.16–2.73× median vs dequant on BOTH
   devices in v3 — consistent with v1, v2, and both sweeps across five
   device instances and three driver generations (580/570/550). k≥6
   held-out shapes (DeepSeek-V3, granite, Qwen3-Next, Qwen2-57B) hold
   ~1.0–1.8×. Fresh blind Grok-1/Mixtral-8x22B (k=2, large): 1.0–1.24×,
   never slower. Energy below dequant and fidelity above the dequant path on
   ALL of these, every time.
2. **The latency-bound classes do not replicate — in either direction.**
   T=1 cells and small-shape cells swing ±2× between instances (and between
   runs on the same home card) because they are clock/launch-bound, not
   bandwidth-bound. The instance-variance that first appeared as one cursed
   census cell (gpt-oss down: now measured 1.75/1.03/1.00/1.47/0.69 across
   five contexts) is the same phenomenon.
3. **Tiny shapes are simply the wrong path for the fused kernel.**
   Switch-Base-128 (6144×768 / 768×3072, T=1): 0.24–0.35× speed and 4–7×
   the energy of the dequant path on both devices. At ~50–100 µs of total
   work, the single fused launch + fp32 workspace reduce + wrapper floor
   loses to three tiny launches on an already-dequantized 2 MB weight. A
   **minimum-bytes dispatch floor** (fall back to the dequant path below it)
   is the correct product behavior — future isolated change, not applied
   here.
4. **Split-K's paired benefit is real where designed** — Scout-down 1.46×,
   Hunyuan-down 1.18× vs the same kernel unsplit on the A5000 (blind,
   n=3 median) — but it cannot stabilize the T=1 class against the dequant
   baseline (see 2), and the starvation-only trigger wrongly splits tiny-K
   cells (Switch gu on the A5000: 0.655× paired — split overhead on 12
   absmax blocks). The trigger needs a per-split minimum-work floor.
5. **The SM-conditional constant should revert.** Its motivating v2
   measurement did not reproduce; on the A2000 the two configs are
   noise-equivalent, and reverting to the universal 64/2 constant removes a
   device-conditional branch that bought nothing. (Not reverted in this
   tree — the freeze stands; recorded for the next change window.)

## Deviations / incidents (A5000 leg, fully disclosed)

- **First attempt stalled pre-output**: the buffered rep-1 process burned
  ~71 CPU-minutes with the GPU near-idle and produced no cell lines; killed
  UNREAD; cause undetermined (single hot main thread, native bnb lib loaded,
  no inductor activity). The identical relaunch ran normally.
- **Diagnostic probe**: one unbuffered (`-u -X faulthandler`) invocation of
  the frozen harness ran to completion for diagnosis; its numbers were
  OBSERVED (first cells scrolled during monitoring) and then DISCARDED —
  the reduction uses only the three subsequent clean fresh-process reps from
  the frozen runner. Code was never modified.
- Pod `yriqaf9v5z8zqt` (driver **550.127.08** — third driver generation in
  the series), torn down + 404-verified after evidence pull.
- A2000: 4 registered NOT-RUN cells (DeepSeek-V3/Grok-1/Hunyuan/Arctic
  gate_up stacks exceed the 12 GB card), thresholds scaled as stamped.

## Cumulative claim after v1 + v2 + v3 (the README basis)

> On sm_86 at batch-1 decode, for MoE expert shapes in the
> **bandwidth-bound regime** (top_k ≥ 4 census-class, and k ≥ 6 off-census;
> per-call weight traffic ≳ 25 MB), the fused kernel is **faster at median
> (1.2–2.7×), strictly more energy-efficient, and strictly more accurate**
> than the dequantize-then-matmul path — replicated blind across five
> device instances and three driver generations. Outside that regime the
> honest answer is: **top_k ≤ 2 / latency-bound cells are instance-unstable
> in both directions** (measured 0.5–1.5× spread on identical code), and
> **tiny shapes (≲5 M weight elements) lose outright** and should dispatch
> to the dequant path. gpt-oss down (2880×2880) remains the named unstable
> census cell (0.69–1.75 across five instances).

## Evidence (committed under `bench/phase1/confirmatory_v3/`, sha256 in
`reduction_v3.json`'s sidecar `SHA256SUMS`)

Per-cell tables: see `reduction_v3.json` (mechanical output). Headline rows
are quoted in the criteria table above; the JSON carries every rep.

## Queue after v3 (all future isolated changes, each requiring its own prereg)

1. **Minimum-bytes dispatch floor** → route tiny cells to the dequant path
   (kills the Switch-class loss by construction).
2. **Split-K trigger: add a per-split minimum-work floor** (keep the
   Scout/Hunyuan gains, stop splitting 12-block cells).
3. **Revert the SM-conditional constant** to universal 64/2.
4. Claims about latency-bound cells should be made ONLY as paired
   (same-instance) statements, never vs-dequant absolutes — the measurement
   methodology lesson of v3.
