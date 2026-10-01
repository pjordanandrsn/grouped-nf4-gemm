# K20 — can K19's plan space carry the RTX 5090's B=16 expert matmul? (registered 2026-10-01, before the 5090 read)

Owner directive (2026-09-30): push ahead on throughput. Licensed by experts4bit-qlora **P87**
(`bench/p87/RESULTS-p87.md`, #806). Its registered verdict was VOID: the calibrated K8 build ran out of its alarm on a
slow host. Its speed arms passed every check of their own and answered the speed question.

On an RTX 5090 at B=16, K19 (`gemm_int4_b32_grouped_smallm`, #419, at its one shipped plan BLOCK_N 64, KC 128, 4 warps,
2 stages) took **6.50 ms per step** against the served int8 GEMV's **7.00**: 1.08× kernel for kernel. The step moved
0.971×. Against the int4-b32 byte floor at the measured routing (4.89 ms), K19 runs at about 75 % and the GEMV at
about 70 %. vLLM's Marlin MoE runs at about 94 % of its own floor (4.78 ms; experts4bit-qlora P86).

K19's plan was chosen on the A2000 and never tried on sm_120. Its loads are small per program (a 64 × 128 int4 slice
per K step) and only two stages deep, for a card with about 1.5 TB/s to hide latency behind. Before a dequant rewrite
or an end-to-end lane, measure the plan space on the target card. Writing no new kernel is the cheapest experiment that
could make K19 worth carrying, or retire it.

## Instrument (`kernel/k20_bench.py`)

- **Routing:** experts4bit-qlora P60's recorded B=16 ids (`bench/p60/receipts/eids_b16.int16.bin`: 128 steps × 48
  layers × 16 rows × top-8), digest-checked. Steps 0–15 are used: 0–7 SELECT, 8–15 EVAL.
- **Weights:** synthetic int4-b32 stores for 128 experts at Qwen3-30B-A3B's expert shapes (gate_up N1536 K2048, down
  N2048 K768). Bytes and L2 behaviour depend on which experts a call touches, not on the values.
- **Arms, one CUDA graph per decode step** (48 layers, gate_up then down), median of 20 replays:
  - **served:** `quant_x_rows` + `gemv_int4_b32` (its own reduce) per call, the route P86 and P87 censused;
  - **k19@plan:** one `build_group_tiles_fused(ids, 128, 16)` per layer, then K19 gate_up (gather) and K19 down at
    that plan;
  - **tiles:** the per-layer tile build alone;
  - **floor:** no kernel. The step's distinct-expert bytes / this box's measured device-to-device bandwidth (a 512 MB
    copy, read + write).
- **The plan grid:** BLOCK_N ∈ {32, 64, 128, 256} × KC ∈ {64, 128, 256} × warps ∈ {4, 8} × stages ∈ {2, 3, 4}: 72 plans.
  - A plan that fails to compile or launch is recorded and skipped.
  - **A plan whose output on step 0, layer 0 is further than 2 % (max abs, relative) from an fp32 dequant oracle or from
    the default plan is not selectable.** A plan changes the fp32 accumulation order across K chunks, so this checks
    closeness, not bit-equality.
- **No winner's curse.** The best plan is chosen on the SELECT steps' median. Every number the rule reads is the EVAL
  steps' median, re-timed.

## Rule (in `k20_bench.py`, 5-case self-test), on the EVAL medians

1. **VOID** if the instrument does not reproduce P87's in-model census within ±15 %:
   - served against 7.729 ms per step (GEMV 7.000 + reduce 0.384 + quantise 0.345);
   - the default plan's k19 arm against 6.993 (K19 6.495 + tile build 0.498).
2. **PROMISING** if best / served ≤ 0.77 (K19 at ≥ 1.3× the served route).
   - A gnf4 PR makes that plan K19's default (sm_120-gated if it loses elsewhere).
   - Then an end-to-end lane in experts4bit-qlora (P88: P87's speed arms, plus the quality arms on a host with a CPU
     floor or a reused pack).
3. **MARGINAL** if 0.77 < best / served ≤ 0.90. Plans alone cannot carry it. The next step is a K19 dequant variant
   (int4 → bf16 by bit construction, scale applied to the per-32-block partial instead of per weight), measured the same
   way before any end-to-end lane.
4. **NO** if best / served > 0.90. K19 is retired for B=16 on the 5090: kept opt-in, recorded measured-not-faster. The
   next lever is the grouping glue (tile build + gather/scatter, about 0.8 ms per step in P87) or a Marlin-layout kernel.

Reported beside the verdict, every one of them:
- each selectable plan's SELECT median;
- the best plan's and the default's efficiency against the floor;
- the tile build's time;
- the plan's numerics.

## Predictions (written before the data)

- **The instrument reproduces P87:** served about 7.7 ms per step, the default k19 arm about 7.0.
- **Some plan beats the default** by 10–25 %. My guess is wider BLOCK_N and 3–4 stages, since loads per program are the
  suspect.
- **The verdict: MARGINAL** (best / served 0.80–0.90). Wider tiles and deeper pipelining recover part of the gap to
  Marlin, but the per-weight fp32 dequant and scale is ALU work Marlin does not pay. Stated so the dequant variant's
  case is on record before the data.

## Rehearsal (correctness only: the A2000 is never a timing instrument)

On the NAS RTX A2000, gnf4 `3351c9d`, a throwaway `pytorch:2.8.0-cuda12.8` container:
- `k20_bench.py --self-test` passed (5 cases);
- `--quick` (4 plans, steps 0–3) ran end to end: graphs captured, plans selected on SELECT and re-timed on EVAL, every
  plan within 0.0033 of the oracle;
- the instrument check fired VOID, as it must, off the target card.

No A2000 time is quoted anywhere.

## Budget

- One RTX 5090 (verified/secure), **0.75 h guard at ≤ $0.75/h (≤ $0.56)**. No model download: the bench needs only
  gnf4 and the committed ids. The guard is under 1 h, so no proving rental is required.
- **Runtime estimate:** install about 3 min; K19's compiled contract tests on the card first (no timing if they fail,
  K17's order); 72 plans × 8 SELECT steps; 4 arms × 8 EVAL steps. About 15–20 min of GPU.
- **Hard stop $1.50.** Lane `k20-5090-<n>`, driven from experts4bit-qlora `bench/k20/` (K18's pattern).
- Receipts in `kernel/receipts-k20/5090/`; results in `kernel/RESULTS-k20-k19-plan-sweep-5090.md`.

Amendments, dated, go below this line before any data is read.
