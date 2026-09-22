# K18 — a grouped split-K int4-b32 expert GEMV: each expert's weight slice read once per program for up to 4 of its rows (registered 2026-09-22, before the 5090 read)

Owner directive (Jordan, 2026-09-22): *"go on"* — the throughput list. Licensed by experts4bit-qlora **P60** (`bench/p60/RESULTS-p60.md`, #686): replaying Qwen3-30B-A3B's **recorded** B=16 routing through the shipped `gemv_int4_b32` on an RTX 5090 reproduced the served kernel row (6.155 vs the census's 6.340 ms/step), and **one row per distinct expert instead of one per routed row runs 0.92 ms/step faster** (6.479 → 5.560) — the cost of re-streaming each expert's slice once per row. Row order is worth nothing (−0.55 %: L2 already serves repeats), so the lever is to read each expert once for all its rows, inside the kernel.

## Prior art, and why this is not it

gnf4 already has a grouped int4-b32 kernel — `_gemm_int4_b32_grouped` (K14/#301): int8 tensor-core MMA over prebuilt expert tiles of `BLOCK_M = 16` rows, no split-K. experts4bit-qlora's P7 measured it **1.92× / 1.28× slower** than this split-K GEMV at B=16 decode: with ~1–2 rows per expert, the 16-row MMA tile wastes ~90 % of its lanes, and without split-K there are too few programs. That is why B=16 decode serves through the per-row GEMV today (`hot_residency._int4_gemv_decode`). K18 keeps everything the GEMV won with and changes one thing:

- **The served GEMV's grid** `(cdiv(N, BLOCK_N), R, SK)` and plan (`_plan`, gnf4 65cb104: split-K 16 for `gate_up`, 6 for `down` on the 5090).
- **The served GEMV's arithmetic, row by row:** exact int32 block dots on CUDA cores, the same fp32 scale products summed over the same `KU` axis in the same k order — only the rows that exist are computed (no MMA padding).
- **What changes:** program `(pid, t, sk)` serves tile `t` of an expert-major tiling (up to `MT = 4` rows of one expert), loading the expert's weight slice and scales **once per K step for all its rows**. The tiling is derived in-register from the call's `R` expert ids (`tl.histogram`, `tl.cumsum`, per-expert ranks over `R ≤ 128` ids): no extra launch, no sort, no host sync, legal under capture. Slots beyond the tile count exit before any load. Each row's fp32 partial is stored at its **original** index, so the served `_reduce_partials` runs unchanged.

## Predictions

- **P1 — bitwise identity (hard gate).** `gemv_int4_b32_grouped(...)` is `torch.equal` to `gemv_int4_b32(..., fused_reduce=False)` on every case: one row, all distinct, one expert over many tiles, skewed routing, R not a power of two, N not a BLOCK_N multiple, SK > 1, `mt ∈ {1, 2, 4, 8}` (`kernel/test_int4_b32_grouped_interp.py`, under the interpreter in CI and compiled on the 5090), plus Qwen3-30B-A3B's expert shapes at R = 128 under three routing patterns and a CUDA-graph capture/replay. *Refuted by* a single differing element — the kernel is refused.
- **P2 — the step saving on recorded routing (the 5090 read).** Replaying P60's 128 recorded steps (`experts4bit-qlora bench/p60/receipts/eids_b16.int16.bin`) with P60's replay (served / grouped / dedup, one CUDA graph per step): `served − grouped ≥ 0.5 ms/step` (band **0.5–0.92**, the ceiling being P60's dedup arm). *Refuted by* a saving under 0.2 ms, or grouped slower than served.
- **P3 — no harm at small R.** At R = 8 (B=1 decode, top-8) and R = 16 on the six K17 shapes, grouped is within +3 % of served (or faster). *Refuted by* > +5 % on any shape → the consumer would route grouped by R.
- **P4 (consumer, a later lane):** with the grouped GEMV in the B=16 serving path, P54's harness step falls ≥ 0.4 ms at B=16 with tokens identical to the control (P1 in the wild). Registered here so it is not forgotten; read in its own lane.

## Decision rule

**¬P1 → refused.** **P1 ∧ P2 → the grouped GEMV ships opt-in in the next gnf4 release, and a consumer lane (experts4bit-qlora) wires it into the B=16 int4 decode branch and reads P4 before any default moves.** P1 ∧ ¬P2 → the kernel is recorded as exact-and-not-faster and not shipped as a lever; the headroom P60 found is then not reachable by load sharing at MT = 4 (P60's P2, dedup 1.21× above the floor, is the next question). ¬P3 → routed by R in the consumer, never a blanket switch.

## Instrument, dry-run before registration

The tile table was checked exact against a Python reference under the interpreter and compiled on the QNAP RTX A2000 (42 tiles from 128 skewed rows). The bitwise suite runs under the interpreter (CPU) and compiled on the A2000 before any rental; the 5090 lane re-runs both contracts on the box before timing (K17's order: rc 21/22 if either fails, no number produced). A2000 timings are never quoted.

## Budget

One RTX 5090 (verified/secure), ≤ 1.0 h guard, estimate ≤ $0.66 (install ~5 min, contracts ~8, replay ~5); no model download — the replay needs only gnf4 and the committed ids. Lane `k18-5090-1`, driven from experts4bit-qlora `bench/k18/` (K17's pattern). Hard stop $2. Receipts in `kernel/receipts-k18/5090/`, results in `kernel/RESULTS-k18-grouped-expert-gemv.md`.

Amendments, dated, go below this line before any data is read.
