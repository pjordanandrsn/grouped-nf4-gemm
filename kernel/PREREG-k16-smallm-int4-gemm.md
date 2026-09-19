# Pre-registration — lane K16: a Marlin-class small-M int4-b32 GEMM for the attention projections (registered 2026-09-18, before any kernel code)

Work item: adertha-agents#110 (throughput + training campaign). Lineage: P42 (experts4bit-qlora `bench/p42/`) attributed the Qwen3-30B-A3B B=16 decode step; experts4bit-qlora#564 decomposed the 4.05 ms gap to vLLM and named the **attention projections** as the gap (the expert GEMV is near its roofline); **K14** (`RESULTS-k14-smallm-int4-gemm.md`) REFUTED the premise that a shipped int4 arm beats dequant-then-GEMM at M=16 (best swept `gemm` arm 1.12–2.00× SLOWER than bf16; the int4 GEMV 1.46–2.40× slower); **K15** (`RESULTS-k15-marlin-comparator.md`) measured the target: vLLM 0.28.0's Marlin GPTQ kernel runs `q_proj` in 6.37 µs and `o_proj` in 8.28 µs at M=16 on the same RTX 5090 where this package's bf16 dequant path takes 10.3 / 16.5 µs, and at matched bytes (g32) still wins (8.18 / 8.24) — **the advantage is the kernel, not the format**. Substituting it where measured is worth 0.58–0.81 ms/step of the gap. This lane builds that kernel on our own int4-b32 format.

## Why the shipped M-tile kernel loses, read from its code (not a guess)

`_gemm_int4_b32_grouped` does exact integer MMA: int8 activations against unpacked int4 nibbles with **one `tl.dot` per 32-wide k-block**, because both scale grids (`scales [E, N, K//32]`, `as [R, K//32]`) are per 32-block and the fp32 scale product must be applied per (row, k-block) AFTER each dot. At M=16 that is a `[16, 32] × [32, BLOCK_N]` dot 64 times per program for K=2048, with the program count `1 × cdiv(N, BLOCK_N)` — a single projection is ONE M-tile and gets none of the expert-count parallelism the kernel was swept for. K14 measured it at 378 GB/s on `q_proj`, **22 % of the 1528 GB/s streaming ceiling**, while the bf16 path it must beat sits at 106 %.

## Design (what K16 builds)

`gemm_int4_b32_smallm(x_bf16 [M, K], packed [N, K//2], scales [N, K//32]) -> [M, N] bf16`, `M <= 16`, one expert (the attention-projection shape; `Int4Linear`'s store), **one launch**:

1. **In-register dequantisation, bf16 tensor-core MMA** (Marlin's arithmetic): each program loads a `[BLOCK_N, KC]` int4 tile as bytes, unpacks nibbles, multiplies by the per-32-block scale *inside the tile* (`KC/32` scale columns broadcast over their 32 k), and runs `tl.dot(a_bf16 [16, KC], w_bf16^T)` with **KC = 128 or 256** — four to eight k-blocks per dot instead of one. Activations stay bf16 (no int8 quantisation; `quant_x_rows` is not on this path). Numerics = the dequant-then-GEMM path's, without materialising the bf16 weight.
2. **Split-K across programs with a fused serial reduction**: grid `(cdiv(N, BLOCK_N), SK)`; each program accumulates fp32 over its K-span, writes its partial, increments a per-N-block counter with `tl.atomic_add`; the program that observes `SK-1` sums the partials, casts to bf16, stores the output tile and resets the counter (stream-K style "last-arriver reduces"). ONE launch — the separate `reduce_partials` launch costs the 3.1–3.6 µs floor, half the target budget, and is why the GEMV's split-K cannot be the answer here.
3. **Config space registered for the sweep**: `BLOCK_N ∈ {32, 64, 128}`, `KC ∈ {128, 256}`, `SK ∈ {1, 2, 4, 8}`, `num_warps ∈ {4, 8}`, `num_stages ∈ {2, 3}` — under **K14's instrument** (graph-replay medians, the launch floor and the measured streaming ceiling on the same box, `kernel/k14_bench.py` gains a `k16` arm), never eager.

Out of scope: the expert tier (52 % of the step, at 80–100 % of its roofline, #564); the grouped multi-expert path (K14's kernel stays for `T > 1` expert tiles); adopting Marlin (a GPTQ repack + dependency + gate; K15).

## Registered predictions (falsifiable; competitor figures biased in Marlin's favour per the standing correction)

- **P1** — on the RTX 5090 at M=16, the best swept K16 config runs `q_proj` (N=4096, K=2048) and `o_proj` (N=2048, K=4096) within **1.4×** of Marlin g128: ≤ 8.9 µs and ≤ 11.6 µs. **Refuted** if either exceeds it.
- **P2** — K16 beats this package's bf16 dequant path on both: < 10.3 µs and < 16.5 µs (the thing K14 found no int4 arm could do).
- **P3** — `k_proj` / `v_proj` (N=512, K=2048) are launch-bound (K14: 1.74× the floor); K16 does not lose to bf16 by more than **1.2×** there (K14's arms lost 1.66–2.00×). Refuted if it does.
- **P4** — the per-32-block in-tile scaling costs nothing measurable against a single-scale control (< 5 % on `q_proj`); **refuted** → the format is the bottleneck, not the kernel, and this lane's conclusion inverts K15's.
- **P5 (model level, the phase-rule requirement)** — routed into `Int4Linear` for `1 < R <= 16` (experts4bit-qlora side, a separate PR; drops the cached bf16 copy #561 holds), P42's census on the same box shows the attention GEMM row falling by **≥ 0.4 ms/step** at B=16 (of the 2.4 ms row; K15's 0.58–0.81 upper bound scaled by P1's 1.4×). Refuted if < 0.4.

## Correctness contract (before any perf number)

- **Deterministic for a fixed config** (same inputs → same bits; the last-arriving program sums the fp32 partials in split order, never in arrival order) and **within one bf16 ulp across `SK`** (fp32 summation order differs between SK=1 and a split, so bit-equality across SK is not the contract) — pinned by a test that runs `SK ∈ {1,2,4,8}` on the same inputs.
- Within bf16 output rounding of the reference `x_bf16 @ dequant_int4_ref(packed, scales).T` on every registered shape and on the non-power-of-two K checkout shapes (K=64, 96) — the same reference K14 used, on CPU.
- Under `TRITON_INTERPRET=1` (the repository's interpreter CI job) for correctness on CPU; on an sm_86 A2000 for the first perf read; the registered numbers only from the 5090 lane.

## Decision rule

P1 ∧ P2 → open the consumer PR (P5's route) and register the kernel-level claim as `measured`; P2 ∧ ¬P1 → ship it as the `1 < R <= 16` route anyway (it beats what runs today) and record the distance to Marlin as the open item; ¬P2 → **refuse the kernel**, record the sweep, and the attention-tier lever goes back to the format/adoption question (K15's "adoption is not established").

## Budget and STOP rules

Development on CPU (interpreter) and the A2000 costs nothing. One 5090 lane for the registered sweep + K14's arms re-run beside it: `k16-5090`, ceiling $0.65/h, guard 1 h, estimate ≤ $0.65; a proving run is not required (guard < 1 h). STOP: a box not of the class; the K14 bench's floor/ceiling instrument failing to reproduce K14's bf16 numbers within 10 % (then the box, not the kernel, is being measured).

## Receipts

`kernel/receipts-k16/` (rows, log, versions-forensics); `kernel/RESULTS-k16-smallm-int4-gemm.md` quoting only the rows. Amendments dated below, before the data they touch.

## Amendments

(none yet)
