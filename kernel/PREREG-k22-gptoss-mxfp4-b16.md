# K22 — gpt-oss-20b at B=16 on an RTX 5090: where does the step go, and can K21 beat the served expert route on its own routing? (registered 2026-10-01, before any data)

Owner directive (2026-10-01): *"once we land somewhere stable lets push out to other family throughput optimization"*.
The Qwen3 line has landed:
- K20 (#421) gave K19 its plan;
- experts4bit-qlora P88 licensed K19 at B=16 (0.905× step, K8 +0.0062 nats);
- experts4bit-qlora #811 made it the default for batched int4 decode.

gpt-oss-20b is the first other family: its quantized expert store is licensed and has no batched kernel.

**The gap.** experts4bit-qlora serves gpt-oss-20b's native MXFP4 store with `gemv_mxfp4_b32` up to 16 rows. The store
is licensed: KL 0.0019 vs 0.0222 for the NF4 control, `e4b.serve.p44.gptoss.store-r12.kl-vs-bf16.2026-09-19`.
- At B=16 a call routes 64 rows (16 × top-4), over `hot_residency._MXFP4_GEMV_ROWS`, so the call falls back to the kept
  NF4 stacks: the fused tile build, the gather, and `gemm_4bit_grouped_captured`.
- bo7 measured that configuration (`store_r12`) at **21.65 ms per B=16 step** (739 tok/s) on a 5090. Qwen3-30B-A3B,
  with more active parameters, steps in 11–12 ms.
- How much of gpt-oss's step is the expert matmul is **unmeasured**. bo7 timed it end to end only.

**K21** (#422, merged `98d17db`) is K19's grouped tensor-core GEMM on the MXFP4 store: exact bf16 weights, the same
tile table and gather. Correctness is held on the A2000; it has never been timed.

## The lane: one RTX 5090, three phases, driven from experts4bit-qlora `bench/k22/`

1. **Census (descriptive).** gpt-oss-20b at bo7's `store_r12` configuration (`E4B_SERVE_EXP_INT4=1`,
   `E4B_INT4_KEEP_NF4=1`, folds r1/r2), B=16, the graph-window timing and P42's replay census. Reported:
   - the step;
   - per-kernel ms per step;
   - the expert role's share;
   - `_gemm_nf4_grouped`'s ms per step, which feeds the instrument check below.
2. **Routing record.** `bench/families/record_eids.py` (e4b #809) on the served model, over 16 wikitext rows tokenized
   by gpt-oss's own tokenizer through `step_decomp._k8_window`, as P37 did. Prefix 384, then 128 teacher-forced B=16
   decode steps, giving `[128, 24, 16, 4]`.
3. **Kernel microbench** (`kernel/k22_bench.py`) on that recorded routing, at gpt-oss-20b's expert shapes (32 experts;
   gate_up N 5760 K 2880; down N 2880 K 2880), one CUDA graph per decode step.
   - **Arms:**
     - **served**: tile build, gather, NF4 grouped GEMM, the route today;
     - **gemv**: `gemv_mxfp4_b32` at 64 rows, reported;
     - **K21 × 48 plans**: BLOCK_N {32, 64, 128, 256} × KC {32, 64} × warps {4, 8} × stages {2, 3, 4}. K = 2880 admits
       KC ≤ 64;
     - **floor**: MXFP4 distinct-expert bytes over a measured copy.
   - **Selection:** the best plan is chosen on steps 0–7 and every arm is read on steps 8–15, re-timed. A plan more than
     2 % from an fp32 oracle or from the default is not selectable.

## Rule (`k22_bench.py`, 5-case self-test), on the EVAL medians

1. **VOID** unless the served arm's `_gemm_nf4_grouped` time per step (torch.profiler over graph replays) is within
   ±15 % of phase 1's in-model census of the same kernel, on the same box. Otherwise the bench is not timing the model's
   work.
2. **PROMISING** if best K21 / served ≤ 0.77. What follows:
   - an experts4bit-qlora consumer PR routes the MXFP4 store's batched rows to K21, opt-in;
   - an end-to-end lane reads the step and a quality gate. The MXFP4 store's KL instrument applies, since K21's weights
     are exactly the store's;
   - only then a default.
3. **MARGINAL** if ≤ 0.90: the census decides whether experts are worth more work on this family.
4. **NO** otherwise: K21 stays dormant for gpt-oss, and the census names the family's real lever.

## Predictions (written before the data)

- **Census:** the NF4 expert GEMM is the largest single item in the B=16 step but under half of it. gpt-oss's decode
  attention runs the f32 path with sinks (bo7's mech tallies), and that is my guess for the rest.
- **K21: PROMISING, best/served 0.55–0.75.** The NF4 grouped M-tile serves about 2 rows per expert per tile (64 rows
  over at most 32 experts), the regime where K14's int4 M-tile lost 1.92× to the GEMV (P7). K21 is K19's structure,
  which beat the GEMV at that row density on Qwen3, and the MXFP4 bytes are about 10 % fewer than NF4 plus fp32 absmax.
- **gemv** reads slower than served at 64 rows, as bo3n's ×0.81 says.

## Rehearsal (correctness only: the A2000 is never a timing instrument)

`k22_bench.py --self-test` passed (5 cases). `--quick` on SYNTHETIC uniform routing ran end to end on the NAS A2000 with
grouped-nf4-gemm `98d17db`:
- every arm captured and timed;
- the profiler found `_gemm_nf4_grouped` in the served graph;
- K21's plans read 0.0032 from the oracle and bit-identical to each other;
- the instrument check VOIDed against a placeholder census, as it must off the target.

No A2000 time is quoted.

## Budget

- One RTX 5090 (verified/secure), any CPU vendor: no calibration build.
- gpt-oss-20b is 13 GB to fetch, and its NF4 bake took bo7 about 6 minutes.
- **Guard 1.5 h at ≤ $0.75/h (≤ $1.125)**, so a proving rental runs first: install, tripwire, K21 + K16 contracts
  compiled, the self-test, no model. 0.5 h, ≤ $0.375.
- **Hard stop $2.00.** Lane `k22-5090-<n>`.
- Receipts in `kernel/receipts-k22/5090/`; results in `kernel/RESULTS-k22-gptoss-mxfp4-b16.md`.

Amendments, dated, go below this line before any data is read.
