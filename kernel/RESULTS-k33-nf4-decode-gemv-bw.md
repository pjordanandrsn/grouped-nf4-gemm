# K33 — results: **LEVER**. The bandwidth-targeted NF4 decode GEMV (`GNF4_GEMV_BW=1`, `prmt32`) runs Qwen3-30B-A3B's single-row expert projections at 0.341× the served route on an RTX 5090 (2.818 → 0.961 ms over 48 layers), within the tolerance contract

Registration: `kernel/PREREG-k33-nf4-decode-gemv-bw.md` (#501, `5a60c37`). The kernel: #500 (`b947c17`). Tracking
issue: experts4bit-qlora#1313. Runner: experts4bit-qlora `bench/k33/` (#1316, with #1317's host-floor fix, `598c0013`).

Code under test: grouped-nf4-gemm at `5a60c37` (0.42.0 with #500 and #501), torch 2.8.0+cu128, triton 3.4.0.

**Verdict by `k33_bench.verdict`: `LEVER`.** The rule's steps, in order:

| step | result |
|---|---|
| VOID | no. The card is an RTX 5090. The copy floor measured 1,515 GB/s. Every arm timed on every Qwen3 projection. All 12 plans compiled and none was refused. Every capture's dispatch tally was its registered route. |
| NOISY | no. `incumbent2 / incumbent` = 0.9999 (gate_up) and 1.0004 (down) |
| FUNCTION_FAIL | no. At every family's selected plan, `bw_prmt32` is bitwise `bw_tree`, and both meet the tolerance contract. Their error against `dequant_ref` equals the scalar GEMV's on all six projections. |
| LEVER | **yes**, with bw = `bw_prmt32`: bw / incumbent **0.332** (gate_up) and **0.357** (down), each ≤ 0.77; **0.341** on the pair, ≤ 0.667; floor / bw **0.74** and **0.63**, each ≥ 0.45 |

## The reading (`k33-5090-1`)

**Host:** one RTX 5090 (sm_120, 170 SMs, driver 595.91.07, 600 W, 3210 MHz) on an AMD Ryzen Threadripper PRO 7965WX
(24 cores, 251 GiB RAM). It was Vast instance 54693772 on machine 151831. **Cost:** $0.037. Teardown was proven at
18:41:57Z.

**Timeline (the box's own log):** install at 18:40:52Z. The tripwire passed, then the rule's self-test (14 cases), then
the premise: `kernel/test_nf4_gemv_bw.py` compiled on the card, **27 passed**, none skipped, in 15.0 s. That covered
`prmt32` bitwise the tree at six plans, the exhaustive one-hot readback, the tolerance contract, the PTX and PDL. The
PTX check read 128-bit loads with no spills (56 registers for `prmt32`, 39 for the tree). The bench started at
18:41:18Z and took 19.9 s. `TP_DONE` landed at 18:41:39Z.

**The work:** one CUDA graph per arm per projection, holding every layer's decode launch at the family's served shape:
8 activation rows, the top-8 routed experts of that layer, 16 synthetic NF4 experts per layer. Medians of 200 rounds,
order reversed each round. Plans were selected on 40 rounds, then timed afresh.

| projection | incumbent | `bw_prmt32` | `bw_tree` | int4-b32 | floor | bw / incumbent | floor / bw | selected plan |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen3 gate_up (48 layers) | 1.8261 | **0.6068** | 0.7886 | 0.6579 | 0.4486 | **0.332** | 0.739 | BN 16, KC 1024, 4 warps, SK 1 |
| Qwen3 down | 0.9921 | **0.3540** | 0.3719 | 0.3808 | 0.2243 | **0.357** | 0.634 | BN 16, KC 256, 4 warps, SK 1 |
| Granite gate_up (32 layers) | 1.0992 | **0.2452** | 0.2848 | 0.2740 | 0.1495 | 0.223 | 0.610 | BN 16, KC 512, 8 warps, SK 1 |
| Granite down | 0.5443 | **0.1401** | 0.1546 | 0.1707 | 0.0748 | 0.257 | 0.534 | BN 16, KC 256, 8 warps, SK 1 |
| OLMoE gate_up (16 layers) | 1.0807 | **0.2436** | 0.3249 | 0.2678 | 0.1994 | 0.225 | 0.818 | BN 16, KC 1024, 4 warps, SK 1 |
| OLMoE down | 0.5373 | **0.1473** | 0.1562 | 0.1519 | 0.0997 | 0.274 | 0.677 | BN 16, KC 256, 4 warps, SK 1 |

All times are ms per graph replay. The incumbent is today's route: dot-pad at Qwen3's shapes on this card, and the
scalar GEMV elsewhere. On Granite and OLMoE the incumbent is the scalar GEMV itself, so their ratios are also bw / scalar.
On Qwen3 the scalar GEMV reads 1.762 and 1.923 ms.

- **Qwen3, the pair:** 2.818 → 0.961 ms, **×0.341**, about 2.9× faster. That is 1.86 ms of kernel time per 48 layers.
  The quartile bands are narrow (within ±0.3 % of the median on every arm) and do not overlap.
- **Per launch at Qwen3's shapes:** `bw_prmt32` takes 12.6 µs (gate_up) and 7.4 µs (down), against dot-pad's 38.0 and
  20.7 µs.
- **The decodes:** `prmt32` beats the tree at every projection, by 0.75–0.95×. The gap is widest where K is largest
  (gate_up).
- **The comparator:** the int4-b32 GEMV reads the same bytes per parameter, and `bw_prmt32` beats it at every projection
  (int4 / bw 1.03–1.22).
- **PDL** (`bw_pdl`, reported): 0.98–1.03× on four projections, but 1.17× (slower) on Qwen3's and OLMoE's down.

## Against the predictions

| prediction (written before the data) | result |
|---|---|
| Q1: numerics and engagement hold everywhere; every plan compiles | **yes** |
| Q2: per launch at Qwen3's shapes, `bw_prmt32` 13–17 µs (gate_up) and 7–9 µs (down); `bw_tree` 19–25 and 10–13 | **partly.** `prmt32` down 7.4 µs held. `prmt32` gate_up 12.6 µs and the tree's 16.4 and 7.7 µs were faster than their bands. |
| Q3: Qwen3 ratios gate_up 0.48–0.63, down 0.36–0.48, pair 0.43–0.57; floor / bw 0.50–0.70 | **missed, faster:** 0.332, 0.357 and 0.341; floor / bw 0.74 (above its band) and 0.63 (held) |
| Q4: `bw_prmt32` is the faster decode at every projection | **yes** |
| Q5: the selected plans have split-K 1 and BLOCK_N 16 | **yes**, on all six |
| Q6: Granite and OLMoE bw / scalar ≤ 0.50 at every projection | **yes** (0.22–0.27) |
| Q7: `int4 / bw_prmt32` in [0.75, 1.0] | **no.** 1.03–1.22: the NF4 kernel is faster than int4-b32 at every projection |
| Q8: `bw_pdl / bw_prmt32` in [0.97, 1.0] | **no** on three of six (1.026, 1.174, 1.171) |
| Q9: instrument within [0.99, 1.01] | **yes** (0.9999 / 1.0004) |
| Q10: the verdict is LEVER | **yes** |

The misses on Q2 and Q3 are in the kernel's favour, so the instruction-issue estimate behind them was pessimistic. Q7
says the arithmetic int4 decode is not the faster route at this shape on this card, so the PREREG's comparator
expectation was wrong. Q8's slowdown on Qwen3's and OLMoE's down projections is reported, not ruled, and not explained
here: Granite's down projection, with the shortest K, did not slow. PDL's served value is read where the step is served,
as P113 read it for int4.

## The registered consequence (LEVER)

- **experts4bit-qlora registers the served lane P116.** It runs the default NF4 `serve_paged` server with
  `GNF4_GEMV_BW=0` against `1`, at W1 and W16, with decode-only slopes. Quality is read under P110's teacher-forced bar,
  because tokens are not expected to match dot-pad's bf16 MMA.
- **`GNF4_GEMV_BW` stays opt-in here** until the served lane reads. On its DEFAULT read, this repository fills
  `_BW_SHAPES`, makes `auto` the default with the `test_m3_defaults` trio and cuts a release, and the consumer floors on
  that release.
- **The register** row `gnf4.kernel.k33-nf4-decode-gemv-bw.5090.2026-10-07` carries the Qwen3 pair's 0.341. It is a
  microbenchmark, not a served speed.

**What it suggests, not what it shows.** SV2's census put the served NF4 expert GEMV at 2.469 ms of a graphed Qwen3 B=1
step, 96 launches. This bench's pair is 2.818 ms for those 96 launches with stores beyond L2. At K6b's measured transfer
from kernel to step, about 0.55, 1.86 ms of kernel time would be about 1 ms of a served step: roughly ×0.89 at W1.
Granite and OLMoE run the scalar GEMV at B=1 today, so their served upside is larger in proportion.

## What it took

| run | status | cost | note |
|---|---|---:|---|
| `k33-5090-1` | **OK, LEVER** | $0.037 | the reading; no proving rental (0.5 h guard) |

**The lane cost $0.037**, inside its $0.75 ceiling.

**Receipts** are in `receipts-k33/5090/`, with `SHA256SUMS`: `k33.json`, `summary.txt`, `forensics.txt`, `versions.txt`,
`logs/bw_contract.log`, `logs/k33_bench.log` and the teardown proof. The launcher's receipt and ledger row are in the
receipt store (adertha-receipts `ffe99352`).

The A2000 rehearsal recorded in the pre-registration is not comparable to these numbers. Under experts4bit-qlora #1133
its timings are not speed evidence, and its synthetic stores come from a device-dependent generator stream.
