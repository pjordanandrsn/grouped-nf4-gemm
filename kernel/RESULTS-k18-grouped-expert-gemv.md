# Results — lane K18: a grouped split-K int4-b32 expert GEMV (RTX 5090, 2026-09-22)

Pre-registration: [`PREREG-k18-grouped-expert-gemv.md`](PREREG-k18-grouped-expert-gemv.md) (#377, merged before the launch). Kernel: [`int4_b32.py`](int4_b32.py) (`_gemv_int4_b32_grouped`, `gemv_int4_b32_grouped`). Rows: [`receipts-k18/5090/k18_rows.json`](receipts-k18/5090/k18_rows.json); the bench log, all three contract logs, versions, forensics and the teardown proof are beside it (the lane receipt itself lives in the private receipts repository, adertha-receipts `542a4ce`). The verdicts are computed by [`k18_reduce.py`](k18_reduce.py) from the rows, and the tables below are its output, unedited.

Lane `k18-5090-1` (experts4bit-qlora `bench/k18`, #687; e4b `58cf0de3`, gnf4 `8379b092`). Rented 20:35Z, `TP_DONE` 20:50Z, destroyed 20:50:53Z (vast instance 52118802, `teardown-proof.json`). Cost **$0.1269** against the $0.66 estimate. Card: NVIDIA GeForce RTX 5090 (170 SMs, driver 580.119.02, 575 W / 3090 MHz), torch 2.8.0+cu128 / triton 3.4.0. The replay input is experts4bit-qlora P60's recorded Qwen3-30B-A3B B=16 routing (`bench/p60/receipts/eids_b16.int16.bin`, sha256 `c050961e…`, 128 steps × 48 layers × 16 rows × top-8), staged under `staged.sha256` and verified on the box.

Contracts ran on the box before any timing:
- interpreter (`TRITON_INTERPRET=1`), `test_int4_b32_grouped_interp.py`: **21 passed** + 7 compiled-only skips;
- interpreter, `test_int4_b32.py` in its own process: **190 passed**;
- compiled on the 5090, `test_int4_b32_grouped_interp.py`: **28 passed** (the real expert shapes at R=128 under three routing patterns, plus a CUDA-graph replay).

**The instrument reproduces P60 on a different host.** The served arm reads 6.520 ms/step against P60's 6.479, and the dedup arm reads 5.564 against 5.560, so the grouped arm is compared on the ground P60 measured.

## The rows

device NVIDIA GeForce RTX 5090 (170 SMs), torch 2.8.0+cu128, 128 recorded steps x 48 layers, eids sha256 c050961e7f7d..., split-K plan {'gate_up': 16, 'down': 6}

| arm (one CUDA graph per step, 20 replays, median) | step ms median | mean | min | max |
|---|---|---|---|---|
| served `gemv_int4_b32` | 6.520 | 6.537 | 6.078 | 6.927 |
| grouped `gemv_int4_b32_grouped` (mt=4) | 9.689 | 9.686 | 9.193 | 10.078 |
| dedup: one row per distinct expert (P60's ceiling) | 5.564 | 5.569 | 4.806 | 6.143 |

served - grouped = **-3.169 ms/step**; served - dedup = +0.956 ms/step (grouped is SLOWER than served, 1.486x)

| shape | N | K | R | P1 equal | served us | grouped us | grouped/served |
|---|---|---|---|---|---|---|---|
| expert_gate_up | 1536 | 2048 | 8 | True | 10.72 | 14.82 | 1.382 |
| expert_gate_up | 1536 | 2048 | 16 | True | 14.85 | 16.90 | 1.138 |
| expert_down | 2048 | 768 | 8 | True | 9.02 | 8.99 | 0.996 |
| expert_down | 2048 | 768 | 16 | True | 8.88 | 10.88 | 1.225 |
| attn_q | 4096 | 2048 | 8 | True | 16.99 | 27.14 | 1.597 |
| attn_q | 4096 | 2048 | 16 | True | 27.04 | 33.09 | 1.224 |
| attn_kv | 512 | 2048 | 8 | True | 8.93 | 8.99 | 1.007 |
| attn_kv | 512 | 2048 | 16 | True | 8.99 | 10.75 | 1.196 |
| attn_o | 2048 | 4096 | 8 | True | 19.63 | 27.30 | 1.390 |
| attn_o | 2048 | 4096 | 16 | True | 27.39 | 34.83 | 1.272 |
| attn_qkv_fused | 5120 | 2048 | 8 | True | 18.94 | 31.23 | 1.649 |
| attn_qkv_fused | 5120 | 2048 | 16 | True | 31.46 | 39.10 | 1.243 |

- **P1 (bitwise identity):** HELD — replay mismatches 0 of 256 (step x projection) checks; small-R rows equal 12/12
- **P2 (served − grouped ≥ 0.5 ms/step):** REFUTED — -3.169 ms/step
- **P3 (grouped ≤ +3 % of served at R = 8/16):** REFUTED — worst attn_qkv_fused R=8 1.649; over +3 %: expert_gate_up R=8 1.382, expert_gate_up R=16 1.138, expert_down R=16 1.225, attn_q R=8 1.597, attn_q R=16 1.224, attn_kv R=16 1.196, attn_o R=8 1.390, attn_o R=16 1.272, attn_qkv_fused R=8 1.649, attn_qkv_fused R=16 1.243

**Decision rule:** P1 ∧ ¬P2 → recorded as exact-and-not-faster, not shipped as a lever; the headroom P60 found is not reachable by load sharing at MT = 4 (P60's P2, dedup 1.21× above the floor, is the next question). ¬P3 as well — moot: routing by R applies only to a kernel that is a lever, and this one is not.

## What the rows establish

**The kernel is exact and slower.** It returns the served GEMV's bits on every recorded step and every small-R row, and it is slower than the served GEMV on its target. On the recorded B=16 routing a step takes 9.689 ms against 6.520 (1.49×), 3.17 ms *worse* where the bar was 0.5 ms *better*. At small R it is 1.00–1.65× the served call.

**The loss exists where there is nothing to share.** In the R=8 rows, the ids cycle over 8 experts, so every row is a different expert, every tile holds one row, and no weight load can be shared. The grouped kernel still costs up to 1.65× the served call (attn_qkv_fused 18.94 → 31.23 µs; attn_q 16.99 → 27.14). So the loss is overhead the kernel adds, not a failure of the sharing it was built for. Where the served call sits near its floor (~9 µs: expert_down, attn_kv at R=8), the ratio is 1.00. Where the call does real work, it is 1.38–1.65×.

Compared with the served GEMV, the kernel adds:
- the per-program tiling over the call's R ids (histogram, cumsum, rank match);
- an `[MT, BLOCK_N]` accumulator that every row's contribution is selected into;
- per-row index and validity extracted by reductions inside the K loop.

That last item is work repeated at every K step. This lane did not profile which of the three costs what, so this list is what the design adds, not an attribution.

**At R=128 the same arithmetic runs on fewer programs.** The served grid gives each of its 128 row slots one row. The grouped grid launches the same 128 slots, but only one slot per tile does work: roughly one per distinct expert (P60's recording averages 54.7 distinct experts per call, 48.8–70.2 by layer), plus the spill for experts with more than four rows. Each working slot serves its rows one after another. That is a structural fact of the design. Its share of the 3.17 ms was not measured separately.

## A correction to P60's reading (and to the premise of this lane)

experts4bit-qlora `bench/p60/RESULTS-p60.md`, this lane's pre-registration, and the milestone reports described P60's 0.92 ms/step dedup gap as *the cost of re-streaming each expert's weight slice once per routed row*. The evidence does not support that reading:
- P60's dedup arm removes three things at once: the repeated rows' weight loads, **their arithmetic**, and their programs.
- P60's sorted arm already showed locality is not the constraint: putting each expert's rows next to each other changed nothing (−0.55 %), because L2 already serves the repeats.
- K18 shares the loads, keeps every row's arithmetic, and is slower.

So what the 0.92 ms measures is the cost of computing ~73 more rows per call (128 routed against 54.7 distinct on average). Reloading is at most part of that, and nothing measured so far separates it out. Those rows are not optional work: each routed row is a different token's activation against the expert, so dedup is a ceiling on what fewer loads could buy, not a computation the model could run. The re-streaming phrasing is withdrawn wherever it appears (RESULTS-p60 carries a dated correction).

## Decision (as registered)

**P1 ∧ ¬P2 → recorded as exact-and-not-faster, and not shipped as a lever.** No consumer lane is opened and P4 is not run. P3 is refuted too, but that is moot: routing by R applies only to a kernel that is a lever.

`gemv_int4_b32_grouped` stays in the tree dormant, as evidence, following the precedent of the refuted split-K decode GEMV (`gnf4.retired.splitk-gemv`). Nothing in this package or the consumer calls it, and its capability entry says it is slower. Removing it is a one-commit follow-up if the surface is not worth its tests.

**Next question:** what bounds the served GEMV at R=128, whether each row's int32 dot on CUDA cores or each expert's load. A served-only sweep over rows per expert at a fixed number of distinct experts would separate the two, on the same recorded routing and at the same cost as this lane. The registered pointer (P60's P2: dedup is 1.21× above the bandwidth floor) stays open.
