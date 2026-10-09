# K34 — results: **NONE**. No K16 plan beats the shipped one on both of Qwen3-30B-A3B's attention projections at a tile; the shipped plan stays. One cell moves: the `o` projection's 64-row tile runs at 0.794× with BLOCK_N 32 and KC 256, a grid-fill effect worth about 1 % of the served step (RTX 5090, exploratory)

Registration: `kernel/PREREG-k34-k16-wide-plan-census.md` (#525, `e21a712`). Tracking issue: experts4bit-qlora#846.
Runner: experts4bit-qlora `bench/k34/` (#1444; launched from the v0.51.0 tag `058f98eb`, where it is byte-identical).

Code under test: grouped-nf4-gemm at `e21a712` (0.44.0 with #524 and #525), torch 2.8.0+cu128, triton 3.4.0.

**Verdict by `k34_bench.verdict`: `NONE`.** The census is exploratory: whatever it found, it licenses nothing. The rule's
steps, in order:

| step | result |
|---|---|
| VOID | no. The card is an RTX 5090. The copy floor measured 1,513 GB/s. Every arm timed at every (projection, tile). All 48 plans compiled, and none was refused on numerics. |
| NOISY | no. `shipped2 / shipped` = 0.9992 (qkv/32), 1.0000 (qkv/64), 0.9979 (o/32), 0.9995 (o/64), each inside [0.98, 1.02] |
| CANDIDATE | no. It needs `best / shipped` ≤ 0.90 on **both** projections of a tile. At 64 rows `o` reads 0.794 but `qkv` 1.005; at 32 rows 0.984 and 1.001 |
| NONE | **yes**: the shipped plan (BLOCK_N 64, KC 128, split-K 4, 4 warps, 2 stages) stays at both tiles |

## The reading (`k34-5090-2`)

**Host:** one RTX 5090 (sm_120, 170 SMs, driver 595.91.07, 575 W, 3105 MHz) on an AMD Ryzen 9 9950X (16 cores, 123 GiB
RAM). It was Vast instance 55038608 on machine 153193. **Cost:** $0.074. Teardown was proven at 14:44:23Z.

**Timeline (the box's own log):** install at 14:42:08Z. The tripwire passed: the installed commit carries #522's 32- and
64-row tiles, and `k34_bench.SHIPPED` is the plan experts4bit-qlora's `Int4Linear` serves (`plan_smallm` at both shapes
with the signature's warps and stages). Then came the rule's self-test (12 cases) and the premise:
`kernel/test_int4_smallm_interp.py` compiled on the card, **25 passed**, none skipped, in 7.8 s. The bench started at
14:42:29Z and took 72 s. `TP_DONE` landed at 14:43:41Z.

**The work:**
- Each graph holds 48 launches of `gemm_int4_b32_smallm`, one per layer, each layer on its own synthetic int4-b32 store,
  so the stores stream from memory.
- The 48 plans were each timed for 40 SELECT rounds; the fastest is `best`.
- The four arms were then timed afresh: 200 rounds, order reversed each round.

All times are ms per graph replay (48 launches), re-timed medians:

| projection / tile | shipped | shipped2 | best | cuBLAS bf16 | floor | best / shipped | best plan (BN, KC, SK, warps, stages) |
|---|---:|---:|---:|---:|---:|---:|---|
| qkv (5120 × 2048) / 32 | 0.3808 | 0.3805 | 0.3813 | 0.7349 | 0.1871 | 1.001 | 64, 128, 4, 4, 2 (the shipped plan) |
| qkv / 64 | 0.6661 | 0.6661 | 0.6692 | 0.8054 | 0.1871 | 1.005 | 64, 128, 4, 4, 2 (the shipped plan) |
| o (2048 × 4096) / 32 | 0.4315 | 0.4306 | 0.4244 | 0.6430 | 0.1497 | 0.984 | 32, 128, 8, 8, 2 |
| o / 64 | 0.7293 | 0.7289 | **0.5791** | 1.0753 | 0.1497 | **0.794** | 32, 256, 4, 4, 2 |

- **The shipped plan is the fastest of 48 at `qkv`, at both tiles.** It ranked first in SELECT at both. The re-timed
  "best" is the same plan in a second graph, so `best / shipped` there is the instrument's spread.
- **At `o`, 64 rows, BLOCK_N 32 with KC 256 runs at 0.794×**, 0.150 ms less per 48 layers. At 32 rows the best plan is
  within 2 %.
- **Against cuBLAS** on the cached bf16 copy, the shipped plan costs 0.52 / 0.83 (qkv, 32 / 64 rows) and 0.67 / 0.68
  (o). The quartile bands are within ±0.35 % of the median on every K16 arm. cuBLAS's widest is −1.1 / +0.7 % at
  qkv/64.
- **No cell is bandwidth-bound.** The K16 arms sit at 21–49 % of the copy floor, and cuBLAS at 14–26 %. The 64-row tile
  is the further from it: the shipped plan at `o`/64 reaches 21 %.
- **Numerics never bound the selection.** At layer 0, every plan's error against the fp32 dequant reference was the
  same to three significant figures within a cell: 0.00197 at `qkv`, 0.00258 at `o`, against the 2^-7 gate. The fp32
  reorderings that split-K and KC introduce did not reach the bf16 output's rounding here. That is the census's sanity
  check, not a quality result.

**Why `o` and not `qkv` (a reading of the SELECT table, not gated).** The kernel's grid is `(cdiv(N, BLOCK_N), split-K)`
programs, on 170 SMs:
- At `o` (N 2048) the shipped plan launches 32 × 4 = **128 programs**, so a quarter of the card has no work.
- The two fastest `o`/64 plans launch 256 programs: BLOCK_N 32 × split-K 4, at 0.580 ms (KC 256) and 0.604 ms
  (KC 128). BLOCK_N 64 × split-K 8 also launches 256 (0.634).
- At 512 programs (BLOCK_N 32 × split-K 8) they slow again (0.639–0.666).
- At the shipped plan's 128 programs, 8 warps with KC 256 recover part of the gap (0.628). So grid fill is the larger
  effect, but not the only one.
- At `qkv` (N 5120) the shipped plan already launches 80 × 4 = 320, and four of the five fastest 64-row plans launch 320
  (the fifth, 640).
- So the gap is mostly grid fill at the narrower projection. It is not the accumulator pressure the PREREG guessed: the
  winner keeps 4 warps and split-K 4.

**One instrument observation.** The SELECT medians at `qkv` read 14–18 % below the re-timed medians of the same plan
(0.327 vs 0.381 ms at 32 rows, 0.547 vs 0.666 at 64). At `o` the two phases agree within 0.2 %. The verdict uses the
re-timed medians only, as registered, and within that phase `shipped2 / shipped` is 0.998–1.000. What differs between
the phases at `qkv` is not established here. The likeliest candidates are the interleaved cuBLAS arm's effect on clocks
or cache, and the graphs' placement. A confirmatory read that times `qkv` should interleave as P124 Amendment 1 does.

## Against the predictions

| prediction (written before the data) | result |
|---|---|
| The instrument holds: `shipped2 / shipped` within 1 % everywhere | **yes** (0.998–1.000) |
| Some plan beats the shipped one at 64 rows by 10–25 % | **partly**: `o` by 20.6 %, `qkv` not at all |
| The winner is 8 warps and a smaller split-K | **no**: 4 warps, split-K 4, BLOCK_N 32, KC 256 (grid fill, not accumulator pressure) |
| The verdict: CANDIDATE at 64 rows; NONE or a smaller gain at 32 | **no** at 64 rows (NONE: `qkv` does not move); **yes** at 32 rows |

## The registered consequence (NONE)

- **The shipped plan stays** at the 32- and 64-row tiles. Nothing in grouped-nf4-gemm or experts4bit-qlora changes.
- **Not pursued as a confirmatory read on its own.** The one real effect is about 1 % of the served step:
  - `o`/64 saves 0.150 ms per 48 layers, against P124's 14.7 ms 64-row step;
  - P124's served bar is 0.98;
  - so a read of that plan alone would most likely not clear it.
- **Where it belongs instead.** A per-shape rule in `plan_smallm` would shrink BLOCK_N while
  `cdiv(N, BLOCK_N) × split-K` is below the SM count. If wanted, it rides with a larger lever under its own
  registration, with P110's teacher-forced bar and a mutant, as the PREREG requires.
- **The register** row `gnf4.kernel.k34-k16-wide-plan-census.5090.2026-10-09` carries `o`/64's 0.794. It is a
  microbenchmark cell, not a served speed and not a default.

## What it took

| run | status | cost | note |
|---|---|---:|---|
| `k34-5090-1` | REFUSED before any rental | $0.000 | the launch process had not armed the live provider; nothing rented |
| `k34-5090-2` | **OK, NONE** | $0.074 | the reading; no proving rental (0.5 h guard) |

**The lane cost $0.074**, inside its $1.00 hard stop.

**Receipts** are in `receipts-k34/5090/`, with `SHA256SUMS`: `k34.json`, `summary.txt`, `forensics.txt`, `versions.txt`,
`logs/smallm_contract.log`, `logs/k34_bench.log` and the teardown proof.

**Reproduce the verdict** from the committed record (any Python; the rule reads the recorded medians):

```
cd kernel && python -c "import json, k34_bench as kb; d = json.load(open('receipts-k34/5090/k34.json')); print(kb.verdict(d) == (d['verdict'], d['verdict_reason']))"
```

The A2000 rehearsal recorded in the pre-registration is not comparable to these numbers: it checked correctness only.
