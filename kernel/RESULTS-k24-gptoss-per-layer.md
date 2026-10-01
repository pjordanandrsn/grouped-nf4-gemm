# RESULTS — K24: VOID by its instrument again; per-layer stores did not close the gap. Descriptively, K21 with its masked-tail plans reads 0.50× the served NF4 route on gpt-oss-20b's B=16 routing, at 50 % of the byte floor

Measured 2026-10-01 under [`PREREG-k24-gptoss-per-layer.md`](PREREG-k24-gptoss-per-layer.md) (#426, `9753a1f`).
- **Lane:** `k24-5090-1`, an RTX 5090 (sm_120, driver 580.126.18) on an AMD EPYC 7K62 host. e4b `2cb0aad` + gnf4
  `9753a1f`, gpt-oss-20b @ `6cee5e81`.
- **Cost:** $0.227. The lane cost $0.283 with its proof (`k24-prove-2`, $0.049: K21's masked-tail and K16's contracts
  compiled on the card, 23 passed) and `k24-prove-1` ($0.006, NOT_RUN on a stale controller host key).
- **Teardown:** destroyed 12:37:04Z, absent.
- **Receipts:** [`receipts-k24/5090/`](receipts-k24/5090/). The routing is K22's
  (`receipts-k22/5090/eids_b16.pt`, sha256 `59b2078b…`, read from the clone at the pin).

```
K24_VERDICT VOID: served _gemm_nf4_grouped 14.850 ms/step is not within 15% of the census 17.903
K24 EVAL best (32, 128, 4, 3) 7.623 ms/step, default 9.215, served 15.178, gemv 23.837, floor 3.810; best/served 0.502
```

## The census (bo7's `store_r12`, B=16, P42 replay census, 8 replays)

The step reads **22.50 ms** (711.1 tok/s); K22's box read 22.52. Of the 22.60 ms of kernel time per step:

| kernel | ms/step | calls/step |
|---|---:|---:|
| **`_gemm_nf4_grouped`** (the experts, NF4 kept for batched rows) | **17.903** | 48 |
| cutlass bf16 GEMM (attention projections) | 1.969 | 97 |
| cutlass bf16 GEMM (router / head) | 0.421 | 24 |
| `_fp8_paged_decode_split` (attention) | 0.366 | 24 |

That is 79 % in the expert GEMM again; K22 read 17.863 ms/step on another box.

## The bench (EVAL steps 8–15, ms per decode step, 24 layers × gate_up + down, per-layer stores)

| arm | median | of the MXFP4 floor |
|---|---:|---:|
| served (tile build + gather + NF4 grouped GEMM) | 15.178 | 25 % |
| gemv (`gemv_mxfp4_b32`, 64 rows) | 23.837 | 16 % |
| K21 default plan (32 / KC 256, 4 warps, 2 stages) | 9.215 | 41 % |
| **K21 best plan (32 / KC 128, 4 warps, 3 stages)** | **7.623** | **50 %** |
| floor (MXFP4 distinct-expert bytes, 1,521 GB/s copy) | 3.810 | |

- **All 54 plans are bit-identical to the default.** That includes every masked-tail plan (KC 128 and 256 on K = 2880)
  against KC 64. The worst plan is 0.0031 from the fp32 oracle.
- **By KC, the best select-window medians are:** KC 64 10.54, **KC 128 7.79**, KC 256 8.19. The masked tail is worth
  1.35× over the KC 64 cap, and KC 128 beats KC 256 here, unlike Qwen3 (K20).
- The top six plans are all KC 128, with BLOCK_N 32 or 64.

## Why VOID

No register row is written from a VOID read.

The bench's served arm ran `_gemm_nf4_grouped` at 14.85 ms per step, against the same box's census of 17.90: 17 % low,
outside the registered 15 %. K22 read the same miss (14.81 vs 17.86).

**K22's inferred cause is refuted.** K24 gave every layer its own weights, so no layer can find another's experts in L2,
and the gap did not move (−17.1 % vs −17.1 %).

**The next candidate, also inferred and not measured:** the bench and the census do not route the same way.
- The bench replays K22's recording: wikitext rows, teacher-forced, 18.2 distinct experts per layer per step.
- The census decodes the harness's own B=16 prompts.
- At 64 rows over 32 experts, the number of distinct experts per layer, and so the weight bytes the step streams, depends
  on the routing. A census at about 21–22 distinct experts per layer would explain the gap.
- On Qwen3 (K20 vs P88) the two read within their band. There, 128 rows over 128 experts sit nearer saturation, so the
  count of distinct experts moves less with the prompts.

A test would record the routing of the census's own decode and replay it.

## Against the predictions

| prediction | outcome |
|---|---|
| the instrument holds with per-layer stores | **refuted** (−17 %, unchanged) |
| K21 PROMISING, best / served 0.45–0.65, KC 256 on top | ratio 0.502, inside the band but VOID, so not read; **KC 128 on top, not 256** |
| K21 at 60–80 % of the MXFP4 floor | **50 %**, below the band |

## What the descriptive numbers say

- **Twice now, on two boxes, K21 beats the served route at gpt-oss's B=16 shapes by a wide margin.** It read 0.68× at
  KC 64 (K22) and 0.50× with the masked tail (K24). Both arms in each bench share the routing, so the ratio is less
  exposed to the routing question than either arm's absolute level. That is an argument, not a measurement.
- **The decision quantity is the model's step, not this bench.** The bench's instrument has missed twice. The next read
  should be end to end, where the routing is the model's own by construction.

## What follows

1. **An experts4bit-qlora opt-in route for the MXFP4 store's batched rows to K21**, at K24's best plan (32 / KC 128 / 4
   warps / 3 stages; descriptive, chosen as the treatment, not licensed by this read). It takes P87/P88's shape:
   - a row-exactness premise on the card (a row's K21 output does not depend on its tile mates);
   - an opt-in `1` that routes T == 1 to K21 too, so the store's T == 1 KL instrument reads the kernel it gates.
2. **An end-to-end lane on gpt-oss-20b at B=16:** OFF vs ON step time, with the store's KL instrument as the quality
   gate, registered before any run.
