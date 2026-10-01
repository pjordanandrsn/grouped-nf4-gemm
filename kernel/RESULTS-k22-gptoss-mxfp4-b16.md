# RESULTS — K22: VOID by its instrument. The census and the bench still say where gpt-oss-20b's B=16 step is: 79 % of it is the NF4 expert GEMM, and K21 reads 0.68× that route at only 37 % of the byte floor

Measured 2026-10-01 under [`PREREG-k22-gptoss-mxfp4-b16.md`](PREREG-k22-gptoss-mxfp4-b16.md) (#423, `1b65a6d`).
- **Lane:** `k22-5090-1`, an RTX 5090 (sm_120, driver 580), e4b `562767e` + gnf4 `1b65a6d`, gpt-oss-20b @ `6cee5e81`.
- **Cost:** $0.135. The lane cost $0.227 with the proof (`k22-prove-1`, $0.092: K21 + K16 compiled on the card, 19 passed).
- **Teardown:** destroyed 11:05:37Z, absent.
- **Receipts:** [`receipts-k22/5090/`](receipts-k22/5090/), including the recorded routing `eids_b16.pt`.

```
K22_VERDICT VOID: served _gemm_nf4_grouped 14.809 ms/step is not within 15% of the census 17.863
K22 EVAL best (32, 64, 4, 3) 10.297 ms/step, default 10.545, served 15.170, gemv 21.649, floor 3.826; best/served 0.679
```

## Phase 1 — the census (bo7's `store_r12`, B=16, P42 replay census)

The step reads **22.52 ms** (710.6 tok/s; bo7 read 21.65). The kernel total is 22.62 ms per step, of which:

| kernel | ms/step | calls/step |
|---|---:|---:|
| **`_gemm_nf4_grouped`** (the experts, NF4 kept for batched rows) | **17.863** | 48 |
| cutlass bf16 GEMM (attention projections) | 1.981 | 97 |
| cutlass bf16 GEMM (24/step; router / head) | 0.424 | 24 |
| `_fp8_paged_decode_split` (attention) | 0.374 | 24 |
| everything else | ≈ 1.98 | |

**79 % of gpt-oss-20b's B=16 decode step is one kernel: the NF4 grouped M-tile GEMM serving the experts.** My
prediction was "the largest item but under half". That is refuted: the paged attention kernel itself is under 2 %, and
the attention projections about 9 %.

## Phase 2 — the recorded routing

The routing is gpt-oss's own, from 16 wikitext rows tokenized by its tokenizer (prompts sha `5bdbcd9c…`): 128
teacher-forced B=16 steps, `[128, 24, 16, 4]`. It averages **18.2 distinct experts per layer per step** (per-layer means
13.7–25.1) of 32, about 3.5 rows per expert. The file is in the receipts for later lanes.

## Phase 3 — the bench (EVAL steps 8–15, ms per decode step, 24 layers × gate_up + down)

| arm | median | of the MXFP4 floor |
|---|---:|---:|
| served (tile build + gather + NF4 grouped GEMM) | 15.170 | 25 % |
| gemv (`gemv_mxfp4_b32`, 64 rows) | 21.649 | 18 % |
| K21 default plan (32 / KC 64, 4 warps, 2 stages) | 10.545 | 36 % |
| **K21 best plan (32 / KC 64, 4 warps, 3 stages)** | **10.297** | **37 %** |
| floor (MXFP4 distinct-expert bytes, 1,515 GB/s copy) | 3.826 | |

- All 48 plans ran and every plan is within 0.0032 of the fp32 oracle.
- Plans at KC 64 are bit-identical to the default. Plans at KC 32 with 8 warps differ by up to 0.0028 relative: on
  sm_120 that plan is not output-identical.
- The top plans are all KC 64, the largest K = 2880 admits.

## Why VOID

No register row is written from a VOID read.


The bench's served arm ran `_gemm_nf4_grouped` at 14.81 ms per step, against the census's 17.86 on the same box: 17 %
low, outside the registered 15 %. So the bench did not time exactly the model's work, and the rule forbids reading its
ratio.

**The likely cause is a design flaw of mine:** the bench uses ONE synthetic weight set for all 24 layers.
- A layer touches about 18 experts, roughly 240 MB of NF4, against the 5090's 96 MB L2. Consecutive layers with
  overlapping expert sets can find part of the previous layer's weights in L2.
- In the model, every layer has its own weights, so no layer reuses another's.
- K20 shared one set too, but read inside its band (−9 %, −11 %), with smaller experts and more of them, so less
  cross-layer overlap.

This is **inferred, not measured**. The amendment below measures it.

## What the descriptive numbers say

- **gpt-oss's B=16 lever is the expert GEMM, by a wide margin.** Even the best route in the bench sits far from the
  floor: K21 37 %, served 25 %.
- **K21's 0.68× is probably capped by its plan.** K = 2880 = 2⁶ × 45 limits KC to 64. On Qwen3, KC 256 was the whole
  difference between 75 % and 89 % of the floor (K20).
- **K21 needs a masked K tail.** KC 128 or 256 with the last chunk masked would lift that cap for gpt-oss (2880 = 11 ×
  256 + 64).

## What follows

1. **A K21 masked-tail variant**: KC need not divide K, and the last chunk is masked. Correctness on the A2000, then
   timing on the 5090.
2. **A re-read (K24)**:
   - this lane's runner, with a bench that gives each layer its own synthetic stores (gpt-oss: about 20 GB for NF4 and
     MXFP4 together, which fits a 5090);
   - the masked-tail plans in the grid;
   - the same instrument against a fresh same-box census;
   - the recorded routing reused from these receipts.
3. **Only then** an experts4bit-qlora consumer route for the MXFP4 store's batched rows, and an end-to-end lane with the
   store's KL instrument as the quality gate.
