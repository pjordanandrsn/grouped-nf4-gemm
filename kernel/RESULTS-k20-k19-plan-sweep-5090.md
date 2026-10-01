# RESULTS — K20: PROMISING. K19 was carrying the wrong plan; at BLOCK_N 32 / KC 256 it takes the 5090's B=16 expert matmul to 0.736× the served route, at 89 % of the byte floor

Measured 2026-10-01 under [`PREREG-k20-k19-plan-sweep-5090.md`](PREREG-k20-k19-plan-sweep-5090.md) (#420, `55bca8a`).
Lane `k20-5090-1`:
- **Box:** RTX 5090 (sm_120, driver 580.119.02, 475 W), torch 2.8.0+cu128 / triton 3.4.0, Vast instance 53653777.
- **Code:** gnf4 at `55bca8a`, driven from experts4bit-qlora `bench/k20/` (#807, `360888a`).
- **Cost:** $0.0615 (estimate $0.5625), about 3 minutes on the box. Destroyed 07:06:26Z, absent.
- **Receipts:** [`receipts-k20/5090/`](receipts-k20/5090/), byte-identical to the adertha receipt store's.

```
K20_VERDICT PROMISING: best/served 0.736 <= 0.77
K20 EVAL best (32, 256, 4, 2) 5.200 ms/step, default 6.206, served 7.062, tiles 0.435, floor 4.640; best/served 0.736
```

## Before any timing

K19's contracts passed on the card:
- `test_int4_grouped_smallm_interp.py` under the interpreter: 11 passed;
- that file and `test_int4_smallm_interp.py` compiled: 20 passed.

The rule's self-test passed (5 cases).

## The read (EVAL steps 8–15, re-timed after selection; ms per decode step, 48 layers × gate_up + down)

| arm | median | vs served | of the floor |
|---|---:|---:|---:|
| served route (`quant_x_rows` + `gemv_int4_b32` + its reduce) | 7.062 | 1.000 | 66 % |
| K19, the shipped plan (BLOCK_N 64, KC 128, 4 warps, 2 stages) | 6.206 | 0.879 | 75 % |
| **K19, the best plan (BLOCK_N 32, KC 256, 4 warps, 2 stages)** | **5.200** | **0.736** | **89 %** |
| of which, the tile build (`build_group_tiles_fused`, one per layer) | 0.435 | | |
| floor: distinct-expert bytes / measured copy (1,532 GB/s) | 4.640 | | |

- **The instrument held.** Served reads 7.062 against P87's in-model census of 7.729 (−8.6 %). The shipped plan reads 6.206 against 6.993 (−11.3 %). Both are inside the registered ±15 %, so this measures the work P87 measured.
- **No winner's curse.** The best plan was chosen on steps 0–7 (4.999 ms) and re-timed on steps 8–15 (5.200). Per EVAL step it reads 4.85–5.28; served reads 6.90–7.15.
- **The best plan is not a lucky draw.** Every plan in the top ten has **KC 256**, and seven plans sit within 2 % of the best on SELECT (4.999–5.098). The slowest are KC 64 with wide BLOCK_N (10.7–11.6 ms).
  - The A2000-chosen KC 128 halved each program's K chunk and doubled the chunk count.
  - **Mechanism not profiled.** The plan grid alone does not separate load size, the loop's iteration overhead and the dequant's register footprint.
- **Two plans did not compile:** BLOCK_N 256 × KC 256 at 4 stages (shared memory out of resources). They are recorded in `k20_rows.json` as errors.

## Numerics: the plan is free

All 70 plans that ran produce output **bit-identical** to the shipped plan on step 0, layer 0 (`rel_vs_default` 0.0), and each is 0.0033 from the fp32 dequant oracle. The MMA accumulates the same products in the same order whatever BLOCK_N and KC are. So moving K19's default plan changes speed only, and needs no quality read.

The new compiled test `test_plans_are_bit_identical_compiled` (five plans against the shipped one, K 768 and 2048) holds the claim. It passes on the A2000 (sm_86) as well. Under the interpreter it is skipped: there the fp32 dot is numpy's and KC can move the last bit, so `test_bitwise_equals_k16_per_expert` now names its plan explicitly.

## Predictions

| prediction | read |
|---|---|
| the instrument reproduces P87 (served ≈ 7.7, default ≈ 7.0) | **held** (7.06, 6.21: both 9–11 % under; within band) |
| some plan beats the default by 10–25 % | **held, at the top of the band:** 16 % (6.206 → 5.200) |
| verdict MARGINAL (best/served 0.80–0.90), the per-weight dequant being the limit | **refuted: PROMISING, 0.736.** At 89 % of the floor, the dequant is not the limit I guessed |

## What follows (the registered PROMISING branch)

1. **K19's default plan becomes BLOCK_N 32 / KC 256 / 4 warps / 2 stages** (this PR). Bit-identical outputs, so no quality question. It is not gated to sm_120: the identity holds on sm_86 too, and K19 is an opt-in kernel either way.
2. **An end-to-end lane in experts4bit-qlora (P88):**
   - P87's speed arms on the new pin;
   - the K8 quality arms on a host with a CPU floor, because the calibration build is CPU-bound and P87's Broadwell host ran out of its alarm.

   In-model, the expert route should fall from about 7.7 ms to about 5.2 plus about 0.3 of glue: a B=16 step near 9.8 ms against today's 12.0. That is an expectation, not a claim.
3. **The grouping glue (about 0.8 ms per step in P87) is the next lever** once K19 is the route. Of it, the tile build is 0.435 here.

## Receipts

`receipts-k20/5090/`:
- `k20_rows.json` (every plan, SELECT and EVAL times, numerics, the verdict);
- `summary.txt`, `forensics.txt`, `versions.txt`, `staged.sha256`;
- the contract and bench logs;
- `teardown-proof.json`.
