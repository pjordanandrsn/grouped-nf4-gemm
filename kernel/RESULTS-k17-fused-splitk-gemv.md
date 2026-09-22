# Results — lane K17: the int4-b32 GEMV's split-K reduce folded into its own launch (RTX 5090, 2026-09-21)

Pre-registration: [`PREREG-k17-fused-splitk-gemv.md`](PREREG-k17-fused-splitk-gemv.md) (#372, merged before any kernel code). Kernel: [`int4_b32.py`](int4_b32.py) (`FUSED_REDUCE` epilogue, `gemv_int4_b32(..., fused_reduce=)`, env `GNF4_GEMV_FUSED_REDUCE`). Rows: [`receipts-k17/5090/k17_rows.json`](receipts-k17/5090/k17_rows.json); bench log, both contract logs, versions, forensics and the teardown proof beside it (the lane receipt itself lives in the private receipts repository, adertha-receipts `4aff79c`). Verdicts computed by [`k17_reduce.py`](k17_reduce.py) from the rows — the table below is its output, unedited.

Lane `k17-5090-1` (experts4bit-qlora `bench/k17`, #663; e4b `0e38cc75`, gnf4 `af1a1376` = this branch before the CI-wiring commits, which touch no kernel or test code). Rented 23:48Z, `TP_DONE` 23:54Z, destroyed 23:54Z (vast instance 51979470, `teardown-proof.json`); **$0.0888** against the $0.65 estimate. Card: NVIDIA GeForce RTX 5090 (170 SMs, driver 595.91.07, 545 W / 3105 MHz), torch 2.8.0+cu128 / triton 3.4.0. Launch floor (K14's empty kernel) **3.67 µs** on this host (K16 read 4.61 on another 5090 host).

Contracts on the box before any timing: interpreter (`TRITON_INTERPRET=1`, `test_int4_b32_fused_reduce_interp.py` + `test_int4_b32.py`) **197 passed**; compiled on the 5090 **27 passed**.

## The rows (µs per call, CUDA-graph replay, 200 iterations, best of two draws A/B per arm)

| shape | N | K | R | SK | P1 equal | two-launch us | fused us | saving us | saving/floor | fused/two | A/A spread us (two/fused) | cnt re-armed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| expert_gate_up | 1536 | 2048 | 1 | 16 | True | 6.20 | 4.16 | +2.04 | +0.56 | 0.670 | 0.01 / 0.01 | True |
| expert_gate_up | 1536 | 2048 | 8 | 16 | True | 8.26 | 8.25 | +0.01 | +0.00 | 0.999 | 0.01 / 0.01 | True |
| expert_gate_up | 1536 | 2048 | 16 | 16 | True | 12.43 | 12.40 | +0.03 | +0.01 | 0.997 | 0.00 / 0.04 | True |
| expert_gate_up | 1536 | 2048 | 128 | 16 | True | 71.82 | 78.60 | -6.78 | -1.85 | 1.094 | 0.01 / 0.11 | True |
| expert_down | 2048 | 768 | 1 | 6 | True | 4.16 | 4.14 | +0.02 | +0.00 | 0.996 | 0.01 / 0.00 | True |
| expert_down | 2048 | 768 | 8 | 6 | True | 6.20 | 6.20 | -0.00 | -0.00 | 1.001 | 0.01 / 0.00 | True |
| expert_down | 2048 | 768 | 16 | 6 | True | 8.25 | 8.16 | +0.09 | +0.02 | 0.989 | 0.00 / 0.04 | True |
| expert_down | 2048 | 768 | 128 | 6 | True | 33.06 | 35.08 | -2.03 | -0.55 | 1.061 | 0.21 / 0.03 | True |
| attn_q | 4096 | 2048 | 1 | 8 | True | 6.20 | 6.20 | +0.00 | +0.00 | 0.999 | 0.01 / 0.00 | True |
| attn_q | 4096 | 2048 | 8 | 8 | True | 14.43 | 14.53 | -0.10 | -0.03 | 1.007 | 0.00 / 0.01 | True |
| attn_q | 4096 | 2048 | 16 | 8 | True | 24.68 | 26.75 | -2.07 | -0.56 | 1.084 | 0.01 / 0.03 | True |
| attn_q | 4096 | 2048 | 128 | 8 | True | 172.24 | 188.58 | -16.34 | -4.46 | 1.095 | 0.34 / 0.19 | True |
| attn_kv | 512 | 2048 | 1 | 16 | True | 5.02 | 4.15 | +0.87 | +0.24 | 0.826 | 0.13 / 0.01 | True |
| attn_kv | 512 | 2048 | 8 | 16 | True | 6.20 | 4.18 | +2.03 | +0.55 | 0.673 | 0.00 / 0.01 | True |
| attn_kv | 512 | 2048 | 16 | 16 | True | 8.25 | 6.21 | +2.04 | +0.56 | 0.753 | 0.01 / 0.00 | True |
| attn_kv | 512 | 2048 | 128 | 16 | True | 26.75 | 28.83 | -2.08 | -0.57 | 1.078 | 0.00 / 0.00 | True |
| attn_o | 2048 | 4096 | 1 | 16 | True | 6.21 | 6.20 | +0.01 | +0.00 | 0.998 | 0.01 / 0.00 | True |
| attn_o | 2048 | 4096 | 8 | 16 | True | 16.08 | 14.66 | +1.41 | +0.39 | 0.912 | 0.03 / 0.00 | True |
| attn_o | 2048 | 4096 | 16 | 16 | True | 24.79 | 26.75 | -1.96 | -0.53 | 1.079 | 0.04 / 0.04 | True |
| attn_o | 2048 | 4096 | 128 | 16 | True | 172.75 | 186.38 | -13.62 | -3.72 | 1.079 | 0.65 / 0.11 | True |
| attn_qkv_fused | 5120 | 2048 | 1 | 8 | True | 6.21 | 6.20 | +0.01 | +0.00 | 0.998 | 0.01 / 0.01 | True |
| attn_qkv_fused | 5120 | 2048 | 8 | 8 | True | 16.57 | 18.49 | -1.92 | -0.52 | 1.116 | 0.00 / 0.00 | True |
| attn_qkv_fused | 5120 | 2048 | 16 | 8 | True | 28.94 | 32.31 | -3.37 | -0.92 | 1.117 | 0.01 / 0.03 | True |
| attn_qkv_fused | 5120 | 2048 | 128 | 8 | True | 214.84 | 235.96 | -21.11 | -5.76 | 1.098 | 0.46 / 0.16 | True |

- **P1 (bitwise identity):** HELD — 24/24 rows torch.equal
- **P2 (R=1 saving 3–6 µs on each of six shapes):** REFUTED — in band 0/6; under 2 µs 5/6; slower 0/6; savings expert_gate_up +2.04, expert_down +0.02, attn_q +0.00, attn_kv +0.87, attn_o +0.01, attn_qkv_fused +0.01
- **P3 (R=128 expert shapes, fused/two in [0.90, 1.15]):** HELD — expert_gate_up 1.094, expert_down 1.061
- **counter re-arm after 800+ replays:** all zero

**Decision rule:** ¬P2 (with P1) → ship as OPT-IN (exact, costs nothing); record that the reduce launch was not the cost.

**Instrument note (read before the numbers):** the timings at these sizes sit on a **~2.05 µs grid** (4.15, 6.20, 8.25, 12.4, 14.5, 16.1, 18.5, 24.7, 26.75, 28.9 …). Whatever quantises the replay — the graph's node scheduling or the event clock — a per-call difference here reads as 0, 1 or 2 grid steps, never as 1.4 or 3.7 µs. The A/A spreads (≤ 0.65 µs, mostly ≤ 0.05) say the grid is stable; it just cannot show a saving between 0 and 2 µs except at `attn_kv` R=1 (+0.87). The 3–6 µs band would have read as 2 or 3 steps (4.1 or 6.2 µs); nothing did.

## Predictions, read against the rows

- **P1 (bitwise identity, `torch.equal` on 24 shape×R rows, compiled on the 5090) — HOLDS.** 24/24, as under the interpreter (197/197) and on the A2000 (27/27). The counter re-arms to zero after 800+ captured replays on every row, so the epilogue is safe under CUDA-graph replay.
- **P2 (R=1: the fused path saves 3–6 µs per call on each of the six shapes) — REFUTED**, by the pre-registered clause "a saving under 2 µs on the majority of shapes (the reduce was not the cost)". Savings at R=1: `expert_gate_up` **+2.04** (one grid step, 0.56 launch floors), `attn_kv` +0.87, and **≤ 0.02 µs on the other four** (`expert_down`, `attn_q`, `attn_o`, `attn_qkv_fused`). 5/6 under 2 µs, 0/6 slower. Two-launch and fused both read 6.20 µs at R=1 on the N ≥ 2048 shapes: removing the `_reduce_partials` launch changed nothing there, so at R=1 that launch was hidden — the graph overlaps it with the GEMV's tail (the same thing P54 found for the K16 row at B=16). Where the fused path did win (N=1536 and N=512 at SK=16 — few column blocks, sixteen partials each) the win is one grid step.
- **P3 (R=128 on the expert shapes: fused/two-launch within −0.10× … +0.15×) — HOLDS, at the wrong end of the band.** `expert_gate_up` **1.094**, `expert_down` **1.061**: the fused path is 6–9 % *slower* at R=128, inside the registered tolerance but a cost, not a saving. Not registered but in the rows: every attention shape is also slower at R=128 (1.078–1.098) and at R=16 on the SK=8 shapes (`attn_q` 1.084, `attn_qkv_fused` 1.117). The last-arriving program's serial read of `SK` partials per column block costs more than the wide-grid reduce it replaces once there are many rows.
- **P4 (model level: `_reduce_partials` at 0 calls/step in the consumer, the B=1 step ≥ 0.15 ms faster, tokens identical) — PENDING**, in the consumer's lane P57 (experts4bit-qlora `bench/p57`, registered #666 before this read). Given P2 the ≥ 0.15 ms clause is expected to fail: the consumer's B=1 step carries ~48 GEMV calls whose two-launch and fused times are equal on four of six shapes.

## Decision rule, applied

**P1 ∧ ¬P2 → ship as opt-in.** `GNF4_GEMV_FUSED_REDUCE` stays **`0` = the two-launch path** by default; `fused_reduce=True` (or the env at `1`) is available, exact, and re-arms its counter under graph replay. Recorded: **the `_reduce_partials` launch was not the cost at R=1** — its launch is overlapped in the graph on the wide shapes, and the fused epilogue's serial partial read is a real cost at R ≥ 16 on SK=8 shapes and at R=128 everywhere (6–12 %). The rule's `P1 ∧ P2 ∧ ¬P3` branch (route by R in the planner) is **not** taken either: P2 failed, so there is no R at which the fused path is licensed as a default.

What P54's census actually measured (the `_reduce_partials` **row** at 0.300 ms/step at B=1) is therefore not recoverable by fusing the reduce: the row's time is GPU time the kernel spends either way, not a launch that removal recovers. The remaining B=1 launch lever named in the K17 prereg — folding the activation quantise (`_quant_x_rows`, 0.153 ms/step at B=1) into the GEMV — is a different fusion and is not licensed by anything here.

## What this does and does not say

- Says: the fused epilogue is exact and safe (P1, counter re-arm), and does not pay for itself on this card at R=1 on the model's shapes. One 5090 host; one card; six shapes; timings on a 2 µs grid.
- Does not say: anything about the consumer's step time (P57 reads that), about other cards, or about the int4_smallm (K16) fused reduce — that kernel's epilogue was measured in K16 against a *separate* reduce it never had; nothing here revisits it.
- Cost of the read: $0.0888, one lane, six minutes; the A2000 pilot (free, `receipts-k17/a2000-pilot.json`) had already shown 0.0–2.8 µs savings at R=1 and was recorded as a warning in the launch manifest.
