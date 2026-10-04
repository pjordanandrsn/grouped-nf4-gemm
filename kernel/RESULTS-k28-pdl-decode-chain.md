# K28 — results: **LEVER**. Programmatic dependent launch (`GNF4_PDL=1`) saves 0.323 µs per gnf4 kernel in a CUDA-graph replay of the served B=1 decode layer's 912 gnf4 kernels on an RTX 5090 (2.658 → 2.363 ms, ×1.125), bit-identically

Registration: `kernel/PREREG-k28-pdl-decode-chain.md` (#449, `afe815d`). The switch: #448 (`4509307`). Tracking issue:
experts4bit-qlora#1015. Runner: experts4bit-qlora `bench/k28/` (#1017, `70844c1`).

Code under test: grouped-nf4-gemm at `afe815d` (0.36.0 with #448 and #449), torch 2.8.0+cu128, triton 3.4.0.

**Verdict by `k28_bench.verdict`: `LEVER`.** The rule's steps, in order:

| step | result |
|---|---|
| VOID | no. The card is sm_120. Every capture launched 912 gnf4 kernels and no other Triton kernel, all 912 with PDL in `chain_on` and `glued_on` and none in the other three. With PDL the probe's dependent started 19.4–19.7 µs before its 20 µs primary ended, in every replay; without PDL it started 0.5–0.8 µs after it |
| FUNCTION_FAIL | no. Every bitwise check held: `chain_on` and `chain_off2` against `chain_off`, `glued_on` against `glued_off`, the eager chain on against off, and eager against the graph. Every probe dependent left its wait at or after its primary's last stamp |
| NOISY | no. `chain_off2 / chain_off` = 1.0007 |
| LEVER | **yes**: (2.6576 − 2.3632) ms / 912 = **0.323 µs** ≥ 0.25 |

## The reading (`k28-5090-1`)

**Host:** one RTX 5090 (sm_120, driver 580.95.05, 600 W, 3135 MHz) on an AMD EPYC 7C13 (256 threads), Vast
instance 54112990 (machine 150527) at $0.58/h. **Cost:** $0.0237 (147 s). Teardown was proven at 06:11:33Z.

**Timeline (the box's own log):** install at 06:10:23Z; the tripwire and the premise (`kernel/test_pdl.py` compiled on the
card: **23 passed**, none skipped) until the bench at 06:10:46Z; the bench took 10.3 s; `TP_DONE` at 06:10:58Z.

| graph | median ms | quartiles ms | gnf4 launches (with PDL) |
|---|---:|---|---|
| `chain_off` | 2.6576 | 2.6562 – 2.6595 | 912 (0) |
| `chain_on` | 2.3632 | 2.3602 – 2.3663 | 912 (912) |
| `chain_off2` | 2.6594 | 2.6583 – 2.6612 | 912 (0) |
| `glued_off` | 2.8468 | 2.8454 – 2.8487 | 912 (0) |
| `glued_on` | 2.6489 | 2.6452 – 2.6525 | 912 (912) |

- **The chain:** 2.658 → 2.363 ms, ×1.125, **0.323 µs per gnf4 kernel**. The quartile bands do not overlap.
- **Glued** (two more ATen kernels a layer, where the served step leaves gnf4): 2.847 → 2.649 ms, ×1.075,
  **0.217 µs per gnf4 kernel**. An ATen kernel is not a programmatic dependent, so the edges next to it keep their
  launch gap. The saving per gnf4 kernel falls by 33 % with 4 ATen kernels a layer in place of 2.
- **The probe** shows what PDL does on this card. The dependent was resident and waiting 19.4–19.7 µs before its 20 µs
  primary ended, so it launched within about 0.3–0.6 µs of the primary's first clock read.

## Against the predictions

| prediction (written before the data) | result |
|---|---|
| Q1: the probe engages (≥ 15 µs early with PDL, at or after without) | **yes** (19.4–19.7 µs early; 0.5–0.8 µs late without) |
| Q2: every bitwise check holds | **yes** |
| Q3: 912 gnf4 launches per capture, PDL exactly on the on captures | **yes** |
| Q4: 0.3–1.2 µs saved per gnf4 kernel | **yes**, at the low end (0.323) |
| Q5: the glued graph saves less per gnf4 kernel | **yes** (0.217 against 0.323) |
| Q6: `chain_off` replays in 1.5–3.0 ms | **yes** (2.658) |
| Q7: LEVER (about 65 %) | **yes** |

## The registered consequence (LEVER)

- **experts4bit-qlora registers a served lane:** `GNF4_PDL=1` on its default decode step, read for tokens (bitwise) and
  decode tok/s at B=1 and B=16. That lane also decides whether e4b's own decode kernels (the KV append and the paged
  decode) take the same preamble.
- **`GNF4_PDL` stays off by default here** until the served lane reads.
- **The register** row `gnf4.kernel.k28-pdl-decode-chain.5090.2026-10-04` carries 0.323 µs per kernel, as a
  microbenchmark, not a served speed.

**What it suggests, not what it shows.** SC1b's B=1 step has 913 gnf4 kernels among about 1,550. At the chain's
0.323 µs that is 0.29 ms of its 4.075 ms span; at the glued 0.217 µs, 0.20 ms. The served step interleaves more
non-gnf4 kernels than the glued graph does, so the served lane may read less. llama.cpp's 1.38 ms of overlap spans
95.5 % of its step's kernel pairs; this switch reaches gnf4's kernels only.

## What it took

| run | status | cost | note |
|---|---|---:|---|
| `k28-5090-1` | **OK, LEVER** | $0.0237 | the reading; no proving rental (0.5 h guard) |

**The lane cost $0.0237**, inside its $0.75 ceiling.

**Receipts** are in `receipts-k28/5090/`, with `SHA256SUMS`: `k28.json`, `summary.txt`, `forensics.txt`, `versions.txt`,
`logs/pdl_contract.log`, `logs/k28_bench.log` and the teardown proof. The launcher's receipt and ledger row are in the
receipt store (adertha-receipts `52253d0`; the ledger row landed with `4124b60`).

The A2000 rehearsal's values in the pre-registration are not comparable to these: the synthetic stores come from the
CUDA generator, whose stream depends on the device.
