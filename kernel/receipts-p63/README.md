# Lane P63's kernel census — RTX 5090, Qwen3-30B-A3B (experts4bit-qlora#708)

Lane P63 asked whether a token's MoE output depends on how many rows share its forward: T = 1 decode against the same
token inside a 16- or 17-row verify or a 160-row prefill. The lane is experts4bit-qlora's. Pre-registration,
instrument and full read: experts4bit-qlora `bench/p63/` (`P63-PREREG.md`, `RESULTS-p63.md`). This directory holds
only the part that is a property of this repository's kernels, so the kernel claims can cite a receipt in this
repository.

## `5090/` — the reading's kernel census

Run `p63-5090-1`, 2026-09-24: one RTX 5090 (sm_120, 170 SMs, driver 595.71.05, 475 W power limit), rented on Vast
verified/secure. Software: grouped-nf4-gemm 0.33.2 at `f88df1e`, experts4bit-qlora 0.37.3 at `75c83bd`, torch
2.8.0+cu128, Triton 3.4.0. The rental cost $0.2881, and its teardown is proven (`teardown-proof.json`). The proving
rental `p63-prove-1` ran first at the same commit, for $0.0115.

`kernel_census.json` is copied unedited from the lane's three receipts (`out/<stack>/p63_arm.json`; the sha256 of each
source file is recorded beside it). For each stack it keeps `kernel_census` and `combine_census`:

- **`kernel_census`:** layer 0's gate_up GEMM (N 1536, K 2048) on the model's own layer-0 activations and routing.
  Row t·8 + j is token t against its j-th expert. Each path it takes is run at T = 1 per token and at T ∈ {16, 17,
  160} tokens (128, 136 and 1,280 rows), and each is scored against the exact fp64 product of its logical operands.
  `pairs` compares each path at T tokens with the T = 1 path it replaces.
- **`combine_census`:** `combine_rows` and torch's chain on random rows at (top_k 8, hidden 2048), T ∈ {1, 2, 16, 17,
  64, 160}.

What it read (the lane's P1 and P3, all predictions held):

| path at T > 1, against the int4 GEMV or NF4 decode GEMV at T = 1 | rows bit-equal at T = 16 / 17 / 160 | class |
|---|---|---|
| `gemv_int4_b32` | 128/128, 136/136, 1,280/1,280 | EXACT |
| `gemm_int4_b32_grouped_captured` | 119/128, 127/136, 1,122/1,280 (max rel L2 4.4e-4, 1 bf16 ULP) | REORDER |
| int4 dequant + bf16 matmul | 0/128, 0/136, 0/1,280 (max rel L2 1.5e-2) | PRECISION |
| NF4 dot-pad decode GEMV | 128/128, 136/136, 1,280/1,280 | EXACT |
| NF4 scalar decode GEMV, `GNF4_GEMV_DOTPAD=0` | 116/128, 124/136, 1,123/1,280 (max rel L2 2.4e-4, 1 ULP) | REORDER |
| NF4 M-tile | 0/128, 0/136, 0/1,280 (max rel L2 2.7e-3) | PRECISION |
| `combine_rows` against itself at T = 1 | every row at T ∈ {2, 16, 17, 64, 160} | EXACT |

Every path is inside the error bound of a correct implementation of its own operand model (bound ratio ≤ 0.896 for
the kernels here; ≤ 0.9954 for `combine_rows`). EXACT means every row is bit-equal to its own T = 1 call. REORDER
means the same operand roundings summed in a different order. PRECISION means the operands are rounded differently:
a different function.

## `a2000-tests/` — `kernel/test_row_invariance_gpu.py` on the NAS RTX A2000

The GPU tests that pin the EXACT rows. `pytest.txt` is their run on the home-lab RTX A2000 12 GB (sm_86, 26 SMs,
driver 575.64.05), torch 2.8.0+cu128 and Triton 3.4.0, with the sha256 of the test file and the kernels it imports.
The result is 18 passed and none skipped.

The tests force the dispatch the 5090 takes: the > 64-SM int4 plan, and the dot-pad route at its census shapes. So
on this 26-SM part they assert the same routes P63 read, on random data. The file's control
(`test_the_check_sees_a_split_k_change`) passed too, which means the comparison does report a split-K change as a
difference. This is not a second reading of the lane.
