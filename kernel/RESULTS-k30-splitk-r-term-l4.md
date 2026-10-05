# K30 — results: the int4-b32 split-K R term on one rented NVIDIA L4 reads **OFF**

Pre-registration: [`PREREG-k30-splitk-r-term-l4.md`](PREREG-k30-splitk-r-term-l4.md), with Amendments 1–2. Every number
below is read from the committed receipts in [`receipts-k30/l4/`](receipts-k30/l4/); nothing is recomputed from memory.

**Verdict: OFF**, by the registered rule.
- KEEP needed S ≤ 0.97 and W ≤ 1.02. The L4 read **S = 1.0101** and **W = 1.1781**.
- No VOID condition fired: right card and SM count, all 48 cells present, and the two passes' S agree to 0.0001 against a
  0.02 bound.
- Under Amendment 2 an L4 OFF decides the lane, so the RTX A4000 was not rented.

## The run

| | |
|---|---|
| run | `k30-l4-1`, 2026-10-05 14:07–14:12Z, status OK, teardown `vast-destroy` complete, **$0.030** |
| proving rental | `k30-prove-1`, 14:02Z, `prove ok` on the same class and image, $0.009 |
| card | NVIDIA L4, 58 SMs, 23,034 MiB, driver 615.71.09, power limit 72 W, PCIe gen 3 x16 (`forensics.txt`); Vast verified/secure machine 77226, Xeon Silver 4114 host |
| stack | torch 2.8.0+cu128, triton 3.4.0, grouped-nf4-gemm 0.41.0 (`versions.txt`) |
| measured cut | grouped-nf4-gemm `aa562718b9eadceeab746f0fb8d07462e23c2b28` (`prereg/k30`). Its `kernel/` and `bench/int4/` trees are byte-identical to main's `ccf4de91` (#478's squash), so the verdict attaches to the registered commit on main too. |
| runner | experts4bit-qlora `13232663631656b28bd5557e1ee41f33e8b0b0e8` (`lane/k30`); `bench/k30/` is byte-identical to main's `ca9141b8` (#1151's squash) |
| owner authorization | [grouped-nf4-gemm#478 issuecomment-5996050362](https://github.com/pjordanandrsn/grouped-nf4-gemm/pull/478#issuecomment-5996050362) |

**The correctness gates passed before any timing** (`summary.txt`):
- 138 GEMV, reduce and plan tests passed, compiled on the card.
- `k30_sk_check.py` passed: every swept sk matches sk = 1 through the served two-launch path.
- The rule's self-test passed 8 of 8 cases.
- The installed `_plan` picks what the reducer prices on all 48 cells at 58 SMs.

The card ran at 2,040 MHz SM throughout the sweep (`logs/smi.csv`).

## What the rule read (`k30_verdict.json`)

**The 24 cells at R ≥ 16, pooled over both passes:**

| quantity | value |
|---|---|
| **S** (R-aware / N-only summed time) | **1.0101** (pass 1: 1.0100, pass 2: 1.0101) |
| **W** (worst cell) | **1.1781**, qwen3_moe gate_up at R = 128 (R-aware sk 1: 943.8 µs; N-only sk 16: 801.2 µs) |
| R-aware pick / per-cell optimum | 1.1122 |
| N-only pick / per-cell optimum | 1.1012 |
| largest pass-to-pass difference on any cell | 1.52 % |

**Every cell, both passes averaged:**

| shape | R | sk (R-aware / N-only) | best sk (p1/p2) | R-aware | N-only | r |
|---|---|---|---|---|---|---|
| qwen3_moe gate_up | 16 | 4 / 16 | 3/3 | 36.5 µs | 37.9 µs | 0.9644 |
| qwen3_moe gate_up | 32 | 2 / 16 | 3/3 | 82.4 µs | 109.5 µs | 0.7522 |
| qwen3_moe gate_up | 64 | 1 / 16 | 3/3 | 425.4 µs | 429.9 µs | 0.9896 |
| qwen3_moe gate_up | 128 | 1 / 16 | 8/8 | 943.8 µs | 801.2 µs | **1.1781** |
| qwen3_moe down | 16 | 2 / 6 | 2/2 | 19.1 µs | 19.7 µs | 0.9693 |
| qwen3_moe down | 32 | 1 / 6 | 1/1 | 35.3 µs | 37.0 µs | 0.9537 |
| qwen3_moe down | 64 | 1 / 6 | 2/2 | 78.8 µs | 85.2 µs | 0.9244 |
| qwen3_moe down | 128 | 1 / 6 | 1/1 | 338.4 µs | 368.0 µs | 0.9195 |
| granitemoe gate_up | 16 | 4 / 12 | 6/4 | 19.5 µs | 20.3 µs | 0.9602 |
| granitemoe gate_up | 32 | 2 / 12 | 2/2 | 36.0 µs | 37.8 µs | 0.9502 |
| granitemoe gate_up | 64 | 1 / 12 | 2/2 | 82.6 µs | 88.5 µs | 0.9333 |
| granitemoe gate_up | 128 | 1 / 12 | 2/2 | 359.5 µs | 389.4 µs | 0.9233 |
| granitemoe down | 16 | 4 / 4 | 2/2 | 11.3 µs | 11.3 µs | 1.0000 |
| granitemoe down | 32 | 2 / 4 | 1/1 | 19.3 µs | 19.6 µs | 0.9874 |
| granitemoe down | 64 | 1 / 4 | 1/1 | 35.2 µs | 36.7 µs | 0.9586 |
| granitemoe down | 128 | 1 / 4 | 1/1 | 68.6 µs | 73.4 µs | 0.9345 |
| olmoe gate_up | 16 | 2 / 16 | 6/6 | 54.6 µs | 55.7 µs | 0.9798 |
| olmoe gate_up | 32 | 1 / 16 | 1/1 | 325.7 µs | 354.2 µs | 0.9195 |
| olmoe gate_up | 64 | 1 / 16 | 8/8 | 648.8 µs | 620.8 µs | **1.0450** |
| olmoe gate_up | 128 | 1 / 16 | 8/8 | 1,251.6 µs | 1,147.9 µs | **1.0903** |
| olmoe down | 16 | 2 / 8 | 2/2 | 24.4 µs | 25.5 µs | 0.9585 |
| olmoe down | 32 | 1 / 8 | 1/1 | 46.1 µs | 48.5 µs | 0.9511 |
| olmoe down | 64 | 1 / 8 | 1/1 | 146.6 µs | 211.1 µs | 0.6946 |
| olmoe down | 128 | 1 / 8 | 4/4 | 477.7 µs | 482.6 µs | 0.9897 |

**Why S lands above 1.** The R-aware pick wins 20 of the 24 cells, but the summed time is dominated by the large gate_up
cells. There it collapses the split to sk = 1 and loses: qwen3_moe gate_up at R = 128 is 1.178, and olmoe gate_up at
R = 64 and 128 is 1.045 and 1.090. Those three cells alone are 2,844 µs of the R-aware sum against 2,570 µs for the N-only
pick, which outweighs every win.

**My registered expectation is refuted.** I expected KEEP, with S below 1 driven by the R ≥ 64 cells, where dropping the
split saves its partials traffic. The opposite holds on the largest cells. On those three gate_up cells the best sk was 8,
not 1. A single split per row at R = 64–128 on 58 SMs is not enough parallelism or memory-level overlap for the wide
gate_up projections, whatever it saves in partials.

## Observations, not findings (post hoc; a new registration is needed to test any of them)

- **The per-R split.** Grouped by R, the R-aware / N-only ratio is 0.9710 at R = 16, 0.8980 at R = 32, 0.9627 at R = 64
  and 1.0543 at R = 128. A term capped so that it never drops below some split, or that stops acting at large R, might win
  on ≤64-SM parts. That is a hypothesis drawn from this data. It is not a result, and testing it is a new registration,
  not a re-read of K30.
- **Neither plan is near the per-cell optimum** (1.10–1.11× it). The best sk on this card is often 1–3 at R ≤ 64 and 8 at
  the largest gate_up cells. One card cannot fit a class constant, so none is fitted here.

## What this lane does not claim

- **No step-level effect.** The term is also off the default server path: since P88, e4b's int4 server sends
  device-grouped B > 1 decode rows to K19's small-M GEMM. The split-K GEMV and its R term are reached only by
  `E4B_INT4_GROUPED_SMALLM=0`, the singleton GEMV at T > 1 and direct callers of `gemv_int4_b32`.
  - So a capped variant is low priority, and I would not register one unless a default-path consumer of the split-K GEMV
    at R ≥ 16 appears.
- **One card is not the class.** An L4 OFF suffices to decide the default under Amendment 2, since a default needs a win
  on every card it ships to. It says nothing about whether an A4000 (48 SMs, sm_86) would have read KEEP.
- **Nothing about the A2000 sweep.** That sweep's timings are not admissible, and its sk = 1 cells omitted the reduce
  launch (Amendment 1). No comparison with them is made here.

## What follows (registered)

- **Done before this read, as Amendment 2 records:** #476 retired the A2000 row
  `gnf4.serve.int4-b32-splitk-row-term.a2000.2026-09-10` and removed its receipt tests.
- **This PR:** the K30 row `gnf4.kernel.k30-splitk-r-term.l4.2026-10-05` (measured), these results, the receipts, and
  `docs/STATUS.md`.
- **The consequence, in its own PR:** `SPLITK_R_TERM_MAX_SMS = 0`, so no part takes the R term and every card plans
  N-only, as the RTX 5090 already does.
  - The mechanism stays, with its tests run under a monkeypatched gate.
  - A structural test enforces the default: for every sm_count in {26, 48, 58, 64} and R ≥ `SPLITK_R_FLOOR`, `_plan`
    equals the N-only plan.
