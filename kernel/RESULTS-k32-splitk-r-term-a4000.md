# K32 — results: K30's split-K R-term sweep on one rented RTX A4000 reads **OFF** (by the worst cell)

Pre-registration: [`PREREG-k32-splitk-r-term-a4000.md`](PREREG-k32-splitk-r-term-a4000.md). The instrument and rule are
K30's, frozen at its measured cut `aa562718`. Every number below is read from the committed receipts in
[`receipts-k32/a4000/`](receipts-k32/a4000/).

**Verdict: OFF**, by K30's registered rule.
- **S passed:** S = 0.9396, against KEEP's S ≤ 0.97.
- **W failed:** W = 1.0965 (olmoe down at R = 16), against W ≤ 1.02. So the lane reads OFF.
- **No VOID condition fired:** the card, its 48 SMs, all 48 cells, and the two passes' S agree to 0.0006 against a 0.02
  bound.
- **What follows, as registered:** nothing changes. K30's OFF is now two-card (sm_89 and sm_86), and the default stays
  N-only on every part (#482).

## The run

| | |
|---|---|
| run | `k32-a4000-1`, 2026-10-05, status OK, teardown `vast-destroy` complete, **$0.016** |
| card | NVIDIA RTX A4000, 48 SMs, 16,376 MiB, driver 535.154.05, power limit 140 W (`forensics.txt`); Vast verified/secure machine 19242, AMD Athlon 3000G host |
| stack | torch 2.8.0+cu128, triton 3.4.0, grouped-nf4-gemm 0.41.0 at `aa562718` (`versions.txt`) |
| runner | experts4bit-qlora `298224fd`, `bench/k30/` with `K30_CARD=A4000` |
| owner authorization | [grouped-nf4-gemm#484 issuecomment-6001106341](https://github.com/pjordanandrsn/grouped-nf4-gemm/pull/484#issuecomment-6001106341) |

**The correctness gates passed before any timing:**
- 138 compiled GEMV, reduce and plan tests;
- every swept sk agrees with sk = 1;
- the rule's self-test, 8 of 8;
- the installed `_plan` matches the reducer on all 48 cells at 48 SMs.

## What the rule read (`k30_verdict.json`)

| quantity | RTX A4000 (K32) | NVIDIA L4 (K30, for comparison) |
|---|---|---|
| **S** (R-aware / N-only summed time) | **0.9396** (pass 1: 0.9393, pass 2: 0.9399) | 1.0101 |
| **W** (worst cell) | **1.0965**, olmoe down at R = 16 | 1.1781, qwen3_moe gate_up at R = 128 |
| R-aware pick / per-cell optimum | 1.0171 | 1.1122 |
| N-only pick / per-cell optimum | 1.0825 | 1.1012 |
| largest pass-to-pass difference on a cell | 0.77 % | 1.52 % |

**Every cell, both passes averaged:**

| shape | R | sk (R-aware / N-only) | best sk (p1/p2) | R-aware | N-only | r |
|---|---|---|---|---|---|---|
| qwen3_moe gate_up | 16 | 2 / 16 | 2/2 | 91.7 µs | 96.5 µs | 0.9497 |
| qwen3_moe gate_up | 32 | 1 / 16 | 1/1 | 174.3 µs | 191.6 µs | 0.9097 |
| qwen3_moe gate_up | 64 | 1 / 16 | 3/3 | 383.5 µs | 394.0 µs | 0.9733 |
| qwen3_moe gate_up | 128 | 1 / 16 | 2/2 | 738.5 µs | 789.7 µs | 0.9351 |
| qwen3_moe down | 16 | 2 / 6 | 6/6 | 50.7 µs | 49.7 µs | 1.0197 |
| qwen3_moe down | 32 | 1 / 6 | 3/3 | 104.0 µs | 106.0 µs | 0.9809 |
| qwen3_moe down | 64 | 1 / 6 | 1/1 | 183.4 µs | 200.0 µs | 0.9170 |
| qwen3_moe down | 128 | 1 / 6 | 1/1 | 365.0 µs | 397.4 µs | 0.9186 |
| granitemoe gate_up | 16 | 4 / 12 | 1/1 | 48.4 µs | 45.3 µs | **1.0670** |
| granitemoe gate_up | 32 | 2 / 12 | 4/3 | 105.6 µs | 107.2 µs | 0.9842 |
| granitemoe gate_up | 64 | 1 / 12 | 3/3 | 196.2 µs | 196.6 µs | 0.9982 |
| granitemoe gate_up | 128 | 1 / 12 | 2/2 | 362.6 µs | 384.6 µs | 0.9428 |
| granitemoe down | 16 | 2 / 4 | 1/1 | 26.2 µs | 25.9 µs | 1.0105 |
| granitemoe down | 32 | 1 / 4 | 1/1 | 45.3 µs | 50.4 µs | 0.8988 |
| granitemoe down | 64 | 1 / 4 | 1/1 | 90.4 µs | 98.6 µs | 0.9162 |
| granitemoe down | 128 | 1 / 4 | 1/1 | 159.4 µs | 185.0 µs | 0.8614 |
| olmoe gate_up | 16 | 2 / 16 | 6/6 | 142.1 µs | 136.4 µs | **1.0418** |
| olmoe gate_up | 32 | 1 / 16 | 3/3 | 274.9 µs | 277.5 µs | 0.9904 |
| olmoe gate_up | 64 | 1 / 16 | 2/2 | 488.3 µs | 520.0 µs | 0.9389 |
| olmoe gate_up | 128 | 1 / 16 | 1/1 | 970.5 µs | 1,079.1 µs | 0.8994 |
| olmoe down | 16 | 2 / 8 | 8/8 | 71.2 µs | 64.9 µs | **1.0965** |
| olmoe down | 32 | 1 / 8 | 3/3 | 135.7 µs | 134.7 µs | 1.0074 |
| olmoe down | 64 | 1 / 8 | 2/2 | 247.8 µs | 257.8 µs | 0.9613 |
| olmoe down | 128 | 1 / 8 | 1/1 | 471.0 µs | 518.3 µs | 0.9087 |

**What decided it.** On the A4000 the R-aware pick wins 18 of the 24 cells and is 6 % faster in summed time, so S
passes. But it loses on 6 cells, five of them at R = 16, and three of those lose by more than the registered 2 %:
- granitemoe gate_up R = 16: 1.0670
- olmoe gate_up R = 16: 1.0418
- olmoe down R = 16: 1.0965

At R = 16 on 48 SMs the R-aware rule cuts the split to 2–4. The per-cell best there ranges from sk 1 to sk 8 by shape,
and in five of the six shapes the R-aware pick lands on a slower sk than the N-only pick. The rule's W bound exists for
exactly this: a default may not cost more than 2 % on any cell it changes.

**My registered expectation, OFF, holds, but not for the reason I gave.** I predicted the large gate_up cells at
R = 64–128 would decide it, as on the L4. On the A4000 those cells are wins: qwen3_moe gate_up R = 128 reads 0.9351,
olmoe gate_up R = 128 reads 0.8994. The losses are at the smallest R the term acts on.

## Observations, not findings (post hoc)

- **The two architectures disagree on where the term hurts.** By R, the R-aware / N-only ratio is:

  | R | L4 (sm_89, 58 SMs) | A4000 (sm_86, 48 SMs) |
  |---|---|---|
  | 16 | 0.9710 | 1.0272 |
  | 32 | 0.8980 | 0.9680 |
  | 64 | 0.9627 | 0.9535 |
  | 128 | 1.0543 | 0.9144 |

  - On the L4 the term loses at R = 128; on the A4000 it loses at R = 16.
  - No single capped variant suggested by one card is supported by the other.
  - This reinforces the two-card rule, and K30's note that one card cannot fit a class constant.
- **On the A4000 the R-aware pick is much closer to the per-cell optimum** than the N-only pick (1.017× vs 1.083×). On
  the L4 they were about equal (1.112× vs 1.101×). Any future per-architecture rule would be a new registration on
  both cards. It is low priority while the split-K GEMV stays off e4b's default server path (P88).

## What this lane does not claim

- **No step-level effect.**
- **No change to the default,** which this lane could not make (registration table).
- **Nothing about other ≤64-SM parts** (a 4060 Ti, 34 SMs, sm_89, for instance).
- **No comparison with the A2000 sweep's timings,** which are inadmissible.

## What follows (registered)

**OFF: nothing changes.** This PR adds:
- the measured row `gnf4.kernel.k32-splitk-r-term.a4000.2026-10-05`;
- these results and the receipts;
- a sentence in `docs/STATUS.md` that K30's OFF now stands on two architectures.
