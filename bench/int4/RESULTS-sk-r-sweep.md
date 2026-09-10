# int4-b32 decode GEMV: split-K vs the row count (A2000, sm_86)

`_plan(N, K)` chose split-K from `N` alone. The launch grid is
`(cdiv(N, BLOCK_N), R, sk)` — so `R`, the number of activation rows, is a
factor of the very block count `sk` exists to top up, and it was absent from
the rule. At `R > 1` every projection therefore ran a config chosen for a
batch it was not in, **and** paid the split-K partials reduce to do it.

The sibling NF4 decode planner has always taken its row count —
`nf4_grouped._decode_plan(N, K, T, sm_count)`. This is that term, restored to
the int4 planner.

## What was measured

RTX A2000 12GB (sm_86, 26 SMs), torch 2.8.0+cu128, CUDA-graph replay —
`int4_b32`'s own two measurement rules, since an eager sweep anti-selects
split-K (the reduce pays a launch the replay does not). Every cell is the
**full** cost including `reduce_partials`; `sk=1` needs no reduce at all, so a
kernel-only number would have flattered the split arms.

6 shapes × 8 row counts = **48 cells**, every `sk` from 1 to the span cap:

| family | proj | N | K | tiles | plan sk |
|---|---|---|---|---|---|
| qwen3_moe | gate_up | 1536 | 2048 | 12 | 16 |
| qwen3_moe | down | 2048 | 768 | 16 | 6 |
| granitemoe | gate_up | 1024 | 1536 | 8 | 12 |
| granitemoe | down | 1536 | 512 | 12 | 4 |
| olmoe | gate_up | 2048 | 2048 | 16 | 16 |
| olmoe | down | 2048 | 1024 | 16 | 8 |

Harness: `bench/int4/sk_sweep.py`. Receipts: `bench/int4/rows/sk_*.json`.

## The floor: B=1 decode does not call this at R=1

This nearly went wrong. B=1 decode does **not** reach the grouped GEMV with
one row — `hot_residency`'s singleton branch calls it once per projection with
`R = top_k`, so 4 or 8. An R term with no floor therefore moves the *licensed
serve config* on real shapes even on a 128-SM part: olmoe gate_up 16 → 8,
qwen3_5_moe gate_up 16 → 8 and down 8 → 4. Split-K changes the grouping of the
fp32 partial sums, so that is a numerics change on the licensed path, arriving
as a side effect of a batch optimisation.

`SPLITK_R_FLOOR = 16` closes it. No MoE served here routes to 16 experts per
token, so every B=1 cell is untouched **by construction**, on every box, rather
than by arithmetic that happens to agree on the shapes anyone checked. `R ≥ 16`
is batched decode (`B*top_k` rows via `_int4_gemv_decode`), which is the regime
the term is for. The floor costs 1.127× → 1.124× overall — the win below R=16
was ≤1.05× — and buys a much stronger property (next section).

`test_the_floor_clears_every_shipped_top_k` fails if a model with `top_k ≥ 16`
is added, so the floor has to be re-justified rather than silently breached.

## Result

Against a per-cell oracle (the best `sk` measured for that cell):

| rule | total vs oracle | worst cell where it acts | cells slower than the N-only rule |
|---|---|---|---|
| N-only (before) | 1.136× | — | — |
| **R-aware + floor** | **1.011×** | **1.064×** | **0 of 48** |

**Never slower than the incumbent on any measured cell** — that is the property
the floor buys, and it is what makes this safe to land rather than merely
favourable on average.

Summed over all six shapes, by row count:

| R | N-only | R-aware + floor | gain |
|---:|---:|---:|---:|
| 1 | 61.1 µs | 61.1 µs | 1.000× — below the floor, untouched |
| 2 | 96.9 | 96.9 | 1.000× — below the floor |
| 4 | 170.4 | 170.4 | 1.000× — below the floor |
| 8 | 324.0 | 324.0 | 1.000× — below the floor (this is B=1 top-8) |
| 16 | 658.9 | 598.1 | **1.102×** |
| 32 | 1302.4 | 1163.9 | **1.119×** |
| 64 | 2595.0 | 2298.2 | **1.129×** |
| 128 | 5116.1 | 4469.3 | **1.145×** |

Over the 24 cells above the floor — the ones the rule actually acts on —
**1.134×**; over all 48, 1.124×.

Per shape at R=128, where the rule picks `sk=1` and the reduce is not launched
at all: qwen3_moe/down **1.305×**, olmoe/down 1.241×, granitemoe/down 1.240×,
granitemoe/gate_up 1.111×, qwen3_moe/gate_up 1.101×, olmoe/gate_up 1.080×.

`SK_R_BOUND = 1.07` in `kernel/test_int4_b32.py` bounds the worst acted cell
(measured 1.064×, granitemoe/down at R=16); the test re-derives it from these
receipts on every run.

## Choice of constant

`8 × sm_count` was picked by scoring the whole 48-cell set, not fitted per
cell. Neighbouring targets are close on totals and differ mainly in worst
case:

| target | total vs oracle | worst cell | cells worse than N-only |
|---|---|---|---|
| 2 × sm | 1.010× | 1.203× | 10 |
| 4 × sm (the NF4 sibling's) | 1.008× | 1.094× | 6 |
| **8 × sm** | **1.008×** | **1.078×** | **2** |
| 16 × sm | 1.012× | 1.192× | 1 |

The sibling spends `4 × sm_count` at `BLOCK_N=64`; this kernel's blocks are
twice as wide, so it covers the same N with half the tiles. That the measured
optimum lands near twice the sibling's target is consistent with that, but the
sweep is the evidence, not the arithmetic.

## Scope, and what this does NOT claim

- **Decode is untouched.** `R < 16` returns exactly the old N-only value, on
  every SM count — pinned by `test_plan_decode_is_unchanged_by_the_r_term`
  against a verbatim copy of the old rule, across *every* row count below the
  floor rather than just R=1. See the floor section: "M=1" does not mean
  "R=1" at this call site, and reading it that way is what the floor fixes.
- **sm_86 only.** The target is measured on one box class. sm_120 is where the
  census licensed this kernel and it is not re-measured here. The constant is
  named, not inlined, for that reason.
- **Kernel-level, not step-level.** These are the expert-projection GEMVs
  timed in isolation at a given row count. What that is worth in a real
  decode step is a separate measurement and is not claimed here.
- **MXFP4 is deliberately not changed.** `mxfp4_grouped.gemv_mxfp4_b32` shares
  this planner and has the same grid, so the same argument applies to it — but
  its inner loop is e2m1, not int4 nibbles, and it was not swept. It keeps
  calling `_plan(N, K)`, which still means decode.
- **`sm_count` enters the config**, so two boxes with different SM counts can
  pick different `sk` for the same call, and split-K changes the grouping of
  the fp32 partial sums. That is the sibling planner's existing bargain rather
  than a new one, but anything asserting bitwise equality *across* boxes must
  pin `sk` rather than plan it. Within a box the plan stays a pure function of
  `(N, K, R)`.

## One thing found and not actioned

At `R = 1` — the licensed decode config, untouched here — the N-only rule is
itself off the A2000 optimum by 1.048×–1.482× (worst: olmoe gate_up, plan
sk16 at 10.2 µs against sk8 at 9.6 µs; qwen3_moe gate_up 14.4 vs 13.4). That
is a config licensed on sm_120 being measured on sm_86, which is not evidence
against it. Recorded, not acted on.

## Harness note — a correction

The first write-up of this section blamed an illegal memory access on "capturing
~40 CUDA graphs in one process". That was wrong, and the real cause was mine: the
harness passed the activation scales as `xs [R, 1]` where the kernel indexes
`xs_ptr + e*KB + kb0 + ku` — per-(row, 32-block), `[R, K//32]`, exactly what
`quant_x_rows` produces. The kernel therefore read `R*(KB-1)` floats past the
buffer: harmless garbage on small shapes, an illegal memory access on the
largest (olmoe gate_up, K=2048, R≥64), surfacing on the *next* shape's
`manual_seed` because the fault is asynchronous. The "fresh process passes"
observation that seemed to implicate graph pools was allocator placement luck.

With `xs [R, K//32]` the full 48-cell sweep ran clean with no fault. Because the
kernel's cost is data-independent (same loads, same integer MACs regardless of
the scale values), the timings were expected to be unchanged, and they were
re-measured rather than assumed: per-(cell, sk) ratio corrected/original median
**1.0012**, p10–p90 0.988–1.016 (one 1.53× outlier on a ~9 µs first-cell
warm-up, granitemoe/down R=1 sk=1). The verdict is identical on the corrected
data — 1.127× over the N-only plan, worst acted cell 1.064×, 0 of 48 cells
slower — and **the receipts in `rows/` are the corrected-harness ones**; the
originals were not kept because a receipt produced by a faulty harness should
not be the one that ships when a clean one exists.

Two disclosures that belong here: the sweep still runs one shape per process
(cheap, and it keeps any future fault attributable), and the A2000 was shared
with other containers holding ~8.9 GB of VRAM at 0 % utilisation throughout —
no SM contention, but the card was not empty.
