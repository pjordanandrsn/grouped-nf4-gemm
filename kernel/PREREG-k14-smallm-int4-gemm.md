# PREREG — K14: at M=16, does an int4 path beat dequant-then-GEMM on the
# attention projection shapes?

Registered 2026-09-11, before measurement. Stage A writes **no kernel**.

## Why this lane exists

P42 (experts4bit-qlora `bench/p42/RESULTS-p42.md`, receipts `p42-census-5`)
measured what the int4 attention store costs at B=16 on an RTX 5090:

| family | nf4 B=1 | int4 B=1 | nf4 B=16 | int4 B=16 |
|---|---|---|---|---|
| cuBLAS `gemvx` (M=1) | 1.954 ms / 241 calls | **0.494 / 49** | 0 | 0 |
| CUTLASS bf16 (M>1) | 0 | 0 | 2.577 / 241 | **2.604 / 241** |

The store is worth 1.46 ms/step at B=1 and **0.027 ms worse than nothing** at
B=16, because `Int4Linear` serves `R > GEMV_ROWS_MAX` (1) by dequantising the
weight once, caching the bf16 copy and calling cuBLAS. It also holds both
representations resident — 1.812 GB bf16 + 0.453 GB int4 for
`Qwen/Qwen3-30B-A3B` — while reading only one. Filed as
experts4bit-qlora#561.

## The premise I checked before proposing a build

`Int4Linear`'s own comment says:

> A batched int4 attention that WINS needs a small-M int4 GEMM with weight-tile
> reuse; not this module.

**That kernel already exists in this repository.**
`gemm_int4_b32_grouped_captured` (`int4_b32.py:335`) is an M-tile grouped int4
GEMM — int8 MMA through `tl.dot`, exact integer accumulation, one fp32 scale
product per (row, k-block), every launch parameter static so the call is legal
under graph capture. It was built for the expert path and has never been pointed
at a single projection.

A projection is the **E = 1** case of it: one group, M rows, local expert index
0 — which is precisely the shape of `Int4Linear`'s own
`packed [1, N, K//2]` / `scales [1, N, K//32]` buffers. So the candidate is a
call site, not a kernel.

K9 was registered, written and voided because its subject was never called.
K11 closed by reading the PTX rather than building. This lane follows both: the
cheapest question first, and the build only if the measurement asks for it.

## What K11 already settled, in this lane's favour

K11 found the emitted instruction is `mma.sync.aligned.m16n8k16` — the M extent
is **fixed at 16 by the hardware**, and it closed the T=1 case because a
one-row GEMV wastes 93.8 % of that tile. It named batch as the one filler that
lives in a different lane. **At B=16, M is exactly 16.** The tile that could
not be filled for single-stream decode is exactly full here. K11 is therefore
not evidence against this lane; it is the reason the lane is worth asking.

## Stage A — three shipped paths on four real shapes

`kernel/k14_bench.py`. Shapes are `Qwen/Qwen3-30B-A3B` @ `ad44e777…` as
`Int4Linear` stores them (N = out_features, K = in_features):

| | N | K | bf16 weight | int4 weight |
|---|---|---|---|---|
| `q_proj` | 4096 | 2048 | 16.78 MB | 4.19 MB |
| `k_proj` | 512 | 2048 | 2.10 MB | 0.52 MB |
| `v_proj` | 512 | 2048 | 2.10 MB | 0.52 MB |
| `o_proj` | 2048 | 4096 | 16.78 MB | 4.19 MB |

Arms, M ∈ {1, 4, 8, 16, 32}:

- **`bf16`** — `x @ dequant(packed).t()`. What ships above M=1.
- **`gemv`** — `quant_x_rows` + `gemv_int4_b32` at R=M. What `GEMV_ROWS_MAX = 1`
  forbids.
- **`gemm`** — `quant_x_rows` + `gemm_int4_b32_grouped_captured` with a
  single group. The kernel nobody called here.
- **`floor`** — one trivial kernel under graph replay. A shape whose baseline
  already sits at the launch floor cannot be helped by reading fewer bytes,
  however few, and that has to be visible rather than argued.

Every arm is timed under **CUDA-graph replay, never eager** (`int4_b32`'s own
rule: the eager host floor hides everything) and carries its **full** cost
including its reduce and its activation quantisation. An arm timed without its
quantisation is a kernel time, not a path time.

The `bf16` arm dequantises the **same packed bytes** rather than using the
original weight, so all three arms compute the same underlying product.

## Registered predictions, each with what refutes it

1. **`gemm` wins at M=16 on the large shapes.** `gemm/bf16 < 1.0` for `q_proj`
   and `o_proj` at M=16. *Refuted by* a ratio ≥ 1.0 on either.
2. **`gemv` loses at M=16 everywhere**, i.e. `GEMV_ROWS_MAX = 1` is correct and
   not merely inherited. *Refuted by* `gemv/bf16 < 1.0` on any shape at M=16.
3. **The small shapes are launch-bound.** `k_proj` and `v_proj` `bf16` time at
   M=16 is within 2× of the launch floor, and no int4 arm improves them by more
   than 10 %. *Refuted by* a `bf16` time above 2× the floor there.

I expect 1 and 3 to hold and 2 to hold; if 2 fails, the cheapest fix in this
whole campaign is a constant.

## The decision rule, fixed before the numbers

Stage A's verdict is computed on the **real mix**, not on the best cell: one q,
one k, one v, one o per layer × 48 layers,

```
saving_ms_per_step = 48 * sum_over_shapes( bf16_ms - min(bf16_ms, gemv_ms, gemm_ms) )
```

- **≥ 0.5 ms/step** → Stage B: a block-config sweep for the winning arm on
  these shapes, and the `Int4Linear` call-site change, each with its own gate.
- **< 0.5 ms/step** → the lane closes REFUTED and `GEMV_ROWS_MAX = 1` plus
  dequant-then-GEMM are **confirmed correct by measurement** rather than
  inherited. That is a receipt where there was an assumption, and it is a
  result, not a failure.

0.5 ms is ~4 % of the 12.849 ms step and ~10 % of the 4.967 ms gap to vLLM
`graph_r1`; below that the call-site change is not worth the serving risk in
the next section.

## What Stage A does NOT authorise

Adopting an int4 arm at M>1 **changes what B=16 serving computes**. Today the
batched path multiplies bf16 activations by a dequantised weight; the int4 arms
quantise activations to int8. The shipped decode path already accepts that at
M=1, so it is not a new class of concession — but it is a new place for it, and
Stage A measures speed only.

**No adoption without a perplexity gate** on the two registered texts, the same
K8 two-text gate the pack lanes use. Stage A reports the relative error per
shape as an observation; it does not clear anything.

## Budget and STOP rules

- **One box, ≤ $0.40, ≤ 45 min.** Lane ceiling $1.50, hard stop $3. No model
  download, no calibration, no bake: this is `pip install` and a kernel loop.
- **STOP-1** — any arm raising on a shape: recorded as an error cell, the run
  continues, and the mix verdict is computed only over shapes where all three
  arms answered. A verdict over a partial mix says so.
- **STOP-2** — `floor` above 5 µs: the box is not quiet enough to separate
  launch-bound from bandwidth-bound, and the run is void rather than
  interpreted.
- **STOP-3** — no second box on a disappointing result. If Stage A refutes the
  lane, that is the finding.

## Receipts

`receipts-k14/` : `k14_rows.json` (every cell, every arm, ms + GB/s + relative
error), the stdout log, `forensics.txt`, `versions.txt`. `RESULTS-k14-*.md` is
written from those and from nothing else.
