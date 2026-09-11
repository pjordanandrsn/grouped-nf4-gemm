# PREREG — K15: what does a Marlin-class 4-bit GEMM cost at M=16 on our
# projection shapes, and is its advantage the FORMAT or the KERNEL?

Registered 2026-09-11, before measurement. Writes no kernel.

## Why

experts4bit-qlora#564 decomposes the 4.05 ms gap to vLLM at B=16. The single
largest addressable item is the attention projections:

| | weight bytes / step | roofline at 1528 GB/s | measured |
|---|---|---|---|
| ours, **dequantised to bf16** | 1.812 GB | 1.186 ms | **1.977 ms** (K14) |
| vLLM, GPTQ-Int4 g128 | 0.467 GB | **0.306 ms** | — |

≈1.67 ms of the gap, and K14 established that **no int4 arm we ship beats our
own dequant path** at M=16: our grouped GEMM runs at 22 % of the bandwidth
ceiling where the bf16 path runs at 106 %.

vLLM completes the entire step in 7.882 ms against our 11.93, with Marlin
kernels on a `Qwen/Qwen3-30B-A3B-GPTQ-Int4` checkpoint (P37's pre-registration
names both). So a 4-bit GEMM that wins at M=16 demonstrably exists. **Before
writing one, measure what one costs on exactly our four shapes.** That is the
discipline that made K14 worth its 6.5 cents, and K11 before it.

## Design

Two venvs on one box, the shape P37 used to put the engines side by side —
vLLM 0.28.0 brings torch 2.13.0+cu130 and our stack is on 2.8.0+cu128, so they
cannot share a process.

- **ours**: `k14_bench.py` unchanged — `bf16` (dequant-then-cuBLAS), `gemv`,
  and the swept `gemm`, on the four Qwen3-30B-A3B projection shapes.
- **theirs**: `k15_marlin_bench.py` — `apply_gptq_marlin_linear`, the function
  vLLM's own GPTQ linear layer calls, at **two group sizes**:

| group size | bits/weight incl. scales | question it answers |
|---|---|---|
| **128** | ~4.125 | what vLLM actually gets |
| **32** | ~4.5 — **our** format | is the advantage the format or the kernel? |

Both sides time under CUDA-graph replay, both measure their own launch floor and
their own streaming ceiling with the same 512 MB copy, so the GB/s columns are
against comparable references.

Correctness is checked in-process against the dequantised reference Marlin's own
quantiser returns, per cell. It could **not** be checked beforehand on the house
A2000 — that host's driver is 12.9 and this torch build needs 13.0 — so a wrong
call would show up as a large relative error rather than a silent number.

## Registered predictions, each with what refutes it

1. **Marlin g128 beats our bf16 path at M=16 on `q_proj`.** Ours measured
   10.35 µs. *Refuted by* Marlin ≥ 10.35 µs there.
2. **Marlin g128 lands within 2× of its own byte-roofline**, i.e. within 2× of
   `max(launch floor, weight bytes / streaming GB/s)`. *Refuted by* a ratio
   above 2, which would mean Marlin is far from memory-bound too and the 0.306
   ms figure in #564 is not reachable by anyone.
3. **At matched bytes (g32), Marlin still beats our best int4 arm.** Ours
   measured 12.47 µs on `q_proj` at its best of twelve configs. *Refuted by*
   Marlin g32 ≥ 12.47 µs — which would mean the advantage is the weight format,
   not the kernel, and the lever is a repack rather than a rewrite.

Prediction 3 is the one I actually care about and the one I am least sure of.

## The decision rule, fixed before the numbers

Per-step projection cost over 48 layers × (q + k + v + o), against our measured
**1.977 ms**:

- **Marlin g128 saves ≥ 1.0 ms/step** → the 1.67 ms in #564 is real and the
  next question is adopt-or-reimplement, with its own lane.
- **saves < 0.5 ms/step** → #564's map is wrong about its largest item and gets
  redrawn before any kernel work is proposed.
- **between** → recorded as inconclusive at this M, and the lane says so rather
  than rounding toward the answer it went looking for.

## What this lane does NOT establish

- **It is not a controlled kernel comparison.** The two sides run different
  torch and CUDA builds in different processes. It compares *engines as
  shipped*, which is what P37's head-to-head measures and what the gap is
  quoted against — but no conclusion here may be stated as "kernel X is faster
  than kernel Y, all else equal", because all else is not equal. The g32 arm
  narrows this by matching bytes; it does not remove it.
- **It authorises no adoption.** Using Marlin would mean a GPTQ-format repack
  and a dependency on vLLM or a vendored kernel, with licence and packaging
  consequences, and a quality gate on the repacked weights. None of that is in
  scope here.
- **It says nothing about the expert tier**, which #564 shows is at 80–100 % of
  its memory roofline and whose distinct-expert count is still unmeasured.

## Budget and STOP rules

- **One box, ≤ $0.35, ≤ 45 min.** Lane ceiling $1.00, hard stop $2. No model
  download, no calibration: two pip installs and two kernel loops.
- **STOP-1** — a relative error above 0.05 on any Marlin cell: the call is
  wrong, the cell is void, and no timing from it is quoted.
- **STOP-2** — launch floor above 5 µs on either side: the box is too noisy to
  separate launch-bound from bandwidth-bound and the run is void.
- **STOP-3** — no second box on a disappointing result.

## Receipts

`receipts-k15/`: `k15_marlin_rows.json`, `k14_rows_same_box.json`, both stdout
logs, `forensics.txt`, `versions.txt` for each venv.
