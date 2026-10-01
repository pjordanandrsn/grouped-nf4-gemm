# K27 — does K25 with the select tree keep its speed at the served kernel's weight precision? TF32 against bf16 at the NF4 families' B=16 shapes on one RTX 5090 (registered 2026-10-01, before any 5090 run)

Issue: experts4bit-qlora#564.

**Why now.**
- **P92** (experts4bit-qlora `bench/p92/RESULTS-p92.md`) read K25 QUALITY_FAIL on OLMoE: its c4val1 K8 moved −0.107 ppl.
  K25 rounds the weight to bf16; the served NF4 GEMM multiplies TF32 on the fp32 weight.
- **K26** (`kernel/RESULTS-k26-nf4-decode-ablation.md`) read the codebook lookup as about 80 % of K25's time. Its exact
  select tree is bit-identical at 0.373 / 0.383 of it, about 0.39× the served kernel. The tree is now K25's default
  (#433).
- **K25's `dot_bf16=False`** keeps the fp32 weight and runs TF32 MMA: the served kernel's precision class. If the tree
  keeps enough of its speed there, both NF4 families can be read for a default.

## The bench (`kernel/k27_bench.py`)

- **Harness.** K26's: one RTX 5090, the NF4 families' expert shapes at B=16 / top-8 over each layer's own synthetic
  stores (Granite 40 × 32 layers, OLMoE 64 × 16), seeded synthetic routing over 8 steps, both GEMMs per layer in one
  CUDA graph, median of 20 replays.
- **Arms:**
  - `pair16`: K25 as P92 ran it (paired lookup, bf16), at the default plan;
  - `tree16`: the tree, bf16;
  - `tree32`: the tree, fp32 weights through TF32 MMA;
  - `served`: `gemm_4bit_grouped_captured`.

  Each tree arm takes the best of six plans (BLOCK_N / KC / warps / stages: 32/256/4/2, 32/128/4/2, 32/128/4/3,
  64/128/4/2, 32/64/4/3, 16/256/4/2), selected on steps 0–3 and read on steps 4–7. `pair16` and `served` run at their
  fixed plans on steps 4–7.
- **The error proxy.** Every row of step 0, layer 0, both projections, against an fp64 product of the fp32 dequant; rms
  error per arm.

**The rule** (`verdict`, 9-case self-test):
- **VOID** if any of these holds:
  - `tree16` is not bit-equal to `pair16`;
  - the served kernel's error is more than 2× `pair16`'s (instrument);
  - an arm is missing;
  - a family was not measured.
- **TF32_PATH** if `tree32 / served` ≤ **0.60** and `err(tree32) / err(served)` ≤ **1.10** in both families. The next
  lane reads K25-tree in TF32 end to end on both families, as a default candidate.
- **BF16_ONLY** otherwise, if `tree16 / served` ≤ 0.60 in both. The next lane reads K25-tree in bf16 end to end, for the
  families whose K8 already passed (Granite).
- **NONE** otherwise.

## What was seen before this page (stated, not hidden)

The bench's correctness pass ran on the NAS RTX A2000 (sm_86) in `--quick` mode (2 layers), with the tree branch
(#433's head). It is not a reading, because the A2000 is correctness-only, but it was seen:
- every plan of both tree arms ran;
- `tree16` was bit-equal to `pair16`;
- `err(tree32) / err(served)` = 1.000 (Granite 2.532e-4 against 2.532e-4; OLMoE 3.269e-4 against 3.269e-4), against
  1.36× for the bf16 arms;
- `tree32 / served` 0.46 (Granite) and 0.64 (OLMoE); `tree16 / served` 0.37 and 0.49.

That pass would have read BF16_ONLY, because OLMoE's TF32 arm missed 0.60.

The runner's rehearsal on the same A2000 (experts4bit-qlora `bench/k27/`, marked REHEARSAL, full mode with 32 / 16
layers) was seen too. It also read BF16_ONLY:
- `tree32 / served` 0.630 (Granite) and 0.637 (OLMoE);
- `tree16 / served` 0.465 and 0.475;
- the error ratio 1.000 in both.

## Predictions

- **The error.** `err(tree32) / err(served)` 0.95–1.05 in both families: the same precision class.
- **Speed.** `tree32 / served` 0.45–0.65. The sm_120 tensor cores' TF32 rate and K26's DECODE margin (the lookup was
  80 % of the time) leave room, but OLMoE is the tighter family, as on the A2000.
- **`tree16 / served`** 0.35–0.45, as K26 read (0.39 / 0.40).
- **The verdict:** TF32_PATH and BF16_ONLY are both plausible. I lean TF32_PATH at about 55 %. The 5090 read the bf16
  tree at 0.39 / 0.40 of the served kernel where the A2000 reads 0.47 / 0.48. Scaling the A2000's TF32 0.63 / 0.64 the
  same way gives about 0.53, under the bar.

## Box and cost

- **`k27-5090-<n>`:** one RTX 5090. **Guard 0.5 h at ≤ $0.75/h (≤ $0.375)**, so there is no proving rental.
- Driven from experts4bit-qlora `bench/k27/`: K26's runner with the bench renamed, the same premise (K25's contract
  compiled on the card, rc 23) and no model.
- **Lane ceiling $0.75; hard stop $1.00.**

Amendments, dated, go below this line before any 5090 data is read.
