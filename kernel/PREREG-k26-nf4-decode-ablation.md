# K26 — is the NF4 codebook decode what holds K25 (and the served NF4 GEMM) back? A decode ablation on the NF4 families' B=16 shapes on one RTX 5090 (registered 2026-10-01, before any 5090 run)

Issue: experts4bit-qlora#564.

**Why now.** experts4bit-qlora lane P92 (`bench/p92/RESULTS-p92.md`, #829) read K25 in-model. Its GEMM ran within 4 %
of the served NF4 GEMM on the families' B=16 steps: Granite 6.464 vs 6.602 ms/step, OLMoE 10.852 vs 10.388. Granite's
×0.937 came from the glue. K25 shares K21's skeleton, and K21 reached about 50 % of its byte floor on MXFP4. What K25
does not share is the decode: K21 decodes with integer shifts, while both NF4 kernels look each nibble up in a codebook.
This lane tests that explanation before any kernel is built for it.

## The bench (`kernel/k26_bench.py`)

- **Shapes.** One RTX 5090. Each family's expert shapes, at B=16 with top-8, over each layer's own synthetic NF4
  stores (K24's lesson):
  - Granite: 40 experts, 32 layers, gate_up 1024 × 1536, down 1536 × 512;
  - OLMoE: 64 experts, 16 layers, gate_up 2048 × 2048, down 2048 × 1024.
- **Routing.** Seeded synthetic top-k routing, 8 steps, each layer its own ids.
- **Timing.** Each arm times only its two GEMMs per layer in one CUDA graph (tile tables and inputs built outside),
  as the median of 20 replays per step, then the median over steps.
- **Floor.** The step's active-expert bytes (packed + fp32 absmax) over a device-copy bandwidth.
- **Arms:**
  - `pair`: K25 as merged (#429, `8cc3510`) at its default plan (BLOCK_N 32, KC 256, 4 warps, 2 stages, the paired
    codebook);
  - `copy`: a bench-local copy of K25's kernel with the same decode. The control: bit-equal outputs, time within 5 %;
  - `affine`: the copy with the codebook replaced by `nibble − 8`. The same bytes and absmax, no lookup. Its outputs
    are wrong by design: it exists only to price the lookup;
  - `noscale`: `affine` without the absmax load and multiply;
  - `bytes`: the packed bytes widened straight to the operand (no nibble split, no scale);
  - `tree`: the copy with the lookup replaced by a 4-level select tree on the nibble's bits over the 16 fp32 codebook
    values. These are the same weights, so it is a candidate exact decode;
  - `served`: `nf4_grouped.gemm_4bit_grouped_captured` (VARIANT 1, TF32, BLOCK_K 64), the route P92's OFF arm ran;
  - `scopy` / `stree`: a bench-local copy of the served kernel's path (the control), and that copy with the select
    tree in place of its `tl.gather` codebook;
  - `load`: K25 with `lut="load"` at KC 128. It overflows shared memory at KC 256. Descriptive.
- **Numerics, before timing:**
  - K25 against the fp32 dequant oracle on 8 rows;
  - `copy`, `tree` (both projections), `scopy` and `stree`, each compared bit for bit with the kernel it copies.

**The rule** (`verdict`, 16-case self-test):
- **VOID** if any of these holds:
  - the numerics fail;
  - `copy` is not bit-equal to K25, or its time is more than 5 % from K25's;
  - an arm is missing;
  - a family was not measured.
- **DECODE** if `affine / pair` ≤ **0.60** in both families: the codebook lookup is the bottleneck.
- **NOT_DECODE** if it is ≥ **0.85** in both.
- **MIXED** otherwise.

**The registered pointer** (`next_lane`). An exact decode qualifies only when it is bit-equal to the kernel it replaces
(then it is a pure speed change and needs no quality read) and fast enough in both families:
- **`tree`** bit-equal and `tree / pair` ≤ **0.80**: K25 takes the select-tree decode. P92's reading of K25 applies to
  it bit for bit, including Granite's K8.
- **`stree`** bit-equal and `stree / served` ≤ **0.90**: the served NF4 kernel takes it, with no arithmetic change for
  any family.

## What was seen before this page (stated, not hidden)

The bench's correctness pass ran on the NAS RTX A2000 (sm_86) in `--quick` mode: 2 layers per family. Its timings are
not a reading, because the A2000 is correctness-only, but they were seen:
- `affine / pair` ≈ 0.13 in both families;
- `tree / pair` ≈ 0.48;
- `stree / served` ≈ 1.07–1.10.

Every numerics check held, except that **`stree` is not bit-equal to the served kernel** on the A2000. The tree's
weights are exact; the layout it leaves changes how the dot lowers, as `tl.gather` did in K25. The predictions below
are informed by that pass.

## Predictions

- **DECODE.** `affine / pair` 0.10–0.40 in both families. The sm_120 load path may hide some of the lookup that sm_86
  could not.
- **`tree` bit-equal; `tree / pair` 0.40–0.75.** The next lane is K25 with the select tree.
- **`stree` not bit-equal** (as on the A2000). The served kernel takes nothing.
- **`served / pair` 0.95–1.10**, as P92's census read (0.96–1.04 in-model).
- **The copy controls hold.**

## Box and cost

- **`k26-5090-<n>`:** one RTX 5090. **Guard 0.5 h at ≤ $0.75/h (≤ $0.375)**, so there is no proving rental.
- The runner installs grouped-nf4-gemm at this lane's merge and runs K25's contract compiled on the card (the
  premise, rc 23). It then runs the bench, which needs no model and no fetch.
- **Lane ceiling $0.75; hard stop $1.00.** Driven from experts4bit-qlora `bench/k26/`.

Amendments, dated, go below this line before any 5090 data is read.
