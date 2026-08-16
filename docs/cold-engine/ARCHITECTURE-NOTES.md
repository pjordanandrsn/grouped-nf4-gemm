# Hybrid CPU/GPU execution tier — architecture notes (gnf4 side)

Pre-work map for the hybrid-tier program (Stage 1 Phases 0–5, gated G0–G5):
where each phase hooks into this repo, which existing seams are reused, and
which decisions are still open. Runtime-side notes live in
`experts4bit-qlora/docs/hybrid/ARCHITECTURE-NOTES.md`; kernels and calibration
live here, per the placement rule (kernels → gnf4, runtime → e4b).

## Lineage: this is the cold-engine program, continued

`docs/cold-engine/PHASE0-premise.md` already measured the phase-0 floors
(naive torch NF4 decode 0.067 GB/s packed = 0.6% of the AVX2 box's ~12 GB/s
ceiling; bnb CPU 0.041 GB/s) and concluded the native kernel is *mandatory*.
The hybrid directive supersedes one of its conclusions: the cold engine was
framed as a **thin-link instrument** ("CPU tier adds little where pipe is
fat — NOT the target"). That held for an AVX2 2-channel box whose kernel
grade (~6–10 GB/s) barely cleared its own link. On a many-channel AVX-512
host, grouped DRAM reads sit an order of magnitude above even a Gen5 x16
link, so DRAM becomes a general *compute tier*, not a last resort. The gate
that decides this per-box is G0 (grouped-scatter as % of STREAM triad), which
formalizes the premise doc's "the kernel grade is the load-bearing unknown."

Phase-numbering reconciliation (avoid two "Phase 2"s):

| hybrid directive | this repo's cold-engine ladder |
|---|---|
| Phase 0 calibration + G0 | extends phase-0 premise (`bench/cold-engine/phase0_*.py`) |
| Phase 2 AVX-512 grouped GEMV/GEMM | the "mandatory ggml-skeleton port", upgraded AVX2→AVX-512 |
| Phase 3 placement solver (e4b) | ADDENDUM-1 A3's `plan_placement()` registration |

New in-repo artifacts use the `hybrid` prefix; gates are G0–G5.

## Phase 0 — calibration (this repo)

- `bench/calibrate.py` — orchestrator: hardware detection (CPUID flags incl.
  AVX512F/VBMI/VNNI, L3/CCD topology from sysfs, THP/hugetlb state, governor,
  GPU inventory + power limit), runs each bench serially, assembles the
  machine-readable calibration blob (`schema: gnf4-hybrid-calib/1`), prints
  the G0 verdict. Torch-side: `B_vram` device triad per GPU, `B_link` pinned
  H2D/D2H at 8 KB and 64 MB both directions.
- `bench/hybrid_calib.c` — plain C11 + pthreads microbench (built on the
  target with `cc -O3 -march=native`): STREAM triad (regular + NT stores,
  thread ladder, compact-vs-spread CCD pinning, first-touch by the timing
  partition), the **G0 gate workload** (k=8 distinct MB-scale expert blocks
  per fetch from a fixed-seed routing trace over E≥128, read-reduced, thread
  and block-size sweep), and O_DIRECT NVMe seq/rand read (`B_nvme`).
- Cross-checks: triad agrees with `bench/cold-engine/phase0_ddr_bench.py`
  (torch-based, not certified STREAM — same caveat applies and is named in
  the receipt); NVMe agrees with `bench/nvme/nvme_microbench.py` where both
  run.
- House discipline: PREREG with the gate thresholds filed and stamped before
  the measured run; receipts JSON banked beside the existing
  `bench/cold-engine/receipts-*.json`; RESULTS reports misses exactly like
  passes. Roofline rule: every later performance claim is achieved GB/s
  against THIS blob's ceilings, never spec sheets.

## Phase 2 — AVX-512 grouped expert GEMV/GEMM (this repo)

Planned modules, following the flat `kernel/` grammar
(`<format>_grouped` / `<format>_pack_ref` precedent):

- `kernel/cpu_grouped.py` — public entry points mirroring the GPU wrappers:
  grouped GEMV/GEMM over packed NF4 (`B [E,N,K//2] u8` + `absmax [E,N,K//64]
  f32`) and MXFP4 (`blocks` + e8m0 `scales`), `sizes`/`expert_ids` grouping,
  FP32 accumulate, fixed reduction order. **Separate entry point, not a
  relaxed CUDA guard**: `gemm_4bit_grouped`'s cuda-or-interpreter refusal
  (`kernel/nf4_grouped.py:480`, pinned by `kernel/test_cpu_refusal.py`)
  stays exactly as is.
- `kernel/cpu_dispatch.py` — runtime ISA detection + compile-at-first-use of
  the native kernel (`cc -O3 -march=native`), cached artifact keyed on
  (source hash, compiler, ISA); scalar/pure-torch fallback = the existing
  oracles, so every box can run the path slowly but correctly.
- Kernel bit-exactness oracles already exist and are the acceptance test:
  `nf4_grouped.dequant_ref` (+ `nf4_pack_ref.make_stack`) and
  `mxfp4_pack_ref.dequant_mxfp4` / `quantize_pack_mxfp4`. CPU output must
  match reference dequant + fp32 matmul exactly at fp32 accumulation, with
  the FMA/reduction order documented and locked.
- Two format facts a CPU kernel must not get wrong (both oracle-pinned):
  **nibble order is opposite** — NF4 decodes high nibble first
  (`nf4_grouped.py:172`), MXFP4 low nibble first
  (`mxfp4_pack_ref.NIBBLE_LOW_FIRST`); and scales differ — NF4 fp32 absmax
  per 64, MXFP4 u8 e8m0 per 32 (`exp2(s-127)`, `0xFF`→ldexp semantics).
- Row layouts are already CPU-legible: the engine's four `as_strided` views
  over one `[k, row_bytes]` u8 buffer (`mxfp4_pipelined._init_geometry`,
  8-byte-aligned segment offsets) and the arena's identical derivation
  (`nvme_arena`, `row_stride` = 4096-aligned `row_bytes`;
  `test_mxfp4_arena_layout.py` guards the agreement). A CPU backend consumes
  the same packed bytes with zero repacking — that is the one-artifact
  invariant made concrete.
- `ColdTier` already supports the CPU tier's memory mode:
  `alloc_landing(pinned=False)` mmap path. The `Mxfp4NvmeResidency` refusal
  of an unpinned tier (`mxfp4_residency.py:381` — correct for the GPU
  address-gather) does NOT apply to a CPU consumer; a CPU-side engine
  branches there rather than relaxing it.
- Stage-2 forward-compat (invariant 8 / Phase 8): the API is grouped
  GEMV *and* small-N GEMM from day one — same entry point, `sizes` > 1 means
  tokens-per-expert > 1. Bandwidth-bound through B≈64; no compute tuning yet.

### Packaging decision (must be resolved in the Phase 2 PR)

The wheel is flat `py-modules` out of `kernel/`; it cannot carry data files,
and repo-root `backends/` currently **ships in no wheel** (not in
`py-modules`; the packaging guard globs only `kernel/*.py` so it cannot see
the omission). Options: (a) embed the C source as a python module string
(wheel-safe, no packaging change, ugly at kernel scale); (b) add a proper
package entry (`packages=["gnf4_native"]` + per-package `package-dir`) whose
`package_data` carries `.c` sources, and widen
`test_packaging_covers_kernel.py` to cover it — which also fixes the
`backends/` gap. Leaning (b); decide in-PR with the guard extended either way.

## Hooks a CPU residency engine reuses (Phase 3 consumes from e4b)

`Mxfp4PipelinedGptOss` is a 5-step template with per-step hooks; the NVMe
subclass already proves the extension pattern. For a DRAM-compute tier the
relevant seams are `_resolve_src` (a CPU tier needs no device address),
`_gather` (no-op / host memcpy when bytes are already resident), `_glu`
(device-agnostic already), `_init_geometry` (reused verbatim). The
address-vs-contents invalidation lesson (`_invalidate`; a slot's address
does not identify its contents) applies to ANY cache the CPU tier adds.

## Test/CI obligations for every new file here

- Every `kernel/test_*.py` must be named in `ci.yml` or listed in
  `_NOT_IN_CI` with a reason (`test_packaging_covers_kernel.py` enforces).
- New modules go in `pyproject.toml` `py-modules` or the guard fails.
- CPU-kernel modules must not import torch/triton at module top (the
  `nvme_*` convention) so they stay importable on macOS/Windows.
- README: new row in "Which entry point? Pick by where the weights live",
  update the cold-engine roadmap paragraph, version-pinned permalinks only
  (`scripts/check_readme_links.py`), and the CPU quickstart marker block
  count is asserted by `test_readme_cpu_block.py` — extend deliberately.
- Determinism: fixed reduction order per backend; same seed + same placement
  manifest ⇒ bit-identical logits per backend on the same hardware.
  Cross-placement variance is documented tolerance only.

## Stop conditions (verbatim from the directive)

G0 <50% · any invariant requires violation · determinism unachievable in a
phase · a dependency forces a weight-format change · CPU router disagrees
with GPU reference beyond tie-break cases. Halt and report; do not improvise.
