# PREREG — rung 3 / K1: the M=1 decode config on sm_120 (gnf4)

Registered 2026-08-24, before any measurement. The rung the b1d cert
licensed: the graph-captured single-stream step is 15.19 ms of nearly
pure device work, and the per-kernel budget (e4b
`receipts-b1d/b1dc_eager_kernels.txt`, #216 attribution convention)
names the mountain:

| kernel | ms/step | calls/step | roofline note |
|---|---|---|---|
| `_gemv_nf4_grouped` (expert GEMV) | **4.84** | 96 | reads ~1.02 GB; floor 0.65 ms @ 1573.9 GB/s ⇒ **13% of roofline** |
| cuBLAS gemvx (dense) | ~1.93 | 241 | ~72% of roofline — low headroom |
| elementwise storm | ~2.5 | ~1900 | fusion lane, NOT this prereg |
| attention (`_fp8_*`) | ~0.57 | 96 | fine |

## The two stages of rung 3 (both registered now)

- **K1 (this cycle)**: the decode launch plan `(BLOCK_N=64, warps=2,
  split_k≤8)` is a universal constant from two-device sweeps that
  never saw sm_120 (170 SMs) or the collapsed M=1 shape census —
  exactly two shapes at B=1: gate_up `(N=1536, K=2048, T=8)` and down
  `(N=2048, K=768, T=8)`, 48× each per step. Sweep the EXISTING
  ablation space — `BLOCK_N ∈ {32, 64, 128, 256} × warps ∈ {2, 4, 8}
  × split_k ∈ {1, 2, 4, 8, 16}` (constraint-filtered) — on real-shape
  synthetic tensors (bandwidth is value-blind), pick the per-shape
  winner by registered rule (min summed median over the two shapes;
  tie → fewer programs), bake it as an sm_120+M=1-guarded branch of
  `_decode_plan`, and confirm end-to-end through the b1d graph
  harness. Config knobs only; zero kernel-body edits.
- **K2 (registered, runs after K1's verdict)**: the kernel-body
  variant work the load pattern demands — each packed byte is loaded
  twice (`kk//2` per-element uint8 loads), no vectorization; a
  uint32/uint4-vectorized dual-nibble mainloop targets the remaining
  gap to roofline. Bars set after K1's receipts fix the baseline.

## K1 instruments

- `bench/m1_decode_sweep.py` (gnf4): CUDA-event timed, 50-iter warmup
  + 200-iter median per config per shape, current-plan baseline row
  included, JSON out. Runs on the same box class as every cert
  (NUMA-pre-gated EPYC + RTX 5090).
- `_decode_plan` gains a SHAPE-KEYED env override
  (`GNF4_DECODE_PLAN="N,K=bn,warps,sk;..."`) so the e2e arms select
  the PER-SHAPE winners without code edits; unlisted shapes fall
  through and unset ⇒ behavior byte-identical to today.
- e4b side unchanged — the env flows through `gemm_4bit_grouped`.

## Fidelity (registered up front, the honest part)

Changing `split_k` or tiling changes the ACCUMULATION STRUCTURE, so
cross-config outputs are NOT bitwise — that is arithmetic, not a bug.
The correctness gate for K1 is therefore: (a) gnf4's kernel property
suite passes ON-BOX with the winner config forced (the kernel is exact
against its own dequant reference per config); (b) the e2e arms REPORT
token agreement length vs the pre-K1 graph run with full disclosure —
agreement is expected to be high but is NOT a bar; a wholesale
divergence (< 100/127) triggers investigation before any ship.

## Bars (before any number)

- **G0**: e2e arm A/A < 7.5%; the sweep's own noise floor: the
  baseline config re-measured at start and end of the sweep must agree
  within 5% or the sweep is NO-VERDICT.
- **H-K (kernel)**: winner summed time for the two shapes ≤ **2/3 of
  the current plan's** (≈ 4.84 → ≤ 3.2 ms/step equivalent).
  PARTIAL (3.2, 4.0]; > 4.0 ⇒ REFUTED-FOR-CONFIG: the plan keeps its
  universal constant, and K2 (kernel body) becomes the lane — the
  ladder does not stall.
- **H-E (e2e)**: graph-loop step ≤ **13.8 ms** (≈ ≥ 72 tok/s) with the
  winner exported, measured by the b1d timed harness, A/A pair per
  arm, GS B=16 sanity in the certified band (the env must not perturb
  the batch point: its shapes fall outside the M=1 guard).
- Consequences: H-K pass ∧ H-E pass ⇒ **CERTIFIED — the winner bakes
  into `_decode_plan` (guarded sm_120 + M=1-census branch) in the
  RESULTS PR** (the #220 in-cycle-default lesson); H-K pass ∧ H-E fail
  ⇒ the kernel win didn't reach the wall — investigate the graph's
  composition before any ship (no silent partial); H-K partial ⇒ bake
  only with H-E pass and full disclosure.

## Verdict calculator

`k1_verdict.py`, self-tested both directions before the box; receipts
in `kernel/receipts-m1-config/` (gnf4) + the e2e arms beside the b1d
receipts (e4b).
