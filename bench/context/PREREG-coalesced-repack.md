# PREREG — repack for coalescing: the only headroom #50 left

**Tier: CONFIRMATORY.** #50 measured the target; this builds to it.

## What #50 established

The decode GEMV is **within 1.01–1.06×** of a stripped kernel doing the same
required work — there is no kernel overhead left. The entire remaining gap is
the **access pattern**:

| | A6000 sm_86 | RTX 4090 sm_89 |
|---|---:|---:|
| strided (the GEMV's pattern) | 150.0 GB/s | 402.9 GB/s |
| coalesced (identical bytes) | 487.7 GB/s | 1404.3 GB/s |
| **penalty** | **3.25×** | **3.49×** |

#50 first called fixing this an on-disk format migration and **that was wrong**
(corrected same day): `repack_from_bnb` builds the packed layout **in memory**
from bnb's quantize output, the loader reads a **bf16** checkpoint, and the only
`torch.save` writes LoRA adapters. Nothing persists the packed layout.

## The change

Today: `B [E, N, K/2]` — a warp spans `N`, so each lane lands on a different row
`K/2` bytes away. Measured at **12.08 sectors/request** pre-#43 and **1.95**
after; the remaining loss is that rows are far apart.

Proposed: **`B [E, K/2, N]`** — for a fixed byte-column, `N` is contiguous, so a
warp reading `BLOCK_N` consecutive experts-outputs touches consecutive bytes.

Touches exactly: `repack_from_bnb`'s output, the kernel's `b_base`/stride
indexing, and the call sites in `bench/phase3/offload_decode_235b.py` and
`bench/phase1/harness.py`.

## Predictions

| quantity | prediction |
|---|---|
| sectors/request after repack | **1.0–1.3** (from 1.95) |
| isolated GEMV speedup | **1.8–3.0×** |
| geomean over the 8 census decode shapes | **1.6–2.5×** |
| worst shape | **≥ 1.2×** — no shape may regress |
| numeric agreement vs shipped | **≤ 8e-03** (bf16 floor) |

The isolated band is deliberately **below** the 3.25–3.49× raw penalty:
coalescing the weight load does not make the reduction, the LUT gather, or the
absmax load free, and #50 showed those are real terms.

## Decision rules, fixed now

- **≥1.6× geomean on BOTH cards AND no shape below 1.2× AND agreement ≤8e-03**
  → land it, and re-measure the 235B ladder, since the expert GEMM is 71.3% of
  that step (#40).
- **1.2–1.6×** → record; do **not** land without a second look. The repack
  changes a layout four result-sets were measured against, and a sub-1.6× win
  does not pay for re-validating them.
- **<1.2× on either card, or any shape regressing** → the coalescing model is
  wrong. Record the negative; the 3.25–3.49× penalty is then **not** recoverable
  by this transform and the structural line closes for good.

## What would make this VOID

- Any arm where the repacked kernel disagrees with the shipped kernel above the
  bf16 floor. A faster kernel that computes something else is not a result — the
  lesson of #44, where a transform was **bit-identical on one card and ~100%
  wrong on another**.
- A single-card conclusion. Two architectures before anything lands (#43 shipped
  a 2.2× regression on one card's evidence).

## Not claimed

- **Nothing about prefill.** `_gemm_nf4_grouped` feeds `tl.dot` and has its own
  layout constraints; #44 already rejected one transform there. Out of scope.
- **Nothing about the 235B step** until measured. #40's 71.3% expert share
  predicts a step-level win, but #43's history is that step-level numbers must be
  measured, not extrapolated.
