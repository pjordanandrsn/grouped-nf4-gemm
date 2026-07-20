# Phase 0 — MXFP4 seam map

Deliverable of the sprint's Phase 0 (sprint plan (private fork), stamped
2026-07-19 sha `448cf954…`). One page: what changes vs NF4, what is untouched.
Every constant here is source-extracted or spec-cited (R6) — none from memory.

## Sources (R6)

- **Operative decode table & unpacking** — `transformers.integrations.mxfp4`
  (the exact path the A4 reference/oracle runs): `FP4_VALUES = [0.0, 0.5,
  1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0,
  -6.0]`; unpack `idx_lo = blk & 0x0F`, `idx_hi = blk >> 4`; entry
  `_convert_moe_packed_tensors(blocks, scales, …)`, class `Mxfp4Dequantize`.
  **This is ground truth for the checkpoint** (disagreement = STOP).
- **Format definition** — OCP Microscaling Formats (MX) Specification v1.0
  (opencompute.org; corroborated by FPRox "OCP MX Scaling Formats"):
  E2M1 = 4-bit element, one zero + one subnormal (0.5) + 6 normals per sign,
  max ±6.0; **block size k = 32**; scale is **E8M0**, an unsigned biased
  float32 exponent, **bias 127**, `0xFF` reserved = NaN (a block with scale
  `0xFF` decodes all-NaN). *(Exact spec § numbers pending a direct PDF read;
  the transformers constant above is the binding one for our tests.)*
- **Checkpoint tensor layout** — live `get_safetensors_metadata(
  openai/gpt-oss-120b)` on the A2000 (687 tensors). Layer-0 experts:
  - `gate_up_proj_blocks`  U8 `[128, 5760, 90, 16]`
  - `gate_up_proj_scales`  U8 `[128, 5760, 90]`
  - `down_proj_blocks`     U8 `[128, 2880, 90, 16]`
  - `down_proj_scales`     U8 `[128, 2880, 90]`
  - `{gate_up,down}_proj_bias` BF16 (epilogue, not GEMM weight)
- **Triton fleet caps** — live on gnf4-v6 (triton 3.4.0): `tl.gather` ✓,
  `tl.exp2` ✓, `tl.fma` ✓, `tl.inline_asm_elementwise` ✓, **`tl.ldexp` ✗**.

## Geometry decode (the load-bearing arithmetic)

16 bytes/block = 32 nibbles = **32 e2m1 elements** (matches MX k=32). 90
blocks × 32 = **K = 2880**. Flatten `[E, N, 90, 16] → [E, N, 1440]`, and
**1440 == K//2 == the existing NF4 `B [E,N,K//2]` packed width** — the shipped
blocks bytes drop onto the current B-tensor layout with **zero reshape/convert**
(the provenance win: the arena bytes ARE the checkpoint bytes). Scales
`[E, N, 90]` = one e8m0 byte per 32-element block (vs NF4 `[E,N,45]` fp32 per
64). Native bytes/expert = 5760·90·(16+1)+2880·90·(16+1) = **13,219,200 =
12.61 MiB** (blocks+scales; 4.25 bits/wt vs NF4's 4.5 — slightly *smaller*).

## What CHANGES vs NF4 (all inside the existing kernels)

| # | change | NF4 now | MXFP4 | cost |
|---|--------|---------|-------|------|
| 1 | decode LUT contents | `NF4_LUT` (16 irregular) | `FP4_VALUES` (16, ±{0,.5,1,1.5,2,3,4,6}) | table swap; VARIANT-1 register-LUT path unchanged in shape |
| 2 | scale scheme | per-64 fp32 absmax, `w = w * am` | per-32 e8m0, `w = w * exp2(e - 127)` (`tl.exp2`, since `tl.ldexp` absent) + `0xFF`→NaN guard | cheaper (pow-2) |
| 3 | block geometry | `BLOCK_K=64`, `g0 = k0//64`, absmax stride over 45 | `BLOCK_K=32`, `g0 = k0//32`, scale stride over 90 | constexpr + stride args |
| 4 | **nibble interleave (STOP)** | bnb: elem 2j = **HIGH** nibble, 2j+1 = LOW (`(kk%2)==0 ? hi : lo`) | transformers `idx_lo=&0x0F, idx_hi=>>4` → **hypothesis: elem 2j = LOW**, opposite bnb → flip the `tl.where` | **must be adjudicated by the A4 oracle in Phase 1, not assumed** |
| 5 | scale dtype in arena | fp32 absmax segment | uint8 e8m0 segment (4× smaller) | row_bytes/segment-offset recompute in the pipelined arena |

## What is UNTOUCHED (anchors — R1)

- Grouped-ragged mainloop tiling (`build_group_tiles`, `t_row0/t_rows/
  t_group`), the `(m_tiles, N/BLOCK_N)` grid, fp32 accumulation, single bf16
  epilogue downcast.
- `gemm_4bit_grouped` device-tensor `expert_ids` calling convention.
- The pipelined engine: `gather_rows_addr` copies bytes and is format-blind;
  slots, `have`-skip, K-dial, CUDA-graph capture — all inherited. Only
  `row_bytes` + the four segment offsets change (item 5).
- gpt-oss hot sets — **reuse verbatim** (routing is format-independent).
- Clamped-GLU epilogue + per-expert biases + harmony surfaces — inherited
  from the port.

## The exploit, confirmed

The plan's load-bearing bet holds: **e2m1 is a 16-entry codebook**, so the
NF4 mainloop's in-register LUT decode (VARIANT 1, `tl.gather` over a
16-entry `lut_reg`) is the right machine — the format swap is items 1–5,
not a rewrite. No structural harness change is required (if Phase 2 finds one
needed, that is the R1 red flag to report, not work around).

## STOP items carried into Phase 1

1. **Nibble interleave order (#4)** — adjudicate `mxfp4_pack_ref` ↔ the A4
   dequant oracle on the SAME bytes; exact agreement or STOP.
2. **e8m0 `0xFF` NaN handling** — confirm the checkpoint contains none (or
   the guard's behavior) against the oracle.
3. **Scale application order** — verify `exp2(e-127)` reproduces the oracle's
   scale to fp32 (the oracle may apply it in a specific dtype/order).

## Phase-1 entry

`kernel/mxfp4_pack_ref.py` mirroring `nf4_pack_ref.py`: pack native-shaped
bytes, dequant-ref, and unit-test *against the A4 oracle* (device-free first,
then GPU). Interpreter-contract tests extend the existing 18 with the format
parameter. No pod spend until the oracle agrees on CPU.
