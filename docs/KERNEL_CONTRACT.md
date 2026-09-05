# Kernel contract — grouped W4A16 GEMM over fused NF4 expert stacks

*Phase 0 deliverable (Gate 0). This document is the successor to the
referenced-but-absent `kbit_gemm_context.md` (see Deviations) and is written against the two
sources that actually exist: the locked Experts4bit design (bnb #1849 discussion / #1965 PR)
and the **merged** #1949 `gemm_4bit` kernel family (bnb main, milestone v0.50.0, merged
2026-05-21).*

> **Status note (2026-09-04).** This contract is the Gate-0 design record of
> the NF4 kernel and is kept as written: "Phase 4" for sm_120 below is that
> schedule (sm_120 shipped in 0.15.0 and is the primary serving target, per
> `docs/STATUS.md`), and the "storage-only asterisk" framing is version-aware
> on `docs/solutions/nf4-grouped-gemm-without-bf16-materialization.md`. The
> next section is the current layout summary for every shipped format; the
> consumer's stores are built to it, and moving a layout is a
> `public-api-change` under `docs/change-impact.json`.

## Layouts at a glance (current)

| format (module) | packed | scales | notes |
|---|---|---|---|
| NF4 (`nf4_grouped`) | `[E, N, K//2]` uint8, high nibble first — the bitsandbytes `gemm_4bit` layout | absmax `[E, N, K//64]` fp32 | `K % 64 == 0`; `nf4_grouped.repack_from_bnb` builds these from per-expert `quantize_4bit` state and de-nests `compress_statistics` |
| MXFP4 (`mxfp4_grouped`) | blocks `[E, N, K//2]` uint8, low nibble first (even element) | e8m0 `[E, N, K//32]` uint8 | `K % 32 == 0`; gpt-oss `[E, N, K//32, 16]` blocks flatten to that width as a contiguous view (`mxfp4_loader.to_kernel_shapes`) |
| int4-b32 (`int4_b32`) | `[E, N, K//2]` uint8, levels -8..7 stored offset-binary, even `k` in the low nibble | `[E, N, K//32]` fp16 | `K % 32 == 0`; `int4_pack_ref.pack_int4_b32` (round-to-nearest) and `gptq_pack.gptq_pack_int4_b32` (calibrated) emit the same bytes |
| fp8 KV (`fp8_kv`) | 16-token packed rows, e4m3 payload plus fp32 scales (`pack_kv_block`, `kv_block_bytes`) | per-(token, head), or `k_groups` sub-row groups | `fp8_paged_attn.fp8_paged_decode_attention` reads them through a block table; `k_row_bytes` / `v_row_bytes` override the stride per layer |

## Boundaries (2026-09-05)

Two limits every shipped kernel now states rather than discovers at run time.

**Offset arithmetic is int64 for every expert base.** A Triton kernel that
loads an expert id (or a block-table row) from an int32 tensor and multiplies
it by a stride evaluates the product in int32 unless the index is widened
first, so a stack whose highest base offset reaches 2^31 bytes wrapped to a
negative offset and faulted (#87, measured at exactly `max(expert_ids) *
stride_be == 2**31`: 256 experts of 8 MiB pass, 257 fault). The rule, in
every kernel that carries the pattern: the expert id is cast to int64 at its
load (`eid = tl.load(...).to(tl.int64)`) *before* any stride product, so the
whole base expression promotes; a stride that itself exceeds 2^31 is passed
as an i64 argument by Triton's specialization. Carriers: `nf4_grouped`
(`_gemm_nf4_grouped`, `_gemv_nf4_grouped`, `_gemv_nf4_grouped_splitk`,
`_gemv_nf4_dotpad`, `_gemv_nf4_dotpad_splitk`, `_dgrad_nf4_grouped`),
`mxfp4_grouped` (`_gemm_mxfp4_grouped`, `_gemv_mxfp4_grouped`,
`_gemv_mxfp4_b32`), `int4_b32` (`_gemv_int4_b32`, `_gemm_int4_b32_grouped`),
`host_gather._gather_rows`, the `mxfp4_pipelined` / `mxfp4_residency` gathers
(slot × row words), the `fp8_kv` appenders (block-table row × row bytes) and,
from this date, the four `fp8_paged_attn` decode kernels, whose block-table
row was still scaled by `k_row_bytes` / `v_row_bytes` in int32 (a pool past
2^31 bytes wrapped on the reader while the writer was already widened). The
M-tile kernels also widen the tile's `row0`, so activation and output row
offsets (`(row0 + offs_m) * K`, `* N`) cannot wrap at `T * max(K, N) >= 2^31`
elements; the decode GEMVs index their rows by program id and are bounded by
their contract (one token per group, `T` in the hundreds), which keeps
`T * max(K, N)` far below 2^31 without a cast. Pack and reference ops are
pure torch and index in int64. The straddling regression is
`kernel/test_expert_offset_boundary.py`: for each carrier, the experts (or
pool rows) whose base offsets sit just below and just above 2^31 are compared
with the pure-torch reference, every above-boundary case in its own process
(an illegal access poisons the CUDA context). `kernel/test_offsets_2gib.py`
keeps the original 258 x 8 MiB reproduction.

**Shared-memory feasibility is decided before the launch.** A tile
configuration whose shared-memory need exceeds the device's limit is refused
or re-dispatched *before* the launch, never surfaced as Triton's
`OutOfResources` (#324). The limit is queried once per device
(`_triton_shim.device_shared_mem_limit`; 0 when unqueryable — no triton, no
device, interpreter mode — and 0 never refuses). Rules by kernel:

| kernel | tile term that scales | rule |
|---|---|---|
| NF4 M-tile (`_gemm_nf4_grouped`) | `stages * (BLOCK_M*BLOCK_K*2 + B tile)` | `nf4_grouped.prefill_fit` steps stages, then `BLOCK_M`, then stages again to the largest configuration that fits under the limit minus `PREFILL_SMEM_HEADROOM`; the smallest configuration still over the limit raises `UnsupportedShapeError` (an explicit `prefill_config=` is launched as given) |
| fp8 paged decode, packed fp8 (`_fp8_paged_decode_packed_f8`) | `(stages-1) * (2*BT*H_kv*D + BT*H_kv*D/k_groups + BT*H_kv*4)` | `fp8_paged_attn.packed_unsupported`: above the limit, the split fp8 kernel serves the call with one `RuntimeWarning` per geometry; the model reproduces the one measured overflow (148 480 B at D=256, 8 kv heads) and admits every packed geometry the suite runs on a 101 376 B card; the launch keeps its overflow catch for what the model misses |
| fp8 paged decode, packed f32 | no calibrated model | an overflow at the launch falls back to the split f32 kernel the same way |
| fp8 paged decode, split (f32 and fp8) | `KTILE * D` per K and V | an overflow at the launch is raised as `UnsupportedShapeError` naming the geometry, Triton's required bytes and the limit (reduce `ktile` or `num_stages`) |
| MXFP4 M-tile / GEMV, int4-b32 GEMV / M-tile, NF4 decode GEMVs | fixed tiles (`BLOCK_K` 32 or 64, `BLOCK_N` ≤ 128) | no runtime dimension scales the tile; every configuration fits a 64 KB LDS |

`UnsupportedShapeError` is a `ValueError` carrying `kernel`, `shape`,
`need_bytes`, `limit_bytes`; the CPU unit test
(`kernel/test_shape_feasibility.py`) drives every selection rule with mocked
limits and, under `TRITON_INTERPRET=1`, checks that a fit-down still matches
`dequant_ref`.

## What the kernel computes

For each MoE block projection (`gate_up`, `down`) and a batch of routed tokens:

```
out[t, :] = act[t, :] @ dequant_nf4(B[e(t)]).T          for every token t routed to expert e(t)
```

with the dequantization fused **inside** the GEMM mainloop — the bf16 expert weight is never
materialized in global memory. This deletes the storage-only asterisk: NF4 stops costing a
dequant round-trip per use.

## Inputs (adopting #1949 conventions wherever they are pinned)

| input | shape / dtype | convention source |
|---|---|---|
| `A` (activations, gathered) | `[T_total, K]` bf16 (fp16 accepted) | #1949: `A ∈ {fp16, bf16, fp32}` |
| `B` (packed experts) | `[E, N, K/2]` uint8 (two NF4 nibbles/byte), per-expert canonical `[out, in] = [N, K]` | #1949 canonicalizes packed weights to `[N, K]`; transposed-quantized layout is deprecated there — we require canonical from day one |
| `shapeB` | `[N, K]` per expert | #1949 op arg |
| `absmax` | fp32, `[E, ceil(N·K / blocksize)]` | #1949 op: `absmax must be float32` |
| `blocksize` | 64 default; `K % blocksize == 0` enforced | e4b locked design (maintainer: divisibility enforced so expert slices land on block boundaries) |
| `quant_type` | `"nf4"` (fp4 accepted for parity with #1949) | #1949 |
| nested absmax (optional, v2) | `absmax_8bit` + `absmax_code` + `absmax_offset` | #1949 op signature trio, adopted **by name**; e4b v1 defers `compress_statistics`, so v1 of this kernel does too |
| `group_offsets` | `[E+1]` int32, prefix-sum of tokens-per-expert after token→expert sort | NEW (the grouping dimension) |
| `expert_ids` | `[G]` int32, the experts with ≥1 token (sparse group list) | NEW |
| `bias` | optional `[E, N]` | mirrors #1949's fused bias |

Output: `[T_total, N]` bf16, grouped in sort order; the caller scatters back via the inverse
permutation (sort + scatter live OUTSIDE the kernel, same contract as every grouped-GEMM).

**Op naming:** `bitsandbytes::gemm_4bit_grouped` — the #1949 signature plus
(`group_offsets`, `expert_ids`) and an expert-major leading dim on `B`/`absmax`. Framing for
Phase 5: *the expert-grouped extension of the #1949 kernel family* — same codebook handling,
same absmax conventions, same dispatch philosophy (conservative heuristics, dequant+linear
fallback above a size threshold).

**Convention correction (recorded):** the roadmap shorthand said "E4M4 absmax". The merged
#1949 source contains no E4M4; the pinned convention at the op boundary is **fp32 absmax**
with an optional 8-bit nested trio (`absmax_8bit`/`absmax_code`/`absmax_offset`). We adopt
what is actually in the code.

## Kernel tiers (mirroring #1949's structure)

| tier | target | dtype | this project's scope |
|---|---|---|---|
| MMA `m16n8k16` | sm80+ (A2000/3090 = sm_86 dev targets; sm_120 in Phase 4) | bf16/fp16 | **primary** |
| SIMT | sm60+ | any | fallback, correctness reference on-GPU |

NF4 dequant in-loop: 16-entry codebook in registers/constant memory (LUT), blockwise absmax
applied per K-tile, MMA accumulate in **fp32**, single bf16 downcast at epilogue.

## The grouping design center

Fine-grained MoE is the battlefield: per-expert token counts at decode are ~1 (see census).
Launch amortization is therefore the design center, not an optimization:

- token sort + `group_offsets` computed once per layer per step (outside the kernel);
- **persistent-kernel scheduling** over variable-size groups — one launch walks all groups,
  tiles sized from the census (small-M tiles dominate);
- skinny-shape configs autotuned from the census table, not intuition — this is precisely
  Marlin's documented failure mode and the reason a rival kernel doesn't already win here.

## Regimes (the three columns every measurement carries)

| regime | M per active expert (census-derived) |
|---|---|
| decode bs1 | ~1 token/expert, k experts active (k=8 of 64/128; k=4 GPT-OSS) |
| prefill (S=2048, bs1) | mean S·k/E: 256 (OLMoE), 128 (Qwen3/Gemma-4), 64 (GPT-OSS); multinomial spread in census |
| training microbatch (mb=1, seq 2048, packed) | same shape as prefill; backward stays on the dequant path in v1 (scope control) |

## Fallback contract

Above the size threshold where dequant+`grouped_mm` wins (roofline: compute-bound cells),
dispatch falls back exactly as #1949 does. The kernel must never be a regression: dispatch
is conservative, calibrated on the Phase-1 baseline table.

## Deviations from the roadmap (recorded at Gate 0)

1. **`kbit_gemm_context.md` does not exist** — not local, not in any pjordanandrsn repo
   (GitHub code search: 0 hits). This contract is written against DESIGN.md (bnb-moe-4bit
   locked design) + the merged #1949 source, and supersedes the missing reference.
2. **"E4M4 absmax" not found in #1949's merged code** — adopted the actual op-boundary
   convention instead (fp32 absmax + nested 8-bit trio). If E4M4 exists in a later kbit
   iteration, Phase 5's rebase picks it up then.
