# Kernel contract — grouped W4A16 GEMM over fused NF4 expert stacks

*Phase 0 deliverable (Gate 0). This document is the successor to the
referenced-but-absent `kbit_gemm_context.md` (see Deviations) and is written against the two
sources that actually exist: the locked Experts4bit design (bnb #1849 discussion / #1965 PR)
and the **merged** #1949 `gemm_4bit` kernel family (bnb main, milestone v0.50.0, merged
2026-05-21).*

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
