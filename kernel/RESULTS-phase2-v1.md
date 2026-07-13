# Phase 2 v1 — the fused NF4 grouped kernel: decode target cleared, prefill is next

**2026-07-13** · A5000 SECURE pod (sm_86, triton 3.4.0, torch 2.8+cu128, bnb
0.49.2) · kernel `kernel/nf4_grouped.py`, suite `kernel/test_nf4_grouped.py`,
receipts `bench/phase1/results/p2_decode_final.json` + `p2_first.json` (prefill)
· pod 404-verified, nothing billing.

First working version of `bitsandbytes::gemm_4bit_grouped` (KERNEL_CONTRACT):
single-launch grouped W4A16, NF4 decoded inside the mainloop (LUT→fp32 in
registers, blockwise fp32 absmax at BLOCK_K==64), fp32 accumulate, one bf16
epilogue downcast. Two kernels behind one entry point: a tensor-core M-tile
path for prefill and a reduction path specialized for decode (every group is
M=1, so an MMA tile would waste 15/16 of its lanes).

## Correctness first — the property suite is green (35/35)

Per TOLERANCE_CONTRACT, run before any timing:

- **Decode exactness vs bnb**: our register LUT + high-nibble-first order decode
  equals `dequantize_4bit` at the kernel's output precision (bf16), and the
  crafted all-16-codes-both-positions probe matches the reference decode — pins
  the codebook values AND nibble order.
- **Census shapes × {M=1, p50=128, p95=290}**, **adversarial absmax** (1e-30 /
  1e30 / mixed / denormal-adjacent, all finite), **boundary cases**
  (single-token groups, all-tokens-one-expert, G<E non-contiguous expert_ids,
  K==blocksize, ragged tail tiles).
- **P-fid HOLDS**: fused error vs the fp64 reference is **0.61–0.77× the dequant
  path's** across every census shape (recorded per shape). The fused kernel is
  *more* accurate than materialize-to-bf16-then-GEMM — the fp32-accumulation
  claim, measured. **B-rel** ≤ 2× dequant and **B-abs** ≤ 1e-2 pass everywhere.

## Decode bs1 — the registered target regime: 6/8 clear ≥1.3× vs dequant

The registered bar is fused vs the dequant+bmm path, same model/card (on sm_86
the native grouped_mm doesn't run, so dequant_grouped — the per-expert loop —
is that path). Fused fidelity 2.2e-3 (vs dequant 1.66e-3, within B-rel).

| model | proj | fused ms | dequant ms | **× vs dequant** | gemv ms | fused vs gemv |
|---|---|---:|---:|---:|---:|---:|
| OLMoE | gate_up | 0.306 | 0.570 | **1.86× PASS** | 0.304 | ~tie |
| OLMoE | down | 0.215 | 0.527 | **2.45× PASS** | 0.309 | **0.70× (fused wins)** |
| Qwen3-30B | gate_up | 0.285 | 0.428 | **1.50× PASS** | 0.256 | 1.11× |
| Qwen3-30B | down | 0.179 | 0.450 | **2.51× PASS** | 0.257 | **0.70× (fused wins)** |
| Gemma-4 | gate_up | 0.355 | 0.437 | 1.23× miss | 0.257 | 1.38× |
| Gemma-4 | down | 0.157 | 0.456 | **2.91× PASS** | 0.259 | **0.61× (fused wins)** |
| GPT-OSS | gate_up | 0.318 | 0.521 | **1.64× PASS** | 0.154 | 2.06× |
| GPT-OSS | down | 0.251 | 0.291 | 1.16× miss | 0.141 | 1.78× |

**6 of 8 clear the ≥1.3× threshold (up to 2.91×), and on the three narrow-N
down-projections the fused kernel beats bnb's hand-tuned `gemv_4bit` outright**
(0.61–0.70× its time) — a single launch reading the packed weight once, versus
gemv's per-expert launch loop. Two honest facts on the other side:

- **Two marginal misses** (Gemma gate_up 1.23×, GPT-OSS down 1.16×) — just under
  the bar, both fixable by the same tiling work the prefill path needs.
- **Wide-N is gemv's territory for now**: at GPT-OSS gate_up (N=5760) bnb's gemv
  is 2× the fused kernel. The decode reduction path reads all N output rows per
  token; gemv is hand-tuned for that shape. Autotuning BLOCK_N per N (a
  descriptor already threaded through) is the v1.1 fix; BLOCK_N=128 was tuned on
  the moderate-N shapes and is left-on-the-table at N≫2048.

## Prefill s2048 — v1 loses, exactly where the roofline said it would

| model | proj | fused ms | dequant ms | ratio |
|---|---|---:|---:|---:|
| OLMoE | gate_up | 17.66 | 3.86 | 0.22× (slower) |
| OLMoE | down | 9.16 | 3.18 | 0.35× |
| Qwen3-30B | gate_up | 13.68 | 6.25 | 0.46× |
| Qwen3-30B | down | 7.30 | 6.23 | 0.85× |

The prefill M-tile path is a naive `tl.dot` K-loop — no software pipelining, no
K-stage prefetch, BLOCK_M/N untuned — so it loses to cuBLAS-backed dequant+mm at
M≈256. This is **not a threshold miss**: `gemm_predictions.json` registered
prefill/train as compute-bound, where the claim is **parity + energy, not
speedup** (roofline ceiling ~1.0× vs bf16). v1 hasn't reached parity because the
mainloop is unoptimized; correctness holds (property suite covers M=128/290).
The MMA-tiled, pipelined mainloop is the v1.1 item — and it's the same work that
closes the two decode misses.

## Gate-2 read

Decode bs1 — the memory-bound regime the whole thesis rests on (roofline 8.1×
ceiling) — **substantially passes**: 6/8 cells ≥1.3× vs the dequant path (to
2.91×), P-fid holds (fused is *cleaner* than dequant), and the kernel matches or
beats bnb's specialized gemv on moderate-N. The recorded narrowings: two
marginal decode misses and a wide-N gap, both tiling-bound; and the prefill path
is a v1-incomplete (compute-bound parity, next iteration), reported at full
volume here rather than hidden. This is a legitimate v1 — the target-regime win
is real and correctness-gated; the compute-bound path and the wide-N autotune
are the honest next steps before the Phase-2 close and the #1949 coordination
comment.

## Next (v1.1, before Gate-2 sign-off)

1. MMA-tiled + pipelined prefill mainloop (num_stages K-prefetch, autotuned
   BLOCK_M/N) → parity at M≈256, and it closes the two decode misses.
2. Autotune BLOCK_N by N for the decode path → the wide-N (GPT-OSS gate_up) gap
   vs gemv.
3. J/token pass on the fused decode cells (the registered energy bar).
4. sm_120 (Blackwell) retune — Phase 4 target.
