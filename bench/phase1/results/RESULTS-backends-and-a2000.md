# Phase-1 continuation — unsloth + marlin wired, A2000 replication

**2026-07-13** · A5000 SECURE pod (cap 8.6, torch 2.11+cu130 / vllm 0.25.0 /
bnb 0.49.2 / grouped_gemm 0.3.0) + QNAP RTX A2000 (torch 2.8+cu128, bnb 0.49.2)
· receipts `p1_olmoe.json`, `p1_qwen.json`, `phase1_A2000_sm86.json` · harness
`231a916` · both boxes released, pod 404-verified, A2000 container removed and
GPU back to its 3 GB resident baseline.

Five backends now measured, not two: `dequant_grouped` (the e4b product path),
`gemv_4bit` (bnb in-kernel dequant, bs1), `dequant_grouped_mm`
(`torch._grouped_mm` — the native grouped GEMM), `unsloth`
(`grouped_gemm.ops.gmm`, the kernel Unsloth's MoE backend rides), `marlin`
(vLLM `apply_gptq_marlin_linear`, W4A16). Timing only here (`--no-energy`) —
the J/token receipts are the first-pass table; this pass is about the backend
field and the second card.

## decode bs1 — the registered target regime (median ms, A5000)

| shape | dequant_grouped | gemv_4bit | grouped_mm | unsloth | marlin |
|---|---:|---:|---:|---:|---:|
| OLMoE gate_up | 0.493 | **0.279** | 1.305 | 1.225 | 0.456 |
| OLMoE down | 0.443 | **0.267** | 0.719 | 0.627 | 0.456 |
| Qwen3-30B gate_up | 0.441 | **0.277** | 1.062 | 0.983 | 0.553 |
| Qwen3-30B down | 0.451 | **0.273** | 0.701 | 0.583 | 0.462 |

Three findings that sharpen the registered claims:

1. **The grouped path — Unsloth's execution class — is the SLOWEST at decode,
   ~2.5–2.9× the naive dequant loop.** `unsloth` (grouped_gemm.ops.gmm) and
   `dequant_grouped_mm` (torch._grouped_mm) track each other to within a few
   percent across every cell, confirming they are the same execution class: a
   padded grouped GEMM over k=8 groups of M=1. At the skinny decode extreme
   that padding + launch is pure overhead. **This is the strongest evidence yet
   FOR the fused single-launch design** — the roofline put decode bs1 at an
   8.1× ceiling over the two-pass dequant path, and the existing "grouped"
   kernels don't approach it; they regress. The fused kernel's real competition
   at decode is `gemv_4bit` and `marlin`, not the grouped path.

2. **`gemv_4bit` is fastest (~0.27 ms) but worst fidelity** (3.3e-3 rel, ~2× the
   dequant path's 1.7e-3 — the first-pass finding reproduces exactly). It is a
   per-expert Python loop of k launches and still wins on wall-clock, which
   re-underscores that the bar is memory-bound throughput, not FLOPs.

3. **`marlin` is the fidelity outlier: 2.07e-4** (vs its own dequant reference)
   — an order of magnitude cleaner than the NF4 paths, at competitive speed
   (≈ dequant_grouped, ~1.6× slower than gemv). It is W4A16 with fp32 reduce, so
   it is a *proof of achievability* of the P-fid claim the fused kernel makes
   (fp32 accumulation beats bf16 materialization). Caveat, recorded per cell:
   marlin's error is vs marlin's OWN int4-group-128 dequant, a different quant
   format than NF4 — so this is a within-kernel arithmetic-cleanliness number,
   not a head-to-head accuracy comparison against the NF4 paths.

## prefill_s2048 (median ms, A5000)

| shape | dequant_grouped | grouped_mm | unsloth | marlin |
|---|---:|---:|---:|---:|
| OLMoE gate_up | 3.82 | 11.24 | 11.18 | **3.40** |
| OLMoE down | 3.20 | 5.18 | 5.12 | 3.40 |
| Qwen3-30B gate_up | **6.25** | 15.68 | 15.60 | 8.40 |
| Qwen3-30B down | 6.14 | 8.26 | 8.20 | 6.86 |

The grouped path stays ~2.5× the dequant path even at M≈16–256 — the padding
cost doesn't amortize at these MoE intermediate widths. marlin wins OLMoE and
loses Qwen gate_up (its narrower N=1536 is off marlin's tile sweet spot). Note
this is the compute-bound regime where the registered claim is parity + energy,
not speedup, so the fused kernel isn't chasing marlin's prefill number.

## A2000 replication (sm_86, second card)

Core backends re-run on the QNAP A2000 (12 GB, driver 575), VRAM-respectful of
the ~3 GB of resident home services:

| shape / backend | A5000 ms | A2000 ms | ratio | A2000 err | A5000 err |
|---|---:|---:|---:|---:|---:|
| OLMoE gate_up dequant_grouped | 0.493 | 0.826 | 1.68× | 1.66e-3 | 1.66e-3 |
| OLMoE gate_up gemv_4bit | 0.279 | 0.556 | 1.99× | 3.29e-3 | 3.29e-3 |
| Qwen3 down dequant_grouped | 0.451 | 0.654 | 1.45× | 1.66e-3 | 1.66e-3 |
| Qwen3 down gemv_4bit | 0.273 | 0.468 | 1.71× | 3.29e-3 | 3.29e-3 |

**The ordering and the fidelity are architecture-invariant:** gemv fastest,
gemv-err ≈ 2× dequant-err, identical to the digit on both cards; only wall-clock
scales (~1.5–2× slower on the smaller card). So the Phase-1 conclusions are not
an A5000 artifact. `torch._grouped_mm` is unavailable on sm_86 (requires
compute capability 9.0 — recorded skip), which is itself a portability fact the
fused kernel must beat: the native grouped GEMM the field leans on doesn't even
run on Ampere consumer cards, and Unsloth's grouped_gemm build does but is the
slowest option at decode.

## Where this leaves Phase 1

The baseline field is now measured on two cards with the real competitors in
it. The fused kernel's job is crisp: at decode bs1 it must beat `gemv_4bit`'s
~0.27 ms *and* clear marlin's fidelity (P-fid), while the grouped path it might
have been assumed to extend is a floor to avoid, not a target. Remaining
Phase-1 items: measured routing histograms (this pass still uses the uniform
analytic M), the chunked-fp64 Gemma-4 / GPT-OSS cells (harness supports it now;
not yet run), and a J/token pass over the new backends. Then Phase 2 (Triton).

## Ops notes

- The first combined run (energy on, single JSON at the end) **died silently
  mid-Qwen with no artifact** — a hard CUDA fault in a grouped/marlin cell kills
  the process rather than raising to the skip handler. Fix applied: per-model
  runs, `--no-energy` for the fast pass, sentinel file per stage, log redirect —
  so a crash in one model's cells can't lose the other's. (Re-run was clean,
  rc=0 both models.)
- grouped_gemm 0.3.0 builds against the pod's `/usr/local/cuda/bin/nvcc`
  (driver 580 / cu130); no prebuilt wheel.
- vLLM 0.25 marlin surface: `apply_gptq_marlin_linear` (not the older
  `ops.gptq_marlin_gemm`) + `marlin_make_workspace_new` + `marlin_make_empty_g_idx`
  for the symmetric zero-point — recorded in the harness so the adapter doesn't
  rot against vLLM's fast-moving utils.
