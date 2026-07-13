# Phase-1: measured routing histograms + Gemma-4 / GPT-OSS-120B cells

**2026-07-13** · A5000 SECURE pod (sm_86, torch 2.8+cu128, bnb 0.49.2,
transformers 4.57.6, grouped_gemm 0.3.0) · receipts `routing_{olmoe,qwen}.json`,
`p1c_{olmoe,qwen}_routing.json`, `p1c_{gemma,gptoss}.json` · harness `fee3498`
+ `routing_hist.py` · pod 404-verified, nothing billing.

Two things this pass replaces with measurement: the uniform-routing assumption
in the prefill regime, and the missing big-census cells.

## Measured routing histograms (real forward, diverse wikitext, seq 2048)

Router gate hooked, per-expert token assignments bincounted per layer. OLMoE
loaded bf16; Qwen3-30B in NF4 (Phase-0 established 4-bit perturbs expert
*selection* negligibly — and it fits a 24 GB card without the CPU-offload path).

| model | occupancy (mean, range) | empty experts | rep-layer skew (cv) | hot group vs uniform-M |
|---|---|---:|---:|---:|
| OLMoE-1B-7B (E64, k8) | 0.999 (0.98–1.00) | 0 | 0.51 | **795 vs 256 (3.1×)** |
| Qwen3-30B-A3B (E128, k8) | 0.800 (0.68–0.99) | 27 | 1.41 | **1112 vs 128 (8.7×)** |

The occupancy reproduces Phase-0 (OLMoE ~1.0, Qwen3-30B 0.80) — but Phase-0 kept
only the union count; the *group-size distribution* is the new information, and
it is heavily skewed. Even OLMoE, which touches every expert, sends 3× the
uniform load to its hottest expert; Qwen3-30B sends nearly 9× to its hottest
while leaving 27 of 128 experts empty per forward.

## What measured routing does to the grouped-GEMM cost (prefill, same 16384 token-rows)

`prefill_measured` replays the representative layer's real per-expert counts
(empty experts dropped); `prefill_s2048` is the uniform 2048·k/E benchmark. Same
total token-rows in both — only the grouping structure differs.

| model | proj | backend | uniform | measured | ratio | n_groups (meas) |
|---|---|---|---:|---:|---:|---:|
| OLMoE | gate_up | dequant_grouped | 3.862 | 4.124 | **1.07×** | 64 |
| OLMoE | gate_up | unsloth (grouped_gemm) | 11.338 | 11.643 | 1.03× | 64 |
| OLMoE | down | dequant_grouped | 3.236 | 3.373 | 1.04× | 64 |
| OLMoE | down | unsloth | 5.223 | 5.584 | 1.07× | 64 |
| Qwen3-30B | gate_up | dequant_grouped | 6.243 | 5.364 | **0.86×** | 101 |
| Qwen3-30B | gate_up | unsloth | 15.848 | 13.461 | 0.85× | 101 |
| Qwen3-30B | down | dequant_grouped | 5.980 | 5.050 | 0.84× | 101 |
| Qwen3-30B | down | unsloth | 8.181 | 6.829 | 0.83× | 101 |

**The uniform benchmark is not conservative in a consistent direction — the
correction depends on occupancy:**

- **OLMoE (occupancy ~1):** measured routing is **3–7% SLOWER** than uniform.
  All 64 experts are non-empty either way, so n_groups is unchanged; the skew
  (one 795-row group + many small ones) is slightly less GEMM-efficient than 64
  even 256-row groups. Uniform *under*-counted here.
- **Qwen3-30B (occupancy 0.80):** measured routing is **14–17% FASTER** than
  uniform. Uniform assumed 128 non-empty groups; real routing has only **101**
  (27 empty), so 21% fewer per-expert launches — and that saving dominates the
  skew penalty, on both the naive loop and the grouped_gemm path. Uniform
  *over*-counted here (it paid for 27 experts that receive no tokens).

So the first-pass prefill numbers (uniform) overstated Qwen3-30B's grouped cost
by ~15% and understated OLMoE's by ~5%. Net for the kernel thesis: the grouped
path's ordering is unchanged (it's still the slow floor — unsloth stays ~2–3×
the naive loop under measured routing too), but the honest prefill baseline for
wide, sparsely-occupied MoEs is ~15% below the uniform benchmark, and the fused
kernel should be measured against the measured-routing number.

**Caveat (recorded):** `prefill_measured` uses one representative layer
(median occupancy); real occupancy ranges 0.68–0.99 across Qwen3-30B's 48
layers, so a per-layer sweep would widen this. The single-layer point is the
honest headline; the full per-layer matrix is in `routing_qwen.json` for anyone
who wants the distribution.

## Big-census cells: Gemma-4-26B + GPT-OSS-120B (decode bs1, synthetic weights)

The chunked fp64 reference (v0.2) unblocked these — no resident stacked
reference, so the GPT-OSS shapes fit a 24 GB card. Synthetic weights of the
census shapes; no real-model load needed for the GEMM measurement.

| model | proj | dequant_grouped | gemv_4bit | gemv speedup | err (dq / gemv) |
|---|---|---:|---:|---:|---|
| Gemma-4-26B (E128 k8, I704) | gate_up | 0.430 ms | 0.252 ms | 1.71× | 1.66e-3 / 3.39e-3 |
| Gemma-4-26B | down | 0.440 ms | 0.247 ms | 1.78× | 1.65e-3 / 3.31e-3 |
| GPT-OSS-120B (E128 k4, I2880) | gate_up | 0.533 ms | **0.151 ms** | **3.53×** | 1.68e-3 / 3.32e-3 |
| GPT-OSS-120B | down | 0.290 ms | 0.135 ms | 2.15× | 1.66e-3 / 3.26e-3 |

The decode-bs1 story holds across the biggest shapes, and **GPT-OSS sharpens
it**: its wide expert FFN (I=2880, ~4× OLMoE's) makes the dequant round-trip the
dominant cost, so bnb's in-kernel-dequant gemv wins **3.5×** on gate_up (vs
~1.7× on the narrower models) — the wider the expert, the more a fused /
in-kernel approach beats materialize-then-GEMM. Fidelity is the same
architecture-invariant pair (dequant path 1.66e-3, gemv ~2× that). GPT-OSS is
also k=4, so decode routes only 4 experts — the skinny-launch regime the fused
single launch targets is even skinnier here.

## Coverage after this pass

- Routing histograms: OLMoE + Qwen3-30B measured (the two E-counts with full
  baseline cells). Gemma-4 (gated) and GPT-OSS-120B (57 GB in 4-bit, needs an
  80 GB card; k=4 differs from Qwen's k=8 so Qwen isn't a clean proxy) — their
  measured routing is **deferred**, noted here rather than faked. Their GEMM
  cells run on uniform routing.
- GEMM cells: all four census models now measured on sm_86.
- Still open before Phase 2: a J/token pass over the measured-routing prefill,
  the per-layer routing sweep, and (opportunistically) GPT-OSS routing on an
  80 GB box.

## Ops

- `routing_hist.py` load path: HF_HOME must be on the container disk, not
  `/dev/shm` (a 24 GB tmpfs the 60 GB Qwen download overran); `expandable_segments`
  must be set in the *runner's* env (the pod-create env didn't reach the nohup'd
  process) to load 4-bit Qwen3-30B on a 24 GB card without fragmentation OOM;
  Qwen's tokenizer needs `protobuf`/`sentencepiece` and cannot run `HF_HUB_OFFLINE`
  on a weights-only cache. All four cost a retry; all recorded so the next run
  is one shot.
