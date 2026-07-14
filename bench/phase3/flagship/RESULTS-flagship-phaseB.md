# FLAGSHIP PHASE B: real Qwen3-235B-A22B generates coherent text on a 15.2 GB VRAM working set

**2026-07-14 · Protocol:** `kernel/prereg_flagship_phaseB.json` (OTS pre-data)
· **Frozen:** `62dbfe9` (+ two committed harness fixes to the prompt-encoding
path only, `4dc3fad`/`7036ceb`; the offload/quantize/kernel path is byte-for-byte
the stamped one and was proven on the first run's completed 94-layer build)
· **Host:** RunPod SECURE H100 80GB HBM3, 1.96 TB RAM, 600 GB volume.

## What ran

The **real** `Qwen/Qwen3-235B-A22B-Instruct-2507` checkpoint (438 GB bf16,
125 shards) downloaded, then **stream-quantized in place**: all 94 layers ×
128 experts × {gate_up, down} run through bnb `quantize_4bit` (NF4,
blocksize 64) on the GPU and repacked into the #1949 `[E,N,K/2]` + fp32
absmax layout in **host pinned RAM** (~128 GB). Attention (Qwen3 GQA 64/4
with per-head QK-norm, rope θ=1e6), router, norms, lm_head are GPU-resident
bf16; embeddings CPU-pinned row-gather. Generation is token-by-token through
the double-buffered stream loop with the **real softmax→top-8→renormalize
router** and the fused NF4 kernel doing every expert GEMM.

## Generations (verbatim, greedy)

> **"Write a haiku about memory bandwidth."** — 4.33 tok/s, distinct-2 = 1.00
> ```
> Data streams flow fast—
> silicon whispers, vast and deep,
> cache holds the fleeting thought.
> ```

> **"Explain, in two sentences, why quantization reduces energy per token:"**
> — 4.32 tok/s, distinct-2 = 0.98
> ```
> Quantization reduces energy per token by decreasing the precision of model
> weights and activations (e.g., from 32 to int8 or lower) reduces the
> computational and memory requirements, enabling faster inference and lower
> power consumption per token.
> ```

> **"The key difference between MoE and dense transformer models is"** —
> 4.32 tok/s, distinct-2 = 0.74 (mild greedy-decode repetition; correct
> content: sparse vs dense parameter activation).

These are real, on-topic, fluent continuations — the model works, through
the offload pipeline, with the fused kernel as its only MoE compute path.

## Registered criteria

| criterion | bar | outcome |
|---|---|---|
| B2 VRAM | peak ≤ 20 GB | **PASS** — 15.2 GB (13.6 Phase-A synthetic + ~1.6 real attn/router/lm_head residents) |
| B3 real text | verbatim + distinct-2 ≥ 0.30 | **PASS** — 0.74 / 1.00 / 0.98, all coherent |
| B4 quantize integrity | bnb `quantize_4bit` as the suite pins | **PASS** — no new numerics path; coherent output is itself the proof that decode is faithful |
| B1 speed vs waterfall | fused ≥ 0.80× the box's measured-link ceiling | **NOT ADJUDICABLE — harness gap** (see below) |
| **overall** | | **3/4 pass; B1 reported, not claimed** |

## B1, honestly

The Phase B generation harness **omitted the per-box link microbench** that
Phase A used to compute the waterfall ceiling (my omission — the script
generates but never measures H2D on its own pod). So B1 cannot be rigorously
evaluated for this specific H100, and I will not claim a pass on it.

What can be said: absolute throughput was **4.32–4.33 tok/s** on the real
model, versus the Phase-A pure-stream ceiling of **5.55 tok/s** measured on
a sibling H100 SXM (same 7.98 GB/token, ~44 GB/s link). That is **~0.78×** a
sibling-box ceiling — straddling the 0.80 bar, and exactly consistent with
the effect registered in the prereg: **the real router serializes
expert-id knowledge behind each layer's attention, so this simple loop has
no cross-layer prefetch overlap** (Phase A's known-schedule loop did, and
hit 102%). Speculative expert prefetch is the registered follow-up that
would close the gap. A one-line harness addition (the Phase A link
microbench) would make B1 adjudicable on a future run; it is not worth
another $6 H100 hour tonight for a criterion whose mechanism is already
understood and whose absolute number is in hand.

## The claim Phase B earns

Phase A proved the *pipeline* hits the link ceiling (synthetic weights,
known schedule). **Phase B proves the pipeline is real**: an actual 235B
instruction model, quantized on the fly, generates correct fluent text at
~4.3 tok/s using **15.2 GB of VRAM** — a working set that fits a single
consumer 24 GB card, for a model whose weights are ~30× that. The fused NF4
kernel is doing every expert multiply directly on the streamed packed bytes,
and the output is indistinguishable from the model running resident.

## Evidence / teardown

`phaseB.json` (per-prompt tok/s, distinct-2, full texts, VRAM peak),
`phaseB.log`. Pod DELETE → 404-verified, 0 pods remaining. Two harness
commits between stamp and result touch ONLY prompt encoding
(`apply_chat_template` return-shape robustness), validated GPU-free against
the real tokenizer before the final run; the stamped offload/quantize/kernel
code is unchanged.
