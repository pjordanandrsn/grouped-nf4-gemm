# Context budgets — the KV cache as a first-class VRAM term

Every VRAM figure this project has published was measured at **short context**
(seq-512 prefill, 128-token decodes — the receipts say so). The KV cache is the
second memory consumer, and it grows linearly in context. This document makes it
a derived, measured, budgeted quantity.

**Status: Phase C0.** Per-layer KV geometry and the bounded/unbounded split are
**measured on the A2000** against each architecture's real `modeling_*.py` (see
*Verification* below). Full-depth real-weight confirmation for the two models
that don't fit the A2000 (gpt-oss-120b, Qwen3-235B) is **rung two — pending**;
rows are labelled accordingly. Kimi K2 is **derived-only** (no local weights).
K3's config does not exist publicly yet — its row lands when it does.

## The budget identity

```
VRAM_total = weights_resident + hot_set + KV(context) + activations + overhead
```

| term | source |
|---|---|
| `weights_resident` | measured per model/mode (e.g. 235B NF4 offload: base ≈ 15.1 GB) — the stamped flagship receipts |
| `hot_set` | `plan_placement()` output — K experts × bytes/expert (C2 will subtract KV *before* sizing this) |
| **`KV(context)`** | **this document**: `slope_KB_per_token × context + bounded_floor` |
| `activations` | transient; measured inside the stamped peak figures |
| `overhead` | allocator + CUDA context; inside the measured peaks |

The first four terms were already instrumented. `KV(context)` was not — it was
implicitly held at ~0 by only ever testing 512-token contexts.

## KV cost per model

`slope` is the **unbounded** per-token growth (full-attention layers only).
`floor` is the **asymptote** that sliding-window layers converge to. The cost is
piecewise in context, because a sliding layer holds `min(context, window − 1)`
tokens:

```
KV(C) = slope × C  +  per_sliding_layer × n_sliding × min(C, window − 1)
```

so the `floor` column applies only once `C ≥ window − 1` (every column in the
table below is past that point for both hybrid models). Below it the floor term
scales with context too — Gemma-4 at C=512 costs 110 MB, not the 220 MB a
flat-floor formula would claim. Cache dtype fp16/bf16
(2 B/elem) — the transformers default; see *KV quantization* for the q8/q4 path.

| model | KB/token | floor | 4K | 8K | 32K | 128K | tier |
|---|---:|---:|---:|---:|---:|---:|---|
| Qwen3-235B-A22B | **188.0** | — | 0.73 GB | 1.47 GB | 5.88 GB | 23.50 GB | measured (per-layer); rung-2 pending |
| Qwen3-30B-A3B | **96.0** | — | 0.38 GB | 0.75 GB | 3.00 GB | 12.00 GB | measured (A2000) |
| gpt-oss-20b | **24.0** | 3.0 MB | 0.10 GB | 0.19 GB | 0.75 GB | 3.00 GB | measured (A2000) |
| gpt-oss-120b | **36.0** | 4.5 MB | 0.14 GB | 0.29 GB | 1.13 GB | 4.50 GB | measured (per-layer); rung-2 pending |
| Gemma-4-26B-A4B | **20.0** | 199.8 MB | 0.27 GB | 0.35 GB | 0.82 GB | 2.70 GB | measured (A2000) |
| OLMoE-1B-7B | **128.0** | — | 0.50 GB | 1.00 GB | 4.00 GB | 16.00 GB | measured (A2000) |
| Kimi-K2-Instruct | **68.6** | — | 0.27 GB | 0.54 GB | 2.14 GB | 8.58 GB | **derived only** |

### The arithmetic, per model

All quantities from each model's own `config.json` — no figure is carried over
from prior conversation or docs.

- **Qwen3-235B-A22B** — 94 layers, uniform full attention (`sliding_window: null`),
  GQA `num_key_value_heads=4`, `head_dim=128`:
  `2 × 4 × 128 × 2 B = 2048 B/layer/token × 94 = 192,512 B = 188.0 KB/token`.
- **Qwen3-30B-A3B** — same geometry, 48 layers: `2048 × 48 = 96.0 KB/token`.
- **gpt-oss-20b / 120b** — strictly alternating `S,F,S,F…`; `sliding_window=128`;
  `num_key_value_heads=8`, `head_dim=64` → `2048 B/layer/token`.
  20b = 12 full + 12 sliding; 120b = 18 full + 18 sliding.
  Unbounded slope counts **full layers only**: `2048 × 18 = 36.0 KB/token` (120b).
  Bounded floor: `2048 × 18 × 127 = 4.46 MB` (sliding layers store `window − 1`).
- **Gemma-4-26B-A4B** — 30 layers in a `S,S,S,S,S,F` pattern (25 sliding + 5 full),
  `sliding_window=1024`. **Two different KV geometries** (see finding #2):
  sliding layers use `num_key_value_heads=8 × head_dim=256` → 8192 B/layer/token,
  bounded at `window−1`; full ("global") layers use
  `num_global_key_value_heads=2 × global_head_dim=512` → **4096 B/layer/token**.
  Slope = `4096 × 5 = 20.0 KB/token`; floor = `8192 × 25 × 1023 = 199.8 MB`.
- **OLMoE-1B-7B** — 16 layers, no GQA (`num_key_value_heads = num_attention_heads = 16`),
  `head_dim = 2048/16 = 128`: `2 × 16 × 128 × 2 = 8192 B/layer/token × 16 = 128.0 KB/token`.
- **Kimi-K2** — MLA: the cache holds a joint compressed latent plus a decoupled
  rope key, **not** separate K and V, so there is no `2×` factor:
  `(kv_lora_rank 512 + qk_rope_head_dim 64) × 2 B = 1152 B/layer/token × 61 = 68.6 KB/token`.

## Verification (rung one — A2000, $0)

Method: instantiate each architecture's **real** model class from its **real**
config with depth truncated to `L_probe` (random weights — the KV geometry is a
function of config and code, not of weight values), prefill at two context
lengths, and diff the actual cache tensors' bytes. This isolates two independent
claims: the per-layer per-token size, and that only full-attention layers grow.

| model | derived B/layer/token | measured | result |
|---|---:|---:|---|
| OLMoE-1B-7B | 8192 | **8192.0** | exact |
| Qwen3-235B-A22B | 2048 | **2048.0** | exact (K shape `(1,4,512,128)`) |
| Qwen3-30B-A3B | 2048 | **2048.0** | exact |
| gpt-oss-20b (2F+2S probe) | 4096 marginal (full only) | **4096.0** | exact — sliding layers **bounded** |
| Gemma-4-26B (1F+5S probe) | 8192 → **corrected to 4096** | **4096.0** | derivation corrected by measurement |

gpt-oss cross-check: at ctx 512 the 2F+2S probe held 2.496 MB =
`2×2048×512 (full) + 2×2048×127 (sliding, window−1)` — both regimes confirmed in
one number.

**Rung two (pending, cloud, standing GO):** full-depth, real-weight slope at
512 / 8K / 32K for gpt-oss-120b and Qwen3-235B — the two models whose depth the
A2000 cannot hold. What it buys over rung one: confirmation that nothing
depth-dependent (e.g. a KV-sharing threshold like Gemma's `num_kv_shared_layers`)
changes the slope at real depth. Until it is green, those two rows are labelled
*rung-2 pending* and **must not** be promoted onto the READMEs or research pages
(C1's gate).

## Exactness tiers

Every result this project has published so far has been *fidelity-neutral*: the
census is bit-accurate per cell, the 235B decode is greedy-identical, the MI300X
run is correctness-confirmed. Those claims are about a kernel reproducing a
reference on given bytes, and they all still hold — the KV work adds files and
does not touch the weight path.

KV quantization is the first feature that is **lossy at the model level**, and
that is a different kind of claim, because the cache is state the model
generates rather than a format the user chose. There is no "the NF4-KV model" to
serve as ground truth the way a 4-bit checkpoint does for weights. So fidelity
now needs its own axis, separate from the engine tiers (which grade performance
maturity):

| tier | meaning | members |
|---|---|---|
| **exact** | greedy-identical to the reference; differences bounded by fp accumulation | every weight-path result: census, property suite, flagship decode, multiarch |
| **approximate** | measurably changes model output; ships only with a fidelity receipt | NF4 KV cache (below) |

The rule this exists to enforce: an approximate feature must not inherit the
bit-accurate framing by sitting next to exact code. That is why the rejected
low-rank path lives under `bench/` and not `kernel/` (finding #6), and why the
NF4 KV dials are off by default.

## Findings

### 1. The published "235B on ≤16 GB" figure covers ~5K of context, not 128K

The stamped flagship number is **15.2 GB peak at seq-512 decode**. Decomposing
with the measured slope: KV at 512 tokens = `188.0 KB × 512 = 0.09 GB`, so
`base = weights + hot + activations + overhead ≈ 15.11 GB`. Adding KV:

| context | KV | total | 16 GB card | 24 GB | 48 GB |
|---:|---:|---:|---|---|---|
| 512 | 0.09 GB | 15.20 GB | fits | fits | fits |
| 4,096 | 0.73 GB | 15.84 GB | fits | fits | fits |
| **4,974** | **0.89 GB** | **16.00 GB** | **ceiling** | fits | fits |
| 8,192 | 1.47 GB | 16.58 GB | over | fits | fits |
| 32,768 | 5.88 GB | 20.98 GB | over | fits | fits |
| 131,072 | 23.50 GB | 38.61 GB | over | over | fits |

So the claim is true and stays true **at its measured scope** — and its scope is
~5K tokens on a 16 GB card. A 24 GB card carries the same model to ~49K. This is
exactly the class of silent-wrongness this directive exists to remove: nothing
was mis-measured, but the context qualifier was never stated. C1 attaches it
everywhere the figure appears.

### 2. Gemma-4's full-attention layers have a different KV geometry than its sliding layers

Deriving Gemma-4 from the top-level `num_key_value_heads`/`head_dim` alone gives
**40.0 KB/token — 2× too high**. The measurement showed full-attention layers
allocating `K(1, 2, ctx, 512)`, i.e. `num_global_key_value_heads=2 ×
global_head_dim=512`, while sliding layers use `8 × 256`. Correct slope is
**20.0 KB/token**. Any KV budget for a hybrid model must read the per-layer-type
fields, not just the top-level pair.

### 3. Sliding-window layers store `window − 1` tokens, not `window`

Measured on both gpt-oss (127 for `window=128`) and Gemma-4 (1023 for
`window=1024`). Small, but it is the difference between a derived and a measured
floor, so the floors above use `window − 1`.

### 4. `num_kv_shared_layers` is a real KV-elision mechanism (0 in this checkpoint)

Gemma-4's attention implements KV sharing: layers at or past
`num_hidden_layers − num_kv_shared_layers` allocate **no** K/V projections and
reuse an earlier layer's cache. It is `0` for gemma-4-26B-A4B, so it does not
reduce this row — but the budget code must honour it, because other Gemma-4
sizes may set it, and it is the one mechanism that breaks `slope ∝ layers`.

### 5. Architecture dominates the KV bill far more than parameter count

OLMoE (**1B active / 7B total**) costs **128 KB/token** — 5.3× gpt-oss-120b's
**24 KB/token** — because it has no GQA at all. gpt-oss-120b is the cheapest
long-context model here per token despite being the second largest, because half
its layers are windowed and it uses 8 KV heads of dim 64. MLA (Kimi) and
hybrid+global-GQA (Gemma-4) are the two cheapest designs per token. "Big model"
and "expensive context" are close to independent axes.

### 6. Low-rank KV codes: built, measured, rejected

Compression *before* quantization looked orthogonal to NF4 — cache rank-r codes
against a pinned per-head basis and absorb the up-projection into the query
(`q @ (C@B).T == (q@B.T) @ C.T`), so the cache is never reconstructed. The
algebra is exact and implemented (11/11), but on real caches the premise fails.
Measured on OLMoE-1B-7B, 1024 wikitext tokens, basis fit on the first half and
scored on the second (`bench/context/lowrank_probe.py`):

| rank | ratio | K held-out | V held-out |
|---:|---:|---:|---:|
| 64 | 2.00x | 48.2% | 43.0% |
| 96 | 1.33x | 35.0% | 27.8% |
| 124 | **1.03x** | 15.2% | **9.1%** |

V reaches NF4-parity error only at a 1.03x saving: there is no operating point,
and the method is dominated by plain NF4 across the whole curve. Ranks the
64-element blocksize cannot pack were measured too, so the packer is
demonstrably not the binding constraint. Even an *oracle* basis fit to the
tokens it is scored on costs 21-23% at rank 64 — so this is not a
generalization failure alone; post-hoc, KV is not low-rank enough at any useful
rank.

The probe did find real structure: keys are strongly low-rank **before** RoPE
(16.9% vs 48.2% at rank 64), because rotation spreads identical content across
directions by position. That independently reproduces why MLA carries a
*decoupled* RoPE key. It is still not cashable post-hoc — pre-RoPE storage
forces an `r -> D` lift before rotating, forfeiting the absorption, and V has no
RoPE excuse at 43%. The generalizable lesson: rank works when the model is
*trained* with the bottleneck; post-hoc SVD does not recover a structure the
model never had.

### 7. Keys are the sensitive tensor — the iid fixture said the opposite

Teacher-forced on OLMoE-1B-7B over 1024 tokens, 4-bit weights held constant so
the cache is the only variable (`bench/context/kv_fidelity.json`):

| config | ppl | delta | argmax agreement |
|---|---:|---:|---:|
| fp16 cache | 5.978 | — | 100% |
| K4 V4 (3.56x) | 6.102 | +0.124 | 93.2% |
| K4 V16 (keys only) | 6.061 | +0.083 | 94.6% |
| K16 V4 (values only) | 5.991 | **+0.013** | 97.3% |

Quantizing K alone costs ~6x more perplexity than quantizing V alone — the
reverse of the iid fixture in C3, which measured V-dominant error and would have
argued for coarse keys. Real keys carry per-channel outliers that blow up a
64-element block's shared absmax; Gaussian noise has none, so the fixture could
not see it. This is why the finding is recorded here and not inferred: the
fixture was measuring a property of the fixture.

Two consequences. First, there is a knee worth exposing as a dial: **values-only
is 1.56x for +0.2% perplexity**, while the remaining 2x costs six times more.
Second, the format lever for keys is granularity, not bit-width — per-channel
scaling, which is what the KV-quantization literature converged on and now has a
local reason.

### 8. Free-running output identity is not a fidelity metric

The same three configs, greedy-decoding 96 tokens from a 256-token prompt:

| config | delta ppl | argmax agreement | free-running |
|---|---:|---:|---|
| K4 V4 | +0.124 | 93.2% | **identical, 96/96** |
| K4 V16 | +0.083 | 94.6% | diverges at token 31 |
| K16 V4 | **+0.013** | **97.3%** | diverges at token **13** |

The ordering inverts: the config that is best on both rigorous metrics diverges
earliest, and the worst one reproduces the reference exactly. Free-running
identity is a single Bernoulli draw over whether some early step happened to be
near-tied — once one flips, the continuations are simply different text, and the
match count after that point measures nothing.

Recorded because it is a live trap rather than a curiosity: the natural
experiment is to run the one interesting config (K4 V4), and on that evidence
alone the honest-looking conclusion is "output is byte-identical, so this is
effectively exact." It is not; the other two arms are what expose it as luck.
Quote teacher-forced agreement and delta-perplexity. Treat a free-running match
as an anecdote about one prompt.

## What this changes downstream

- **C1** — every published VRAM figure gains its context qualifier; serving docs
  gain a 512-vs-32K worked example.
- **C2** — `plan_placement()` takes `context_len` and subtracts `KV(context)`
  from the budget **before** hot-set sizing, and records the planned context in
  its receipt (a plan computed at 512 and run at 32K is the failure mode).
- **C3** — KV quantization. **Implemented and measured** for nf4: `kernel/nf4_kv.py`
  (attention that reads a 4-bit cache in the mainloop) with `kernel/test_nf4_kv.py`,
  21/21 on the A2000. Corrections to the estimate this document originally carried:
  the saving is **3.56×, not 4×** — the fp32 blockwise absmax is a side channel
  (per token per head: 64 nibble-bytes + 2×4 B absmax = 72 vs 256 bf16) — so the
  235B 32K case measures **5.88 GB → 1.65 GB**, not 2.94. And it is not free:
  the decode attention step costs **2.5–3× fp16 SDPA** (15.4 vs 6.3 ms at 32K),
  because v1 reads the cache twice (scores, then weighted sum). Fidelity on an
  iid fixture: 9.3% relative error end-to-end, decomposing to **K-only 1.3% /
  V-only 9.2%** — the softmax *contracts* K error (9.2% logit → 1.4%) while V
  error passes through unattenuated. That inverts the usual "K is the sensitive
  one" guidance, which derives from per-channel outliers an iid fixture does not
  have; the asymmetric-precision decision therefore belongs to the real-scale
  perplexity gate, not to this fixture.
- **C4** — for the batch regime, KV becomes a streamed tier alongside the weights;
  the transfer law gains a KV term (`bytes_per_token = cold_weights + 2 × KV_layer_slice`).

## Reproducing

`bench/context/kv_budget.py` (derivation, from config.json only) and
`bench/context/kv_verify.py` (the A2000 rung-one probe). Receipts:
`bench/context/receipts-c0-20260724/`.
