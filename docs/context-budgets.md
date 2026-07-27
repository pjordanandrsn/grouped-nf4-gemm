# Context budgets — the KV cache as a first-class VRAM term

Every VRAM figure this project has published was measured at **short context**
(seq-512 prefill, 128-token decodes — the receipts say so). The KV cache is the
second memory consumer, and it grows linearly in context. This document makes it
a derived, measured, budgeted quantity.

**Status: Phase C0.** Per-layer KV geometry and the bounded/unbounded split are
**measured on the A2000** against each architecture's real `modeling_*.py` (see
*Verification* below) — and since 2026-07-25 that includes **full depth for all
seven models**, gpt-oss-120b and Qwen3-235B included, on the same 12 GB card.
What is still not measured for those two is real *weights* and real *width*;
rows are labelled accordingly. Kimi K2 is **derived-only** (no local weights).
K3's config does not exist publicly yet — its row lands when it does.

## The budget identity

```
VRAM_total = weights_resident + hot_set + KV(context) + activations + overhead
```

| term | source |
|---|---|
| `weights_resident` | measured per model/mode (e.g. 235B NF4 offload: base ≈ 15.1 GB) — the stamped flagship receipts |
| `hot_set` | K experts × bytes/expert, chosen by the **caller** — see C2 below for why no planner ships |
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
| Qwen3-235B-A22B | **188.0** | — | 0.73 GB | 1.47 GB | 5.88 GB | 23.50 GB | measured (**full depth**, narrowed width) |
| Qwen3-30B-A3B | **96.0** | — | 0.38 GB | 0.75 GB | 3.00 GB | 12.00 GB | measured (A2000) |
| gpt-oss-20b | **24.0** | 3.0 MB | 0.10 GB | 0.19 GB | 0.75 GB | 3.00 GB | measured (A2000) |
| gpt-oss-120b | **36.0** | 4.5 MB | 0.14 GB | 0.29 GB | 1.13 GB | 4.50 GB | measured (**full depth**, narrowed width) |
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

### Rung 1.5 — full depth on the small card (2026-07-25)

Rung two was scoped as cloud work on the premise that gpt-oss-120b and
Qwen3-235B cannot be held at real depth by a 12 GB card. **That premise was
wrong, and it cost nothing to check.** Depth is the *only* thing the rung-one
probe truncates — vocab, MLP and experts were already shrunk, on the stated
grounds that they do not touch the geometry under test. Narrowing `hidden_size`
the same way removes the last obstacle, because the cache is
`[B, num_key_value_heads, T, head_dim]` and **both fields are read from the
config**, so width is not part of what is being measured. That holds only where
`head_dim` is explicit — otherwise transformers derives it from `hidden_size`
and the shrink would move the number — so the probe refuses when it is absent,
and additionally re-derives after shrinking and requires an exact match.

Run at real depth with the real model classes:

| model | real depth | measured B/token | derived | err |
|---|---:|---:|---:|---:|
| Qwen3-235B-A22B | **94 layers** | 192,512.0 | 192,512 | **0.00%** |
| gpt-oss-120b | **36 layers** (18F+18S) | 36,864.0 | 36,864 | **0.00%** |

So the mechanism rung two exists to catch — something depth-dependent altering
the slope at real depth — **is not there**, and gpt-oss-120b's 18F+18S split is
confirmed at full depth rather than inferred from a 4-layer window.

**What is still not established**, and why the rows are not simply promoted:
real weights and real width. Both are *argued* irrelevant — geometry is a
function of config and code, which is rung one's founding premise and now holds
across seven model×depth combinations, and the width shrink is self-checked to
leave the derivation identical — but argued is not measured. Rung two as
originally specified (full-depth **real-weight**) remains open, and whether
rung 1.5 clears C1's publication gate is a judgement about how much that
residual is worth, not a measurement.

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

### 9. Per-channel key scaling loses — and it fails for the same reason low-rank did

Finding #7 said keys are the sensitive tensor and blamed per-channel outliers
blowing up a 64-element block's shared absmax. The obvious remedy is to group
the absmax along **tokens** instead, giving every channel its own scale — the
move the KV-quantization literature converged on. Implemented (one kernel,
two constexpr divisors; `quantize_kv_perchannel`) and measured at identical
bytes, since both schemes store one fp32 scale per 64 quantized values:

| config | ppl | delta | argmax | cache |
|---|---:|---:|---:|---:|
| K4 V4 per-token | 6.102 | +0.124 | 93.2% | 36.00 MB |
| K4 V4 per-channel | 6.311 | **+0.333** | 90.5% | 36.00 MB |
| K4 V16 per-token | 6.061 | +0.083 | 94.6% | 82.00 MB |
| K4 V16 per-channel | 6.253 | **+0.275** | 92.4% | 82.00 MB |

3.3x worse at the same cost. The group-size sweep says why — degradation is
monotone in how many tokens share a scale (+0.035 at group 8, +0.210 at 16,
+0.269 at 32, +0.275 at 64). So key magnitude varies strongly **across tokens**,
and sharing a channel's scale over 64 of them lets one loud token spoil the
other 63: precisely the failure per-channel scaling was adopted to fix, moved
to the other axis. Group 8 does beat per-token (+0.035) but at 96 MB, where the
absmax equals the packed bytes — 8 bits/value, outside 4-bit territory
altogether, and int8 would serve that budget better.

**The unifying result.** Both rejected schemes group along the TOKEN axis: a
basis fit across tokens (#6) and a scale shared across tokens (#9). Both lose.
Per-token blockwise groups *within* a token and wins. Post-RoPE keys are
hostile to cross-token grouping, and rotation is a sufficient explanation for
both — it makes a channel's value oscillate with position, which inflates
apparent rank and inflates within-group magnitude spread at the same time. The
prediction that falls out: cross-token schemes should be applied to **pre-RoPE**
keys, which is what MLA does and why it needs a decoupled rope key. For a
post-hoc cache on a RoPE model, group within the token.

**Method note, twice earned.** Both times a synthetic fixture endorsed the idea
and the model rejected it. The iid fixture had no outliers, so it made V look
dominant (#7); the outlier fixture had token-invariant channel gains, so it made
per-channel look like a win. Each fixture confirmed the hypothesis built into
it. `_outlier_cache` in `kernel/test_nf4_kv.py` carries this caveat inline, and
its test is named for the precondition rather than the conclusion.

The dial (`key_scaling="per_channel"`) is kept, defaulted off: it is correct at
the kernel level (matches its oracle to 1e-5), the cost is genuinely equal, and
an architecture that does not rotate its keys may land differently. It is a
measurement others can repeat, not a recommendation.

### 10. Cross-architecture validation: ~2% is the generalizable number, the K/V knee is not

Findings #7-#9 all came from OLMoE-1B-7B, which has **no GQA** (16 kv heads = 16
attention heads) while every other model in the table above uses it. Re-ran the
teacher-forced gate on three more architectures
(`bench/context/validate_arch.py`), each with a **control arm** — the same cache
class with quantization disabled — so cache semantics are separated from
quantization:

| model | GQA | head_dim | base ppl | K4V4 rel | keys-only | values-only | K/V gap |
|---|---|---:|---:|---:|---:|---:|---:|
| OLMoE-1B-7B | 1:1 | 128 | 5.978 | +2.07% | +0.083 | +0.013 | 6.4x |
| Gemma-4-26B-A4B | 2:1 | 256 | 3.824 | +2.17% | +0.070 | +0.041 | **1.7x** |
| SmolLM2-135M | 3:1 | 64 | 10.507 | +16.6% | +1.294 | +0.139 | 9.3x |
| gpt-oss-20b | 8:1 | 64 | *see below* | — | +325.9 | -0.079 | direction only |

**What replicates.** Keys are the sensitive tensor in 4/4. And the headline cost
is stable where it matters: full NF4 KV is **+2.1% relative perplexity on both
real MoE models** — two independent architectures, different GQA, different
head_dim, one hybrid. SmolLM2's +16.6% is a 135M model with little redundancy to
spare, not a counterexample to the trend.

**What does NOT replicate — and corrects finding #7.** The values-only "knee"
was sold off OLMoE as 1.56x for +0.2%. On Gemma-4 it is 1.56x for +1.1%, while
full quantization buys 3.56x for +2.2% — so on Gemma you take the full
quantization and the knee is pointless. The K/V gap ranges 1.7x to 9.3x and is
**architecture-dependent**; treat values-only as a dial to measure per model,
not a recommended default.

Also worth recording as a dead hypothesis: GQA does not explain the gap. The
ratio was the reason for running this at all (one kv head feeding many query
heads should amplify its error), but 1:1 -> 6.4x, 2:1 -> 1.7x, 3:1 -> 9.3x is
not a trend in either direction.

**gpt-oss is excluded on fixture grounds, not model grounds.** Its wikitext
perplexity is 143.8 where a 135M model scores 10.5, which initially looked like
the mxfp4 -> nf4 expert requant corrupting weights. It is not: the model answers
correctly ("The capital of France is" -> " Paris.", and it continues
"1, 2, 3, 4," -> " 5, 6, 7, 8, 9,") and then switches into assistant mode
mid-continuation. It is a chat model and raw wikitext is out of its
distribution, so perplexity on this fixture is not a meaningful metric for it.
Its K/V direction (keys catastrophic, values free) is consistent with the other
three; its magnitudes are not usable.

**The control arm earned its place.** It matched the fp16 reference to four
decimals on all four models, which settles the hybrid-cache question: the cache
does not implement sliding-window truncation, but transformers builds the mask
from `config.layer_types` independently, so outputs are correct. The open cost
is memory, not correctness — past the window a sliding layer should hold
`window - 1` tokens (finding #3) and this cache holds all of them.

Two bugs in the cache surfaced here that OLMoE could not reach: `get_mask_sizes`
omitted the tokens about to be written, giving a zero-width mask on any model
using an explicit additive mask (gpt-oss, via attention sinks), and the harness
held two full fp32 logit tensors, which OOMs at Gemma's 262k vocab. Neither was
a quantization bug; both were only findable on a second architecture.

### 11. Token-axis sparsity: composes cleanly, but loses badly at matched bytes

Sparsity was the recommended next lever precisely because it is the one axis
that does NOT group across tokens (#6, #9) — eviction removes tokens rather
than sharing structure between them, so it should compose with NF4 instead of
competing. Implemented as sink+recent retention
(`NF4KVCache(keep_sink=, keep_recent=)`) and measured with chunked teacher
forcing so eviction happens between forwards, the reference arm chunked
identically:

| arm | ppl | delta | cache | ratio |
|---|---:|---:|---:|---:|
| full fp16 | 5.968 | — | 128.00 MB | 1.00x |
| full nf4 | 6.086 | **+0.118** | 36.00 MB | 3.56x |
| sink4+rec512 fp16 | 6.624 | +0.656 | 64.50 MB | 1.98x |
| sink4+rec512 nf4 | 6.793 | +0.824 | 18.14 MB | 7.06x |
| sink4+rec256 fp16 | 9.284 | **+3.316** | 32.50 MB | 3.94x |
| sink4+rec256 nf4 | 9.469 | +3.501 | 9.14 MB | 14.00x |
| sink4+rec128 fp16 | 10.504 | +4.536 | 16.50 MB | 7.76x |
| sink4+rec128 nf4 | 10.679 | +4.711 | 4.64 MB | 27.58x |

**Quantization dominates eviction at matched bytes.** NF4 buys 3.56x for
+0.118; eviction buys 3.94x for +3.316 — 28x the quality cost for the same
memory, and NF4 wins at every matched point measured. The prediction that
sparsity was "the biggest remaining lever" is wrong on this workload.

**Composition is real, and is the usable result.** NF4's cost is near-constant
regardless of how much has been evicted: +0.168 on top of rec512, +0.185 on
rec256, +0.175 on rec128, against +0.118 alone. The two axes are genuinely
independent — exactly what low-rank and per-channel scaling failed to be — so
quantizing on top of an eviction policy you already run is close to free.

**Scope, which matters more than usual here.** Wikitext perplexity is the least
favourable possible task for eviction: next-token prediction over contiguous
prose depends on the dense recent context the policy deletes. StreamingLLM's
claim is not "equal quality at lower memory" but *stable* perplexity on streams
longer than training length — a different objective this fixture cannot test at
1024 tokens. And sink+recent is the weakest selection rule; H2O/SnapKV retain
the heavily-attended tokens rather than merely the newest. Read this as "naive
recency eviction is a bad trade for dense next-token prediction", not as
"sparsity does not work".

**Both confounds were tested in #13 and neither rescues eviction.** A better
selection rule narrows the 28× gap to 8.8× and a sparse long-range fixture does
not reverse it; the 28× above should be read as rule-specific.

**Two cache bugs, both found by this measurement.** `get_query_offset` returned
a hardcoded 0, which is correct only from an empty cache — chunked prefill
scored ppl 330 against 5.97 single-shot, and every prior single-forward control
arm on four architectures had matched its reference exactly while saying
nothing about the accumulating path. And eviction initially ran inside
`update()`, contradicting a mask already built for the pre-eviction length; it
is now explicit and called between forwards. A third was avoided rather than
measured: once tokens held != tokens seen, `cache_position` must follow the
TRUE count or newly written keys get RoPE rotations inconsistent with retained
ones — that would have looked like "sparsity degrades quality" instead of a
bookkeeping error.

### 12. The 4-bit cache reads at or below fp16 cost — once dequant stops running per query head

> ## ⚠ ERRATUM (2026-07-25) — this finding's headline is WRONG
>
> **The fp16 baseline it was measured against materialized a 16× replicated
> cache.** `torch.scaled_dot_product_attention` has taken `enable_gqa=True`
> since torch 2.5, which broadcasts kv heads inside the kernel instead. Same
> device, same shape (T=32768, H_kv=4, H_q=64, bf16), measured 2026-07-25:
>
> | bf16 SDPA baseline | time | effective KV bandwidth |
> |---|---:|---:|
> | `repeat_interleave` (what #12 measured) | **6.205 ms** | 10.8 GB/s |
> | **`enable_gqa=True`** (correct) | **0.324 ms** | **206.8 GB/s** |
> | fused NF4 `attend_nf4_kv_gqa` | 3.760 ms | 5.0 GB/s |
>
> 6.205 ms reproduces #12's reported 6.055 ms, which is how the baseline was
> identified. The correct baseline runs at **72% of the A2000's ~288 GB/s**, i.e.
> it is memory-bound as a decode kernel should be; the NF4 kernel runs at **1.7%
> of peak** and is nowhere near memory-bound despite moving 3.56× fewer bytes.
>
> Both baselines are **correct** — each lands 2.34e-3 from an fp32 reference, so
> `enable_gqa` is not skipping work.
>
> **So "0.82× fp16 SDPA" should read "≈11.6× SLOWER than fp16 SDPA".** #12's
> pre-committed decision — make the GQA-batched kernel the decode default and
> remove the latency caveat — was taken on that wrong baseline and does not
> stand. Even at tf32 (#12's 2.355 ms) the kernel is 7.3× slower.
>
> The finding is left below **as written and as scored**, because editing a
> falsified conclusion to match later evidence is the one thing this document
> forbids. What is corrected is the claim, here, in front of it.
>
> **This also opens a hole nothing in #1–#16 measured**: every latency arm in
> this document compares NF4-cache configurations against *each other*. None
> compares against a **bf16 cache with attention invoked properly** — and the
> shipped path (dequant a layer, then SDPA) measures **10.750 ms** against that
> baseline's 0.324. The memory dial's true latency cost is unmeasured and is
> evidently large. See finding #17.


Three registered attempts, the first two falsified (see the stamped
`bench/context/PREREG-kv-context.md`):

| attempt | hypothesis | result |
|---|---|---|
| B1/B2 | single-pass fusing removes the scores intermediate | **falsified** — 0.79×, *slower* than two-pass |
| B5 | fusing lost parallelism; split the token axis | **falsified** — 0.90×, still slower |
| B6 | dequant runs per QUERY head; batch the GQA group | **confirmed** — 2.81× vs two-pass, **0.82× vs fp16** |

Both failures were models of *why* the kernel was slow, and both were wrong.
What settled it was a measurement, not a model: holding H_kv=4 and varying H_q
on **byte-identical** input, time scaled linearly with query heads above the
occupancy knee (4.60 ms at 4:1 → 13.83 ms at 16:1). `grid=(H_q, ...)` made each
query head re-dequantize the same kv bytes, so GQA 16:1 paid 16× redundant ALU.
The path was never memory-bound — it moves ~19 MB against fp16's ~67 MB.

The fix is a grid over **kv** heads: dequantize each block once, then serve all
its query heads with one `[GQA, D] × [D, BLOCK_T]` dot. At T=32768, H_kv=4,
D=128 (A2000, median of 25):

| H_q | GQA | two-pass | gqa-batched | fp16 SDPA |
|---:|---|---:|---:|---:|
| 16 | 4:1 | 5.943 ms | 5.317 | 1.158 |
| 64 | 16:1 | 13.997 ms | **4.975** | 6.055 |

**Scope.** 0.82× holds at GQA 16:1; at 4:1 the kernel is 4.59× *slower* than
fp16, since there is little redundancy to remove. The claim is "readable at or
below fp16 cost in the high-GQA long-context regime" — where current models sit
(Qwen3 16:1, gpt-oss 8:1) — not "4-bit attention is free". One device.

**Two implementation notes worth carrying.** The shared-memory failure was fixed
by reusing `_device_shared_limit` from the CDNA3 LDS work, but a static estimate
was insufficient: 82 KB modelled against a 99 KB cap still failed, because
`num_stages` pipelining multiplies staging by a factor the caller does not set.
It now pins `num_stages=2` and steps `block_t` down on the actual
`OutOfResources`. And `tl.dot` silently defaults to **TF32**, which measured
2.7e-3 relative error at extreme logits — small, but a second error source
stacked on the quantization error this module exists to characterize, so
`input_precision="ieee"` is the default at a measured 2.11× cost, with a test
pinning that tf32 is worse so the default cannot be flipped silently.

### 13. A better eviction rule narrows the gap 3× and still loses to quantization

#11 closed with two named confounds: sink+recent is the weakest selection rule,
and wikitext next-token prediction is the least favourable task. Experiment A
(registered and stamped in `bench/context/PREREG-kv-context.md` before it ran)
tested both — H2O-style selection by accumulated attention, and an `induction`
fixture where the dependency is sparse and 256 tokens away, which a 128-token
recency window cannot serve by construction.

**The confounds were real and they do not rescue eviction.**

| wikitext, fp16, chunk 128, matched bytes | held | resident | ppl | Δ |
|---|---:|---:|---:|---:|
| full cache | 1024 | 128.00 MB | 5.968 | — |
| **full cache, NF4** | 1024 | **36.00 MB** | 6.091 | **+0.124** |
| recency (sink4+rec256) | 260 | 32.50 MB | 9.294 | +3.326 |
| H2O (sink4+rec64+top192) | 260 | 32.50 MB | 7.062 | **+1.094** |
| static (sink256+rec4) | 260 | 32.50 MB | 6.936 | +0.968 |

Selection is worth a factor of 3: #11's 28× quality gap at matched bytes narrows
to **8.8×** under H2O. It does not close. Quantization still wins at every
matched-byte point measured, which is the result that carries.

**On the induction fixture, no policy at a 132-token budget comes close.**
Second-copy perplexity, geometric mean over 3 seeds; the full-cache induction
gain is 97,778× (first copy 147,627 → second copy 1.510), so the fixture is
capable of testing this:

| policy (132 tokens held, chunk 128) | 2nd-copy ppl | induction gain | log-gain retained |
|---|---:|---:|---:|
| recency (sink4+rec128) | 103,484 | 1.4× | **3.1%** |
| H2O (sink4+rec64+top64) | 8,051 | 18.3× | 25.3% |
| **oracle** — scored on the whole sequence | 9,832 | 15.0× | 23.6% |
| static (sink128+rec4) | 456 | 323.7× | 50.3% |

The oracle is the load-bearing row. It scores with attention accumulated over
the **entire** sequence, including the second-copy queries that have not run
when eviction fires — a policy nobody could deploy, and an upper bound on what
this signal can buy. It is **no better than causal H2O**. The failure is not
that importance is unobservable when it is needed.

**It is that accumulated attention mass is confounded by opportunity.** Summing
over query positions rewards a token for how many queries *could* attend to it,
and that count falls linearly with position: token 5 is scored by ~500 queries,
token 400 by ~100. Dumping the keep-sets shows it directly — H2O and the oracle
both retain ~50 **contiguous** tokens from the start of the sequence, and agree
with each other far more than either agrees with any notion of importance. At
chunk 128 that makes H2O barely distinguishable from a static re-split of the
same budget: sink128+rec4 scores 8.715 against H2O's 8.773.

**But H2O is not only that, and the chunk size is what separates them.** Chunked
teacher forcing hands every query its own chunk in full, so up to 127 tokens of
local context arrive free of the budget. Shrinking the chunk removes that
subsidy. The static optimum then moves violently — the *same* split goes from
best to worst — while H2O, which re-selects at every boundary, barely moves:

| wikitext, budget 132 | chunk 128 | chunk 32 | chunk 8 |
|---|---:|---:|---:|
| full cache (control) | 5.968 | 5.952 | 5.959 |
| sink4+rec128 | 10.515 | 11.035 | 11.292 |
| sink68+rec64 | 9.129 | 9.529 | 9.776 |
| sink128+rec4 | **8.715** | 10.389 | 12.812 |
| H2O (sink4+rec64+top64) | 8.773 | **8.974** | **9.264** |

So the tidy version of this finding — "H2O is StreamingLLM with extra steps" —
was available after three diagnostics and is **wrong**; the fourth killed it.
H2O's *advantage over recency* is largely budget re-allocation toward early
tokens, and a static split captures it at chunk 128. Its adaptivity is
nonetheless real and shows up exactly where a static rule cannot follow: from
chunk 128 to chunk 8 H2O gives up 0.49 ppl against the best static split's 1.06
and a fixed sink128+rec4's 4.10.

The control moves by ≤0.016 — the known bf16 chunk-size kernel effect, two
orders of magnitude below the eviction arms' swings — so the protocol change is
doing what it claims. **Every eviction arm degrades as the chunk shrinks toward
true decode while the quantization arm does not**, so the 8.8× gap measured at
chunk 128 is the *most* favourable reading eviction gets here, not the least.

**Quantization's own cost is task-dependent, which the "~2%" headline hides.**
NF4 costs 0.020 nats/token on wikitext and **0.120 on induction** — 6× more on a
task that turns on matching an exact token 256 positions back. Finding #10's
~2% generalizes across *architectures*; it does not generalize across *tasks*,
and retrieval-shaped workloads should expect several times that.

**Scope.** One model (OLMoE-1B-7B, MHA), one device, 1024 tokens. `induction` is
a **mechanism fixture** and is labelled as such: `sink128+rec4` wins on it partly
because the fixture's dependency sources *are* the first 128 tokens, so "keep the
oldest" is degenerate with "keep the answer" — which is why the wikitext column,
where no such degeneracy exists, is the one that ranks policies. Predictions A1
and A3 (at its primary budget) were confirmed, A2 and A4 falsified; the scored table and both
disagreements between a prediction's prose and its falsification test are in the
prereg. Per its pre-committed decision, **token-axis sparsity is closed out for
this project** and the remaining lever is quantization granularity. The
arbitrary-keep-set primitive (`NF4KVCache.evict_index`) is in e4b and tested; the
H2O policy stays in `bench/context/`, unpromoted, exactly as the rejected
low-rank probe did.

### 14. Streaming the KV cache (C4) is worth nothing at batch 1, and the link decides the rest

C4 says the cache becomes a streamed tier alongside the weights. Derived before
building any of it, because the arithmetic is cheap and the conclusion is not
the expected one.

**The asymmetry that makes this its own problem.** Streamed *weights* amortize
over a batch — one H2D copy of an expert serves every sequence in the step.
Streamed *KV does not*: each sequence owns its cache, so

```
step_bytes = W + batch × KV(ctx)          tok/s = batch × link / step_bytes
```

Two numbers follow, and they are the whole decision:

- **ceiling = link / KV(ctx)** — per-token KV bytes are constant in batch, so
  this caps aggregate throughput at *every* batch size. It is also exactly what
  you get when the weights are resident and only the cache streams.
- **batch\* = W / KV(ctx)** — where KV traffic overtakes weight traffic. Past it
  the thing you increased to amortize the weights is what you are now paying for.

**NF4 moves batch\* by exactly 32/9.** bf16 KV costs 2 B/element; NF4 costs
0.5 + 4/64 = 0.5625. For Qwen3-235B at 32K against the one anchored weight
stream (~11.7 GB/token, back-solved from `RESULTS-ikllama-ab.md`'s 26.74 GB/s ÷
2.29 tok/s, valid only if that arm was purely transfer-bound), batch\* goes
**1.9 → 6.6**. That is a justification for KV quantization with nothing to do
with fitting in VRAM: in a streamed regime it buys *batch headroom*.

**The measured link, and one refuted concern.** The A2000 sustains **6.20 GB/s**
pinned H2D — consistent with PCIe 3.0 x8, and **4.3× below** the 26.74 GB/s box
the transfer law was measured on. The fit gives a 25.6 µs per-copy fixed cost,
which looked alarming (a 94-layer model paying `94 × c` per token), but the
decode shape refutes it: **94 separate slices cost 1.00× one blob of the same
bytes**. Queued copies pipeline, so `L × c` is not a term — it only bites a
design that synchronizes per layer, which is therefore the design to avoid.

**Whether C4 pays at all, as a function of batch.** A cell is a genuine window
only if the resident NF4 cache does *not* fit **and** the streamed ceiling still
meets the target. 21 model×context cells, 8 GB free for KV, 5 tok/s target:

| target batch | link 6.20 (A2000) | link 26.74 (reference box) |
|---|---:|---:|
| 1 | 0 / 21 | **0 / 21** |
| 8 | 1 / 21 | 6 / 21 |
| 32 | — | 10 / 21 |

At batch 8 on the reference box the breakdown is **14 resident-already-fits, 6
window, 1 impossible** (Qwen3-235B at 128K, whose 3.8 tok/s ceiling no batching
can lift). So:

- **At batch 1, C4 is unnecessary everywhere** — Finding #10's NF4 cache already
  fits every cell, so C3 solved the single-sequence case outright. This is the
  one headline here that is *not* robust to its assumption, and it is the free-KV
  budget it turns on, not the link: windows at batch 1 go **3 / 1 / 0 / 0** for
  2 / 4 / 8 / 16 GB free. Below ~4 GB — a 12 GB card with streamed weights — the
  single-sequence case reopens.
- **C4 is a batch-regime capacity feature**, exactly as the C4 line said, and its
  value is monotone in batch.
- **On the A2000 the window is 1/21 and that cell is OLMoE-1B-7B.** Building and
  measuring C4 here would characterize a regime that does not exist; the honest
  perf number needs a gen4/5 ×16 link.

**Scope.** The window criteria (free VRAM, target batch, target tok/s) are
parameters, not facts — the table above uses plausible defaults, and a real
deployment target changes which cells light up. `kv_stream_budget.py` takes them
as arguments for that reason. The KV side of every figure is derived exactly
from config; the weight side is a single anchored point and is labelled wherever
it is used.

### 15. #14's transfer model is measured, not assumed — and `pinned` is only half of `fast`

#14 derived C4's whole window table from one unmeasured claim:
`step_time = bytes / link`. `NF4KVCache(residence="host")` now exists and is
byte-exact, so that claim was preregistered (`PREREG-kv-streaming.md`, stamped
before any arm ran) and tested against the geometry of Qwen3-235B — 94 layers,
4 kv heads, head_dim 128, synthetic, no weights, which is what lets a
235B-shaped cache be measured on a 12 GB card.

**The model holds to 1%.** Per-step transfer overhead, `t_host − t_gpu`, against
the packed byte count over the measured 6.20 GB/s asymptote:

| arm | bytes | predicted | measured | ratio |
|---|---:|---:|---:|---:|
| nf4 32K | 1.774 GB | 286.2 ms | 288.6 ms | **1.009** |
| nf4 32K (replicate) | 1.774 GB | 286.2 ms | 286.4 ms | **1.001** |
| bf16 8K | 1.578 GB | 254.4 ms | 271.4 ms | 1.067 |
| bf16 4K | 0.789 GB | 127.2 ms | 136.2 ms | 1.071 |

So #14's ceilings and `batch*` are describing the tier that actually exists, and
**a faster link changes only the constant** — the shape of that window table
does not need a rented box to be trusted. This also reverses the scoping call in
#14: a *slow* link is the better place to test the law, because the transfer term
dominates everything else.

**The run's most valuable output was a bug it was not looking for.** The
unquantized host path allocated its arena `[1, H, cap, D]` and handed out prefix
views sliced on **dim 2** — non-contiguous. `is_pinned()` returns **True** for
such a view, so every guard passed while the DMA collapsed to **0.09 GB/s
against 0.95** on the same device, a 17× penalty across the full cache. Fixed by
making the arena token-major `[cap, H, D]`, matching the packed layout.
**Pinned is necessary and not sufficient; contiguous is the other half, and no
API surface says so** — the only symptom is a number that looks like bad
hardware.

**Capacity delivers, at 4.9× rather than the 20× predicted.** Peak GPU allocated
at 32K is 429 MB streamed against 2112 MB resident, and cache residency itself
is zero. The registered `< 0.10` threshold was falsified because peak is
dominated by ~340 MB of prefill transients present in *both* arms, which a ratio
cannot cancel — a defect in how the prediction was operationalized, not in the
feature.

**Two other registered predictions failed and one is a methodology warning.**
The resident path's append *is* O(T) (`torch.cat` reallocates the packed store
per layer per step) but at device bandwidth that is ~9 ms, so per-call overhead
across 94 layers dominates and the ratio runs backwards. And `t_host − t_gpu` is
a difference of two separately-timed loops carrying ~±15 ms of noise: fine
against a 286 ms overhead, useless against a 36 ms one. One prediction confirmed
at 3.961 in the first run and would falsify at 4.970 in the second **from
identical code**. Any future claim off this harness needs a difference large
relative to that floor, or a direct measurement instead of a subtraction.

**Scope.** One device, one geometry, no prefetch — copies and dequant serialize
on the default stream, which is exactly what makes an *additive* model the right
one to test. A prefetched implementation would break additivity by design and
must not be scored against this.

### 16. Prefetch hides 96% of the transfer in a synthetic harness — and none of it in a real decode

#15 measured the streamed KV tier obeying `t = bytes/link` and left it there:
+288 ms per step at 32K, additive. Registered in `PREREG-kv-stream-faster.md`
(stamped before any of it was built) and now measured, **two changes take most
of that back.**

**Prefetch (B1) — all three predictions confirmed IN A SYNTHETIC HARNESS, and
the result does not transfer.** The registered follow-up (E1) ran the same
mechanism on a real OLMoE decode and measured **−22.5%**: prefetch made it
slower. The harness timed *loads only* — it never called `update()`, so it never
paid the append, never paid the assembly, and never exercised the interleaving
that turned out to make the design incorrect. **Read the table below as a
property of that harness, not of decode.** The real-model numbers, and the two
defects E1 exposed, are in the block after it.

| ctx | resident | streamed | streamed + prefetch | transfer hidden |
|---:|---:|---:|---:|---:|
| 8192 | 245.4 ms | 318.8 | **248.5** | **95.8%** |
| 32768 | 967.9 ms | 1256.3 | **979.8** | **95.9%** |

Exposed transfer at 32K falls from **288.4 ms to 11.9 ms**. The streamed tier
costs **1.2%** over holding the whole cache resident while keeping **zero** bytes
of it on the device (443 MB peak against 2108 MB — 4.76× less). The hidden
fraction is identical at both contexts, which is what a mechanism looks like
rather than a lucky ratio. Cost: one layer in flight, **+18 MB**.

**Split residency (A1) — the dial works, its predictions did not.** Keeping the
oldest f of the cache resident makes residence continuous, and the bytes track
the dial exactly (0.4435 / 0.8871 / 1.3306 GB measured against the same
expected). But all three numeric predictions were falsified, and only one of
those was the code's fault: `_materialize` assembled with a `cat` over
already-materialized halves, ~60 ms/step of avoidable copying. Fixing it moved
the fitted constant 88.0 → 54.0 ms and the fitted bandwidth 8.21 → 7.14 GB/s.
The other two failures were mine — an analysis plan that fitted one line through
two regimes (f=0 does no concatenation at all), and an interval written with the
sign inverted, since splitting trades VRAM *for* bandwidth and resident VRAM
must therefore **rise** with f.

**What this changes about #14, in the favourable direction — and why it is not
re-derived.** #14 treated transfer as strictly additive and fully exposed. With
overlap the step is `max(compute, transfer)`, so **`link/KV(ctx)` is the
transfer-bound rather than the bound**: it binds only when `KV/link` exceeds the
per-step compute, which here it does not, by 3.4×. #14's window counts are
therefore conservative. Re-deriving them needs a per-model compute estimate this
project does not have, and inventing one to widen a table in my own favour is
the move the preregistration exists to prevent.

**The substitution that will bite later, registered before either ran.** B1 hides
the transfer behind **dequantization** — work that exists precisely *because* the
data arrives packed. D1 (wiring in the fused `attend_nf4_kv_gqa`, #12) removes
that dequant. The two are **substitutes, not complements**: if D1 lands, B1's
95.9% falls, because there is less left to hide behind, and the two results must
never be multiplied together.

**What a real decode actually costs (E1, OLMoE-1B-7B, 4-bit weights resident,
4096-token prompt).** Registered before it ran, and it falsified the headline:

| arm | ms/step | KV on device |
|---|---:|---:|
| resident cache | 311.1 | 151.9 MB |
| streamed | 357.5 | **0** |
| streamed + prefetch | 367.9 | **0** |

**+18.3% for zero KV bytes on the device** is the honest number for the tier.
Prefetch contributes nothing here and costs 3%.

**Two defects, both invisible to a synthetic harness and to a green unit suite.**
The arena append used `copy_` into a CPU destination without `non_blocking`,
which **blocks the host** — 32 times per step, dragging a 6.20 GB/s link down to
a measured **1.87 GB/s**. That is the same failure `kernel/host_gather.py`
records for the expert path (B3, ~94 syncs/token), reappearing in the KV path.
And prefetch staged the cache *before* the step's token was appended, then used
it as the whole layer — silently dropping the newest token from attention. **The
unit suite was 25/25 green throughout**, because its prefetch test completed all
updates before any prefetch and so never produced the order a decode uses. A
run that was *faster and wrong* is what caught it.

> **REOPENED AND CONFIRMED 2026-07-25 (finding #18 changed the input).** The
> closure below rests on "transfer is 9% of a step". #18's fused dequant removed
> most of the step, taking that share to **46.7%** at 16K — and the *same
> prefetch code*, unmodified, flipped from −38.9% to **+29.0% hidden**, making
> the streamed arm **0.865×**. It still costs **9.6%** at 4K (17.8% share), so it
> ships **opt-in and off by default**, with the regime stated. The reasoning
> below was right about its ratio; the ratio moved.

**Prefetch is closed after a third attempt (E2).** The obvious diagnosis for
E1's loss was that the safety wait is a whole-stream barrier where a per-layer
event would do. Registered, built, measured: the narrowed version lost by
**more** (353.10 ms/step against 327.49 without it). The barrier was not the
problem.

**The reason generalizes.** The transfer is 24.4 ms of a ~262 ms step — **9%** —
and the machinery to hide it (an extra allocation per layer per tensor, a
staged-history concatenation the plain path never pays, cross-stream
bookkeeping) costs more than the 9% it chases. **At 9% of a step, transfer is
not worth machinery.** Prefetch would pay where transfer is a large fraction of
the step: long context with cheap compute — the opposite of this model, and of
most of what #14's table covers.

So the streamed tier ships at **+18.3% for zero KV bytes on the device**, and
scheduling is recorded as tried and rejected across three registered attempts
rather than left as a promising TODO. The levers that remain are the ones that
move **bytes**: NF4 (3.56×) and split residency — both shipped, both exact.

**Confirmed off this card (2026-07-25, A100-SXM4-80GB, $0.70).** #18's headline
reproduces at **1.144×** against the A2000's **1.133×** — 1% apart across a 7×
gap in memory bandwidth and a different link generation — and the decomposition
travels with it (wrapper 1.002/0.987, dequant 1.142/1.148). **So it carries no
device qualifier.** Variance on a dedicated card is **MAD 0.28%** against this
A2000's ±12%, which is where every error band in #13–#18 came from.

**Both of that run's open items are now closed (2026-07-25, second A100, $0.46).**
The dequant kernel measures **1423.8 GB/s amortized — 69.8% of an A100's
~2039**, against 248.7 (86%) on the A2000: **bandwidth-bound on both**, and the
earlier 276.5 GB/s was 5.15× low because it was synced per call at the smallest
problem size. There was never a portable kernel limit. And prefetch's behaviour
is now four clean points rather than a disputed law: it **loses at ctx 2048
(1.064) and wins monotonically from 4096 (0.943 → 0.885 → 0.872 at 16384)**,
crossover near 4096. NF4's cost rises only **0.042 across an 8× context
increase** (1.114 → 1.156), so the "grows with context" warning is real but
mild. A follow-up
attempt to model that crossover on the A2000 (K2) came back **void** — the
prefetched arm measured *faster than a fully resident cache*, and the resident
baseline was non-monotonic in context, both impossible. Two runs of identical
code on that card disagree about the sign of the effect. **All A2000-based
prefetch guidance is withdrawn**; what stands is the A100's two points, taken
where variance was 0.28%.

**Scope.** Two devices now. The synthetic harness's "compute" was
dequantization rather than attention, and the argument that a real decode has
more to hide behind was made in advance, tested, and **wrong**.

### 17. The control that was never run: NF4 KV costs ~1.13× decode

> **RESTATED 2026-07-25 (finding #18).** This finding first published
> **1.9–2.6×**. That number was measuring `dequant_kv_ref` — a function whose
> own docstring calls it a *test oracle* — sitting in the decode hot path.
> Replacing it with a fused kernel (bit-identical, 12.6× faster) brings the same
> measurement to **1.133× at 4K** and **1.244× at 16K**. The tables below are
> left as measured, with the pre-fix figures intact; the headline is corrected
> here. What the finding got right is the *shape* of the omission — there was no
> bf16 control — and that stands.

Sixteen findings of KV latency, every control another NF4 configuration. This
one is `DynamicCache` — transformers' own bf16 cache, what a user runs by
default. Registered in `PREREG-kv-vs-bf16.md` and stamped before the harness was
written. OLMoE-1B-7B, 4-bit weights resident, greedy decode:

| ctx | cache | ms/step | vs bf16 | peak VRAM | KV bytes |
|---:|---|---:|---:|---:|---:|
| 4096 | bf16 | 143.58 | 1.00× | 5406.4 MB | 541.1 MB |
| 4096 | **NF4** | 270.96 | **1.89×** | 5039.6 MB | 152.2 MB |
| 4096 | NF4 streamed | 312.17 | 2.17× | 4900.1 MB | **0 on device** |
| 16384 | bf16 | 237.27 | 1.00× | 8169.5 MB | 2151.7 MB |
| 16384 | **NF4** | 605.58 | **2.55×** | 6697.5 MB | 605.2 MB |
| 16384 | NF4 streamed | 741.88 | 3.13× | 6121.5 MB | **0 on device** |

All four predictions confirmed (F1a 1.887, F1b 1.887→2.552, F1c +366.8 MB,
F1d 2.174).

**So the trade is not "3.56× memory for ~2.1% perplexity".** It is 3.56× memory
for ~2.1% perplexity **and a decode cost** — measured at 1.9× with the oracle in
the path, and **1.133× once that is a real kernel** (#18). Both numbers travel
with the claim, in `kv_cache.py` and in C3 below.

**And the cost grows with context — the worst possible direction**, because the
dial exists *for* long context. ~~1.89× at 4K, 2.55× at 16K, still climbing…
expect worse than 2.6× at 128K.~~ **Superseded.** With the oracle out of the path
and measured on a quiet card, the growth is **1.114× at 4K → 1.156× at 32768**:
**0.042 across an 8× context increase**. The direction was right and the
magnitude was the A2000's noise plus the oracle. The warning stands as "mild and
real", not as "expect worse than 2.6×".

**Greedy ids diverge at position 1 of 33.** The first generated token can
already differ. That is what a lossy cache means and it is consistent with
#10's perplexity number — it belongs *next to* it rather than in a separate
finding, because together they are the actual trade.

**Decomposed (G1), because "upper bound" is not actionable.**
`NF4KVCache(quantize=False)` runs the same object, the same bookkeeping and the
same append/load path with no arithmetic, which splits the ratio cleanly:
at 4096 the **wrapper costs 1.138×** and the **dequant 1.475×**. So the dequant
is the target and the wrapper is not worth attacking — and `dequant_kv_ref` is
a *reference* implementation by name.

**Two honesty notes on the numbers above.** The 16384 decomposition is
**contaminated** and is not used: it puts the wrapper at 0.736×, i.e. faster
than `DynamicCache` while doing strictly more work on identical data, which is
not physical — peak is 8.17 GB of ~8.6 GB free and the arms are fighting the
allocator. And re-measuring F1 gave **1.679** where the first run gave 1.887, so
the honest headline is **~1.7–1.9× at 4K** and **~2.2–2.6× at 16K**; the third
digit never existed on a shared card. One model, one device,
GQA 1:1 — which is neutral for this path, since none of #12's `enable_gqa`
effect applies at 1:1.

**Why this took seventeen findings to run.** Every earlier comparison had an NF4
cache on both sides, so the dequant cancelled and became invisible. A control
that shares your feature with the treatment measures everything except your
feature.

### 18. Three quarters of #17's slowdown was a test oracle in the hot path

G1 split #17's cost into **wrapper 1.138×** and **dequant 1.475×**, which named
the target. The thing doing the dequantizing was `dequant_kv_ref`, whose own
docstring says *"Reference dequant of a packed cache. Test oracle."* — and which
is written like one: an int32 widening, two masks, a `stack`, a float32 LUT
gather, a `repeat_interleave` expanding each scale into 64 copies, the product,
and a cast. **Seven full-size intermediates, most of them 4-byte, for a 2-byte
result.**

A fused Triton kernel doing the same arithmetic in registers
(`dequant_kv_fused`), registered and stamped before it was written:

| `[4096,16,128]` → bf16 | reference | fused | speedup |
|---|---:|---:|---:|
| synced per call (as first measured) | 2.729 ms | 0.187 ms | 14.6× |
| **amortized** (K1's corrected stopwatch) | **2.544 ms** | **0.098 ms** | **26.1×** |
| effective bandwidth, amortized | 8.4 GB/s | **220.3 GB/s** | |

The synced figure **understates the speedup by 1.79×**: sync round-trips are ~48%
of the fused arm and only ~7% of the reference, so the instrument flattered the
thing it was measuring against. The honest number is **26×**, not the 12.6×
first recorded.

**Bit-identical**, `torch.equal`, across five shapes and both dtypes — including
`777×3×64` and `1×1×128`, because a dequant correct only on round numbers is
wrong in production. Both paths multiply an fp32 codebook value by an fp32 scale
and round once, so equality was the right gate rather than a tolerance.

End-to-end at 4096, OLMoE-1B-7B: **1.679× → 1.133×**, and the decomposition
collapses to **wrapper 0.987, dequant 1.148**. The wrapper is free; what remains
is real arithmetic.

**Two honest limits — one of which turned out to be my stopwatch.** The 99.9 and
113 GB/s figures were timed with `torch.cuda.synchronize()` around a single call.
Amortizing the launch (K1) puts the same kernel at **248.7 GB/s, 86% of this
card's ~288** — **~50% of the original measurement was sync round-trips**. So
there is no unexplained headroom and no kernel deficit; the kernel is essentially
at memory bandwidth. Every synced bandwidth recorded in #18 and in the rented run
is inflated in the same direction, most at the smallest problem sizes. And per-channel keys keep the
reference: their absmax is grouped over *runs of tokens*, a different indexing
problem, and that dial is off by default and measured worse (#9).

**What this reopens.** #16 closed prefetch because "the transfer is 9% of a step
and machinery to hide 9% costs more than 9%." With the dequant gone, the
streamed arm at 16384 now exposes **46%** of its step. That closure rested on a
ratio this change invalidated. Prefetch is *not* reopened here — doing so on an
argument rather than a registration is the failure mode this document set
exists to prevent — but it is no longer settled, and it would inherit three
prior falsifications.

**The generalizable lesson.** A function named `_ref` or `_oracle` is a
correctness artifact, and this one was load-bearing in decode for as long as the
KV cache has existed. #17 measured it faithfully and attributed it to 4-bit KV.
The control was right; the conclusion inherited whatever was in the path.

## What this changes downstream

- **C1** — every published VRAM figure gains its context qualifier; serving docs
  gain a 512-vs-32K worked example.
- **C2** — **CLOSED, not built.** The original item was `plan_placement()`
  taking `context_len` and subtracting `KV(context)` from the budget before
  hot-set sizing. That function does not exist, and building it would put the
  **first policy** into a library that is deliberately mechanism-only
  everywhere else: `enable_hot_residency` and `enable_pipelined_residency` both
  *take* `hot_sets` rather than computing them, `expert_profile` emits routing
  data without deciding anything, `serve.py`'s `vram_fraction` is a cap rather
  than a plan, and both rejected KV policies (low-rank, H2O selection) were kept
  in `bench/` for exactly this reason.
  A planner also needs facts the library cannot have — batch size, target
  throughput, whether weights stream, and how much VRAM belongs to some other
  process. That last one is not hypothetical: the A2000 these results were
  measured on permanently holds ~3 GB for an unrelated home-lab service, so a
  planner reading free VRAM would plan against a number that moves for reasons
  the model knows nothing about. **Placement policy belongs to the deployment.**
  The failure mode C2 named — a plan computed at 512 and run at 32K — is real,
  and what it actually needs is the KV term being *visible* at plan time, which
  `bench/context/kv_budget.py` already provides from config alone. No accessor
  was added to e4b to wrap it: nothing in that library sizes hot sets, so it
  would be API with no caller.
- **C3** — KV quantization. **Implemented and measured** for nf4, and the trade
  is memory-for-latency, not memory-for-free: 3.56× smaller, ~2.1% perplexity,
  and **1.13× slower decode at 4K, 1.24× at 16K** against a bf16 cache
  (findings #17 and #18 — #17's original 1.9–2.6× was an oracle in the hot
  path). `kernel/nf4_kv.py`
  (attention that reads a 4-bit cache in the mainloop) with `kernel/test_nf4_kv.py`,
  21/21 on the A2000. Corrections to the estimate this document originally carried:
  the saving is **3.56×, not 4×** — the fp32 blockwise absmax is a side channel
  (per token per head: 64 nibble-bytes + 2×4 B absmax = 72 vs 256 bf16) — so the
  235B 32K case measures **5.88 GB → 1.65 GB**, not 2.94. Latency was 2.5–3×
  fp16 SDPA in v1 and is now **0.82× fp16 in the high-GQA long-context regime**
  (4.975 vs 6.055 ms at 32K, GQA 16:1) — see finding #12; the cost was redundant
  dequant, not the 4-bit format. At GQA 4:1 it is still 4.59× slower, so the
  saving is free where current models live and not free everywhere. Fidelity on an
  iid fixture: 9.3% relative error end-to-end, decomposing to **K-only 1.3% /
  V-only 9.2%** — the softmax *contracts* K error (9.2% logit → 1.4%) while V
  error passes through unattenuated. That inverts the usual "K is the sensitive
  one" guidance, which derives from per-channel outliers an iid fixture does not
  have; the asymmetric-precision decision therefore belongs to the real-scale
  perplexity gate, not to this fixture.
- **C4** — for the batch regime, KV becomes a streamed tier alongside the weights;
  the transfer law gains a KV term. **Scoped by finding #14 and not built**: the
  term is `W + batch × KV(ctx)` (KV does not amortize over batch the way weights
  do), which makes C4 worth **nothing at batch 1** — the resident NF4 cache
  already fits every cell measured — and worth progressively more as batch grows
  (6/21 cells at batch 8, 10/21 at batch 32, on a 26.74 GB/s link). Two design
  constraints fall out of the probe: never synchronize per layer (queued copies
  pipeline; 94 slices cost 1.00× one blob), and do not measure this on the
  A2000, whose 6.20 GB/s gen3 ×8 link opens only 1/21 cells.

## Finding #19 — at 235B, context is free, because the step is not byte-bound

Qwen3-235B-A22B, NF4 experts streamed from 122 GB of pinned host RAM, on an
A100-SXM. Step time is **flat across a 64× change in context**:

| ctx | ms/step | peak | KV on device |
|---:|---:|---:|---:|
| 512 | 6266.5 | 18.62 GiB | 0 |
| 32768 (KV resident) | 6095.7 | 28.46 GiB | 1774.8 MB |
| 32768 (KV streamed) | 6124.1 | 26.81 GiB | **0** |

Streaming 1.77 GB of KV per step costs **28 ms — 0.5% of the step**, against a
predicted 6–25%. Whatever dominates that 6.1 s **does not scale with
context**, so KV's bytes disappear inside it. So "context is affordable when
weights stream" holds at flagship scale — but for a reason not yet identified.

**Withdrawn attribution.** An earlier draft of this finding explained the 6.1 s as
"94 layers × ~65 ms of `c_box`". That is wrong and is retracted: `c_box` is a
**whole-box** constant, fixed by the gen5 receipts at **53.5–114 ms** for the
entire per-token forward, not a per-layer quantity — two orders of magnitude
below what is measured here. The 65 ms was back-solved from the result (6.1 s /
94) and then presented as derivation. The flatness and the 0.5% are measurements
and stand; the mechanism is **unidentified**, and is almost certainly the same
unknown as the 27× gap below.

**The working-set claim did not survive.** Peak at 32K is **28.79 GB** against a
≤16 GB target, and the seq-512 arm alone measured **18.62 GiB / 0.160 tok/s**
where the stamped flagship records 15.2 GB / 4.3–4.4 tok/s. That gap — 27× in
time — is between *this* harness (loader defaults) and the tuned hot-residency
configuration the flagship was measured on; it is **not** evidence the stamped
number is wrong. Until it is resolved, the 235B row is the least trustworthy
figure in the project, and the README flagship heading keeps its `~5K context`
qualifier rather than gaining a 32K one.

## Finding #20 — the tier idea does not transfer to training

Decode's per-token VRAM term is the KV cache; training's is the activation stack.
The obvious translation — offload activations to pinned host instead of
recomputing them — **loses, and then turns out to be forbidden.**

| seq | policy | s/step | peak |
|---:|---|---:|---:|
| 32768 | `none` | **OOM** | — |
| 32768 | `recompute` | **7.439** | 28.15 GiB |
| 32768 | `offload` | **20.882** | 30.01 GiB |

With weights **resident** — nothing else on the link, offload's best case —
offload is **2.81× slower and saves no memory**. And with weights **streamed** it
cannot run at all: a streamed expert is evicted after its forward, so a backward
not re-staged by a checkpoint recompute has nothing to dequantize from. On a
streaming box, **gradient checkpointing is mandatory, not preferable.**

Caveat that bounds the claim: the offload arm used `save_on_cpu`, which moves
*every* saved tensor, so these numbers are a **lower bound on offload's quality**
— a boundary-only scheme would move far less. What is established is that the
naive translation loses badly and the streamed path forbids it, not that no
offload scheme could win.

## Finding #21 — the 27× is the offload path staging EVERY expert, not the routed ones

#19 left a 27× gap between an untuned 235B decode and its stamped configuration,
and named it the least trustworthy figure in the project. It is now located, and
it is neither of the two candidates that looked obvious.

Qwen3-235B-A22B, 2×A100-SXM-80GB, link **21.84 GB/s measured**, greedy:

| arm | ctx | s/token | tok/s | peak |
|---|---:|---:|---:|---:|
| no prefetch | 512 | 5.782 | 0.173 | 18.62 GiB |
| prefetch | 512 | 5.228 | 0.191 | 19.88 GiB |
| prefetch + grouped kernel | 512 | 5.144 | 0.194 | 20.07 GiB |
| prefetch + grouped kernel | 32768 | 5.165 | 0.194 | 35.60 GiB |

```
routed experts only (top_k=8):   7.98 GB/token -> 0.366 s at this link
FULL expert stack (all 128):   127.74 GB/token -> 5.849 s
measured:                                         5.144 s   (ratio 1.14)
```

**The offload pre-hook stages a layer's entire expert tensor — all 128 — when
routing needs 8.** That is 16× the necessary bytes on every token, and it is the
gap. The residual 14% is prefetch overlap, which is independently measured at
**1.11×** in the table above, so the two figures corroborate.

**Two attractive explanations died here.** Prefetch is worth 11%, not 27× — the
loader defaulting `prefetch=False` while `infer.py` defaults it on is a real
footgun but a small one. And the **grouped kernel is worth 1.6%** on this path:
`enable_fast` now patches 94 modules where it structurally patched zero before
(it only ever looked for `ExpertsNbit`, while the streaming loader builds
`ExpertsLoRA`, which never calls `base.forward()`). That fix is correct and
verified 10/10, but it is a **correctness fix, not a speedup**, because no
expert-compute dial can matter while the step moves 16× the bytes it needs.

**Consequence for the exclusion lattice.** The optimization that would actually
close this — staging only the routed experts — is what hot residency does, and
hot residency is mutually exclusive with the grouped kernel *and* refuses the
`ExpertsLoRA` base the streaming loader builds. So the one configuration that
should be fastest is currently unreachable by construction. That exclusion is the
central open problem, not a footnote.

**Not yet claimed:** that fixing the staging recovers the stamped 4.3–4.4 tok/s.
Routing-only bytes imply 0.37 s/token at this link, which would be ~2.7 tok/s
here, but that is arithmetic on an unbuilt path and belongs in its own prereg.

## Finding #22 — routed-only staging: 5.95×, bit-identical, same memory

#21 found the offload pre-hook staging all 128 of a layer's experts while the step
routed to 8. `enable_routed_staging` makes the copy follow the router.

Qwen3-235B-A22B, 2×A100-SXM-80GB, link 22.21 GB/s, `prefetch=False` both arms:

| arm | ctx | s/token | tok/s | peak |
|---|---:|---:|---:|---:|
| bulk (today's default) | 512 | 5.570 | 0.180 | 18.62 GiB |
| **routed** | 512 | **0.936** | **1.068** | 18.62 GiB |
| routed | 32768 | 0.986 | 1.015 | 26.81 GiB |

**5.95× faster, greedy token ids identical, peak memory identical (1.000).** The
destination keeps the full `[E, …]` shape with only routed rows filled, so every
consumer still indexes by original expert id and nothing downstream changes.

It is **mutually exclusive with prefetch**, structurally: layer L+1's routing is
decided by a router reading layer L's output, so there is nothing to prefetch. It
gives up that 1.11× to move 16× less. Prefill and training fall back to bulk
automatically.

**#21's "the fastest configuration is unreachable by construction" is retired.**
It is reachable, and it required neither the grouped kernel nor hot residency.

**The byte model still does not describe it (2.1× off).** Predicted 0.445 s,
measured 0.936 — a residual of **5.2 ms per layer** from the per-layer host sync
and ~32 small copies replacing 4 large ones. More of the routed step is now
overhead than is bytes, which makes that the next target and keeps any throughput
prediction unquotable for the moment.

**A qualifier #19 now needs.** At 512 the routed step is 0.936 s and at 32768 it
is 0.986 — context costs **5%**, where #19 measured 0.4%. Nothing about the KV
tier changed; the weight term shrank 6× and the same context cost became a
visible share of a smaller step. "Context is free when weights stream" was true
*of a step carrying 16× surplus weight traffic*.

## Finding #23 — the routed-staging residual is the per-expert loop, and the grouped kernel was never useless

#22 left routed staging at 0.936 s/token where the routed bytes imply 0.359 s at
the probed 22.21 GB/s — a **0.577 s residual**, provisionally blamed on a
per-layer host sync and on 32 small copies replacing 4 large ones. Both were
named as confounds. **Both are wrong**, and so was a third guess.

Decomposed on an RTX 4090 with real Qwen3-235B per-layer shapes (E=128, top_k=8,
10.62 MB/expert), CUDA-event timed:

| candidate | measured | verdict |
|---|---:|---|
| sync: `torch.unique(...).tolist()` | **0.053 ms/layer** | 0.8% of the step — not it |
| 32 small copies vs 8 large, identical bytes | **1.01×** | no penalty — not it |
| `torch.empty` full destination per layer | **−0.001 ms** | free (caching allocator) — not it |
| **per-expert Python loop in `ExpertsLoRA.forward`** | **4.403 ms/layer** | **0.414 s/token** |

Every copy variant achieved the same 13.2–13.3 GB/s, which is what made the first
three explanations collapse: there is no small-copy penalty and no allocation
cost. *(A first version of this microbench timed the **enqueue** of async copies
and reported 1.359 GB moving in 0.021 ms — 65 TB/s. Caught before it was used;
CUDA-event bracketing gives the numbers above.)*

**The grouped kernel takes that loop from 4.403 to 2.174 ms/layer — 2.03×,
worth 0.210 s/token.**

### What this corrects about #21

#21 measured the grouped kernel at **1.6%** and concluded it was "a correctness
fix, not a speedup". That was true *of a step in which transfer outweighed
compute 15:1*. Once routed staging removes the 16× surplus bytes, the compute
half is no longer hidden and the same kernel is worth **2.03×** on it. The kernel
was never useless — it was **masked**, and the byte fix is what unmasks it.

The two are complementary rather than competing, and they compose: routed staging
fixes the bytes, the grouped kernel fixes the launches. Verified together
(`tests/test_routed_staging.py`, 6/6) — the kernel reads exactly the rows it was
given `expert_ids` for, out of a destination where only those rows were written.

**Not yet measured end to end.** 0.936 − 0.210 ≈ 0.73 s/token projected, and that
projection carries a 4090's loop cost onto an A100. It needs the pair run
together on the real model before any number is claimed. ~0.16 s of the residual
is still unattributed (attention, router, norms, KV).

## Finding #24 — the divergence did not reproduce, and the instrument was wrong

#23's pair run left `bulk+grouped` != `routed+grouped` with the kernel held
constant — which should be impossible, since routed staging is bit-identical.
Three targeted probes, all negative:

| probe | shape | result |
|---|---|---|
| kernel determinism, 4 repeats | E=32, h=256 | **deterministic** |
| unrouted rows poisoned `0x00` vs `0xFF` | E=32, h=256 | **rel 0.000e+00** — never read |
| routed vs bulk, shape sweep | **E=128, h=4096, i=1536** | **rel 0.00e+00** at every shape |
| routed vs bulk, real full model | OLMoE, 16 layers, attn+router+KV | **identical**, self-consistent ×3 |

So it reproduces neither at flagship per-layer shape nor on a real end-to-end
model. What the receipts actually show is that **the instrument was wrong**:

```
bulk+ref        [388, 13, 220, 16, 15, 15, 15, 15, ...]
routed+ref      [388, 13, 220, 16, 15, 15, 15, 15, ...]   <- IDENTICAL
bulk+grouped    [ 68, 197, 197, 322, 220, 17, 15, 16, ...]
routed+grouped  [ 13, 220,  17,  15,  16, 22, 13, 15, ...]
```

`bulk+ref` vs `bulk+grouped` differ at **11 of 13 positions** across a boundary
whose documented error is `rel < 2e-2`. The prompt is **random tokens**, so the
logits are near-uniform, greedy argmax sits on ties, and the first flip
re-conditions every token after it. Greedy ids under those conditions measure
chaos amplification, not agreement.

**Correct instrument: compare LOGITS (max abs / relative error on the first
forward), not sampled ids** — and on a natural prompt, where the distribution is
peaked, rather than random tokens. Every "greedy ids identical" gate in this
document set inherits this flaw; the ones that PASSED are still informative
(identical ids imply identical logits at that length), but a FAIL says much less
than it appears to.

**What stands.** Routed staging is bit-identical: `bulk+ref == routed+ref` on the
real 235B, and `rel 0.00e+00` at flagship shape in isolation. **What is open.**
Whether `routed+grouped` differs from `bulk+grouped` in the *logits* on the 235B,
which no measurement so far has actually asked.

## Finding #25 — logit gates: routed staging is clean, and the kernel's error compounds

Re-ran the gates on **logits** rather than sampled ids, on a **natural prompt**
(OLMoE-1B-7B, 47 tokens, top-1 probability **0.911** — where the random-token
prompts of #24 sat near uniform).

| comparison | max\|Δlogit\| | rel | greedy ids |
|---|---:|---:|---:|
| **bulk+ref vs routed+ref** | **0.000e+00** | **0.000e+00** | 0/9 differ |
| **bulk+grouped vs routed+grouped** | **0.000e+00** | **0.000e+00** | 0/9 differ |
| bulk+ref vs bulk+grouped | 1.488 | **1.293e-01** | 0/9 differ |
| routed+ref vs routed+grouped | 1.488 | **1.293e-01** | 0/9 differ |

**Routed staging is bit-identical at model scale under BOTH kernels.** That
closes #24's open question: the 235B `bulk+grouped != routed+grouped` was the
instrument, not the code. The unit suite's composition gate has been tightened
from `rel < 2e-2` to **bit-identity** on exactly this pair and passes (216 tests
green).

**And the gate the ids could never have shown**: the grouped kernel moves the
model's logits by **12.9%** while leaving greedy ids **identical**. Both facts
come from the prompt being peaked — a 0.911 top-1 absorbs a large logit shift
without changing argmax. On #24's random prompts the same kernel changed 11 of 13
ids. **Neither id measurement was informative about fidelity in either
direction.**

**This is compounding, not a defect.** The kernel's documented `rel < 2e-2` is a
*per-layer* bound; 16 layers of it compounds to 0.373, and the measured 0.129 sits
comfortably inside that. The gap is that **nothing ever measured the composed
error**, and depth is exactly where it grows — a 94-layer 235B has ~6× OLMoE's
compounding budget.

**Consequence for the pair's 7.88×.** The speed number stands; a *fidelity* claim
does not yet exist. The right instrument is **perplexity** — the same one #10 used
to price the NF4 KV cache at ~2.1% — measured with `enable_fast` on and off at
depth. Until that exists, `enable_fast` should be described as a speedup with an
**unquantified** accuracy cost at model scale, not a free one.

## Finding #26 — the grouped kernel was nondeterministic; fixed, and priced at +0.023%

Applying perplexity — the instrument #10 used to price the KV cache — to
`enable_fast` on OLMoE-1B-7B, 24 independent 2048-token chunks, 3 repeats/arm.

The gate caught a bug in the kernel path rather than in the measurement:

| arm | ppl | spread over 3 repeats |
|---|---:|---:|
| reference (per-expert loop) | 7.45474 | **0.00e+00** |
| grouped, atomic `index_add_` | 7.45928 | **9.01e-04** |
| **grouped, deterministic scatter** | **7.45645** | **0.00e+00** |

**The reference path is bit-deterministic; the grouped path was not.** The
weighted combine used `index_add_`, which accumulates with CUDA atomics in
run-varying order. A stable sort alone did not fix it. Since `order` is a
permutation, scattering by **assignment** and reducing with a fixed-axis `sum`
is deterministic — and measured **0.038 pp more accurate** as a side effect.

**Priced: `enable_fast` costs +0.0229% perplexity for its 1.32×.** The NF4 KV
cache costs ~2.1% (#10) — **92× larger**. #25's "unquantified accuracy cost"
was right to demand the number and wrong about its size.

**A claim that did not survive composition.** `fast.py` states the fused path
"measured *more* accurate than the reference" — true on the kernel's per-op
property suite, false through 16 layers, where it is consistently slightly worse.
A per-op accuracy claim is not a model-level one, and the two were conflated.

**Bounded:** OLMoE is 16 layers against the flagship's 94, so +0.023% is a lower
bound on the 235B's compounded cost.

## Finding #27 — #23's divergence is closed: instrument plus a real kernel bug

The last open correctness question. OLMoE-1B-7B, natural prompt, logit-level
comparison, with the deterministic scatter of #26 in place:

| comparison | max\|Δlogit\| | rel | ids |
|---|---:|---:|---:|
| **bulk+grouped vs routed+grouped** | **0.000e+00** | **0.000e+00** | 0/9 |
| routed+grouped vs itself | **0.000e+00** | 0.000e+00 | 0/9 |
| bulk+grouped vs itself | **0.000e+00** | 0.000e+00 | 0/9 |
| bulk+ref vs routed+ref | **0.000e+00** | 0.000e+00 | 0/9 |
| bulk+ref vs bulk+grouped *(documented inexact)* | 1.488 | 1.293e-01 | 0/9 |

**Routed staging is bit-identical under both kernels, and the grouped kernel is
now deterministic under both staging policies.** #23's `bulk+grouped !=
routed+grouped` had two causes stacked, which is why four probes chased it:

1. **the instrument** — greedy ids on a random-token prompt, where near-uniform
   logits let any perturbation flip the first token and re-condition the rest
   (#24), and
2. **a real bug** — the kernel's atomic `index_add_`, which made it genuinely
   nondeterministic run to run (#26).

Either alone would have produced the symptom; together they made it look like a
staging bug, which it never was. **Not one artifact — an artifact hiding a defect.**

**Scope, stated plainly:** 16 layers. The 235B re-run that would confirm this at
94 was swept by a concurrent session before its gates computed, and its timing
arms (6.69–7.76× end to end) show only that the determinism fix costs no
throughput.

## Finding #28 — the Kimi-K2 KV row was 35.56x optimistic

`Kimi-K2-Instruct` was the last **derived only** row in the published KV table:
68.6 KB/token, computed from `config.json` with no probe behind it, and the only
MLA geometry in the table. A rung-1.5 probe (real model class, full MLA widths,
truncated depth, dense layers so the MoE is out of frame) against `transformers`'
native DeepSeek-V3 classes:

```
measured : 40960 B/token/layer   = (64x192 + 64x128) x 2B, decompressed K and V per head
derived  :  1152 B/token/layer   = (kv_lora_rank 512 + qk_rope 64) x 2B, the compressed latent
ratio    : 35.56x
```

**The derivation encoded MLA as designed; the reference implementation does not
implement it that way.** MLA's entire premise is caching a joint compressed
latent, and `transformers` materializes full per-head K/V instead — forfeiting
it. The measured value matches the decompressed arithmetic *exactly*, so this is
not an approximation gap.

| Kimi-K2 at 61 layers | KB/token | @4K | @32K | @128K |
|---|---:|---:|---:|---:|
| published (compressed / MLA-as-designed) | 68.6 | 0.29 GB | 2.30 GB | 9.21 GB |
| **measured (`transformers`)** | **2440.0** | **10.23 GB** | **81.87 GB** | **327.49 GB** |

At 32K this is the difference between "fits on one card" and "fits on no single
card". README corrected, with the two-stack caveat stated inline — engines that
implement the compressed cache (vLLM, SGLang, DeepSeek's own) do get 68.6.

**The general lesson, and it applies beyond MLA:** a KV figure derived from
`config.json` describes an architecture's *permission*, not a runtime's
*behaviour*. Every other row in that table was probe-verified, which is why this
was the only place the gap could hide — and "derived only" was the right label
for exactly this reason.

## Finding #29 — the flagship never disagreed with these measurements; they measured different mechanisms

#19 opened a 27× gap between the stamped flagship (**4.3–4.4 tok/s, 15.2 GB**)
and every attempt to reproduce it (**0.18 tok/s, 18.6 GiB**, five independent
pods), and called the 235B row "the least trustworthy number in the project".
**That was wrong, and the resolution needed no measurement — only reading the
flagship's own code and link speed.**

`bench/phase3/offload_decode_235b.py:188` stages **"one layer's active experts"**,
double-buffered across a dedicated `copy_stream`. The e4b offload hook stages the
**whole layer's expert stack, synchronously** (#21). Two different mechanisms.
And the flagship ran on a **44.3 GB/s** link where these pods measure 21.7–23.3.

Against the additive law `t = c_box + bytes/L`, with the flagship's own receipted
`c_box = 53.5 ms`:

| | bytes | link | predicted | measured |
|---|---:|---:|---:|---:|
| flagship, **routed** bytes | 7.98 GB | 44.3 GB/s | **4.28 tok/s** | **4.3–4.4** ✓ |
| flagship if it were bulk | 127.74 GB | 44.3 GB/s | 0.34 tok/s | — |
| these pods, **bulk** bytes | 127.74 GB | 22.5 GB/s | **0.17 tok/s** | **0.18** ✓ |
| these pods if routed | 7.98 GB | 22.5 GB/s | 2.45 tok/s | 1.20–1.39 |

**Both numbers were right. Comparing them was the error.** The flagship always
streamed only the routed experts; the e4b offload path never did until routed
staging. The "27×" was 16× of surplus bytes (#21) times ~2× of link.

**The remaining gap in row 4 is overlap.** phase3 double-buffers its staging so
transfer hides compute; routed staging is synchronous and cannot prefetch — the
next layer's routing does not exist yet (#22). That ~2× is the honest distance
left, and it is a scheduling difference, not a defect.

**Caveat, because phase3 is not a drop-in:** its own source notes MoE compute on
"stale staged bytes" — it is a pipeline benchmark measuring an achievable rate,
not a correctness-preserving inference path. Routed staging is bit-identical
(#22, #27); that is a property phase3 does not claim.

**Retracted:** #19's "the least trustworthy number in the project", and the
"still 4× short of the stamped 4.3" caveats in #22 and #23. The flagship figure
stands as measured, on its own mechanism and its own link.

## Finding #30 — bit-identity holds at 48 layers, and the four "lost" 235B runs had one cause

Qwen3-30B-A3B (**48 layers**, 128 experts, top_k 8) — triple OLMoE's depth, and
the only depth-dependent risk the 94-layer run was meant to close:

| gate | max\|Δlogit\| | |
|---|---:|---|
| bulk+grouped vs routed+grouped | 0.000e+00 | **PASS** |
| routed+grouped vs itself | 0.000e+00 | **PASS** |
| bulk+ref vs routed+ref | 0.000e+00 | **PASS** |
| bulk+grouped vs itself | 0.000e+00 | **PASS** |

| cell | s/token | tok/s |
|---|---:|---:|
| bulk+ref | 1.555 | 0.643 |
| bulk+grouped | 1.219–1.256 | 0.80–0.82 |
| routed+ref | 0.810 | 1.234 |
| **routed+grouped** | **0.515–0.687** | **1.46–1.94** |

**2.3–3.0× end to end here, against 7.88× on the 235B** — the 16× byte saving
pays in proportion to how much the bytes dominate, and a 30B expert stack is far
smaller relative to its compute. The speedup is model-dependent; the
*correctness* is not.

### The four swept 235B runs: `gen1-sweeper.sh`

Four 235B attempts died at ~45–60 min and were attributed in turn to provider
flakiness, a "RunPod curse", and cross-session interference. **All three readings
were wrong.** `~/gen1-sweeper.sh` on the Mac mini — an account-wide **45-minute
age cap** written for the gen1 hunt, whose pods were 10–15 minute probes — had
been running as a **bare bash loop for 13 days**. It was never a launchd label,
so the 2026-07-23 cleanup that removed every `gnf4.*` label never touched it, and
neither `launchctl list` nor `crontab -l` showed anything.

Its own log names every casualty at `age≈2700s`, including another Claude
session's pod. Disabled 2026-07-26. Two traps recorded for anyone who re-arms it:
`~/gen1-hunt-stop` is **not** a safe off switch (it sets `KILL_ALL=1` and
terminates every pod on the account), and `~/gen1-keep-<podid>` is what extends a
long run to 3 h.

**The methodological point is the same one #29 made:** the evidence fit three
plausible mechanism stories, and the answer came from `ps` on the right machine.
~$15 and four experiments went to not looking there first.

## Finding #31 — the 94-layer gate: closed, and compounding did NOT grow with depth

With `gen1-sweeper.sh` disabled (#30), the run that four attempts could not
finish completed. Qwen3-235B-A22B, natural prompt, logit gates:

| gate | max\|Δlogit\| | verdict |
|---|---:|---|
| `bulk+grouped` vs `routed+grouped` | **0.000e+00** | **PASS** |
| `routed+grouped` vs itself | **0.000e+00** | **PASS** |
| `bulk+ref` vs `routed+ref` | **0.000e+00** | **PASS** |

**Routed staging is bit-identical at 94 layers under both kernels, and the
grouped kernel is deterministic there.** The last open correctness question in
this document set, closed at the scale it was actually asked about. End-to-end
**6.97×** (5.938 → 0.852 s/token), kernel gain **1.05× under bulk → 1.39× under
routed**, independently reproducing #23's masking interaction.

### The surprise: divergence did not compound

The reference↔grouped logit `rel` is **0.1189 at 94 layers** against **0.1293 at
16** (#25). It did not grow — it shrank slightly. #26 attached a caveat to its
+0.023% perplexity figure reasoning that "deeper models compound further", and
#25 computed that 2% per layer over 16 layers gives 0.373 while 94 gives 5.4.
**That reasoning is not supported.** Whatever bounds the reference↔grouped
divergence is not accumulating linearly in depth.

Two candidate explanations, neither tested: the per-layer perturbation may be
mean-reverting rather than additive, or the final-logit norm may grow with depth
fast enough to hold the *relative* error flat. **Perplexity at 94 layers remains
unmeasured** — this is a logit-norm observation, not a fidelity measurement. But
the depth caveat on #26 should be read as *unquantified*, not as *presumed
larger*, and #25's compounding arithmetic is withdrawn as a prediction.

## Finding #32 — expert routing is predictable 2–4 layers ahead, and the MoE output is why

Can layer L+d's routing be guessed early enough to prefetch its experts? Measured
on Qwen3-30B-A3B (44 MoE layers, 128 experts, top_k 8), coverage of the true
top-8 by a predictor using layer L's *real* router on an earlier hidden state:

| predictor | K=8 | K=16 | K=32 | prefetch window |
|---|---:|---:|---:|---|
| d=1 | 0.9089 | 0.9930 | 0.9987 | 0 MoE layers |
| **d=2** | **0.8471** | 0.9754 | 0.9940 | **1 MoE layer** |
| d=3 | 0.8072 | 0.9543 | 0.9884 | 2 layers |
| d=4 | 0.7721 | 0.9357 | 0.9808 | 3 layers |
| from the token embedding | 0.1815 | 0.2784 | 0.4335 | all layers |
| from the previous token's final state | 0.0998 | 0.2081 | 0.4197 | unbounded |

**What determines routing is the MoE output, not attention.** Predicting L+1
while dropping only L+1's *attention* holds 0.9089; additionally dropping layer
L's *MoE* collapses it to 0.2439. Routing is set by what the experts wrote to the
residual stream.

**Which kills the wide horizons.** Predicting from the token embedding or the
previous token's final state gives an unbounded window and 0.42 coverage even at
K=32 — everything that decides routing happens in the layers being skipped. A
naive temporal baseline (reuse the previous token's *decision*) scores 0.4513,
beating a recomputed router on a stale residual: the *decision* is temporally
stable even where the router is highly input-sensitive.

**Decay with distance is very slow** — 0.9987 → 0.9940 → 0.9884 → 0.9808 at
K=32 across d=1..4, so lookahead is nearly free out to 3 layers of window.

### The design conclusion is not the one coverage suggests

Coverage says stage K=16 or K=32. The arithmetic says otherwise, because staging
K experts costs `K/8 ×` synchronous routed staging's bytes and the step is
transfer-bound:

```
                          @22.5 GB/s          @44.3 GB/s
  synchronous routed        0.559 s             0.385 s
  spec d=2, K=8             0.409 s  1.37x      0.232 s  1.66x
  spec d=2, K=16            0.718 s  0.78x      0.365 s  1.05x
  perfect-overlap ceiling   0.355 s  1.58x      0.204 s  1.88x
```

**K=16 is slower than not speculating** on a slow link. The design is **d=2,
K=8, misses staged synchronously** — 1.37–1.66×, which is 87–88% of the
perfect-overlap ceiling.

**This also prices #29's residual.** Overlap is worth 1.88× at 44.3 GB/s and
1.58× at 22.5 — phase3's advantage over routed staging was never purely
scheduling; roughly half of it was link speed.

**Unbuilt.** A coverage measurement plus an arithmetic model whose ancestors were
wrong by 10.7× and 2.1×, and which omits the miss path's *latency*. No speedup is
claimed until measured.

## Finding #33 — speculative routed staging: built, bit-identical, 1.330×

#32 predicted routing is guessable 2 layers ahead (0.847 coverage at top-8) and
modelled **1.37×** from hiding compute behind the prefetch. Built as
`enable_speculative_staging` and measured on Qwen3-30B-A3B, 48 layers,
21.74 GB/s:

| policy | s/token | tok/s | peak |
|---|---:|---:|---:|
| routed (synchronous) | 0.3218 | 3.107 | 3.84 GiB |
| **routed + speculative d=2, k=8** | **0.2420** | **4.132** | 4.14 GiB |

- **Bit-identical** to synchronous routed staging: `max|Δlogit| = 0.000e+00`.
  Correctness cannot depend on the guess — the destination keeps its full
  `[E, …]` shape and every truly-routed row the guess missed is staged before
  the forward runs. A wrong guess costs bandwidth, never an answer.
- **1.330×** against #32's modelled 1.37× — the first time a model in this
  document set predicted a speedup and the measurement agreed (its predecessors
  were wrong by 10.7× and 2.1×).
- **In-situ hit rate 0.8536** against the offline 0.8471 — the prediction
  behaves in production as it did in measurement.
- **Memory +8%**, not the doubling a second resident buffer would suggest.

### The first measurement said 0.920× and it was a leak, not a verdict

The initial build measured a **slowdown** at 2.4× the memory, with an entirely
plausible story available: extra copies cost more than the overlap saves, exactly
as #32 warned for K=16. It was two defects instead.

**Prefill routes nearly every expert**, so routed staging takes its bulk fallback
there — but the speculative hook fired anyway and allocated a full-shape buffer
that `stage_routed` never consumed, and the evict guard then refused to drop it
(correctly, by its own rule). Every prefill leaked one buffer per layer. Second,
any staging path that does not go through the speculative branch has to release
an unconsumed speculation or the guard pins it for the whole run.

Both fixed; the number moved 0.920 → 1.330. **The lesson is the shape of the
failure**: a real bug produced a result that agreed with a plausible pessimistic
theory, which is the easiest kind of wrong answer to accept.

### Where this leaves the stack

```
bulk (default)                              1.00x
  + routed staging          #22             5.95x   bit-identical
  + grouped kernel          #23/#31         6.97x   +0.023% ppl
  + speculative d=2 k=8     #33            ~9.3x    bit-identical
```

The last figure composes a 1.330× measured on a 48-layer model with a 6.97×
measured on the 235B, on different boxes. **It is not a measured end-to-end
number** and is written here only to show where the pieces sit.

## Finding #34 — the full stack on the 235B: 10.21×, every rung bit-identical

All four optimizations composed in one process, one box, one link
(Qwen3-235B-A22B, 2×A100-80GB, 21.53 GB/s):

| rung | s/token | tok/s | vs bulk | step |
|---|---:|---:|---:|---:|
| bulk (shipped default) | 5.6918 | 0.176 | 1.00× | — |
| + routed staging | 1.0804 | 0.926 | 5.27× | 5.27× |
| + grouped kernel | 0.9818 | 1.018 | 5.80× | 1.10× |
| + speculative d=2 k=8 | 0.6263 | 1.597 | 9.09× | 1.57× |
| **+ expert cache** | **0.5573** | **1.794** | **10.21×** | 1.12× |

**Every rung is bit-identical to the one before** — `max|Δlogit| = 0.000e+00` at
94 layers for both new mechanisms. The only fidelity cost in the stack remains
the grouped kernel's +0.023% perplexity (#26).

Peak VRAM 18.58 → 27.26 GiB; the cache is 7.98 GB of that and is the only rung
that trades memory for speed.

**Speculation delivered 1.568× at 94 layers against 1.330× at 48** (#33) — more
compute to hide per layer at the same link.

### The cache thrashed until its partitioning was fixed

Its first measurement was a **loss**: 0.904×, hit rate **0.0002 (6/34,719)**. One
752-slot pool is *exactly* one token's working set (94 layers × 8 experts), and a
decode step touches every row in layer order, so global LRU evicted layer 0's
rows for layer 93's — one token before layer 0 needed them. Cache size equal to
working set is the classic thrash, and it is easy to build by accident when the
natural sizing rule is "one token's worth".

Per-layer partitioning: hit rate **0.1322**, step **1.120×**. Still far under the
0.4513 reuse #32 measured, so partition size is an unexplored knob.

**And the model overshot again.** I predicted 1.78× for the cache from #32's
reuse figure; it delivered 1.12×, because speculation already moves bytes off the
critical path, so part of what the cache saves was hidden anyway. Three modelled
gains have now overshot — 10.7×, 2.1×, 1.6× — and each time the error was an
**interaction between optimizations**, not the isolated physics.

## Finding #35 — the expert cache works and is worth nothing once speculation is on

#34's cache rung was measured on 2 reps at 1.120×. Re-measured at 6 reps with
ranges, sweeping the partition size (235B, 25.94 GB/s):

| slots/layer | median s/tok | range | hit rate | speedup [range] |
|---|---:|---|---:|---|
| none | 0.8308 | 0.8013–0.8640 | — | 1.000× |
| 16 | 0.8776 | 0.7547–0.9199 | 0.3521 | 0.947× [0.903, 1.101] |
| 24 | 0.7314 | 0.5816–0.8368 | 0.4452 | 1.136× [0.993, 1.428] |
| 32 | 0.8187 | 0.7751–0.8539 | 0.5384 | 1.015× [0.973, 1.072] |

**The mechanism is confirmed and the payoff is not there.** Enlarging the
partition removes within-token eviction exactly as predicted — hit rate
0.132 → 0.352 → 0.445 → 0.538, clean and monotonic, because hit rates are counts.
**Every timing range overlaps the baseline**, and 16 slots is *worse* than no
cache at its median.

**Because speculation already moved those bytes off the critical path.** The
cache and speculative prefetch target the same term, and speculation reaches it
first, so eliding bytes that were already overlapped saves nothing. Both are
"don't pay for this transfer now" — one by prefetching it, one by not needing it.

**#34's 1.120× is retracted.** It was two reps against a spread that runs to 35%.
The 235B end-to-end figure is **9.09× (through speculation)**, not 10.21×, and
the recommendation is **not to enable the expert cache** alongside speculation:
8–32 GB of VRAM for no measured return.

**Method note.** The 2-rep curve looked orderly enough to believe — and was
non-monotonic in a way that should have been the tell, since 24 slots beat 32
while hitting less often. Hit rates and timings came from the same runs; only the
timings were noise.

## Finding #36 — the step is GPU-saturated; CUDA graphs is dead and attention is the target

After speculation, 67% of the 235B step was unmeasured (#35). Decomposed with
CUDA-event ranges on disjoint categories, 94 layers, 48-token context:

| category | s/token | % of wall |
|---|---:|---:|
| **experts** | 0.2921 | **46.4%** |
| **attention** | 0.2803 | **44.5%** |
| norms | 0.0305 | 4.8% |
| router | 0.0056 | 0.9% |
| lm_head | 0.0008 | 0.1% |
| **GPU busy** | 0.6093 | **96.7%** |
| gap (launch + Python) | 0.0207 | **3.3%** |

**Not launch-bound — 96.7% GPU-busy.** The prediction was <60%, reasoning from
~1,880 kernel launches per token at bs=1 and from per-layer cost scaling only
1.32× between the 30B and the 235B. The fixed term is real; it is **GPU work**,
not scheduling. **CUDA graphs is dropped without being built** — its ceiling is
3.3%.

**Attention nearly equals expert compute** (44.5% vs 46.4%) at a context where
the KV is ~9 MB. It is not intrinsic: the benchmark ran KV `residence="host"`, so
every attention call pulls the cache across the link and dequantizes NF4. **This
session has been optimizing expert streaming inside a configuration whose KV dial
was set wrong for the context being measured** — the project's own planner would
put KV resident at 48 tokens. Roughly half the step was paying for that.

**Instrumentation bug, recorded:** `wrap(model.model)` was written where
`model.model.norm` was meant, making a non-leaf range that contained every other
category. It reported 259.9% GPU-busy and a *negative* gap, which is how it was
caught — an impossible number is a better failure than a plausible one. The five
disjoint leaves were unaffected.

## Finding #37 — the KV cost is quantization, not residence; and #36's attribution was wrong

Swept cache kind x context on the 235B at routed+grouped+speculative:

| ctx | bf16 | nf4_resident | nf4_host |
|---:|---:|---:|---:|
| 48 | **0.6209** | 0.7877 | 0.8134 |
| 4096 | **0.5715** | 0.8139 | 0.6743 |
| 32768 | **0.7421** | 0.8230 | 0.8982 |
| peak @32K | 43.71 GiB | 39.49 GiB | 37.84 GiB |

**Host residence is nearly free and unrankable** — 1.03× at 48 tokens, 1.09× at
32K, and the ordering *inverts* at 4096. Against this session's observed 10–35%
run-to-run spreads that is noise, not a result.

**bf16 beats NF4 at every context, 3 for 3** (1.27× / 1.42× / 1.11×). The KV
dial's cost is the **dequantization**, not where the bytes live — #17's 1.7–1.9×
reappearing smaller against a faster step.

**#36 is corrected.** It attributed attention's 44.5% share to the host-KV
setting and estimated "roughly half the step" was paying for it. The best KV
setting buys **1.31×**, not ~1.8×. Most of attention's cost is intrinsic, so the
next target is attention itself.

**The trade, stated:** at 32K, NF4 KV saves 4.2 GB for 1.11× time; host residence
saves a further 1.65 GB for another 1.09×. On an 80 GB card at 32K **bf16 is the
right answer**, and every benchmark in this session ran `nf4_host` — the wrong
one for the contexts measured.

**Cross-pod caution:** `nf4_host` at 48 tokens measured 0.8134 here against
0.6263 in #34 on a different host. Absolute step times do not transfer between
pods; only within-pod ratios do.

## Finding #38 — NF4 attention abandoned: the kernel was never bandwidth-bound

Wrote the single-pass online-softmax NF4 attention that #37's profile called for,
against a pre-committed stopping rule. A40, T=32768, GQA 16:1:

| path | ms | KV GB/s | rel err |
|---|---:|---:|---:|
| bf16 SDPA (`enable_gqa`) | **0.1399** | 134.9 | — |
| old `attend_nf4_kv_gqa` | 1.1613 | 16.3 | 2.899e-03 |
| new `flash_nf4_kv_gqa` | 1.7309 | 10.9 | **5.613e-07** |
| achievable bandwidth | — | 491.9 | |

**The profile was wrong.** It said 69% of the old kernel's traffic was a
materialized `[H_q, T]` score matrix and that removing it would close the 12.7×
gap to SDPA. The new kernel removes it entirely — moving 3.2× fewer bytes — and
is **1.5× slower**. At 2.2% of achievable bandwidth neither kernel was ever
memory-bound; the binding term is the inner loop's nibble unpack, LUT gather and
`ieee` `tl.dot`, none of which a byte-count profile can see.

**A byte-count profile diagnosed a kernel that bytes did not limit.** That is the
lesson worth carrying: traffic analysis identifies what a kernel *should* be
bound by, not what it *is*.

**Stopping rule honored, path abandoned.** N1a fell far below its 25% line after
one diagnosed fix (split-T parallelization, worth 19.6× — the first version put
4 programs on an 84-SM card). NF4 KV is a **memory** play, not a speed one, and
#37's "bf16 unless VRAM binds" stands.

**The one thing produced:** the new kernel is correct to **5.613e-07**, ~5,000×
more accurate than the old one, and is kept as the correctness oracle for any
future attempt.

## Finding #39 — the NF4 decode is near-optimal; the grouped GEMM's cost is elsewhere

#38's post-mortem blamed the inner loop — nibble unpack and LUT gather — for a
kernel that a byte-count profile had misdiagnosed. The grouped NF4 GEMM shares
those primitives and is now **46%** of the 235B step at **1.3%** of A100 HBM, so
the same suspicion applied. Measured directly, on identical packed bytes (A40,
8 experts of [3072, 4096], 56.62 MB):

| kernel | ms | GB/s | % of gemm |
|---|---:|---:|---:|
| read (bytes only) | 0.0826 | 685.5 | 14.5% |
| decode (unpack + LUT + scale) | 0.1163 | **487.0** | 20.4% |
| gemm (shipped grouped) | 0.5710 | 99.2 | 100% |
| achievable (pure read) | 0.0992 | 570.9 | |

**The decode is not the problem — it runs at 85.3% of achievable bandwidth and
costs 1.4× a raw byte read.** A1a predicted ≥50% of the GEMM and measured
**20.4%**, falsified below its 25% line. The LUT gather that #38 blamed is
essentially free.

Decomposing the grouped GEMM:

```
reading the bytes    0.0826 ms   14.5%
+ the NF4 decode     0.0337 ms    5.9%
+ everything else    0.4547 ms   79.6%   <- the target
```

**It is bound by none of the three things it was suspected of.** Not memory
(99 of 487 GB/s reachable on the same bytes), not compute (201 MFLOP in 0.571 ms
= 352 GFLOP/s, **0.95%** of an A40's ~37 TFLOP/s), not decode (20.4%). The
remaining 79.6% is tiling and dispatch — at bs=1 the GEMM is an M=1 GEMV padded
to `tl.dot`'s M≥16 minimum, so most of every tensor-core tile is wasted work on
padding.

**Headroom against its own decode floor: 4.9×**, and the pre-committed branch for
a falsified A1a fires — **the target is the GEMM's tiling and dispatch, not the
decode primitive.**

**This also re-reads #38.** That kernel was not decode-bound either; the shared
inner-loop *primitives* were never the issue in either case. What both have in
common is a `tl.dot` on a shape tensor cores cannot use well. The lesson from #38
("traffic analysis says what a kernel should be bound by, not what it is")
survives, but its follow-on guess — that the decode was the culprit — is now
falsified by direct measurement rather than inherited.

## Finding #40 — on the right KV setting, experts are 71.3% of the step

#36 decomposed the 235B step while running `nf4_host` KV, the setting #37 then
showed is wrong for these contexts. Repeated on bf16:

| category | s/token | % of wall | was (#36) |
|---|---:|---:|---:|
| **experts** | 0.6613 | **71.3%** | 46.4% |
| attention | 0.1473 | **15.9%** | 44.5% |
| norms | 0.0828 | 8.9% | 4.8% |
| router | 0.0322 | 3.5% | 0.9% |
| lm_head | 0.0008 | 0.1% | 0.1% |
| **GPU busy** | 0.9245 | **99.7%** | 96.7% |

**Attention was never the target** — its 44.5% was a config artifact, and it is
**15.9%** once the KV dial is set the way #37 recommends. The GPU is **99.7%**
busy, so #36's CUDA-graphs verdict holds *more* strongly: the whole scheduling
gap is **0.3%**.

Composed with #39: experts are 71.3% of the step and 79.6% of the GEMM is neither
reading nor decoding, so **56.8% of the entire step is grouped-GEMM tiling and
dispatch**. If the GEMM reached its own decode floor the step goes
**0.9691 → 0.4428 s/token = 2.19×**.

## Finding #41 — three hypotheses for that gap, all falsified

The 2.19× above has survived three attempts to explain it, each registered and
each wrong:

1. **`tl.dot` M-padding.** I claimed the expert GEMM was an M=1 GEMV padded to
   `tl.dot`'s M≥16 minimum. **Wrong** — `gemm_4bit_grouped` already dispatches
   `_gemv_nf4_grouped` when `max(sizes) == 1` and skips the M-tile entirely. The
   99.2 GB/s in #39 *was* the GEMV path. I nearly rebuilt what already exists.
2. **The decode primitive** (#39). Measured: the decode runs at **85.3%** of
   achievable bandwidth, costs 1.4× a raw byte read, and is 20.4% of the GEMM.
3. **Occupancy.** Swept BLOCK_N × warps × split_k — 64 configs, 0 failures, max
   rel err 4.57e-05. Default `(64, 2, 1)` at **9.1 warps/SM** gives 95.5 GB/s;
   best `(128, 4, split_k=8)` at **73.1 warps/SM** gives 110.9. **8× the
   occupancy bought 16%**, falsifying G1a at 1.162× against a 1.2× abandon line.

**Config tuning is abandoned per the stopping rule.** The GEMV moves 56.62 MB at
110.9 GB/s where a flat decode of the same bytes reaches 487, and it is not
memory-bound, not compute-bound (0.95% of peak), not decode-bound, and not
occupancy-bound.

**I do not understand this kernel.** Three registered mechanisms, three
falsifications. The next step is a real profiler — Nsight Compute on
`_gemv_nf4_grouped` — not a fourth guess.

## Finding #42 — the ladder is 9.19× on bf16 KV, and the KV dial barely matters

#34's 9.09× ladder ran `nf4_host` KV throughout — the setting #37 called wrong.
Re-run on bf16, one process, one load, median of 3:

| rung (bf16 KV) | s/token | vs bulk | step |
|---|---:|---:|---:|
| bulk+ref | 5.9041 | 1.00× | |
| routed+ref | 1.0917 | 5.41× | 5.41× |
| routed+grouped | 0.8384 | 7.04× | 1.30× |
| **routed+grouped+spec** | **0.6423** | **9.19×** | 1.31× |
| same rung, `nf4_host` | 0.6658 | | **1.037×** |

Every rung bit-identical (`max|Δlogit| = 0`); speculation hit rate 0.8973.

**The ladder did not widen: 9.19× against 9.09×.** L1a predicted ≥10.0× on the
reasoning that cheaper attention helps the fast rung disproportionately. It does
not — the KV setting is close to irrelevant to the ladder.

**And the within-pod KV gap is 1.037×, not the 1.27–1.42× of #37.** Both are
within-pod measurements of the same comparison at the same context, on different
boxes, and they disagree by 4×. With 3 reps against this session's observed
10–35% spreads, **#37's KV gap is not reliably separable from noise** and should
be read as ≲1.3× rather than as a measured 1.31×.

**Which weakens #40's framing.** #40 contrasted attention at 44.5% (#36,
`nf4_host`) against 15.9% (bf16) — but those were **different pods with different
absolute step times** (0.63 s vs 0.93 s). If the KV dial is worth 1.04× here, a
44.5%→15.9% shift cannot be attributed to it. The bf16 decomposition stands as a
measurement; the *contrast* with #36 does not.

**What survives all of it:** experts dominate the step, the ladder is ~9.2×
regardless of KV setting, and every rung is bit-identical. **The headline is
9.19× on bf16 KV** — the configuration the project recommends — with #34's 9.09×
kept as the `nf4_host` measurement.

## Reproducing

`bench/context/kv_budget.py` (derivation, from config.json only) and
`bench/context/kv_verify.py` (the A2000 rung-one probe). Receipts:
`bench/context/receipts-c0-20260724/`.

Finding #14: `bench/context/pcie_probe.py` (link characterization — device
measurement, no hypothesis, so no prereg) and `bench/context/kv_stream_budget.py`
(the C4 derivation; `--link`, `--vram-free`, `--target-batch`, `--target-tps`).
Receipts: `bench/context/receipts-c4-20260725/`.

Finding #13: `bench/context/attn_select.py` is the registered harness (both
fixtures, all policies) with `attn_select_smoke.py` as its pre-flight; the four
post-hoc diagnostics are `_oracle.py` (future-knowing selection), `_sinks.py`
(static sink/recent sweep), `_chunk.py` (protocol confound) and `_h2o_chunk.py`
(the one that falsified the tidy conclusion). Receipts, including a copy of each
script as it ran: `bench/context/receipts-attnsel-20260725/`.

Finding #19: `bench/context/flagship.py` under `PREREG-flagship-context.md`
(2×A100-SXM-80GB, taken for 2 TB of host RAM rather than the GPUs).

Finding #20: `bench/context/train_context_bench.py` under
`PREREG-training-context.md`. The streamed-weight half of that harness did not
run — it caught only `OutOfMemoryError`, and the guard that fired was a
`RuntimeError` — so T1c and T1d are recorded VOID rather than back-filled.

Finding #21: `PREREG-planner-validation.md` in the private planner repo drove
this run; the harness is `tuned.py` (staged, not committed — it is 60 lines of
arm-runner around the public loader). The `off` arm was measured on a prior pod
that vanished mid-run and reproduced #19 to the decimal (0.173 tok/s, 18.62 GiB),
which is why it is carried rather than re-paid for.

Finding #22: `bench/context/PREREG-routed-staging.md`; implementation is
`enable_routed_staging` in `experts4bit_qlora/offload.py`, unit-verified by
`tests/test_routed_staging.py` (5/5, full suite 34 passed on an RTX 4090).

Finding #23: decomposition harnesses were run ad hoc on a 4090 (staged, not
committed): copy/sync/allocation variants at real per-layer shapes, then one
Qwen3-235B-shaped `ExpertsLoRA` layer timed with and without `enable_fast`.

Finding #24: shape sweep and full-model determinism probes were run ad hoc on an
A5000 (staged, not committed). Receipts for the pair run they interrogate:
`bench/context/receipts-pair-20260726.json`.

Finding #25: `logit_gates.py` (staged, not committed) — OLMoE-1B-7B, natural
prompt, 2x2 over staging x kernel, comparing first-forward logits. Unit-level
gate is `tests/test_routed_staging.py::test_composes_with_the_grouped_kernel`,
now asserting bit-identity between bulk+grouped and routed+grouped.

Finding #26: `bench/context/PREREG-fast-perplexity.md`. The deterministic combine
is `_scatter_combine` in `experts4bit_qlora/fast.py`, shared by both fused paths.

Finding #27: `gate.py` (staged, not committed) — OLMoE, natural prompt, 6 cells
(bulk/routed x ref/grouped, plus self-repeats), comparing first-forward logits.

Finding #28: `k2probe2.py` (staged, not committed) — K2's config into
`transformers`' native `DeepseekV3ForCausalLM` (K2's own remote code imports
`is_torch_fx_available`, removed in transformers 5.x), 2 dense layers at full MLA
width, cache bytes measured after one forward.

Finding #31: `bench/context/PREREG-pair-deterministic.md` (resumed);
receipts `bench/context/receipts-gate94-20260726.json`.

Finding #32: `bench/context/PREREG-speculative-routing.md`; receipts
`bench/context/receipts-speculative-20260726.json`.

Finding #33: `enable_speculative_staging` in `experts4bit_qlora/offload.py`;
receipts `bench/context/receipts-specstaging-20260726.json`.

Finding #34: `bench/context/PREREG-full-stack-235b.md`; receipts
`bench/context/receipts-fullstack-20260726.json`.

Finding #35: `bench/context/PREREG-cache-slots.md`; receipts
`bench/context/receipts-cacheslots-20260726.json`.

Finding #36: `bench/context/PREREG-step-decomposition.md`; receipts
`bench/context/receipts-decomp-20260727.json`.

Finding #37: `bench/context/PREREG-kv-dial-sweep.md`; receipts
`bench/context/receipts-kvdial-20260727.json`.

Finding #38: `bench/context/PREREG-nf4-flash.md`; kernel `flash_nf4_kv_gqa` in
`kernel/nf4_kv.py`; receipts `bench/context/receipts-nf4flash-20260727.json`.

Finding #39 (part A): `bench/context/PREREG-decode-bound.md`; receipts
`bench/context/receipts-decodeiso-20260727.json`.

Findings #40 and #41: `bench/context/PREREG-decode-bound.md` and
`bench/context/PREREG-gemv-occupancy.md`; receipts
`bench/context/receipts-redecomp-20260727.json`.

Finding #42: `bench/context/PREREG-bf16-ladder.md`; receipts
`bench/context/receipts-bf16ladder-20260727.json`.

## Finding #43 — the GEMV was loading every packed byte twice, and I nearly shipped the wrong fix

#41 ended with "I do not understand this kernel" after three registered
falsifications. The fourth hypothesis came from reading the source rather than
guessing at a resource, and it was right:

```python
bytes_ = tl.load(b_base + (kk[None, :] // 2), ...)   # kk spans BLOCK_K consecutive k
```

`kk` runs over **consecutive** k, so the packed-byte tile is addressed by
`kk//2`. Loading `BLOCK_K/2` bytes once and extracting both nibbles is the whole
fix. **The mechanism stated here originally — "halves sector efficiency, issues
2× the loads" — was falsified by Nsight Compute in #45:** sectors fall **4.63×**
(12.08 → 1.95 per request, i.e. the old access was *scattered*, not merely
redundant) and requests actually **rise 1.34×**. The speedups below all stand;
the causal story did not.

### The trap: a falsified component rode along on a confirmed one

Three arms were registered. On the A2000 the composition looked best, so that is
what I first landed. It is a **severe regression** on the card the flagship
actually runs on:

| arm | A2000 sm_86 / triton 3.4 | A100 sm_80 / triton 3.0 |
|---|---:|---:|
| **H4 — load each byte once** | 2.375× | **4.080×** |
| H5 — one reduction, not per-iteration | 0.970× | 0.806× |
| H6 — `BLOCK_K` unpinned | 0.987× | 0.809× |
| **composed (first landed)** | 2.838× | **0.361×** |

**H5 and H6 were falsified by my own preregistered measurement** (0.970×,
0.987×) and I shipped them anyway because their *composition* with a confirmed
mechanism measured well **on a single card**. The cause is almost certainly
register spilling: the `[BLOCK_N, 32]` accumulator stays live across the
unrolled `NSUB` inner loop, which triton 3.4 schedules on sm_86 and triton 3.0
does not on sm_80.

> **A falsified component does not earn a ride on a confirmed one.** Ship the
> mechanism that was confirmed, alone, and re-measure a composition on every
> target it claims.

**Nothing in the test suite could have caught this.** All 44 tests passed on the
composed kernel, because it was *correct* — just slow. The `err_vs_shipped` gate
proves equivalence and says nothing about time. Only a second card did.

### What H4 alone delivers

Through the real `gemm_4bit_grouped` API, all eight census decode shapes,
original vs H4-only on the same card in the same container:

| | A100-SXM4-80GB (dedicated) | RTX A2000 12GB (shared) |
|---|---:|---:|
| geometric mean | **2.272×** | 2.257× |
| worst shape | 1.443× (gptoss_dn) | 1.254× (olmoe_dn) |
| best shape | 3.470× (olmoe_gu) | 6.154× (qwen_gu) |

Two architectures, two Triton majors, **never a regression on any shape**.
`kernel/test_nf4_grouped.py` passes **44/44** against a **44/44 control** on the
unmodified kernel in the same image — including split-K exactness and Gemma's
K=704.

> The A2000 is a **shared production GPU** and its per-shape numbers are noisy:
> the *same* original kernel measured 0.687 ms and 1.519 ms for `qwen_gu` in two
> runs, a 2.2× spread. Its geomean is directionally consistent with the A100 but
> **the A100 figures are the trustworthy ones.**

### Step level, measured rather than extrapolated

94 layers, E=128, top_k=8, hidden 4096, inter 1536 — the flagship geometry with
synthetic NF4 weights — on 2×A100-SXM, `--no-stream` so compute is exposed and
no link is in frame:

| kernel | c_box | vs original |
|---|---:|---:|
| original | 191.7 ms/token | 1.00× |
| composed (reverted) | 424.8 ms/token | **0.45×** |
| **H4-only** | **102.2 ms/token** | **1.876×** |

Registered prediction for this cell was **1.4–2.0×**; measured **1.876×**.
**Confirmed.**

The streamed cells are **not quotable**: the two runs saw different link speeds
(24.9 vs 22.5 GB/s). The confound favoured the fix and it still lost, so the
direction is sound, but no ratio is claimed from them.

### What is still NOT closed

- #39 measured **4.9×** of headroom to the GEMM's decode floor; this recovers
  ~2.3×, so roughly **2.1× remains**.
- **#40's 2.19× is still unmeasured on its own mechanism.** It was derived by
  decomposing the *e4b* step, where experts are 71.3% of wall. Everything above
  is `bench/phase3`, a different mechanism (#29) that double-buffers. The
  prereg recorded this exclusion before the run, and it stands.
- `_gemm_nf4_grouped` (prefill, `nf4_grouped.py:145`) keeps the **same
  duplicated byte load**. This was attempted and **closed as a measured
  negative** — see **#44**: both transforms are bit-identical and ~2× faster on
  sm_86/triton-3.4 and produce **~100% relative error** on sm_80/triton-3.0.
- `_gemv_nf4_grouped_splitk` got H4 only, for the same reason it is the only
  part shipped anywhere: its span arithmetic counts 64-element absmax blocks, so
  `BLOCK_K` there is load-bearing rather than a knob.

### Two harness errors worth keeping

**The first metric failed every arm, including a verbatim copy of the shipped
kernel.** It used per-element relative error, and the output sums K random-signed
terms, so entries land near zero and the ratio explodes. A *known-good* kernel
failing is what surfaced it.

**My own registered diagnostic was falsified.** I predicted the shipped kernel
would be **super-linear** in loop trip count; holding bytes constant it is
**sub-linear and saturating** (1.808 → 2.260 ms across an 8× trip increase) —
which is exactly why the two trip-count hypotheses did nothing. The sweep also
carries a confound: holding bytes constant forces N to vary inversely with K, so
grid size moves 8× alongside trip count.

Receipts: `bench/context/receipts-gemv-steplevel-20260727/`.

## Finding #44 — the prefill path stays broken-by-design, and a bit-identical result on one card meant nothing

#43 left `_gemm_nf4_grouped` (prefill) carrying the same duplicated-byte load it
fixed in the decode GEMV. Closing it was preregistered with a **pessimistic**
expectation — prefill is compute-bound (#39), each weight element is reused
`BLOCK_M` = 64 times, so halving *load instructions* should not matter.

Two transforms were built. Both load `BLOCK_K/2` bytes once and split the
nibbles; they differ in how they get back to a contraction:

- **P-A** — `tl.join(w_hi, w_lo)` → `[BN, BK/2, 2]` → reshape `[BN, BK]`,
  reproducing the original K layout. A operand and the single `tl.dot` untouched.
- **P-B** — two dots of `K = BK/2`, no join, stride-2 A loads.

### On the A2000 this was the most convincing result of the whole arc

| shape | P-A speedup | P-A agreement vs shipped |
|---|---:|---:|
| olmoe_gu | 1.847× | **0.000e+00** |
| qwen_gu | 1.753× | **0.000e+00** |
| qwen_dn | 2.237× | **0.000e+00** |
| gemma_gu | 2.246× | **0.000e+00** |
| gptoss_dn | 1.915× | **0.000e+00** |

**Geomean 1.989×, and bit-identical on every shape.** My registered prediction
was 1.00–1.20×, so this was a large miss in the flattering direction — the
`BLOCK_M`-reuse argument was simply wrong, the byte load is first-order even at
AI ≈ 64. A faster kernel with *zero* numerical difference is about as strong as
single-card evidence gets.

### On the A100 both transforms compute garbage

Same code, sm_80 / triton 3.0.0, relative error against the shipped kernel:

| shape | P-A | P-B |
|---|---:|---:|
| olmoe_gu | **9.294e-01** | 9.466e-01 |
| qwen_gu | 8.472e-01 | 7.966e-01 |
| qwen_dn | 9.591e-01 | 8.966e-01 |
| gemma_gu | 8.837e-01 | 9.523e-01 |
| gptoss_dn | **9.966e-01** | 9.417e-01 |

A relative error near **1.0** means the output is unrelated to the answer. This
is not "less accurate" and not "slower" — it is **wrong**. The timings collected
alongside are meaningless and are not quoted: you cannot time an incorrect
kernel. `tl.join`+`tl.reshape` evidently does not carry the same semantics from
Triton 3.4 back to 3.0, and P-B fails there too despite using no join at all.

### Verdict, per the pre-committed rule

The registered stopping rule was "**< 1.00× on either card → falsified, leave
the prefill path alone permanently**". It fires. **Neither transform ships.**
`_gemm_nf4_grouped` keeps its duplicated-byte load, and that is now a
**recorded, measured decision** rather than the deferral #43 left open.

### Why this is the most important result in the arc

#43 shipped a composed kernel on one card's evidence and it was a **2.2×
regression** on the other. The rule adopted afterwards — *nothing ships on one
card* — was written for a **performance** failure. Here it caught a
**correctness** failure, and one that no gate in the repo would have flagged:

- P-A was **bit-identical** on the development card. Not "within tolerance" —
  exactly equal.
- `kernel/test_nf4_grouped.py` runs on whatever card is present, so on the
  A2000 it would have passed 44/44.
- The defect is invisible until the code is *compiled by a different Triton*.

> **A bit-identical result on one toolchain is not evidence of correctness on
> another.** Portability of *numerics* has to be measured, exactly like
> portability of *speed*.

Cost of learning this: **$0.31** of A100 time, because the prereg said to test
the free card first and rent only if it cleared the bar.

Receipts: `bench/context/receipts-prefill-20260727/`.

## Finding #45 — Nsight Compute: #43's speedup is real, its stated mechanism was wrong

#41 pre-committed the next step to "a real profiler — not a fourth guess". Run
at last, on the A2000 (`RmProfilingAdminOnly: 1`, so a root container with
`--cap-add=SYS_ADMIN` — **no rental needed**, counters collect fine).

Old vs new GEMV, same shape, one profiled launch each:

| metric | `v0_shipped` | `v1_h4` (shipped now) |
|---|---:|---:|
| sectors per request | **12.08** | **1.95** |
| global-load sectors | 58,212,003 | **12,568,942** |
| global-load **requests** | 4,817,664 | **6,439,680** |
| SM throughput | 15.80 % | **66.17 %** |
| DRAM throughput | 6.28 % | 29.15 % |

### Two claims in #43 are now falsified by direct measurement

1. **"halving sector efficiency."** Sectors fall **4.63×**, and sectors/request
   improves **6.2×** (12.08 → 1.95). 12 sectors for one warp-wide request means
   the old access was scattered across cache lines, not merely 2× redundant. A
   `[BLOCK_N, BLOCK_K]` byte tile indexed by `kk//2` puts lanes on rows that are
   `stride_bn` apart; halving the K-extent makes each row exactly one 32 B
   sector. **1.95 is near-ideal.**
2. **"issuing 2× the loads."** Requests **rose 1.34×** (4.82 M → 6.44 M). The
   fix issues *more* instructions and is still 2.3× faster.

**The mechanism is coalescing/sector efficiency, not instruction count.** The
speedup and every timing number in #43 stand; the causal story did not.

### It also explains #41's dead end

#41 concluded "not memory-bound" from DRAM throughput — **6.28 %**, plainly not
DRAM-bound. But the kernel was bound by **L1TEX sector throughput**, a different
resource entirely. Four falsified hypotheses came from checking DRAM bandwidth,
FLOP/s, decode cost and occupancy while the limiter was transaction efficiency
one level up. *"Not memory-bound" needs to name which memory resource.*

### The current limiter, now that the old one is gone

| | |
|---|---|
| top stall | **Long Scoreboard — waiting on L1TEX** |
| cost | **9.8 cycles/warp = 51.0 %** of the 19.2-cycle issue gap |
| achieved occupancy | **51.44 %** vs 66.67 % theoretical |
| theoretical cap | 8 warps/scheduler vs hardware 12; limited by blocks/SM + shared mem |

ncu's own estimates: **50.95 %** from removing the L1TEX stalls, **22.84 %**
from closing the achieved-vs-theoretical occupancy gap, **33.33 %** from raising
theoretical occupancy.

The kernel is now **latency-bound on global-load dependencies** — the textbook
fixes are more loads in flight (ILP) or more warps to hide the latency.

> **This legitimately reopens occupancy, which #41 falsified** (8× occupancy
> bought 16 %). That falsification was against the *sector-bound* kernel, where
> more warps could not help because the transactions themselves were the wall.
> With sectors at 1.95/request the bottleneck has moved, and latency hiding is
> exactly what occupancy buys. **Re-testing a falsified hypothesis is correct
> when the measured bottleneck has changed** — but it must be re-registered, not
> assumed.

Receipts: `bench/context/receipts-ncu-20260727/`.

## Finding #46 — occupancy falsified a second time, and ncu's 50.95% is not reachable by config

#41 falsified occupancy on the *sector-bound* GEMV. #45 showed the post-#43
kernel is **latency**-bound (Long Scoreboard = 51.0% of the issue gap), which
legitimately reopened the question — and added the lever #41 never swept,
`num_stages`. Re-registered before measuring, then swept 72 configs
(`BLOCK_N × warps × stages`) on two architectures.

**Paired measurement was mandatory and the first attempt was void.** A sweep
with one up-front baseline reported the **default config as 1.283× faster than
itself** — the shared A2000 drifts that much between runs. Re-run with the
default re-timed immediately before every candidate, ratio taken per pair. The
self-pair then reads **0.996×** (A2000) and **1.009×** (4090), which is the
validation that makes the rest of the table meaningful.

| lever | predicted | A2000 sm_86, 26 SM, triton 3.4 | RTX 4090 sm_89, 128 SM, triton 3.0 |
|---|---|---:|---:|
| `num_warps` alone | 1.10–1.35× | 1.106× | **1.009×** |
| `num_stages` alone | 1.10–1.40× | 1.187× | **1.009×** |
| best combined | ≥1.25× to land | 1.358× | **1.155×** |

### Both isolated levers are falsified

On the 4090 neither moves the kernel at all. `num_stages` at the *winning* tile
runs 1→6 as **1.155 / 1.145 / 1.125 / 1.136 / 1.146 / 1.146** — flat inside
noise. The entire residual win is **`BLOCK_N=32, warps=1`**: a **tile-shape**
effect, not occupancy and not instruction-level parallelism. My prediction that
`num_stages` would be the *stronger* lever was wrong on both cards.

### The landing rule fires as a refusal

Best combined is **1.155×** on the 4090, inside the pre-committed
1.10–1.25× "do not land" band. **`_decode_plan` is not touched.** Four
registered result-sets depend on it and a card-dependent 1.15× does not justify
the risk — especially when the same config gives 1.358× on one card and 1.155×
on another, which is itself evidence the win does not generalise.

### What this says about profiler estimates

ncu reported **"Est. Local Speedup: 50.95%"** for removing the Long-Scoreboard
stalls. Neither occupancy nor ILP captures any stable part of it.

> **A profiler's "estimated speedup" is an upper bound on what removing a stall
> would be worth, not a claim that the stall is removable — least of all by
> configuration.** ncu names the resource; it does not promise a knob exists.

**Occupancy is now falsified twice, on two different bottlenecks, across three
architectures.** Config tuning is closed for this kernel. The remaining ~2.1×
against #39's decode floor is **structural** — shared-memory staging or a
packed-byte layout change — not something a launch parameter reaches.

Cost: **$0.12** on a 4090, chosen over an A100 because a config sweep needs
*architectural* diversity (sm_89, 128 SM) more than it needs the flagship's
exact card, and because the free A2000 had already screened the question.

Receipts: `bench/context/receipts-occ-20260727/`, harness
`bench/context/occ_sweep.py`.

## Finding #47 — scoping the three open items, and the "remaining 2.1×" is measured against an unreachable bound

Exploration only; no kernel changed. Each item was scoped far enough to price it
or to kill it.

### (a) The e4b ladder harness is unrecoverable and must be rewritten

The receipts behind #34/#40/#42 (`receipts-fullstack-*.json`,
`receipts-bf16ladder-*.json`) contain **results only** — `link`, `rows`, `e2e`,
`spec_rate`, `cache_rate` — and no `cmd`, `script`, or `argv`. No committed
script in **either** repo calls `enable_routed_staging`; the only hits in e4b are
its unit tests. The script was written on a pod and lost with it.

So the project's **headline 9.19× ladder has no reproducible harness.** Rewriting
it is straightforward — the four rungs are documented API calls — but validating
it needs a **real 235B checkpoint on the e4b path**, not `bench/phase3`'s
synthetic weights, which is a materially bigger rental than anything run today.

### (b) Triton portability: no live bug, but the fallback is a lie

`tl.join` appears **only** in `bench/context/prefill_singleload.py` — the
transform #44 rejected. **No shipped kernel uses it**, so #44's
wrong-answer-on-3.0 defect never reached the package.

`tl.gather` is real though: 11 uses, guarded by `hasattr(tl, "gather")` in three
files. Measured on a Triton 3.0 container against the shipped code:

```
triton 3.4:  PREFILL OK
triton 3.0:  PREFILL FAILED — AttributeError: module 'triton.language' has no attribute 'gather'
```

The guard selects `prefill_variant=0`, but the **kernel source still contains
`tl.gather`**, and Triton 3.0 resolves it during the AST walk even in a dead
`constexpr` branch (the same mechanism that broke #44's bench). So the
gather-less fallback **cannot run on precisely the Tritons it exists to
rescue** — in `nf4_grouped.py` *and* `mxfp4_grouped.py`.

**Severity is low, and stating it accurately matters:** `pyproject.toml`
declares `torch>=2.8, triton>=3.4`, so a gather-less Triton is *out of declared
support*. Nobody on a correct install is broken. The defect is that three files
advertise a fallback that is **misleading rather than protective** — dead code
that looks like a safety net. Fix is either to delete the guards (and say
plainly that ≥3.4 is required) or to split the kernel so the fallback actually
compiles. **This is correctness work and needs no rented hardware.**

### (c) The structural GEMV work is weakly motivated — and the target is wrong

Two candidate structural changes, both undercut by evidence already in hand:

1. **Shared-memory staging is partly pre-falsified.** Triton's `num_stages`
   *is* software pipelining through shared memory. #46 swept it 1→6 on two
   architectures and found it **flat** (1.009× on the 4090). Hand-rolling the
   same idea should not be expected to beat the compiler's version of it.
2. **A layout change has little coalescing left to win.** #45 measured
   **1.95 sectors per request** post-#43, which is near the 1.0 ideal. The
   scattered-access problem is already solved.

**And the headline number is measured against a bound the project itself calls
unreachable.** `PREREG-gemv-occupancy.md`, confound 2:

> "The 487 GB/s ceiling is a flat streaming read with no reduction and no
> output — **genuinely unreachable, useful only as a bound**."

Every "4.9× headroom" and "~2.1× remaining" figure is distance to *that*. A GEMV
must also reduce along K and write an output; neither is in the bound.

> **Before any structural work, establish a *reachable* target.** The honest
> next step is not shared memory — it is a reference point that includes the
> reduction and the store: a hand-written CUDA GEMV on the same packed bytes, or
> a roofline that prices the reduction. Optimising toward an acknowledged-
> unreachable number is how a project talks itself into work with no payoff.

**Recommendation: (b) first** — cheap, correctness-only, free testbed. **(a)
second**, because the flagship claim currently rests on a lost script. **(c) is
not ready to start** until it has a target worth aiming at.

## Finding #48 — the ladder harness is rebuilt and working, and it corrects the README on bit-identity

#47 found the driver behind the README's headline 9.19× had been lost with the
pod it was written on. Rebuilt as `bench/context/e4b_ladder.py` and validated
end-to-end on **OLMoE-1B-7B** (RTX 3090, torch 2.13 / triton 3.7 / transformers
5.14, **$0.09**):

| rung | s/token | vs bulk | Δlogit vs rung below |
|---|---:|---:|---:|
| bulk | 0.2017 | 1.00× | — |
| `enable_routed_staging` | 0.0807 | 2.50× | **0.000e+00** |
| `+ enable_fast` | 0.0562 | 3.59× | **3.125e-01** |
| `+ enable_speculative_staging` | 0.0516 | **3.91×** | **0.000e+00** |

Speculation hit rate **0.824** (738 hits / 158 misses / 896).

### The correction

The README says **"every rung is bit-identical"**. That is wrong, and the same
README contains the contradiction: it also prices the grouped kernel at
**+0.023% perplexity**. A kernel that moves perplexity cannot be bit-identical.
Nobody had a harness to notice.

Measured per-rung, exactly **one of three transitions** changes the numbers, and
it is the priced one:

- **routed staging — bit-identical.** It moves *which bytes* are copied, not
  what is computed.
- **the grouped kernel — NOT bit-identical**, by design, at the documented
  +0.023% ppl.
- **speculative staging — bit-identical.** It moves *when* the copy starts.

A cumulative-only comparison hides this: measured against rung 1, `spec` also
shows 3.125e-01 and looks non-identical, when it introduced nothing of its own.
The harness now reports **both** deltas for that reason. The accurate claim is
*"the staging rungs are bit-identical; the kernel rung is priced"*, not
*"every rung is bit-identical"*.

### Two defects caught while building it

**The first draft measured the wrong kernel.** It re-ran a full forward over the
whole prompt each step — a prefill-shaped call (M>1) dispatching the M-tile
GEMM, not `_gemv_nf4_grouped`. It would have produced a plausible ladder for a
path the ladder is not about. `decode()` now uses a real KV cache and feeds one
token per step.

**The loader does not return the offload handles.**
`load_moe_4bit_streaming` returns `(model, config)`, but
`enable_routed_staging` *requires* handles. They live at `layer.mlp._offload`,
reachable only by the private walk `enable_speculative_staging` does internally,
which `_collect_handles` now mirrors. **That API gap is the most likely reason
the original harness was ad hoc and never committed** — reproducing the headline
requires reaching into a private attribute. Worth fixing in e4b by returning
handles or exposing an accessor.

### Not yet done

This validates the **mechanism and the harness**, not the flagship number. The
9.19× is a 235B claim; OLMoE is a different model at a different scale, and its
3.91× is not a substitute. Confirming the headline needs a 2 TB-host-RAM box and
a ~470 GB checkpoint stream — materially more than any run today.

Receipt: `bench/context/receipts-ladder-20260727/ladder_olmoe.json`.

## Finding #49 — the 235B ladder reproduces independently, and the bit-identity correction holds at flagship scale

The rebuilt harness (#48) run on the real Qwen3-235B-A22B, 2×A100-SXM4-80GB,
2015 GB host RAM, torch 2.13 / triton 3.7 / transformers 5.14. Load (stream,
quantize 94×128 experts to NF4, pin 122 GB) took **3445 s**; the measurement
itself is seconds.

| rung | s/token | tok/s | vs bulk | Δlogit vs rung below |
|---|---:|---:|---:|---:|
| bulk | 5.7974 | 0.172 | 1.00× | — |
| `enable_routed_staging` | 0.9223 | 1.084 | **6.29×** | **0.000e+00** |
| `+ enable_fast` | 0.6800 | 1.471 | **8.53×** | **3.750e-01** |
| `+ enable_speculative_staging` | **0.5764** | **1.735** | **10.06×** | **0.000e+00** |

Speculation hit rate **0.9046** (5326 / 562 / 5888), against 0.8973 recorded in
#42. 94 offload handles recovered, `enable_fast` patched 94 modules,
speculative hooks on 92 layers.

### The headline corroborates

This is an **independent** reproduction: the original harness was lost (#47) and
this one was rebuilt from the API, on a different pod, a different toolchain,
and a different transformers major. It lands at **10.06×** against the
**9.19×** recorded in #42 (bf16 KV) and **10.21×** in #34.

The rungs bracket well: bulk **5.7974** here vs 5.9041 in #42 — within 2%. The
spread is at the top: 0.5764 vs 0.6423 s/token, i.e. the speculative rung ran
faster here, which is consistent with its higher hit rate (0.9046 vs 0.8973).
Per the project's own law only **within-run ratios** transfer, and 9.19 / 10.06 /
10.21 across three pods is agreement, not disagreement — but the honest headline
remains a **band, not a point**: the ladder is ~9–10× and the exact figure is
pod- and hit-rate-dependent.

### The correction from #48 holds at 235B

OLMoE showed `routed = 0`, `fast ≠ 0`, `spec = 0`. The 235B reproduces **exactly
that shape** (0.000e+00 / 3.750e-01 / 0.000e+00), so #48's finding was not a
small-model artifact:

- **routed staging — bit-identical.** Changes which bytes move.
- **the grouped kernel — NOT bit-identical**, by design, priced at +0.023% ppl.
- **speculative staging — bit-identical.** Changes when the copy starts.

**The README's "every rung is bit-identical" is wrong and must be narrowed to
"the staging rungs are bit-identical; the kernel rung is priced".** Two models,
two scales, same result.

Cost **$2.98** (~64 min, most of it the ~470 GB checkpoint stream at ~139 MB/s).
Receipts: `bench/context/receipts-ladder-20260727/ladder_235b.json`.

## Finding #50 — the "headroom" was never kernel overhead; it is the access pattern

Every headroom figure in this project — #39's **4.9×**, #46's **~2.1× remaining**
— measured distance to a floor of **487 GB/s**. That floor is a **coalesced**
read. The GEMV's access is **strided by N**. The comparison was apples to
oranges from the start.

Measured directly, same bytes, two access patterns, one process:

| | A6000 sm_86 / 84 SM | RTX 4090 sm_89 / 128 SM |
|---|---:|---:|
| strided (the GEMV's pattern) | 150.0 GB/s | 402.9 GB/s |
| coalesced (identical bytes) | **487.7 GB/s** | **1404.3 GB/s** |
| **access-pattern penalty** | **3.25×** | **3.49×** |

The A6000's coalesced figure — **487.7 GB/s** — reproduces #39's 487 almost
exactly. #39 measured a coalesced stream and the project has been treating it as
a target for a strided kernel ever since.

### There is no removable kernel overhead

A stripped GEMV doing exactly the required work and nothing else (`R5`: unpack,
LUT, scale, multiply, K-reduce, store) against the shipped kernel:

| | A6000 | 4090 |
|---|---:|---:|
| shipped / R5 | **1.058×** | **1.009×** |

Both under the pre-committed **1.3×** line. **The structural-kernel line CLOSES**
as a measured negative: there is nothing left to remove from this kernel's
tiling or dispatch, and the hand-written CUDA GEMV should **not** be written —
it would be re-deriving a kernel that is already at its structure's limit.

### The preregistered design partly failed, and the decisive result was a follow-up

Recording this because the write-up would otherwise imply a cleaner experiment
than actually happened.

**The term-by-term ladder did not work.** R4 measured *faster* than R3 (0.889×)
and R5 faster than R4 (0.935×) — adding work cannot speed a kernel up. The cause
is the sink: R1–R3 hold a `[BLOCK_N, 32]` accumulator live across the whole K
loop, and that register pressure costs more than the reduction it was meant to
isolate. **Rungs R2–R4 are confounded and are not usable as bounds.**

**R5 is not independent of what it measures.** It is structurally the shipped
kernel, so `shipped/R5 ≈ 1.0` is close to tautological *by construction*. It
does establish "no removable overhead in this structure" — a real but narrower
claim than "no faster kernel exists".

**The load-bearing number came from a follow-up**, not the registered design: the
coalesced-vs-strided comparison. It was not preregistered, so it is reported as
what it is — a strong measurement made after the planned one under-delivered,
confirmed on two architectures, not a confirmation of a prior hypothesis.

### What is actually left

The ~3.3–3.5× is reachable **only by changing the packed layout** so a warp's
k-range is contiguous by construction — `[E, N, K/2]` puts consecutive `k` in one
row while a warp spans `N`.

> **CORRECTION (same day).** This finding first called that "a format change …
> touching the on-disk layout that the PyPI packages, four registered
> result-sets and every existing checkpoint depend on … a different project with
> a migration story". **That is wrong.** The packed layout is **never
> persisted**: `repack_from_bnb` builds it in memory from bnb's quantize output,
> and the loader `safe_open`s a **bf16** checkpoint and quantizes on the way. The
> only `torch.save` in either repo writes LoRA adapters. So a repack is
> `repack_from_bnb`'s output layout + the kernel's indexing + four call sites
> (`bench/phase3/offload_decode_235b.py`, `bench/phase1/harness.py`) — an
> ordinary code change with no compatibility story. **It is substantially more
> attractive than this finding originally claimed**, and the "don't start it
> casually" framing was based on a blocker that does not exist.

**Retire the 4.9× and ~2.1× figures.** They describe a distance to a bound the
kernel's access pattern forbids.

Receipts: `bench/context/receipts-roofline-20260727/`, harnesses
`bench/context/reachable_target.py` and `bench/context/access_pattern_penalty.py`.

> Caveat: the 4090 self-pair read **0.969×** rather than ~1.000×, a ~3% harness
> wobble. The conclusions here rest on 3.25–3.49× and ~1.0×, both far outside
> that, but a future run wanting tighter numbers should tighten the pairing first.

## Finding #51 — the repack is FALSIFIED, and #50's attribution was too generous

`PREREG-coalesced-repack.md` predicted the transposed store would win
**1.8–3.0× isolated / 1.6–2.5× census geomean**. Measured, same kernel, arms
differing only in `B.stride()` (logical bytes asserted identical):

| | A6000 sm_86 | A100 sm_80 |
|---|---:|---:|
| geomean | **0.957×** | **0.843×** |
| worst shape | 0.901× | 0.794× |
| best shape | 1.019× | 0.911× |
| self-pair | 1.004× | 1.007× |

**It is slower on every shape but two, on both cards.** Agreement stayed inside
the bf16 floor everywhere, so the layout is *correct* — it is simply not faster.
Prediction missed by ~2×, in the wrong direction.

The pre-committed rule fires: *"<1.2× on either card, or any shape regressing →
the coalescing model is wrong … the structural line closes for good."*
**It closes.**

### Why — and it corrects #50

#50 compared the GEMV's access against a **flat linear sweep** of the same bytes
and attributed the whole **3.25–3.49×** to layout. That was too generous. A tiled
GEMV reads a `[BLOCK_N, 32]` byte tile; **no layout makes that a linear sweep**:

- contiguous `[E, N, K/2]` → 64 segments of 32 B (rows `K/2` apart)
- transposed `[E, K/2, N]` → 32 segments of 64 B (columns `N` apart)

Transposing trades segment count for segment length and buys nothing measurable;
the extra address arithmetic costs more than the trade returns. **A large part of
#50's 3.25–3.49× is the cost of tiled access at all, not of this layout** — and
that part is not addressable by repacking.

> **Corrected reading of #50:** its measurement stands (strided 150.0 vs
> coalesced 487.7 GB/s), but "the remaining gap is the access pattern, reachable
> by a repack" was over-claimed. The reachable part of that gap is now measured
> at **≤ 1.02×**. #50's own conclusion — that the decode GEMV sits within
> 1.01–1.06× of a stripped kernel — was the durable result; the repack corollary
> was not.

### The branch was deleted, deliberately

The work made all three kernels stride-generic (an explicit `stride_bk`, nothing
assumed contiguous) — harmless, arguably more robust, and **unused**. It was
abandoned with the hypothesis rather than kept as tidy-looking generality: the
project already shipped a 2.2× regression by letting components ride along on a
result they did not earn (#43). Ship the confirmed mechanism, alone, or ship
nothing.

**Cost of closing the last performance line: $0.30 and one afternoon**, against a
layout migration that #50 had (wrongly) framed as the remaining opportunity.

Receipts: `bench/context/receipts-layout-20260727/`, harness
`bench/context/layout_ab.py`.

## Finding #52 — the routed-residual arms, and a metric that reported faster-than-light

`PREREG-routed-residual` ran on Qwen3-235B-A22B, 2×A100-SXM, 4 arms × 6 reps,
interleaved in one process, spec + cache off.

### Gates

**R1 (bit-identity) — PASSES.** Identical greedy ids and `max_delta_logit = 0.0`
in all 24 records. The gate that matters: a mismatch means a routed row was never
staged and the kernel read uninitialized memory.

**R2 (engagement) — PASSES.** The ablation switches genuinely flip:

| arm | ids path | row plan | s/token (median of 6) |
|---|---|---|---:|
| C | device 1410 / host 0 | dict 1406 | 0.9087 |
| T1 | **host 1410 / device 0** | **flat 1406** | 0.9117 |
| T1s | host 1410 | dict 1406 | 0.9085 |
| T1c | device 1410 | flat 1406 | 0.9136 |

Registered because equality testing is structurally blind here — both branches
return identical ids, so a fast path that never fires passes every correctness
check and reports "no measurable change". That is exactly how `enable_fast`
stayed dead on every offloaded model until #22.

**R6 — `T1/C = 0.9967`**, inside the registered 0.95–1.00 band, arms within a
1.2–1.7 % spread. The sync + dict-churn fix is real and worth ~0.3 %, as the
prereg predicted when it said removing an 8-element device sort "saves
kernel-launch time, not milliseconds".

### R4: the metric was broken, and fixing it CONFIRMS the prediction

`routed_gbps` reported **22.83 GB/s**. The probed ceiling was **14.68**, and an
independent size-swept re-probe on the same pod gave **12.65–18.46 GB/s** — so
the ceiling was right and the *implied rate exceeded the physical link*. That is
not a result; it is a broken gauge.

**The bytes are correct.** An independent model — 94 layers × top_k 8 ×
(gate_up + down + absmax), NF4-packed — gives **7.984 GB/token**, matching the
harness's own `7.98 GB/token`.

**The denominator is the problem.** `routed_gbps` divides by the summed
*copy-window* time (`start_ev`→`end_ev` bracketing each stage), not by step wall
time. Those answer different questions, and the window figure does not bound the
transfer it names.

Recomputed soundly: **7.984 GB ÷ 0.9061 s = 8.81 GB/s = 0.60× ceiling.**
R4 registered **≤ 0.70×**. **R4 HOLDS.**

> **Limitation, stated because R5 hangs off this.** The sound number divides by
> *step* wall time, whose denominator also contains attention, norms and the
> expert GEMM. It therefore cannot cleanly separate "transfer inefficiency" from
> "host stall" — which is what R4's *rationale* claimed to distinguish, even
> though its *criterion* was only a ratio. The criterion is met; the
> interpretation is weaker than the prereg's wording implies. R5 says R4 holding
> licenses building the expert-major coalescer, and it does — but its ceiling is
> the measured gap, and that gap is now known to be under 0.70× on a metric that
> conflates two costs.

### Corrections this finding makes to itself

I first said the **ceiling probe** was anomalous and the routed number
trustworthy, reasoning that #22 measured 22.21 GB/s so 22.83 looked plausible.
The re-probe exonerated the ceiling and indicted the routed metric. **#22's
22.21 was a different pod**; carrying it across boxes is the exact error the
project's own additive law forbids.

Receipts: `bench/context/receipts-routed-residual-20260727/`.

## Finding #53 — re-profiling the unchanged GEMV: the H4 fix moved the bottleneck, it did not only shrink it

The decode GEMV is **byte-identical to what #45 profiled** — `741308f` (reduce to
H4 alone) was its last change and #45 came after it. So this run asks what #45
did not: *what* is the Long-Scoreboard stall waiting on, now that #46 has
falsified occupancy/ILP and #51 has falsified the layout.

### The memory hierarchy reconciles end to end

| | |
|---|---:|
| unique bytes needed (packed 50.33 + absmax 6.29) | **56.62 MB** |
| L1 sector requests | 12.57 M = **402 MB — 7.1× the unique data** |
| L1 sector hit rate | **75.07 %** |
| → miss to L2 (predicted 3.13 M / measured) | **3.15 M sectors** |
| L2 sector hit rate | **9.00 %** |
| → miss to DRAM (predicted 91.70 MB / measured) | **91.82 MB** |
| DRAM amplification | **1.62× unique** |

Predicted-vs-measured agrees at both levels, so the model is sound. Two things it
says plainly: **L1 is doing the real work** (absorbing 7.1× request amplification
at 75 %), and **L2 is inert** (9 % — this is a pure stream with no reuse, exactly
as expected, and it means L2-oriented tuning has nothing to bite on).

### The stall split is the new result

| stall | cycles / issue-active | share |
|---|---:|---:|
| **long scoreboard** (waiting on data) | 9.23 | **65.6 %** |
| **lg throttle** (LSU pipe saturated) | **3.98** | **28.3 %** |
| mio throttle | 0.86 | 6.1 % |

#45 reported only the Long-Scoreboard figure. **`lg_throttle` at 28 % is new, and
it is a different kind of limit**: not waiting for data to arrive, but the
load/store pipe unable to retire requests as fast as they are issued.

That connects directly to #43, which measured the H4 fix **increasing** request
count (4.82 M → 6.44 M) while cutting sectors 4.63×. **The fix traded sector
efficiency for request count, and the request count is now costing about a
quarter of the stall.** So H4 did not merely shrink the bottleneck — it *moved*
part of it, from sector throughput into LSU issue pressure.

### Deliberately not proposing a fix

The obvious next thought — fewer, wider loads to relieve the LSU — is exactly the
shape of idea this kernel has already refuted four times: occupancy (falsified
twice, #41/#46), the prefill single-load transform (#44), and the coalescing
repack (#51). It also runs into #50's finding that the kernel is already within
**1.01–1.06×** of a stripped version doing the same required work, which bounds
how much *any* remaining change can return.

**This finding is diagnostic only.** Anything acted on here gets its own prereg,
with a landing bar and a two-card rule, like the four before it.

Profiled free on the QNAP A2000 (`gnf4-ncu:1`, `--cap-add=SYS_ADMIN`); counter
ratios, not wall times, so the shared box is a valid testbed for it.

