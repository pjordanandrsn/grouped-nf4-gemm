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
