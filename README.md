# grouped-nf4-gemm — single-launch 4-bit codebook GEMM over fused MoE expert stacks (NF4 + native MXFP4)

[![CI](https://github.com/pjordanandrsn/grouped-nf4-gemm/actions/workflows/ci.yml/badge.svg)](https://github.com/pjordanandrsn/grouped-nf4-gemm/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/grouped-nf4-gemm)](https://pypi.org/project/grouped-nf4-gemm/)

A Triton kernel that runs the grouped expert GEMM **directly on 4-bit-packed
weights** — one launch for all active experts, LUT decode to fp32 in
registers, blockwise fp32 scaling, fp32 accumulation, bf16 epilogue. No
per-expert dequantize-then-`bmm` round trip, no bf16 weight materialization.
Both 16-entry codebooks ship: **NF4** on the canonical bitsandbytes
`gemm_4bit` layout ([#1949](https://github.com/bitsandbytes-foundation/bitsandbytes/pull/1949))
— `[E, N, K/2]` uint8 + fp32 blockwise absmax — and **MXFP4 (OCP e2m1 +
e8m0 per-32 scales), computing on a checkpoint's exact released bytes**
(see the native-byte lane below).

**Why:** for frozen 4-bit MoE experts, the standard path pays to decode the
weights into bf16 and then reads them again — at batch-1 decode that round
trip (plus ~3 kernel launches per active expert) dominates. Fusing the decode
into the GEMM deletes it. The measured side effect worth stating plainly:
**fp32 accumulation makes the fused path *more accurate* than the
materialize-to-bf16 baseline — the fused path has never measured less
accurate than the baseline.**

## See it on your own hardware first

Every number below this section is one **I** measured. This one you measure:

```bash
pip install grouped-nf4-gemm bitsandbytes
python examples/dequant_tax.py          # ~1 min, one GPU, no model download
```

[`examples/dequant_tax.py`](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/v0.17.0/examples/dequant_tax.py)
is one file under 150 lines. It times the dequantize-then-GEMM round trip against
computing on the packed bytes, at a census shape, across three points on the M axis
— so the decay is visible rather than asserted. It prints a **self-pair** (the fused
arm timed against itself) beside every ratio, because a ratio inside the instrument's
own spread is not a measurement, and it prints what the run does *not* show. Without
`bitsandbytes` it falls back to the reference decode and labels the ratio an upper
bound. No GPU? It names what it needs and exits clean.

## Install

```bash
pip install grouped-nf4-gemm
```

`pip install nf4gemm` and `pip install gnf4` are equivalent aliases.
Published via trusted publishing; every wheel carries a PEP 740 attestation.

## Which entry point? Pick by where the weights live

This package is one kernel plus the machinery to feed it. What you call depends
on where the expert bytes are when you need them — nothing else.

| the bytes are in… | call | needs |
|---|---|---|
| **VRAM**, NF4-packed | `nf4_grouped.gemm_4bit_grouped(...)` | CUDA + triton |
| …and you need its **backward** | `nf4_grouped.dgrad_4bit_grouped(...)` | CUDA + triton |
| **VRAM**, native MXFP4 | `mxfp4_grouped.gemm_mxfp4_grouped(...)` | CUDA + triton |
| **host DRAM**, all rows pinned | `mxfp4_pipelined.Mxfp4PipelinedGptOss` | CUDA + triton + RAM ≥ all experts |
| **NVMe**, too big for DRAM | `mxfp4_residency.Mxfp4NvmeResidency` | a baked arena (below) |
| **NVMe**, and you want a real model wired up | `arena_moe_patch.enable_arena_experts(model, arena)` | a baked arena |
| nowhere yet — you need to *make* an arena | `nvme_arena.bake_expert_tensors(...)` | the checkpoint + disk |
| a checkpoint you want to **verify**, not run | `verify_provenance` | torch only |

**Do not quantize-bake a checkpoint that is already MXFP4.** The bake has two
modes and they are not interchangeable. `nvme_arena.bake_expert_tensors` is a
*relocation* — it copies the existing MXFP4 bytes into arena order, so the
residency engine hands packed nibbles straight to the fused kernel.
`nvme_bake_nf4.bake_nf4` *re-quantizes* to NF4, which then has to be
dequantized to bf16 per expert on every read. Measured on the same host and the
same task (DeepSeek-V4-Flash, 43L × 256E): **8.7 s per request on the MXFP4 lane
against 34.9 s on the NF4 lane — ~4×.** Quantize-bake only when the source is
bf16 or block-FP8 and there is no MXFP4 to relocate. `source=` on `bake_nf4`
picks the reader, and the two formats share tensor *names* on DeepSeek-V4
(`.weight`/`.scale` either way), so that flag is the only thing separating them —
both readers assert their format and name the other in the error.

**`--absmax-dtype bf16` takes 5.6% off every arena row, losslessly.** An NF4 row
is 11.1% fp32 absmax (294,912 of 2,654,208 B on Qwen3-30B). For a **bf16**
checkpoint that absmax is exactly representable in bf16 — it is `|w|.amax()` over
a block, so it *is* one of the source magnitudes, and the maximum of a set of
bf16 values is a bf16 value. Measured on the real model: **80/80 expert tensors
bitwise identical** after a round-trip, against an fp32-source control that is
correctly *not* identical. So the bytes shrink and nothing the model computes
changes. `auto` picks it only for sources where that proof holds, and the cast
*refuses* rather than rounding if it ever does not. The default stays `f32`,
because the arena index is self-describing but readers older than this refuse the
segment. int8/double-quant would take 8.3% instead — for a numerics change, a
re-bake accepted as a different quantization config, and a kernel contract that
excludes nested absmax. Consuming a bf16-absmax arena needs `experts4bit-qlora`
new enough to widen it back to fp32 at staging; VRAM and the kernel contract are
unchanged either way.

**The one ordering trap, because it costs 1.45 TB to get wrong.** An arena's
segment order has two legitimate forms. `arena_experts.K3_KINDS` is the
released-K3 spelling and interleaves per projection — fine for
`ArenaExpertSource`, which slices by suffix. `mxfp4_residency.K3_RESIDENCY_KINDS`
puts the two blocks segments adjacent and the two scales segments adjacent,
which the residency engine needs because it reads gate_up at **one computed
offset**. **`K3_RESIDENCY_KINDS` serves both consumers — bake with it.** As of
0.3.0 the gather can also permute a mis-ordered arena on the fly, so an existing
bake is readable either way; the order still decides whether you pay for that.

**Training** (LoRA over frozen 4-bit experts) goes through `nf4_qlora` /
`mxfp4_qlora`, which is what the sibling package
[`experts4bit-qlora`](https://pypi.org/project/experts4bit-qlora/) drives —
`enable_fast()` for inference, `enable_fast_train()` for the differentiable path.
Division of labour: this package makes one expert-stack matmul cheap; e4b decides
which bytes are where.

**Scope, unhedged:** the NVMe tier is a *batch* tier. At a measured per-box
`S ≈ 3.45 GB/s` a fully cold 235B streams ~2.3 s/token and a K3-class model
~7.5 s/token. If you need interactive latency, this is the wrong tier — what it
buys is reachability and provenance.

## Try it on CPU right now

No GPU needed for the pack/decode/provenance surface — the fused GEMM is
CUDA-only, but the reference decode and the provenance hashing are pure torch.

> **On Linux this works from a bare `pip install`; on macOS and Windows it does
> not, today.** `nf4_pack_ref` imports `nf4_grouped`, which does a module-level
> `import triton` — and triton is declared
> `triton>=3.4; platform_system == 'Linux'`, so it is simply absent elsewhere and
> these blocks raise `ModuleNotFoundError`. The *math* is pure torch; the import
> graph is not. CI executes these blocks on Linux, where triton is present, so it
> validates the code without validating this sentence. Tracked as a real defect —
> the reference decode should not need the kernel's dependency.
These three blocks are extracted and executed by CI (`test_readme_cpu_block.py`),
so they cannot drift from the API.

<!-- CPU-QUICKSTART-START -->
**1. NF4 round-trip** — pack a weight, decode it back, check the error:

```python
import torch
from nf4_pack_ref import quantize_pack_nf4
from nf4_grouped import dequant_ref

w = torch.randn(256, 512)                      # a per-expert weight [N, K]
packed, absmax = quantize_pack_nf4(w)          # [256, 256] uint8, [256, 8] fp32
wq = dequant_ref(packed, absmax, 256, 512)     # decode back to [N, K]
print("nf4 rel-err:", round(((wq - w).norm() / w.norm()).item(), 3))     # ~0.09
print("nf4 re-pack idempotent:", torch.equal(quantize_pack_nf4(wq)[0], packed))  # True
```

**2. MXFP4 round-trip** — the gpt-oss expert format, same shape story:

```python
import torch
from mxfp4_pack_ref import quantize_pack_mxfp4, dequant_mxfp4

w = torch.randn(128, 256)                      # [.., K], K a multiple of 32
blocks, scales = quantize_pack_mxfp4(w)        # [128, 8, 16] u8, [128, 8] u8 (e8m0)
wq = dequant_mxfp4(blocks, scales)             # [128, 256]
print("mxfp4 rel-err:", round(((wq - w).norm() / w.norm()).item(), 3))   # ~0.12
```

**3. Provenance in four lines** — hash on-disk bytes, catch a tampered one:

```python
import torch, json, struct, tempfile, os
from mxfp4_loader import file_tensor_sha256, tensor_sha256

t = torch.arange(64, dtype=torch.uint8)        # stand-in for an expert's packed bytes
hdr = json.dumps({"w": {"dtype": "U8", "shape": [64], "data_offsets": [0, 64]}}).encode()
path = tempfile.mktemp(suffix=".safetensors")
with open(path, "wb") as f:
    f.write(struct.pack("<Q", len(hdr))); f.write(hdr); f.write(t.numpy().tobytes())
print("prov bytes match:", file_tensor_sha256(path, "w") == tensor_sha256(t))    # True
b = bytearray(open(path, "rb").read()); b[-1] ^= 0xFF; open(path, "wb").write(bytes(b))
print("prov tamper detected:", file_tensor_sha256(path, "w") != tensor_sha256(t))  # True
os.remove(path)
```

That's the same instrument the 144/144 training receipt used.
<!-- CPU-QUICKSTART-END -->

## The MXFP4 native-byte lane (0.2.0)

gpt-oss ships its experts as **MXFP4 blocks** — e2m1 is a 16-entry codebook,
so the same in-register-decode mainloop serves it by table swap. The lane's
point is **provenance**: compute on the checkpoint's *exact released bytes*
(no requantization), which makes the served weights verifiable and deletes
the conversion tax. Stamped, receipts in `docs/mxfp4/`:

- **Serve** ([`RESULTS-mxfp4-serve.md`](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/v0.17.0/docs/mxfp4/RESULTS-mxfp4-serve.md)):
  fused-native exact-chunk ppl **26.72** on gpt-oss-120b = the
  shipped-precision reference (26.75) — the measured **+9.4% ppl / KL 0.066
  NF4-requant tax is deleted**; per-shard provenance
  `sha256(loaded bytes) == sha256(file range)` on a **4-tensor spot sample** of
  real 120b shards (4/4). Its own receipt grades this a sample, not shard-level
  coverage — read it as a spot check that the byte path is honest, not as
  "all four shards verified".
- **Train** ([`RESULTS-mxfp4-train.md`](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/v0.17.0/docs/mxfp4/RESULTS-mxfp4-train.md)):
  **gpt-oss-120b QLoRA at 9.82 GB peak VRAM on native bytes**
  (recompute-in-backward + per-expert LoRA), step-0 ppl inside the stamped
  serve band, **144/144** `sha256(file) == sha256(loaded) == sha256(post-train)`
  — the frozen base is byte-identical after training.
- **Verify it yourself**: `verify_provenance` re-hashes a checkpoint's expert
  byte ranges against a served/trained arena from the artifact alone
  (96/96 on the real shipped 20b bytes).
- **Kimi K3** — released, and the per-model STOP gate **PASSED**. That gate said
  no K3-specific number would be claimed until the oracle re-adjudicated our
  decode against K3's *own* declared reference. It did, on 2026-07-30:
  compressed-tensors 0.17.1, format `mxfp4-pack-quantized`, **33,030,144
  elements across w1/w3/w2, max abs delta 0, exact**
  ([`docs/RESULTS-k3-phase1-oracle.md`](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/v0.17.0/docs/RESULTS-k3-phase1-oracle.md)).
  A real-bytes arena round-trip on a byte-verified 1.56 TB store (96 shards
  checked against Moonshot's LFS hashes) came back **48/48 segments identical**
  with a byte-flip negative control, fixing the released row at **17,547,264 B**
  ([`docs/RESULTS-k3-slice-roundtrip.md`](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/v0.17.0/docs/RESULTS-k3-slice-roundtrip.md)).
  `moonshot_gather` is no longer merely K2-verified: it carries a `K3_SCHEME`
  measured against the real checkpoint, and K3's SiTU epilogue is registered
  from the release's own modeling code rather than inferred — none of the
  guesses had been right.

  **What the gate does not cover, stated because the receipts state it:** the
  oracle gates the *reference* decode (`mxfp4_pack_ref`), not the Triton kernel;
  it covers one expert of one layer; the round-trip is a slice (8 of 82,432
  rows) and carries no throughput claim; and both `.ots` stamps were applied
  *after* their runs, so these sit at this project's **`measured`** tier, not
  `confirmed`.

The engine composes with the hot/cold serving work in
[experts4bit-qlora](https://pypi.org/project/experts4bit-qlora/): hot sets
are format-independent, and the pipelined-residency integration rail is the
next e4b increment.

**Using it inside a model?** [experts4bit-qlora](https://pypi.org/project/experts4bit-qlora/)
ships this kernel as its optional inference path:
`pip install "experts4bit-qlora[fast]"` then `enable_fast(model)` routes the
frozen NF4 expert projections through `gemm_4bit_grouped` (measured 3.65× over
its reference per-expert loop at bs=1 decode, OLMoE geometry, A2000) with
automatic fallback for training and ineligible modules.

```python
from nf4_grouped import gemm_4bit_grouped, dequant_ref
```

### sm_120 census: faster than PyTorch's own grouped engine — on half the bytes (0.17.x)

At the Qwen3-30B-A3B serving cell (E=128, top-8, B=16, real expert shapes,
RTX 5090), `gemm_4bit_grouped` runs the routed expert GEMM in
**0.42 / 0.21 ms** where **`torch._grouped_mm` on unquantised bf16 — the
engine transformers v5 ships for MoE — takes 1.28 / 1.30 ms: 3.0–6.0×,
at 2× the weight bytes** (rel err vs the NF4 truth ≤ 5e-3). Three more
challengers lost at the same cell (an SMEM-dequant mainloop, the per-row
GEMV path, per-expert dequant+`mm`), and both kernels' configuration
spaces are swept closed on sm_120. Numbers, gates, receipts, and the
probe scripts:
[bench/sm120-census/RESULTS-sm120-grouped-census.md](bench/sm120-census/RESULTS-sm120-grouped-census.md).

### Training: the backward is a kernel too (0.7.0)

`gemm_4bit_grouped` is forward-only. `nf4_qlora` wraps it so `dL/dx` flows, and until
0.7.0 that backward was a Python loop over experts — one `dequant_ref` + matmul each,
~10k pairs per step at 256 experts over 40 layers, measured at **78–84% of a training
step**.

`dgrad_4bit_grouped` is that backward in one launch: `grad_out @ dequant(B)`, decoding in
registers exactly as the forward does, so it **materializes nothing**. Against the
per-expert decode oracle on an A2000 (T_cat=4096): gate_up E=256 **5.92 ms vs 61.78 ms**,
down E=256 **3.28 ms vs 85.12 ms**. Tuned it runs at 0.91× the *forward* kernel's time on
the same problem — it reaches the forward's ceiling.

`lora_delta_grouped` was the other per-expert Python loop, in the *forward*, putting `2E`
matmul nodes per projection per layer on the autograd graph. It is batched as of 0.7.0
(2.96× end-to-end), with a `_PAD_WASTE_LIMIT` fallback so pathological router skew cannot
cost more than before.

Together, one training step at E=256 goes **403.7 → 26.5 ms (~15×) at 134 MB peak**.

```python
from nf4_qlora import fused_grouped_lora
from nf4_grouped import dgrad_eligible

out = fused_grouped_lora(a_cat, packed, absmax, sizes, expert_ids,
                         lora_A, lora_B, scaling=alpha / r)
                         # dgrad_kernel defaults to True since 0.9.1
```

**On by default since 0.9.1**, opt-in before that. The loop decodes with the same oracle
the reference uses, so its gradient is *exact*; the kernel accumulates fp32 in a different
order and lands near 2.9e-3 — inside the bf16 budget, not zero — and that non-zero was the
whole case for making it opt-in. What the case never priced is the gap measured two
paragraphs above: an order of magnitude on the isolated backward, and 403.7 → 26.5 ms on
the composed step. Shipping the loop as the default meant the backward paid back the very
round trip the fused forward exists to avoid. `dgrad_kernel=False` restores the exact loop
and is the right choice for gradient-equivalence work — a bit-exact A/B against a reference
trainer, or convergence forensics. Ask `dgrad_eligible()` before
committing rather than catching: it falls back to the loop for non-bf16
gradients, a `BLOCK_K` that does not divide the quant blocksize, empty/evicted storage, and
offload-staged weights on another device — where the kernel would need the whole stack
resident, which is what offload exists to avoid.

Layer-composed fidelity is **measured** (experts4bit-qlora's
[`bench/dgrad-gate/`](https://github.com/pjordanandrsn/experts4bit-qlora/blob/main/bench/dgrad-gate/RESULTS-dgrad-gate.md),
2026-08-06): at 48 layers on Qwen3-30B-A3B the dgrad kernel adds **nothing** to the fused
lane's composed gradient error (4.97e-2 → 4.99e-2 mean vs the reference loop) and is the
fastest training option at real width (2.52× vs 1.72× without it). An fp32-truth arm over
the same NF4 bytes further shows every lane — the reference loop included — sitting on the
composed bf16 noise floor (~5.2e-2 at 48 layers), with the fused lane landing *closest* to
truth at 16 layers; divergence between lanes is two valid bf16 roundings, not one being
looser. Loss trajectories sit ≤0.003 median |Δ| against a 0.05 band.

From inside a model, [experts4bit-qlora](https://pypi.org/project/experts4bit-qlora/)
≥ 0.11.0 exposes it as `enable_fast_train(model, dgrad=True)`.

### Benchmark this on real text. Random token ids understate it by 1.6–1.7×

A MoE trainer benchmarked on random token ids is measuring a routing distribution
no user will ever have — and the error is **against** this kernel. Measured inside
a real QLoRA finetune (OLMoE-1B-7B, 16 layers / 64 experts, seq 512, LoRA r=8,
grad checkpointing, e4b 0.17.5 + published wheels), fused vs the per-expert
dequant-and-project loop, on two architectures
([receipts](https://github.com/pjordanandrsn/grouped-nf4-gemm/tree/v0.17.0/bench/phase1/results/dequant_forward/leg-e2e),
[write-up](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/v0.17.0/bench/phase1/results/dequant_forward/RESULTS-e2e-training.md),
prereg stamped pre-data):

| experts resident | RTX 4090 (sm_89) | H100 (sm_90) |
|---|---:|---:|
| **real prose** (wikitext-2) | **4.50×** | **4.75×** |
| random token ids | 2.75× | 2.81× |

The mechanism is routing, and it is measurable off the live router during the
timed run: prose hits **98.4%** of experts at **cv 0.687**, random ids only
**87.5%** at **cv 1.463** — fewer experts, far more unevenly. That is the opposite
of the intuition that random input spreads load, and it matters because fewer hit
experts means fewer iterations of exactly the Python loop this kernel replaces.
The fiction flatters the baseline. Prose routing reproduces across both cards to
the third decimal (0.984, cv 0.686/0.687), as it should — routing is a property
of model and data, not silicon; the random-id cells sit a little apart
(0.875/1.463 vs 0.883/1.471), which is bf16 non-determinism flipping marginal
routing decisions on inputs that carry no real structure to route on.

Under expert offload the same cells read 2.53×/4.06× (prose) and 1.81×/2.38×
(random): host↔device streaming is paid by both arms and compresses the ratio.
Which makes the honest note about our own prior number: `dgrad-gate`'s **1.99×**
for OLMoE was measured with random ids *and* offload, and it replicates here at
1.81×/2.38× — but it understated the same kernel on the same model by more than
half, purely through the fixture.

**What this does not claim.** The baseline is experts4bit-qlora's own per-expert
loop, not any third party's implementation. **Peak VRAM does not improve** — the
fused arms peak *higher* (5.31 → 5.65/5.87 GB), and a self-pair of the reference
arm against itself varied peak by 1.33× on identical work, so every peak
difference at this scale is allocator noise; only the ~1.9× *transient* (the
bytes held across forward-to-backward) is real, and it does not reach peak.
Absolute s/step is not comparable across the two rented hosts, because the
per-expert loop is host-bound and their CPUs differ; only within-host ratios are
reported. All eight self-pairs landed in 0.967–1.032. One model, 24 steps, seq
512 — steps are cheaper, which is not a claim that the adapter trains to a
better model.

**And the caveat that costs the most: that baseline is not CUDA-graphed.** A
large part of what the fused kernel removes at small batch is Python launch
overhead, and a user can remove it themselves — the per-expert loop captures
cleanly, the fused path needed work in 0.13.1 before it could. Racing a
*graphed* baseline instead, at the same routing-faithful fixture
([leg 4](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/v0.17.0/bench/phase1/results/dequant_forward/RESULTS-leg4-routed.md),
prereg stamped pre-data), the picture changes and is reported here rather than
left in the receipts:

| | RTX 4090 (sm_89) | H100 (sm_90) |
|---|---:|---:|
| decode band, T=32, ungraphed → **graphed** | 11.71 → **0.949** | 6.76 → **0.858** |
| training shape, T=2048, ungraphed → **graphed** | 2.94 → **1.489** | 1.63 → **1.059** |

**At the decode band the fused path loses to a graphed baseline on both cards,
and no speed claim there survives.** What survives is the memory-traffic
component, which graphing cannot touch: **1.489× at training shape on the
4090**, against parity (1.059) on the H100.

That split is now explained rather than merely observed. The fused kernel runs
at a roughly fixed, issue-limited rate — measured at ~168 GB/s on an H100
(≈5% of HBM3 peak) and ~214 GB/s on a 4090 (≈21% of its peak), R-flat on both
— while the baseline it replaces dequantises **one expert at a time**, a
4.2–8.4 MB working set that sits inside 50–72 MB of L2 and largely never pays
DRAM for its extra bytes (its apparent rate exceeds the 4090's physical peak,
which is how that was caught). **So the fewer-bytes thesis converts into speed
in proportion to how starved the baseline's memory system actually is** —
substantially on consumer GDDR6/GDDR6X, barely on HBM3 with a cache-resident
per-expert working set. Cross-architecture receipts and the falsified bands
behind that sentence are in
[`RESULTS-graphed-buckets.md`](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/v0.17.0/bench/phase1/results/dequant_forward/RESULTS-graphed-buckets.md).

The position this package holds is therefore unchanged and deliberately
narrow: **competitive at equal VRAM, and it wins when VRAM binds** — plus a
real speed win at training shape on bandwidth-limited cards.

## The NVMe tier: compute on packed bytes that never fit in RAM (0.2.5 / 0.2.6)

`nvme_arena` relocates a checkpoint's per-expert tensors into an expert-major
arena — hash-preserving, because every row segment is one whole source tensor
range. `arena_experts` then turns a row into the fused `[E, N, K//2]` blocks and
`[E, N, K//32]` e8m0 scales `gemm_mxfp4_grouped` already takes:

```python
from arena_experts import ArenaExpertSource, moe_layer_forward

src = ArenaExpertSource("k3.arena", device="cuda")      # O_DIRECT, async, qd-deep
out = moe_layer_forward(src, layer, a_cat, sizes, expert_ids)   # gate → GLU → down
```

Those shapes are not a coincidence worth glossing: a DeepSeek-V3-lineage MXFP4
release ships each expert as **exactly** `weight_packed [N, K//2]` +
`weight_scale [N, K//32]`, which *is* the kernel's input contract. So the bytes
travel disk → arena → GEMM with **no dequantize round trip and no
requantization** — what gets multiplied is what shipped.

`arena_moe_patch.enable_arena_experts(model, arena)` wires it into a real model
by rebinding `KimiSparseMoeBlock.moe_infer`, which already produces the kernel's
inputs (group-sorted tokens + per-expert counts) and then loops one matmul per
expert. The patch collapses that loop and changes nothing else — sorting,
weighting, unsorting and shared experts stay upstream's.
`arena_call_stats(model)` reports **`patched` and `calls` separately**, because a
patch count is not a call count.

**Scope, unhedged:** this is a *batch* tier. At a measured per-box
`S ≈ 3.45 GB/s` a fully cold 235B streams ~2.3 s/token and a K3-class model
~7.5 s/token. Interactive use is not the claim — see
[`docs/nvme-ceilings.md`](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/v0.17.0/docs/nvme-ceilings.md). What the tier buys is
reachability and provenance, not latency:
[`docs/K3-PROVENANCE-CHAIN.md`](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/v0.17.0/docs/K3-PROVENANCE-CHAIN.md) composes the
receipts from a publication hash to the multiply.

## The claim (blind-confirmed, receipts in-repo)

Everything below is from **pre-registered, OpenTimestamps-stamped blind
confirmatory runs** (protocol + pass/fail criteria stamped before data; two
devices; n=3 fresh-process reps; worst/median-rep reduction; failures
reported at full volume). On sm_86 at batch-1 decode, versus the
dequantize-then-matmul baseline on the same stacks:

- **Fidelity:** property suite green on every device, every run (35 → 44
  tests as the kernel grew); fused output error **below the baseline's in
  every cell ever measured** (fp32 accumulate).
- **Energy:** fused J/token **below the baseline in 104 of 112
  confirmatory-grade cells across v1–v3**. Six of the eight misses are the
  `top_k=1`/tiny class (named below); the other two are parity-margin
  readings (1.005, 1.010) on a single instance. On bandwidth-bound cells the
  energy win has never failed to replicate.
- **Speed:** census MoE shapes (OLMoE, Qwen3-30B, Gemma-4, GPT-OSS-120B,
  gate_up + down) run **1.16–2.73× at median** (one census cell —
  gpt-oss `down`, 2880×2880 — is instance-sensitive: 0.7–2.0× across five
  instances). Fresh off-census shapes with `top_k ≥ 6` (DeepSeek-V3,
  granite-3.1, Qwen3-Next) run **1.0–1.8× at median**; `k=2`-large shapes
  (Grok-1, Mixtral-8x22B) 1.0–1.24×, never slower.
- **Versus the other execution classes** (same-run census on the v6 kernel —
  an **exploratory census, not a blind confirmatory run**, as its own receipt
  says; treat these as measured, not confirmed, [receipts](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/v0.17.0/bench/phase1/results/comparators_v6/RESULTS-comparators-v6.md)):
  the grouped-bf16-GEMM **execution class** (`grouped_gemm.ops.gmm` —
  tgale96's standalone package — dequant inside the timed path as 4-bit
  storage requires) loses to the fused kernel on **every census cell — decode
  median 4.67×, prefill median 3.02×** (that class targets bf16-resident
  training, a job it is excellent at; this comparison is the 4-bit-storage
  regime, which both must serve when weights are quantized).

  > ⚠️ **That is an execution class, not Unsloth.** Unsloth's own MoE kernel is
  > `unsloth/kernels/moe/grouped_gemm/interface.py::grouped_gemm`, and the
  > backend above has never executed it — it returns early on tgale96's package
  > where that is installed, and raises `TypeError: 'module' object is not
  > callable` where it is not. **The proxy is also slower than the real thing:
  > 1.33× at median on an H100 (up to 3.40×), worst on the widest FFNs.** So the
  > 4.67× above was measured against a weaker opponent than "unsloth's MoE
  > backend" implies. Superseded by the head-to-head below, not rescaled.
- **Head-to-head against Unsloth's own kernel** — same pod, same process, arms
  interleaved with the fused kernel re-timed immediately before each comparator.
  Unsloth runs with `autotune=True` (their autotuner, their best config per
  shape) against gnf4's *shipped default*. Protocol
  [`prereg_unsloth_head_to_head.json`](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/v0.17.0/kernel/prereg_unsloth_head_to_head.json)
  + amendments, stamped pre-data; full write-up and per-cell matrix in
  [`RESULTS-unsloth-head-to-head.md`](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/v0.17.0/kernel/RESULTS-unsloth-head-to-head.md).
  **`H2H_CONFIRMED` on both devices**, in the **4-bit-storage regime**:

  | device | TMA | decode | prefill | J/token |
  |---|---|---:|---:|---:|
  | H100 80GB HBM3 (sm_90) | live | **1.70×** | 1.67× | 2.51× better (23/24 cells) |
  | RTX 4090 (sm_89) | unavailable | **2.79×** | 2.79× | **3.32× better (24/24)** |

  Three things travel with those numbers, and quoting them without these is
  quoting them wrong:
  - **The margin is card-dependent.** With Unsloth's TMA path live the decode
    margin drops from 2.79× to 1.70× — 40% of it. An H100 was rented
    specifically so their fast path was not compiled out.
  - **Unsloth wins their own regime.** Against their **bf16-resident** kernel —
    weights already bf16, nothing to dequantize — they run **2.6–5.3× faster at
    prefill on the H100**. gnf4's advantage is the 4-bit-storage regime
    specifically and is *not* a general claim. Their kernel is excellent at the
    job it was built for.
  - **It is not a simple decay in M.** Median `unsloth/fused` runs 2.32 → 1.48 →
    1.67 across `decode_bs1` → `decode_m8` → `prefill` (H100), so the minimum is
    at `decode_m8`. The advantage tracks how *bandwidth-bound* a cell is, not how
    small it is.

  Forward pass only. A training-axis leg exists but is **exploratory** and
  licenses no claim — see the results doc.

  Axolotl/PEFT
  QLoRA forwards run bitsandbytes `Linear4bit` — see the flagship bnb
  baseline. GPTQ-Marlin is fidelity-excellent but per-expert
  (launch-storm at MoE decode) and format-incompatible with NF4 checkpoints.
- **Known losers:** `top_k=1` cells are **instance-unstable in both
  directions** (Scout `down` measured 0.47–1.12 across six contexts on
  identical code — split-K helps paired but can't stabilize the class), and
  **tiny shapes (≲5 M weight elements) lose outright** (0.24–0.35× speed,
  4–7× energy). v4 adds a dispatch floor that routes tiny cells back to the
  dequant path.
- **Prefill** (compute-bound M): the v6 register-LUT mainloop rewrite
  (blind-CONFIRMED) runs **1.39–1.54× the prior mainloop on every census
  prefill cell**; against the dequant path the census reads 1.14–2.78× with
  all three large gate_ups above 1.15 — gate_up is no longer a loser class.
  One caveat carried at full volume: the dequant *baseline* itself swings
  ~25% between cloud instances (the fused kernel holds within 0.2 ms), and
  OLMoE gate_up (the smallest-expert shape) remains below parity at ~0.6×.

Six blind confirmatories have run; the first five **did not fully pass as
registered**, each results doc says exactly what failed and why, and the
sixth passed clean:
[v1](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/v0.17.0/kernel/RESULTS-gate2-confirmatory.md) (caught the original per-shape
config table overfitting its census), [v2](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/v0.17.0/kernel/RESULTS-v2-confirmatory.md)
(validated the replacement single-constant config on 64-SM parts and the
off-census `k≥6` wins), [v3](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/v0.17.0/kernel/RESULTS-v3-confirmatory.md) (found the
v2-era SM-conditional premise was measurement noise, quantified the
`top_k=1` and tiny-shape loss classes, and established the methodology rule
that latency-bound cells only support paired claims),
[v4](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/v0.17.0/kernel/RESULTS-v4-confirmatory.md) (dispatch floor + split-K work floor
+ prefill config; caught its own dispatch-point regression),
[v5](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/v0.17.0/kernel/RESULTS-v5-confirmatory.md) (the load-time dispatch fix, clean on
the A5000 11/11 with energy 8/8 on both devices; one contended-A2000 noise
cell kept it from a full pass — the dispatch line is closed),
[v6](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/v0.17.0/kernel/RESULTS-v6-confirmatory.md) (**CONFIRMED**, all five criteria:
the register-LUT M-tile mainloop, adjudicated on the instance-robust paired
rewrite ratio after the dress rehearsal exposed the dequant baseline's
host lottery). The preregs,
amendments, evidence JSONs, sweeps, and mechanical reducers are all
committed; `.ots` files anchor the protocols to Bitcoin. An anchor proves the
registered bytes existed before its block — an upper bound, so for runs that
finish faster than Bitcoin confirms, the pre-data evidence is the public push
receipt instead; `kernel/ATTESTATION-TIMELINE-2026-08-15.md` audits that day's
protocols timestamp by timestamp.

## Flagship: a 235B MoE decoding at the PCIe physical limit on ≤16 GB of VRAM

`bench/phase3/` runs Qwen3-235B-A22B with **all expert weights NF4-packed in
host pinned RAM (~128 GB)** and streamed per-token over PCIe, with this
kernel as the sole MoE compute. Same discipline (prereg + OTS, receipts
in-repo):

- **[Phase A](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/v0.17.0/bench/phase3/flagship/RESULTS-flagship-offload.md)** (synthetic
  weights, real GQA attention + router): **5.57 tok/s = 102–103% of the
  measured 44.3 GB/s link's waterfall ceiling** — the stream fully hides
  compute — on a **13.6 GB** working set. The dequantize-then-matmul path on
  the identical pipeline: 1.81 tok/s (34% of ceiling). ALL PASS. (Fractions
  marginally above 100% are microbench conservatism: the 1 GiB×10 ceiling
  measurement brackets every copy with a host sync, paying launch +
  sync-return latency the pipeline's continuously-queued copy stream never
  pays.)
- **[The gap is architectural](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/v0.17.0/bench/phase3/flagship/RESULTS-flagship-bnb-baseline.md)** —
  we registered the prediction that bnb's own CUDA dequant kernel would
  also hide under the copy shadow (which would have narrowed our claim),
  and it was **refuted**: the standard path reaches **40% of waterfall**
  (per-expert dequant+GEMM compute outlasts the shadow), versus 93–94%
  fused on the same pod. Against the strongest standard comparator the
  fused path is **2.33× tokens/s and 2.21× J/token**.
- **[Phase B](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/v0.17.0/bench/phase3/flagship/RESULTS-flagship-phaseB.md)** (the real
  438 GB checkpoint, stream-quantized to NF4 in place): **coherent greedy
  text at 4.3–4.4 tok/s on 15.2 GB VRAM**, replicated across five pods —
  all at 45–55 GB/s datacenter links. The per-token rate is **link- and
  host-dependent**: `t_token ≈ c_box + bytes/link`, with the per-box floor
  `c_box` measured at 53.5–114.0 ms across seven hosts (gen4 desktop
  L40S: 2.6 tok/s; gen5 bare-metal H100 PCIe: 3.9 tok/s — same kernel,
  greedy-identical; see
  [the gen5 doc](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/main/bench/phase3/flagship/RESULTS-flagship-gen5-metal.md)).
  A fixed "fraction of waterfall" is NOT the law — the two 0.77 readings
  that once suggested one were a two-host coincidence, retired 2026-07-22.
- **Expert prefetch is measured CLOSED, negative** — four registered arcs
  ([B2](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/v0.17.0/bench/phase3/flagship/RESULTS-flagship-phaseB2.md) speculation:
  token-to-token expert stickiness is only 0.44;
  [B3](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/v0.17.0/bench/phase3/flagship/RESULTS-flagship-phaseB3.md) early routing: the
  pre-attention router predicts the post-attention top-8 at **0.93** but the
  CPU sync tax is the **leading hypothesis** for why the win does not land —
  the receipt labels it a suspect, not a measured cause, and the successor
  experiment sized that whole sync class at ~1 %;
  [B4](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/v0.17.0/bench/phase3/flagship/RESULTS-flagship-phaseB4.md) threaded issuance:
  GIL tax, 0.57×;
  [B5](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/v0.17.0/bench/phase3/flagship/RESULTS-flagship-phaseB5.md) GPU-driven
  zero-copy gather: hit rate H makes speculation move (2−H)× the bytes, and
  the observed loss matches that law to ~1% — break-even needs H ≳ 0.95,
  above this model's 0.93 predictor ceiling).
- **Recommended configuration: `--prefetch-mode gpu`** — expert ids stay
  GPU-resident and a triton kernel ([`kernel/host_gather.py`](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/v0.17.0/kernel/host_gather.py))
  gathers expert rows straight from pinned host RAM over UVA (zero-copy),
  with no per-layer memcpy launches and no GPU→CPU syncs. It is the fastest
  measured arm (4.39–4.41 tok/s, +1.5% over serialized memcpy, byte-identical
  greedy output 6/6) and validates SM-issued UVA reads at ≥ copy-engine
  throughput at 7.98 GB/token.

  **Scope on that recommendation:** those figures are **one host** — a SECURE
  H100 80GB HBM3 whose on-box link measured 45.0 GB/s, the slowest-link box in
  the set. +1.5 % is a margin thin enough that a different link could reorder the
  arms, and this is a *default* being recommended on a single-host result. Prefer
  it, but measure on your own box before treating it as settled.

Every comparative "first/only/faster" claim above is backed by a verified, dated
comparison against the named alternative's own published numbers or a same-box
A/B (see [`docs/RESULTS-ikllama-ab.md`](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/main/docs/RESULTS-ikllama-ab.md)
for the ik_llama run) — no receipt, no claim.

## Reproduce

See [REPRO.md](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/v0.17.0/REPRO.md) — suite, benchmark, and verdict reduction are each
one command from a frozen tree. Requires an sm_86 GPU, `torch ≥ 2.8`,
`bitsandbytes`, and a C compiler on PATH (triton builds launcher stubs at
runtime).

```
python -m pytest kernel/test_nf4_grouped.py -q        # 44 tests, ~2.5 min
python bench/phase1/harness.py --models OLMoE --regimes decode_bs1 \
    --backends dequant_grouped fused_nf4 --out receipts.json
```

## Layout

- `kernel/nf4_grouped.py` — the kernel (decode gemv path + M-tile path),
  packing helpers, torch reference decode
- `kernel/test_nf4_grouped.py` — property suite (bnb decode exactness at
  bf16 output precision, fidelity ordering, adversarial absmax, boundaries)
- `kernel/prereg_*.json` + `.ots` — pre-registered protocols, stamped
- `kernel/RESULTS-*.md` — results, including the failures
- `bench/phase1/` — backend-registry harness (dequant/gemv/grouped-mm/
  unsloth/marlin/fused), confirmatory evidence, reducers
- `bench/phase2/` — decode config sweeps (both devices); `arch/` —
  cross-architecture census (sm_86/89/90)
- `bench/phase3/` — the 235B offload flagship: `offload_decode_235b.py`
  (Phase A, synthetic), `offload_generate_235b.py` (real checkpoint,
  generation, prefetch arms), `flagship/` — results + receipts
- `kernel/host_gather.py` — GPU-driven zero-copy gather from pinned host
  memory (UVA), the recommended offload copy path
- `docs/KERNEL_CONTRACT.md`, `docs/TOLERANCE_CONTRACT.md` — op contract and
  fidelity spec; `census/`, `roofline/` — shape census + ceilings

Regenerate the machine-generated artifacts:

```
python3 census/make_census.py     # census/shape_census.json
python3 roofline/roofline.py      # roofline/ceilings.json
```

## Status / roadmap

Landed through v6: universal decode constant (the dense-sweep result),
split-K for starved grids (with a per-split work floor), a load-time
min-bytes dispatch floor (tiny cells route to the dequant path via
`decode_dispatch()`), the register-LUT prefill mainloop (v6, confirmed),
and the flagship offload pipeline (Phase A/B + the closed prefetch program
+ the UVA gather path + the bnb-CUDA-dequant baseline, whose registered
prediction was refuted — see the flagship section). The v6 A2000
report-only addendum landed 2026-07-20
([`kernel/RESULTS-v6-a2000-report.md`](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/v0.17.0/kernel/RESULTS-v6-a2000-report.md)):
paired prefill medians inside the confirmed band on 7/8 cells at 26 SM
(the eighth 0.9% below the floor, quantified in-doc) — the mainloop is
bracketed 26→170 SM with zero retune. Pending: a bare-metal gen4
replication when stock returns. Parked: sm_120 (three consecutive cloud
provisioning failures on 5090s — availability, not code). Ecosystem landing
is calendar-gated on the bitsandbytes v0.50.0 release; see the coordination
note on #1949.

**Downstream serving** lives in the sibling package
[`experts4bit-qlora`](https://pypi.org/project/experts4bit-qlora/): its
`[fast]` extra routes frozen-expert inference through this kernel (the measured
3.65× above), and its hot-expert residency runs hot and
cold stacks on the same kernel — with 2026-07-20 receipts showing decode
gain tracks routing coverage (informed hot sets +56–120% on gpt-oss, +44%
on Gemma-4). Division of labor: this kernel makes one expert-stack matmul
cheap; e4b decides which bytes are where.

**Cold-engine exploration** (CPU-resident third tier for the coldest
experts) is at phase-0: premise measurements on the target NAS host are in
[`docs/cold-engine/PHASE0-premise.md`](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/v0.17.0/docs/cold-engine/PHASE0-premise.md).
Honest status: the "free floor" premise (bnb's CPU `dequantize_4bit` as a
ready-made decode arm) is **refuted** on that box — no AVX-512 means bnb
falls back to its reference path at 0.041 GB/s against a ~12 GB/s DDR
ceiling — so an AVX2 decode port is the mandated phase-2 step before any
integration work. Design-stage; no registered claims.

## Cross-vendor projections (stamped, PROJECTED tier — help us confirm them)

The waterfall arithmetic doesn't care which vendor's bus you're on, so we've
extended it — under the same receipts discipline — into a stamped, pre-silicon
projection table for AMD, Intel, and NVIDIA unified-memory parts:
[`PROJECTIONS-multiarch.md`](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/v0.17.0/PROJECTIONS-multiarch.md) (protocol:
[`PROTOCOL-multiarch.md`](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/v0.17.0/PROTOCOL-multiarch.md); model + R1 anchor gate:
[`projections/`](https://github.com/pjordanandrsn/grouped-nf4-gemm/tree/v0.17.0/projections/)). Both docs are OpenTimestamps-anchored (`.ots`)
**before any of this silicon was run** — the projections are a falsifiable
prediction, not a marketing table.

**Streaming rows are now graded against measurements** (Addendum 2): the
original pure-waterfall gen5 row (6.0–6.9) is **falsified**, as Addendum 1
pre-registered it would be; Addendum 1's revised **gen4 band (2.4–3.0) is
CONFIRMED** (desktop L40S measured 2.60–2.61) and its revised **gen5 band
(4.0–5.0) missed narrowly** (bare-metal H100 PCIe measured **3.924** — below
the band, above the 3.6 falsification line; that box's floor `c_box = 114 ms`
lies outside the five-pod fitted range). The standing model is **additive,
per-box**: `t_token ≈ c_box + bytes/link` with `c_box` measured (53.5–114.0 ms
across seven hosts, not ordered by link speed) — a fixed fraction-of-waterfall
is retired as a law (`PROJECTIONS-multiarch.md` Addendum 2,
`bench/phase3/flagship/RESULTS-flagship-gen5-metal.md`). Unified-memory rows
remain *ceilings only*: **17–22 tok/s ceiling** on 128 GB unified boxes
(Strix Halo / DGX Spark / Jetson Thor), real decode below them by that same
per-box floor. NF4-vs-bf16 is a **3.56×** byte reduction (absmax-inclusive),
not the round 4×.

**Call for confirmatories.** If you own any listed part, run
`PROTOCOL-multiarch.md` and file the result — **pass or fail** — as an issue.
A refuting measurement is as welcome as a confirming one; that's the point.
Template:

```
Title: [confirmatory] <platform> — <model>
Environment: vendor / device / driver / runtime / triton / torch / bnb;
  link measured via lspci + on-box microbench (streaming) OR mem-band spec
  (unified)
Correctness gate: max rel-err vs dequant_ref = <value>  (pass < 1e-2)
Measured decode: <tok/s> per census cell   Projected band: <from table>
Verdict: within band? / refutes row?   Attach: results JSONL
```

## License & attribution

MIT ([LICENSE](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/v0.17.0/LICENSE)). Portions developed with Claude Code as an AI
assistant under the author's direction and review — see
[ATTRIBUTION.md](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/v0.17.0/ATTRIBUTION.md). All claims are the author's responsibility.

## Portability program

The kernel is single-source Triton; everything that must differ per vendor
is being pulled into `backends/` — device detection, warp/wavefront/sub-group
width, per-arch autotune search spaces. `bench/hw_contract.py` validates
kernel correctness on any torch device **without a bitsandbytes build**; if
you have ROCm or XPU silicon, that is the entry point. `docs/PORTABILITY.md`
is the pre-port hazard register. Per the repo's tier language, every
non-CUDA row is `port target` until a confirmatory passes on that silicon.

## Router-predictability probe

`router_probe/` asks whether the measured H = 0.93 one-layer-lead prediction
ceiling is the router's conditional entropy or the probe's capacity limit.
The charter and procedure were OTS-stamped before any real-model capture; the
Phase-0 instrument gate passed 4/4 on planted fixtures. Phase 1 has run on
**five MoE families** (see `router_probe/RESULTS.md`, exploratory tier):
low-expert-count families pin cleanly at first data volume (gpt-oss-20b E=32
→ 0.83, granite E=40 → 0.90, OLMoE E=64 → 0.91, all model-limited ×3 from the
committed reducer), while **both E=128 families are data-unpinnable** (Qwen3-30B
k=8 ≥0.845 after **two** data doublings — 147,456 → 294,912 → 589,824 records;
gpt-oss-120b k=4 ≥0.787 after **one**) —
high expert count doesn't just lower H, it makes H unmeasurable by data
scaling on this ladder, at both k. Every observed plateau sits far below the
≈0.95 wire-law break-even for speculative expert streaming.

## Contact

Cerin Amroth Research takes contract and pilot engagements on this work —
kernel ports, offload integration, and sponsored research lanes with
stamped receipts. Contact **jordan@cerinamroth.com**.
