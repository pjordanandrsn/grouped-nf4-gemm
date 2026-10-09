# grouped-nf4-gemm

### Run MoE expert math directly on 4-bit weights.

[![PyPI](https://img.shields.io/pypi/v/grouped-nf4-gemm)](https://pypi.org/project/grouped-nf4-gemm/)
[![CI](https://github.com/pjordanandrsn/grouped-nf4-gemm/actions/workflows/ci.yml/badge.svg)](https://github.com/pjordanandrsn/grouped-nf4-gemm/actions/workflows/ci.yml)

**One GPU launch for the active experts. No full-size bf16 weight copies.**

Triton kernels for grouped **NF4 and native MXFP4** matrix multiplication, training gradients,
INT4 decode, FP8 attention, and streaming expert weights from RAM or SSD.
The packers, reference implementations, and byte-level checks ship alongside the kernels.

**Want to train a model? [Start with Loggetta](https://github.com/pjordanandrsn/loggetta).**
It installs this package through the [experts4bit-qlora runtime](https://github.com/pjordanandrsn/experts4bit-qlora).
Use grouped-nf4-gemm directly to build or tune a kernel integration.

> **Where it loses.** A per-expert baseline can be faster at small shapes and in some graphed decode workloads. With
> weights already resident in bf16, Unsloth's H100 prefill kernel won by 2.6–5.3×. A faster kernel can leave training
> time unchanged. [Details](#where-it-loses).

[Install](#install) · [Results](#what-is-measured) · [Kernel API](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/main/docs/KERNEL_CONTRACT.md) · [Task guides](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/main/docs/SOLUTIONS.md) · [Hugging Face](https://huggingface.co/spaces/pjordanandrsn/research)

![grouped-nf4-gemm against Unsloth's MoE kernel: faster decode with weights stored in 4-bit on RTX 4090 and H100; Unsloth faster at H100 prefill with weights resident in bf16. OLMoE QLoRA on real prose against this project's per-expert loop.](https://raw.githubusercontent.com/pjordanandrsn/grouped-nf4-gemm/main/docs/assets/speed-vs-unsloth-rtx4090-h100.svg)

Measured on one RTX 4090 and one H100. The receipts record the GPU, torch and (for the Unsloth comparison) the NVIDIA
driver; clock locking, ECC state and the Triton version were not recorded. [How these were measured](#what-is-measured).

## Install

```bash
pip install grouped-nf4-gemm
```

GPU kernels need **Linux, an NVIDIA sm_80+ GPU, torch ≥ 2.8 and Triton ≥ 3.4**.
CI tests Python 3.11. CPU pack/decode and provenance tools work without CUDA;
macOS and Windows are not exercised by CI. ROCm and XPU are port targets.

**New in 0.45.0:** four opt-in kernel options for a top-k MoE decode, each bitwise the launches it replaces: int64 expert
ids and in-place token rows for `gemm_4bit_grouped`, bf16 routing weights from `router_epilogue`, one-launch q/k norm
and rotary (`rope_norm_qk`), and the residual add in `combine_rows`. The cumsum tile table can split over several
programs (`programs=P`). No default changes.
[Release notes](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/main/CHANGELOG.md)

## Try it on your GPU

From a checkout of this repository; about one minute, no model download:

```bash
pip install grouped-nf4-gemm bitsandbytes
python examples/dequant_tax.py
```

The [example](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/main/examples/dequant_tax.py)
compares packed compute with dequantize-then-GEMM at three workload sizes and prints a repeated-baseline control.
On CPU it explains which GPU measurement is unavailable.

## Which entry point?

| Task | API or guide |
| :--- | :--- |
| Grouped NF4 forward and backward | `nf4_grouped.gemm_4bit_grouped`, `dgrad_4bit_grouped` |
| NF4 single-row decode, bandwidth route (default at Qwen3-30B-A3B's shapes on >= 160-SM parts) | `GNF4_GEMV_BW` (`auto`; `1` everywhere, `0` off) with `gemm_4bit_grouped`; [NF4 guide](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/main/docs/solutions/nf4-grouped-gemm-without-bf16-materialization.md) |
| Native MXFP4 expert math | `mxfp4_grouped.gemm_mxfp4_grouped` |
| INT4 decode | [INT4 guide](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/main/docs/solutions/int4-decode-gemv.md) |
| FP8 paged attention | [Attention guide](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/main/docs/solutions/fp8-paged-attention-for-moe-serving.md) |
| Stream experts from RAM or NVMe | [Streaming guide](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/main/docs/solutions/stream-moe-experts-from-host-or-nvme.md) |
| Check checkpoint bytes | [Provenance guide](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/main/docs/solutions/verify-quantized-checkpoint-provenance.md) |

[Layouts and signatures](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/main/docs/KERNEL_CONTRACT.md) ·
[All supported entry points](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/main/docs/capabilities.json)

<details>
<summary><strong>Try the packers and byte checks on CPU</strong></summary>

These examples are executed by CI on Linux. They use pure PyTorch; the GPU kernels remain CUDA-only.

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

</details>

## What is measured

Each result below is **confirmed** by a pre-registered run with public evidence.
CI checks the values against [claims.json](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/main/docs/claims.json).

| Experiment | Result | Tier | Claim ID |
| :--- | :--- | :--- | :--- |
| OLMoE QLoRA on real prose, fused vs this project's per-expert loop | **4.50×** on RTX 4090; **4.75×** on H100 | confirmed | `gnf4.kernel.e2e-training-real-prose` |
| Decode vs Unsloth's kernel, with weights stored in 4-bit | **1.70×** on H100; **2.79×** on RTX 4090 | confirmed | `gnf4.kernel.h2h-unsloth` |
| gpt-oss-120b native-MXFP4 QLoRA experiment, L40S | **9.82 GB** peak VRAM; **144/144** frozen expert tensors hash-identical after training | confirmed | `gnf4.mxfp4.train-9.82gb` |

[OLMoE training](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/main/bench/phase1/results/dequant_forward/RESULTS-e2e-training.md) ·
[Unsloth kernel comparison](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/main/kernel/RESULTS-unsloth-head-to-head.md) ·
[120B experiment](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/main/docs/mxfp4/RESULTS-mxfp4-train.md)

<details>
<summary>How these were measured</summary>

- **Unsloth comparison:** same pod and process, three repetitions per GPU. The receipts record the GPU and compute capability, torch 2.8.0+cu128, the
  NVIDIA driver (570.195.03 on the RTX 4090, 580.159.03 on the H100), bitsandbytes 0.50.0, Unsloth 2026.8.15, and whether
  Unsloth's TMA path was available (no on the RTX 4090, yes on the H100). [Receipts](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/main/bench/phase1/results/unsloth_h2h/)
- **OLMoE training:** the receipts record the GPU, compute capability, VRAM, torch 2.8.0+cu128 and experts4bit-qlora
  0.17.5 from published wheels; the grouped-nf4-gemm version was not recorded. [Receipts](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/main/bench/phase1/results/dequant_forward/leg-e2e/)
- **Not recorded in either:** clock locking, ECC state, the Triton version.
- **Analytic ceiling, not a measurement:** at the design stage, batch-1 decode was estimated memory-bound with a ceiling of
  about 8.1× over the two-pass dequant path ([phase 1 notes](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/main/bench/phase1/README.md)).
- **The chart** is generated from `docs/claims.json` by `scripts/build_readme_chart.py`; CI fails when it is stale.

</details>

The first comparator is this project's per-expert dequantize-to-bf16 loop, without CUDA graphs.
The Unsloth row is a kernel comparison, not end-to-end training. The 120B row is an experiment
using host memory; it does not mean the whole model fits in 9.82 GB or that Loggetta runs this path.

## Where it loses

- **Small shapes and some graphed decode workloads:** a per-expert baseline can be faster.
- **Weights already resident in bf16:** Unsloth's H100 prefill kernel won that comparison, by 2.6–5.3×.
- **Kernel speed vs whole-model speed:** a faster kernel can leave training time unchanged.
  The decoded training route stays opt-in: its end-to-end training-step read found no measurable saving.

Use the [current status](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/main/docs/STATUS.md)
to choose a route, and benchmark on real text rather than random token IDs.

## Documentation and machine-readable data

[Task guides](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/main/docs/SOLUTIONS.md) ·
[Capabilities](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/main/docs/capabilities.json) ·
[Claims and evidence](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/main/docs/claims.json) ·
[Package compatibility](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/main/docs/system-manifest.json) ·
[Reproduce the benchmarks](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/main/REPRO.md) ·
[llms.txt](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/main/llms.txt) ·
[Contributing](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/main/AGENTS.md)

## License & attribution

MIT. Developed with AI assistance under the author's direction and review;
[attribution](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/main/ATTRIBUTION.md).
For ports, integration, or research work: [jordananderson.work](https://jordananderson.work).
