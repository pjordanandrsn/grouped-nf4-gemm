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

[Install](#install) · [Results](#what-is-measured) · [Kernel API](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/main/docs/KERNEL_CONTRACT.md) · [Task guides](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/main/docs/SOLUTIONS.md)

## Install

```bash
pip install grouped-nf4-gemm
```

GPU kernels need **Linux, an NVIDIA sm_80+ GPU, torch ≥ 2.8 and Triton ≥ 3.4**.
CI tests Python 3.11. CPU pack/decode and provenance tools work without CUDA;
macOS and Windows are not exercised by CI. ROCm and XPU are port targets.

**New in 0.42.0:** bucketed padding of the LoRA update is on by default (`auto`) for calls with at least 16,384
routed rows, such as packed 4,096-token training rows; smaller calls keep the single padded block.
Its training-speed evidence is on torch 2.12; its effect on torch 2.8 is still unmeasured.
`NF4_QLORA_PAD_BUCKETS=0` restores the single block.
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
