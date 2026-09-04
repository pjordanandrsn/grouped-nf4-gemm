# How do I run a grouped GEMM directly on NF4 packed MoE expert weights, without dequantizing to bf16 first?
<!-- summary: gemm_4bit_grouped runs the routed MoE expert GEMM on bitsandbytes NF4 packed weights in one Triton launch, decoding nibbles in registers with fp32 accumulation and no bf16 expert tensor. -->

Use `nf4_grouped.gemm_4bit_grouped` from `grouped-nf4-gemm`: one Triton launch computes the routed expert GEMM on the bitsandbytes `gemm_4bit` NF4 layout, decoding nibbles to fp32 in registers and accumulating in fp32, so no bf16 copy of any expert is written to memory. The pure-torch `nf4_grouped.dequant_ref` is the oracle the kernel is asserted against, and it runs on CPU.

## Measured boundaries

Every figure here is quoted from [`claims.json`](../claims.json) under the named ID with its tier, and each row carries the cells that lose beside the cells that win.

| what was measured | result | claim ID | tier |
|---|---|---|---|
| Batch-1 decode on sm_86, census MoE shapes (OLMoE, Qwen3-30B, Gemma-4, gpt-oss-120b) vs the registered dequant comparator | 1.16–2.73× at median; **loses:** `top_k=1` cells (instance-unstable) and tiny shapes under about 5 M weight elements | `gnf4.kernel.decode-speed-census` | confirmed |
| A real OLMoE QLoRA finetune on prose, fused vs the per-expert loop | 4.50× (RTX 4090), 4.75× (H100) | `gnf4.kernel.e2e-training-real-prose` | confirmed |
| Head-to-head against Unsloth's own kernel, 4-bit-storage regime, decode | 1.70× (H100), 2.79× (RTX 4090); **loses:** Unsloth is 2.6–5.3× faster at prefill in its own bf16-resident H100 regime | `gnf4.kernel.h2h-unsloth` | confirmed |
| Fidelity vs the dequantize-to-bf16-then-GEMM comparator | has not measured less accurate in any registered confirmatory cell (fp32 accumulation, bf16 epilogue); a CUDA tensor-core statement, not a universal proof | `gnf4.kernel.fused-more-accurate-than-dequant-bf16` | confirmed |

The other registered loser, a CUDA-graphed per-expert baseline at the decode band, is carried by ID under Limitations (`gnf4.kernel.graphed-baseline-decode-loses`). The comparator in the first and last rows is this repository's own per-expert dequantize-then-GEMM loop as each receipt ran it; the dated note under "Why it happens" says what that does and does not say about current bitsandbytes.

## Symptoms

- A 4-bit MoE (bitsandbytes NF4, `load_in_4bit`) runs its experts as a per-expert loop, one launch per active expert per projection, and wherever that loop dequantizes (bitsandbytes releases before 0.50.0, cells outside its packed `gemm_4bit` inference forward, the conventional 4-bit backward) every active expert is decoded to a bf16 `[N, K]` tensor, multiplied, and discarded.
- Transient memory is dominated by bf16 expert materialization, not by the packed weights.
- The expert loop is a per-expert `dequantize_4bit` + `bmm` pair, and top-k routing over many experts makes launch count the bottleneck.
- You want a W4A16 grouped GEMM that takes NF4 packed weights as stored.

## Why it happens

bitsandbytes stores NF4 as a 16-entry codebook index per weight plus a blockwise fp32 absmax. What its forward does with those bytes depends on the release, the workload and the shape:

- **Ordinary 2-D inference, bitsandbytes 0.50.0 and later.** Recent bitsandbytes CUDA inference paths can compute directly from packed 4-bit weights for supported ordinary 2-D matrices: upstream commit `5453368bed15d19cbcfba4426ed118de33dc3d94` ("[CUDA] New 4bit GEMM kernels for inference (#1949)", committed 2026-05-21T14:05:48Z; 75 commits after tag 0.49.2 and 44 before tag 0.50.0) added `torch.ops.bitsandbytes.gemm_4bit`, so 0.50.0 is the first stable release containing the direct packed 4-bit CUDA inference forward. Full bf16 materialization is no longer inherent to every `Linear4bit` forward.
- **Older releases and unsupported cells.** Releases before 0.50.0, and cells the packed kernel does not cover (unsupported shapes, devices or configurations), may still dequantize: the bf16 weight is materialized and the GEMM runs on that copy. That round trip is the dequant tax this page's comparators measured.
- **Fused MoE execution is a different contract.** Routed tokens address many expert matrices with different group sizes: a few rows for one expert, hundreds for another, most experts idle. Upstream has no grouped routed-MoE contract, so a naive implementation is a sequence of per-expert operations, one launch per active expert per projection per layer, whether or not each of those operations dequantizes.
- **Training also needs dX.** The conventional 4-bit backward dequantizes the weight to compute the input gradient, so a QLoRA step over fused experts pays the bf16 round trip in the backward even where the forward does not.

`grouped-nf4-gemm` keeps the expert-major NF4 stack packed, groups the routed work, decodes nibbles inside the kernel, accumulates in fp32, and writes no bf16 expert tensor; `dgrad_4bit_grouped` does the same for dX. [`KERNEL_CONTRACT.md`](../KERNEL_CONTRACT.md) is the op contract.

| Path | Workload | Packed forward | Grouped routing | Training dgrad |
|---|---|---|---|---|
| bitsandbytes ≥ 0.50 supported CUDA path (`torch.ops.bitsandbytes.gemm_4bit`) | ordinary 2-D inference matrix, on supported shapes/devices/configs, inference (no-grad) | yes | no | backward dequantizes |
| naive fused-MoE loop | per-expert operations, one per active expert | path-dependent (release and cell) | no | path-dependent |
| grouped-nf4-gemm (`gemm_4bit_grouped`) | expert-major stack + routed groups | yes | yes | one-launch path available (`dgrad_4bit_grouped`) |

Dataflow, per layer per projection:

```text
per-expert materializing loop                grouped packed path (this package)
-----------------------------                ----------------------------------
for each active expert e:                    packed B [E, N, K/2] u8 + absmax [E, N, K/64] f32
    W_e = dequant(B[e]) -> bf16 [N, K]       routed groups: a_cat [T, K] sorted by expert,
    out_e = a_e @ W_e.T                      sizes + expert_ids
    (write W_e, read W_e, discard)                   |
                                                     v
                                             one grouped kernel dispatch:
                                               nibble -> LUT decode in registers,
                                               absmax scale, fp32 accumulate,
                                               bf16 epilogue
                                                     |
                                                     v
                                             output [T, N] in group order
```

Sorting tokens by expert and scattering the output back remain outside the kernel, as for every grouped GEMM.

*Dated note (2026-09-04).* Every comparator called "the dequant path" or "dequantize-to-bf16-then-GEMM" in this repository's receipts is the per-expert loop as each receipt ran it: bitsandbytes `dequantize_4bit` per active expert followed by a bf16 matmul (`bench/phase1/harness.py`, `bk_dequant_grouped`; [`examples/dequant_tax.py`](../../examples/dequant_tax.py) labels which decode arm it ran). Those numbers stay what they measured; they are not a statement that current bitsandbytes dequantizes on every `Linear4bit` forward, and no registered cell times bitsandbytes' own packed 2-D inference forward.

## Which project solves it

`grouped-nf4-gemm` owns the kernel, its CUDA-graph-capturable variant, the one-launch backward, the packers and reference decode, and the repack from bitsandbytes state. It does not load models or route tokens. [experts4bit-qlora](https://github.com/pjordanandrsn/experts4bit-qlora) ([PyPI](https://pypi.org/project/experts4bit-qlora/)) owns model loading, quantization orchestration, adapters, QLoRA training, residency integration and serving, and drives these kernels through `enable_fast()` / `enable_fast_train()`. A model-level symptom ("bitsandbytes MoE still OOMs after `load_in_4bit`") starts there.

## Install

Kernel package (the minimum route):

```bash
pip install grouped-nf4-gemm
```

Linux, NVIDIA GPU of compute capability sm_80 or newer (sm_120 is the primary serving target), `triton>=3.4` (Linux-only distribution), `torch>=2.8` (pre-releases accepted); CI tests Python 3.11. `nf4gemm`, `gnf4` and `grouped-mxfp4-gemm` are lookup aliases, not separate packages. Through the model consumer:

```bash
pip install "experts4bit-qlora[fast]"
```

## Smallest correct example

CPU-only: pack one expert and decode it with the reference.

```python
# CPU-only (pure torch; no GPU, no triton launch)
import torch
from nf4_pack_ref import quantize_pack_nf4
from nf4_grouped import dequant_ref

N, K = 256, 512                                  # K % 64 == 0
w = torch.randn(N, K)
packed, absmax = quantize_pack_nf4(w)            # [N, K//2] uint8, [N, K//64] fp32
wq = dequant_ref(packed, absmax, N, K)           # [N, K] fp32
assert wq.shape == (N, K)
assert torch.equal(quantize_pack_nf4(wq)[0], packed)   # re-pack is idempotent
```

GPU: the grouped kernel, asserted against the reference decode group by group.

```python
# GPU (sm_80+) + triton
import torch
from nf4_pack_ref import make_stack
from nf4_grouped import gemm_4bit_grouped, dequant_ref

E, N, K = 8, 256, 512
B, absmax = make_stack(E, N, K, device="cuda")   # [E, N, K//2] u8, [E, N, K//64] f32
sizes, expert_ids = [3, 1, 4], [5, 0, 2]         # group-sorted tokens per active expert
a_cat = torch.randn(sum(sizes), K, device="cuda", dtype=torch.bfloat16)

out = gemm_4bit_grouped(a_cat, B, absmax, sizes, expert_ids)     # [T, N] bf16

row, num, den = 0, 0.0, 0.0
for m, e in zip(sizes, expert_ids):
    ref = a_cat[row:row + m].double() @ dequant_ref(B[e], absmax[e], N, K).double().t()
    num += (out[row:row + m].double() - ref).norm().item() ** 2
    den += ref.norm().item() ** 2
    row += m
assert (num ** 0.5) / (den ** 0.5) <= 1e-2       # B-abs bound, docs/TOLERANCE_CONTRACT.md
```

Calling `gemm_4bit_grouped` on CPU tensors raises and names `dequant_ref`; nothing falls back silently.

## Expected result

Both blocks finish without an assertion. The GPU block returns `[T, N]` bf16 in the group order of `a_cat`, with relative Frobenius error against the fp64 product of the reference decode inside the registered bound. Output is not bit-identical to dequantize-then-`matmul`: the per-element decode is identical, the fp32 reduction order is not ([`TOLERANCE_CONTRACT.md`](../TOLERANCE_CONTRACT.md)).

## Supported scope

- Layout: bitsandbytes `gemm_4bit` NF4, canonical `[N, K]` per expert, blocksize 64, `K % 64 == 0`; expert-major `B [E, N, K//2]` uint8, `absmax [E, N, K//64]` fp32. `nf4_grouped.repack_from_bnb(packed_list, states, N, K)` builds these from per-expert `quantize_4bit` output and de-nests `compress_statistics`.
- `a_cat [T, K]` bf16/fp16 group-sorted; `sizes` all `> 0`; `expert_ids` a list or int32 tensor. Sort and scatter live outside the kernel. One-token groups take the GEMV path, larger groups the M-tile path.
- Capture: `nf4_grouped.build_group_tiles_device(expert_ids, n_experts, block_m)` builds static device tiles; `nf4_grouped.gemm_4bit_grouped_captured(a_sorted, B, absmax, t_row0, t_rows, t_group, block_m)` runs against them.
- Training: `nf4_grouped.dgrad_4bit_grouped(grad_out, B, absmax, sizes, expert_ids)` is the input gradient in one launch; `nf4_qlora.gemm_4bit_grouped_train` and `nf4_qlora.fused_grouped_lora` wrap it in autograd, the latter adding the LoRA delta pre-activation.
- [`examples/dequant_tax.py`](../../examples/dequant_tax.py) times the round trip against the fused path on your GPU, with a self-pair beside every ratio.

## Limitations

- Against a CUDA-graphed per-expert baseline the fused path loses at the decode band; the memory-traffic win survives at training shape on bandwidth-limited cards (claim `gnf4.kernel.graphed-baseline-decode-loses`). Read [`STATUS.md`](../STATUS.md) before quoting a decode win.
- Every speed figure is per card: the census is sm_86, the head-to-head and the real finetune are RTX 4090 / H100, the `torch._grouped_mm` cell is RTX 5090 (claim `gnf4.kernel.sm120-census-vs-grouped-mm`). None is an architecture-wide statement, and no cell here is a measurement of bitsandbytes' own packed 2-D inference forward.
- Shapes under a weight-byte floor lose outright; `nf4_grouped.decode_dispatch(N, K, T, sm_count)` returns `("dequant",)` for them and the caller routes those cells back. The op itself never switches algorithm.
- `top_k=1` cells are instance-unstable; peak VRAM does not improve, only the forward-to-backward transient shrinks ([`STATUS.md`](../STATUS.md)).
- The fidelity ordering is a CUDA tensor-core statement; other backends must re-measure. No ROCm or XPU ([`PORTABILITY.md`](../PORTABILITY.md)).
- CUDA + Triton only for the kernel. `nf4_pack_ref` imports `nf4_grouped`, which binds triton through `_triton_shim`, so the pure-torch surface (pack references, `dequant_ref`, provenance, arena bake/verify) imports and runs without triton, and a `gemm_4bit_grouped` call on CPU tensors raises naming `dequant_ref` on a triton-less box too; the Triton kernels need a CUDA GPU; macOS and Windows are not exercised by CI.
- Open: `#87`, int32 offset overflow at large `max(expert_ids)`.

## Related

[`KERNEL_CONTRACT.md`](../KERNEL_CONTRACT.md) · [`TOLERANCE_CONTRACT.md`](../TOLERANCE_CONTRACT.md) · [`STATUS.md`](../STATUS.md) · [`claims.json`](../claims.json) · [`REPRO.md`](../../REPRO.md) · [`native-mxfp4-moe-inference.md`](native-mxfp4-moe-inference.md) · [`int4-decode-gemv.md`](int4-decode-gemv.md) · [`stream-moe-experts-from-host-or-nvme.md`](stream-moe-experts-from-host-or-nvme.md)

## Evidence

Register: [`claims.json`](../claims.json). Confirmed: claim `gnf4.kernel.fused-more-accurate-than-dequant-bf16` (has not measured less accurate than the dequantize-to-bf16-then-GEMM comparator in any registered confirmatory cell), claim `gnf4.kernel.decode-speed-census`, claim `gnf4.kernel.energy-104-of-112`, claim `gnf4.kernel.e2e-training-real-prose`, claim `gnf4.kernel.h2h-unsloth` (4-bit-storage regime; Unsloth wins its own bf16-resident regime), claim `gnf4.kernel.graphed-baseline-decode-loses`. Measured: claim `gnf4.kernel.sm120-census-vs-grouped-mm`, claim `gnf4.kernel.dgrad`. Receipts under `kernel/RESULTS-*.md` with their `prereg_*.json`; property suite `kernel/test_nf4_grouped.py`.
