# How do I run a grouped GEMM directly on NF4 packed MoE expert weights, without dequantizing to bf16 first?

Use `nf4_grouped.gemm_4bit_grouped` from `grouped-nf4-gemm`: one Triton launch computes the routed expert GEMM on the bitsandbytes `gemm_4bit` NF4 layout, decoding nibbles to fp32 in registers and accumulating in fp32, so no bf16 copy of any expert is written to memory. The pure-torch `nf4_grouped.dequant_ref` is the oracle the kernel is asserted against, and it runs on CPU.

## Symptoms

- A 4-bit MoE (bitsandbytes NF4, `load_in_4bit`) spends its expert step in dequantize-then-GEMM: every active expert is decoded to a bf16 `[N, K]` tensor, multiplied, and discarded.
- Transient memory is dominated by bf16 expert materialization, not by the packed weights.
- The expert loop is a per-expert `dequantize_4bit` + `bmm` pair, and top-k routing over many experts makes launch count the bottleneck.
- You want a W4A16 grouped GEMM that takes NF4 packed weights as stored.

## Why it happens

bitsandbytes stores NF4 as a 16-entry codebook index per weight plus a blockwise fp32 absmax, and its matmul path is storage-only: it materializes the bf16 weight and hands it to cuBLAS. For a fused-expert MoE that is one round trip per active expert per layer; writing and re-reading those bf16 weights is the dequant tax. [`KERNEL_CONTRACT.md`](../KERNEL_CONTRACT.md) names the fix: fuse the decode into the GEMM mainloop so the packed bytes are the only weight bytes read.

## Which project solves it

`grouped-nf4-gemm` owns the kernel, its CUDA-graph-capturable variant, the one-launch backward, the packers and reference decode, and the repack from bitsandbytes state. It does not load models or route tokens. [experts4bit-qlora](https://github.com/pjordanandrsn/experts4bit-qlora) ([PyPI](https://pypi.org/project/experts4bit-qlora/)) owns model loading, quantization orchestration, adapters, QLoRA training, residency integration and serving, and drives these kernels through `enable_fast()` / `enable_fast_train()`. A model-level symptom ("bitsandbytes MoE still OOMs after `load_in_4bit`") starts there.

## Install

```bash
pip install grouped-nf4-gemm
```

Linux, NVIDIA GPU of compute capability sm_80 or newer (sm_120 is the primary serving target), `triton>=3.4` (Linux-only distribution), `torch>=2.8` (pre-releases accepted); CI tests Python 3.11. `nf4gemm`, `gnf4` and `grouped-mxfp4-gemm` are lookup aliases, not separate packages. Through the consumer:

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
- Shapes under a weight-byte floor lose outright; `nf4_grouped.decode_dispatch(N, K, T, sm_count)` returns `("dequant",)` for them and the caller routes those cells back. The op itself never switches algorithm.
- `top_k=1` cells are instance-unstable; peak VRAM does not improve, only the forward-to-backward transient shrinks ([`STATUS.md`](../STATUS.md)).
- The fidelity ordering is a CUDA tensor-core statement; other backends must re-measure. No ROCm or XPU ([`PORTABILITY.md`](../PORTABILITY.md)).
- CUDA + Triton only for the kernel. `nf4_pack_ref` imports `nf4_grouped`, which binds triton through `_triton_shim`, so the pure-torch surface (pack references, `dequant_ref`, provenance, arena bake/verify) imports and runs without triton, and a `gemm_4bit_grouped` call on CPU tensors raises naming `dequant_ref` on a triton-less box too; the Triton kernels need a CUDA GPU; macOS and Windows are not exercised by CI.
- Open: `#87`, int32 offset overflow at large `max(expert_ids)`.

## Related

[`KERNEL_CONTRACT.md`](../KERNEL_CONTRACT.md) · [`TOLERANCE_CONTRACT.md`](../TOLERANCE_CONTRACT.md) · [`STATUS.md`](../STATUS.md) · [`claims.json`](../claims.json) · [`REPRO.md`](../../REPRO.md) · [`native-mxfp4-moe-inference.md`](native-mxfp4-moe-inference.md) · [`int4-decode-gemv.md`](int4-decode-gemv.md) · [`stream-moe-experts-from-host-or-nvme.md`](stream-moe-experts-from-host-or-nvme.md)

## Evidence

Register: [`claims.json`](../claims.json). Confirmed: claim `gnf4.kernel.fused-more-accurate-than-dequant-bf16`, claim `gnf4.kernel.decode-speed-census`, claim `gnf4.kernel.energy-104-of-112`, claim `gnf4.kernel.e2e-training-real-prose`, claim `gnf4.kernel.h2h-unsloth` (4-bit-storage regime; Unsloth wins its own bf16-resident regime), claim `gnf4.kernel.graphed-baseline-decode-loses`. Measured: claim `gnf4.kernel.sm120-census-vs-grouped-mm`, claim `gnf4.kernel.dgrad`. Receipts under `kernel/RESULTS-*.md` with their `prereg_*.json`; property suite `kernel/test_nf4_grouped.py`.
