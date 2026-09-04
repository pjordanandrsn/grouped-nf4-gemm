# How do I run MoE expert inference natively on MXFP4 (e2m1 + e8m0) weights, straight from the released checkpoint bytes?
<!-- summary: gemm_mxfp4_grouped multiplies gpt-oss and Kimi-class experts on their released e2m1 blocks and e8m0 scales, with no requantization to NF4 and no bf16 materialization. -->

Use `mxfp4_grouped.gemm_mxfp4_grouped` from `grouped-nf4-gemm`: the grouped expert GEMM runs on the checkpoint's own MXFP4 blocks and e8m0 scales, so a gpt-oss or Kimi-K3-class expert is multiplied as shipped, with no requantization to NF4 and no bf16 materialization. `mxfp4_pack_ref.dequant_mxfp4` is the pure-torch reference the kernel is gated against.

## Symptoms

- gpt-oss ships its experts as MXFP4 (`*_blocks` uint8 e2m1 nibbles, `*_scales` uint8 e8m0), and your only 4-bit path requantizes them to NF4 or dequantizes to bf16, which changes the numbers and costs a decode per read.
- You want to serve or fine-tune on the exact released checkpoint bytes and prove it afterwards (hash before == hash after).
- A per-expert-tensor MXFP4 release (Kimi K3, DeepSeek lineage: `weight_packed [N, K//2]` + `weight_scale [N, K//32]`) has no fused-expert kernel that consumes it directly.

## Why it happens

MXFP4 (OCP MX v1.0) is a different codebook from NF4: sixteen e2m1 values, ±{0, 0.5, 1, 1.5, 2, 3, 4, 6}, one power-of-two e8m0 scale per 32 elements, even element in the low nibble, against NF4's per-64 fp32 absmax and high-nibble-first packing. A kernel built for one layout cannot read the other, so the usual answer is to convert, and every conversion either loses information (MXFP4 to bf16 to NF4) or reintroduces the bf16 round trip the fused kernel exists to delete. `docs/mxfp4/PHASE0-seam-map.md` records the four places the decode differs; the MXFP4 kernel is the NF4 kernel with exactly those swapped.

## Which project solves it

`grouped-nf4-gemm` owns the native MXFP4 kernels (`gemm_mxfp4_grouped`; `gemv_mxfp4_b32`, the packed MXFP4 decode GEMV — optimising that GEMV is a change in this kernel repository, per AGENTS.md section 8, not in the consumer), the pack/decode reference, the loader helpers that map checkpoint shapes to kernel shapes without copying (`mxfp4_loader.to_kernel_shapes`), the arena source for per-expert releases (`arena_experts.ArenaExpertSource`), and the training wrapper (`mxfp4_qlora`). [experts4bit-qlora](https://github.com/pjordanandrsn/experts4bit-qlora) ([PyPI](https://pypi.org/project/experts4bit-qlora/)) drives them from a model; `mxfp4_native_load.build_native_qlora_model(snap, r, alpha)` here is the gpt-oss-specific loader that builds a QLoRA model without entering the dequant path.

## Install

Kernel package (the minimum route):

```bash
pip install grouped-nf4-gemm
```

Linux, NVIDIA GPU sm_80 or newer (sm_120 is the primary serving target), `triton>=3.4` (Linux-only distribution), `torch>=2.8` (pre-releases accepted); CI tests Python 3.11. The pack/decode reference and loader hashing are pure torch. Through the model consumer:

```bash
pip install "experts4bit-qlora[fast]"
```

## Smallest correct example

CPU-only: pack, decode, and confirm the checkpoint-to-kernel reshape moves no bytes.

```python
# CPU-only (pure torch; no GPU, no triton launch)
import torch
from mxfp4_pack_ref import quantize_pack_mxfp4, dequant_mxfp4
from mxfp4_loader import to_kernel_shapes, tensor_sha256

E, N, K = 2, 64, 256                                  # K % 32 == 0
blocks, scales = quantize_pack_mxfp4(torch.randn(E, N, K))   # [E,N,K//32,16] u8, [E,N,K//32] u8
wq = dequant_mxfp4(blocks, scales)                    # [E, N, K] fp32
assert wq.shape == (E, N, K)
kb, ks = to_kernel_shapes(blocks, scales)             # [E, N, K//2] view; scales unchanged
assert tensor_sha256(kb) == tensor_sha256(blocks)     # a view, not a reorder
```

GPU: the grouped kernel against the reference decode of the same bytes.

```python
# GPU (sm_80+) + triton
import torch
from mxfp4_pack_ref import quantize_pack_mxfp4, dequant_mxfp4
from mxfp4_loader import to_kernel_shapes
from mxfp4_grouped import gemm_mxfp4_grouped

E, N, K = 4, 128, 256
blocks, scales = quantize_pack_mxfp4(torch.randn(E, N, K) * 0.3)
kb, ks = (t.cuda() for t in to_kernel_shapes(blocks, scales))
sizes, expert_ids = [3, 1, 4], [1, 0, 3]              # group-sorted tokens per expert
a_cat = torch.randn(sum(sizes), K, device="cuda", dtype=torch.bfloat16)

out = gemm_mxfp4_grouped(a_cat, kb, ks, sizes, expert_ids)   # [T, N] bf16

row, refs = 0, []
for m, e in zip(sizes, expert_ids):
    W = dequant_mxfp4(blocks[e].cuda(), scales[e].cuda())     # [N, K] fp32, same bytes
    refs.append(a_cat[row:row + m].float() @ W.t()); row += m
ref = torch.cat(refs)
assert ((out.float() - ref).abs().max() / ref.abs().max()).item() < 2e-2
```

## Expected result

The CPU block completes: the packed shapes are the gpt-oss tensor shapes and the kernel-shaped view hashes identically to its source. The GPU block returns `[T, N]` bf16 in group order, within the bound `kernel/test_mxfp4_grouped.py` uses. On a real checkpoint, `mxfp4_loader.file_tensor_sha256(path, name)` over the shard's byte range equals `tensor_sha256` of the loaded view ([`verify-quantized-checkpoint-provenance.md`](verify-quantized-checkpoint-provenance.md)).

## Supported scope

- Format: e2m1 blocks `[E, N, K//2]` uint8 (low nibble = even element), e8m0 scales `[E, N, K//32]` uint8, `K % 32 == 0`; gpt-oss `[E, N, n_blk, 16]` blocks flatten to that width as a contiguous view.
- `gemm_mxfp4_grouped(a_cat, blocks, scales, sizes, expert_ids)`: the NF4 kernel's calling convention; all-ones `sizes` take the GEMV reduction, mixed groups the M-tile path.
- `gemv_mxfp4_b32(xq, xs, blocks, scales, eids, N, K)`: the decode-grade GEMV on int8 activation rows from `int4_b32.quant_x_rows`, split-K partials reduced by `int4_b32.reduce_partials`.
- Per-expert releases: `nvme_arena.bake_expert_tensors` relocates them into an expert-major arena; `ArenaExpertSource.fused_stacks(layer, expert_ids, proj)` returns the kernel's `(blocks, scales)` for one projection.
- Training: `mxfp4_qlora.ExpertsMxfp4LoRA` trains LoRA over frozen native bytes with recompute-in-backward; `build_native_qlora_model` returns the model, wrappers and per-tensor file hashes.
- Engines: `mxfp4_pipelined.Mxfp4PipelinedGptOss` (host-resident), `mxfp4_residency.Mxfp4NvmeResidency` (NVMe-backed) — see [`stream-moe-experts-from-host-or-nvme.md`](stream-moe-experts-from-host-or-nvme.md).

## Limitations

- CUDA + Triton only for the kernels; no ROCm or XPU ([`PORTABILITY.md`](../PORTABILITY.md)). `mxfp4_grouped` binds triton through `_triton_shim`, so the pure-torch surface (`mxfp4_pack_ref`, the `mxfp4_loader` hashing, the relocation arena bake/verify) imports and runs without triton; the Triton kernels need a CUDA GPU; macOS and Windows are not exercised by CI.
- Do not quantize-bake a checkpoint that is already MXFP4 ([README](../../README.md)); relocation keeps the bytes, re-quantizing to NF4 costs a decode per read and breaks provenance.
- The e8m0 `0xFF` byte decodes as transformers' oracle does (ldexp, no NaN reservation); real checkpoints do not contain it.
- The reference decode agrees with two independent implementations (transformers' gpt-oss path; compressed-tensors for K3). Agreement rules out a convention mismatch, not a shared misreading of the OCP spec ([`K3-PROVENANCE-CHAIN.md`](../K3-PROVENANCE-CHAIN.md)).
- `gemv_mxfp4_b32` has a correctness gate in the tree but no entry in [`claims.json`](../claims.json): a capability without a published measurement. The MXFP4 GEMM has no separate speed census; the NF4 page's decode-speed limits apply structurally.

## Related

[`STATUS.md`](../STATUS.md) · [`claims.json`](../claims.json) · [`K3-PROVENANCE-CHAIN.md`](../K3-PROVENANCE-CHAIN.md) · [`INDEX.md`](../INDEX.md) (the `docs/mxfp4/` pre-registrations and results) · [`verify-quantized-checkpoint-provenance.md`](verify-quantized-checkpoint-provenance.md) · [`int4-decode-gemv.md`](int4-decode-gemv.md)

## Evidence

Register: [`claims.json`](../claims.json). Confirmed: claim `gnf4.mxfp4.serve-tax-deleted` (gpt-oss-120b served on its released bytes against the shipped-precision reference; its P1 sub-clause missed as stamped and the receipt says why), claim `gnf4.mxfp4.train-9.82gb` (QLoRA on native bytes; every expert hash identical before and after training). Measured: claim `gnf4.k3.oracle-exact` (reference decode reproduces Kimi K3's declared reference). Receipts: [`mxfp4/RESULTS-mxfp4-serve.md`](../mxfp4/RESULTS-mxfp4-serve.md), [`mxfp4/RESULTS-mxfp4-train.md`](../mxfp4/RESULTS-mxfp4-train.md), [`RESULTS-k3-phase1-oracle.md`](../RESULTS-k3-phase1-oracle.md).
