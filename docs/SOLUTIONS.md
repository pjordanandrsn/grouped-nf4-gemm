# Solutions — one page per problem

Each page answers one ordinary problem in the same order: the direct
answer, the symptoms people search for, why it happens, which of the two
projects solves it, the canonical install command, the smallest correct
example, the observable result, the supported scope, the limitations, and
the evidence (active claim IDs in [`claims.json`](claims.json), or the
statement that the page describes a capability without a measurement).

| the problem | page |
|---|---|
| A per-expert 4-bit loop dequantises each expert to bf16 and re-reads it; compute the grouped GEMM on the NF4-packed weights | [nf4-grouped-gemm-without-bf16-materialization.md](solutions/nf4-grouped-gemm-without-bf16-materialization.md) |
| Serve MXFP4 experts (gpt-oss, DeepSeek-V4) from the released bytes | [native-mxfp4-moe-inference.md](solutions/native-mxfp4-moe-inference.md) |
| Single-token int4 expert / attention projections near the memory ceiling | [int4-decode-gemv.md](solutions/int4-decode-gemv.md) |
| An fp8 paged decode attention and the fused decode glue | [fp8-paged-attention-for-moe-serving.md](solutions/fp8-paged-attention-for-moe-serving.md) |
| The experts do not fit VRAM, or host RAM: stream them from pinned memory or an NVMe arena | [stream-moe-experts-from-host-or-nvme.md](solutions/stream-moe-experts-from-host-or-nvme.md) |
| Prove the served bytes are the checkpoint's released bytes | [verify-quantized-checkpoint-provenance.md](solutions/verify-quantized-checkpoint-provenance.md) |

## Which of the two packages

- **`grouped-nf4-gemm`** (this repository, `pip install grouped-nf4-gemm`)
  owns the kernels and primitives: grouped NF4 and native MXFP4 GEMMs, the
  int4-b32 decode GEMV and its calibrated packer, the FP8 paged attention,
  the decode glue, the pure-torch references, the arena bake/reader/tiers
  and the provenance verifier. It is a kernel package, not a serving
  engine: it runs on Linux with Triton ≥ 3.4 on NVIDIA sm_80+ GPUs (sm_120
  is the primary serving target), and the CPU-reachable surface is pure
  torch.
- **[`experts4bit-qlora`](https://github.com/pjordanandrsn/experts4bit-qlora)**
  (`pip install "experts4bit-qlora[fast]"` installs this package) owns
  model loading, quantisation orchestration, adapters, training, residency
  integration and serving. Questions that start from a model — "my MoE
  still OOMs after `load_in_4bit`", "QLoRA on fused experts", "run a MoE
  larger than VRAM", "offload experts to RAM or NVMe" — belong there; its
  own solution index is
  [docs/SOLUTIONS.md](https://github.com/pjordanandrsn/experts4bit-qlora/blob/main/docs/SOLUTIONS.md).

Lookup aliases (`nf4gemm`, `gnf4`, `grouped-mxfp4-gemm`) install this
package; always install and cite `grouped-nf4-gemm`.

## Environment, in one line

Linux; NVIDIA sm_80+ with Triton ≥ 3.4 and torch ≥ 2.8 for the kernels;
the pack references, `dequant_ref`, the relocation arena bake and the
provenance hashing are pure torch (the NF4 quantise-bake needs bitsandbytes
and CUDA unless a `quantize_fn` is injected). CI tests Python 3.11 on Linux
only. No macOS or Windows for the kernels (the Triton kernels need a CUDA
GPU and triton is Linux-only); the pure-torch surface (pack references,
dequant, provenance, arena bake/verify) imports and runs without triton via
`_triton_shim`, but macOS and Windows are not exercised by CI. No ROCm or
XPU (port targets, see
[`PORTABILITY.md`](PORTABILITY.md)). The current position, including the
three limits where the fused path loses and what is measured-private, is
[`STATUS.md`](STATUS.md).

## Limitations that apply to every page

- Linux + NVIDIA sm_80+ with Triton ≥ 3.4 for the kernels; Python 3.11 is
  what CI tests. The pure-torch surface (references, packers, relocation
  bake/verify, provenance) has no GPU requirement, imports without triton via
  `_triton_shim`, and is exercised by CI on Linux only; the NF4 quantise-bake
  needs bitsandbytes and CUDA unless a `quantize_fn` is injected.
- Nothing falls back silently. `nf4_grouped.gemm_4bit_grouped` and
  `dgrad_4bit_grouped` refuse CPU tensors with an error that names
  `dequant_ref`; `gemm_mxfp4_grouped` and the `int4_b32` kernels carry no
  device guard and fail inside the Triton launch, and the `fp8_kv` appends
  and the paged attention refuse with a CUDA+Triton message that names no
  reference.
- The fused path loses at small shapes and to a CUDA-graphed per-expert
  loop at some decode shapes; the NVMe tier is batch-only; expert prefetch
  closed negative. The register records each of these.
- Numbers live in [`claims.json`](claims.json), tiered in the public
  register (confirmed, measured, measured-private, projected, open,
  superseded, retired); a page quotes a figure only beside the claim ID
  and tier it comes from, and a retired or superseded claim is never
  current.
- The fp8 paged attention is two support states, carried as two entries
  in [`capabilities.json`](capabilities.json): the fp8 compute path
  (sm_89+ precondition, measured on the RTX 5090 only) is supported; the
  f32 compute path (the sm_80–sm_88 default and every explicit f32
  request) is open under #319 and `unsupported` until it closes.
