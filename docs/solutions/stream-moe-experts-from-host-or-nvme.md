# How do I run a MoE whose experts do not fit in VRAM, streaming them from pinned host RAM or an NVMe arena?
<!-- summary: The kernel-side storage primitives for experts that do not fit VRAM: the NVMe arena bake and O_DIRECT reader, the pinned-DRAM row tier and low-level residency, and the GPU-driven host gather. -->

**Role of this page: kernel/storage primitives.** The arena bake and verifier, the O_DIRECT reader, the pinned-DRAM row tier and the low-level residency engines. It is not the decision page and not the model-level integration; those live in the consumer. An NVMe primitive (arena, reader, row tier, residency engine) belongs here; model-level NVMe integration — binding those primitives to a loaded model and deciding which bytes live where — belongs to experts4bit-qlora.

**Use this page when…** you are building or debugging the storage layer itself: baking an expert-major arena, sizing a pinned row tier, reading rows with `ArenaReader`, or wiring `ColdTier` / `ArenaExpertSource` under your own forward. Start in [experts4bit-qlora](https://github.com/pjordanandrsn/experts4bit-qlora) instead when the question starts from a model:

- **Should this MoE stream at all, and from where?** — the decision/router page, [run-moe-larger-than-vram.md](https://github.com/pjordanandrsn/experts4bit-qlora/blob/main/docs/solutions/run-moe-larger-than-vram.md).
- **Offload a loaded model's experts to CPU RAM or NVMe and serve or train it** — the model-level integration page, [offload-moe-experts-to-cpu-or-nvme.md](https://github.com/pjordanandrsn/experts4bit-qlora/blob/main/docs/solutions/offload-moe-experts-to-cpu-or-nvme.md).

`grouped-nf4-gemm` ships the tiers under a fused-expert forward: a GPU-driven gather from pinned host memory over UVA (`host_gather.gather_expert_rows`), an expert-major on-disk arena you bake once (`nvme_arena`, `nvme_bake_nf4`), an O_DIRECT reader (`nvme_reader.ArenaReader`), a pinned-DRAM residency tier over that arena (`nvme_residency.ColdTier`), and the MXFP4 engines that consume them (`mxfp4_pipelined.Mxfp4PipelinedGptOss`, `mxfp4_residency.Mxfp4NvmeResidency`). Which bytes live where is the consumer's decision.

## Symptoms

- The routed experts exceed the card: the model runs only if experts stream from host RAM per token.
- The experts exceed host RAM too (Kimi-K3-class), so a subset must come from disk on demand.
- A per-expert-tensor checkpoint (safetensors shards, tensor-major) makes one expert several seeks; you need an expert-major layout on NVMe and proof the relocation kept the bytes.
- Pinned host allocations OOM at a capacity that fits on paper.

## Why it happens

Top-k routing touches a small fraction of expert bytes per token, so residency, not capacity, is the binding constraint; but a serving loop needs those bytes to arrive as one aligned request landing where the kernel will read them. Safetensors is tensor-major; the page cache duplicates a DRAM tier and hands eviction to the OS; a pinned row costs more host memory than its stride (`nvme_residency.PINNED_ROW_FACTOR`). The tiers here fix the layout at bake time and never copy a row more than the link requires.

## Which project solves it

`grouped-nf4-gemm` owns the primitives: bake, verify, read, residency, gather, the writable KV row pool, and the two MXFP4 residency engines. [experts4bit-qlora](https://github.com/pjordanandrsn/experts4bit-qlora) ([PyPI](https://pypi.org/project/experts4bit-qlora/)) owns residency integration into a model, the NF4 offload path that serves a checkpoint from pinned host RAM on a small-VRAM card, and serving. To run a MoE larger than VRAM end to end, install the consumer; this page is the primitives it stands on.

## Install

Kernel package (the minimum route):

```bash
pip install grouped-nf4-gemm
```

The relocation bake and verifier are stdlib plus the byte-range hasher and run on any host; the reader and `ColdTier(pinned=False)` are CPU-only. Pinned buffers, the gather and the engines need Linux with an NVIDIA GPU (sm_80+), `triton>=3.4` (Linux-only), `torch>=2.8` (pre-releases accepted); CI tests Python 3.11. `nvme_bake_nf4` needs bitsandbytes and CUDA to quantize. Through the model consumer:

```bash
pip install "experts4bit-qlora[fast]"
```

## Smallest correct example

CPU-only: bake a toy gpt-oss-shaped checkpoint into an arena, verify it against the source, make two experts resident through the mmap tier, and check the bytes.

```python
# CPU-only (stdlib + torch import chain; no GPU, no triton). O_DIRECT falls back to
# buffered reads with a one-time warning on filesystems that refuse it.
import json, os, struct, tempfile
from mxfp4_loader import EXPERT_SUFFIXES
from nvme_arena import bake, verify, load_index
from nvme_residency import ColdTier, capacity_for_bytes

E, L = 4, 2
shapes = {EXPERT_SUFFIXES[0]: (8, 16), EXPERT_SUFFIXES[1]: (8, 4),   # gate_up blocks / scales
          EXPERT_SUFFIXES[2]: (6, 8), EXPERT_SUFFIXES[3]: (6, 2)}    # down blocks / scales
snap = tempfile.mkdtemp()
src, hdr, blobs, off = {}, {}, [], 0
for layer in range(L):
    for suf, (n, w) in shapes.items():
        name, data = f"model.layers.{layer}.{suf}", os.urandom(E * n * w)
        src[name] = data
        hdr[name] = {"dtype": "U8", "shape": [E, n, w], "data_offsets": [off, off + len(data)]}
        blobs.append(data); off += len(data)
hj = json.dumps(hdr).encode()
with open(os.path.join(snap, "model.safetensors"), "wb") as f:
    f.write(struct.pack("<Q", len(hj))); f.write(hj); f.write(b"".join(blobs))

arena = os.path.join(snap, "toy.arena")
bake(snap, arena, log=lambda *a: None)             # + toy.arena.index.json, toy.arena.manifest.json
assert verify(arena, against_source=snap, log=lambda *a: None)["ok"]

index = load_index(arena)
stride = index["row_stride"]
tier = ColdTier(arena, hot_rows=capacity_for_bytes(8 * stride, stride, pinned=False), pinned=False)
slots = tier.ensure(1, [3, 0])                     # layer 1, experts 3 and 0 -> slot indices
assert len(slots) == 2 and tier.resident(1, 3)
row = tier.row(1, 3)                               # the row's bytes, one memoryview
seg = {g["suffix"]: g for g in index["segments"]}[EXPERT_SUFFIXES[0]]
n, w = shapes[EXPERT_SUFFIXES[0]]
assert bytes(row[seg["seg_off"]:seg["seg_off"] + seg["length"]]) == src["model.layers.1." + EXPERT_SUFFIXES[0]][3 * n * w:4 * n * w]
tier.close()
```

GPU: the host-RAM primitive, a device-side gather from a pinned expert stack.

```python
# GPU (sm_80+) + triton
import torch
from host_gather import gather_expert_rows

E, ROW = 64, 4096                                          # 64 expert rows of 4 KiB
host = torch.randint(0, 256, (E, ROW), dtype=torch.uint8).pin_memory()   # pinned host RAM
ids = torch.tensor([9, 3, 9, 41], dtype=torch.int32, device="cuda")
dst = torch.empty(ids.numel(), ROW, dtype=torch.uint8, device="cuda")
gather_expert_rows(dst, host, ids)                          # the GPU reads host memory over UVA
torch.cuda.synchronize()
assert torch.equal(dst.cpu(), host[ids.cpu().long()])
```

## Expected result

The CPU block bakes, verifies the full chain (source range hash, manifest, arena bytes), and hands back a row whose segment equals the source slice. The GPU block copies exactly the requested rows with no CPU involvement in choosing them.

## Supported scope

- Bake: `nvme_arena.bake(snapshot, out, layers=None, prefix="model.layers", align=4096)` relocates gpt-oss-style fused expert tensors; `nvme_arena.bake_expert_tensors(snapshot, out, name_template=..., kinds=...)` relocates per-expert tensors (K3, DeepSeek lineage; bake in `mxfp4_residency.K3_RESIDENCY_KINDS` order if the residency engine will read the arena); `nvme_bake_nf4.bake_nf4(snapshot, out, ...)` quantizes a bf16 checkpoint to NF4 at bake time with a two-hop manifest. CLI: `python -m nvme_arena bake|bake-experts|verify ...`; `python -m nvme_bake_nf4 --snapshot ... --out ... [--layers 0-93] [--prefix ...] [--moe mlp] [--absmax-dtype f32|bf16|auto]`.
- Read: `nvme_reader.ArenaReader(arena_path, qd=None)` (thread pool over `os.preadv`, O_DIRECT where available); `nvme_reader.alloc_landing(n_bytes, pinned=...)` for page-aligned landing buffers.
- Residency: `ColdTier(arena, hot_rows=..., pinned=True)` with `ensure(layer, experts)`, `row`, `pinned_tensor()`, `stats()`; size `hot_rows` from measured free RAM with `capacity_for_bytes(usable_bytes, row_stride, pinned=True)`.
- Compute: `arena_experts.ArenaExpertSource(arena, qd=16, device="cuda")` returns kernel-shaped `(blocks, scales)` via `fused_stacks`; `Mxfp4PipelinedGptOss` (all rows pinned) and `Mxfp4NvmeResidency` (tier-backed) run the fused MXFP4 forward; `cpu_grouped` is the hybrid CPU path over the same packed bytes (`cpu_kernels_available()`, `gemv_nf4_grouped_cpu`, `gemv_mxfp4_grouped_cpu`).

## Limitations

- The NVMe tier is a batch tier: at the measured per-box read constant it is seconds per token for a fully cold large model; it buys reachability and provenance, not latency (claim `gnf4.nvme.tier-batch-only`; [`nvme-ceilings.md`](../nvme-ceilings.md)).
- Expert prefetch is closed, negative, over four registered arcs; the recommended copy path is the GPU-side gather, not speculation (claim `gnf4.flagship.prefetch-closed-negative`).
- Per-token time from host RAM is additive and per box, `t ≈ c_box + bytes/link`; a fixed fraction-of-waterfall is not the law ([`STATUS.md`](../STATUS.md)).
- `Mxfp4NvmeResidency` is not CUDA-graph capturable: a miss is a host-side disk read.
- Do not quantize-bake a checkpoint that already ships MXFP4 ([README](../../README.md)); relocate it.
- The cold-engine premise (bitsandbytes' CPU dequant as a free decode arm) is refuted on a box without AVX-512 (claim `gnf4.cold-engine.phase0-premise-refuted`).
- Open issues on arena efficiency (#73, #60, #58) and the pinned-row factor (#71: `PINNED_ROW_FACTOR` is conservative on cgroup v1; cgroup v2 is unmeasured) are listed in [`STATUS.md`](../STATUS.md). No ROCm or XPU.

## Related

[`nvme-ceilings.md`](../nvme-ceilings.md) · [`K3-PROVENANCE-CHAIN.md`](../K3-PROVENANCE-CHAIN.md) · [`RESULTS-nvme-determinism.md`](../RESULTS-nvme-determinism.md) · [`STATUS.md`](../STATUS.md) · [`claims.json`](../claims.json) · [`native-mxfp4-moe-inference.md`](native-mxfp4-moe-inference.md) · [`verify-quantized-checkpoint-provenance.md`](verify-quantized-checkpoint-provenance.md)

## Evidence

Register: [`claims.json`](../claims.json). Confirmed: claim `gnf4.flagship.235b-phaseA` (synthetic-weight offload at the link's waterfall ceiling), claim `gnf4.flagship.235b-phaseB` (the real checkpoint served from pinned host RAM across five pods, with the additive per-box law), claim `gnf4.flagship.prefetch-closed-negative`. Measured: claim `gnf4.nvme.tier-batch-only`, claim `gnf4.k3.oracle-exact` (arena round trip on a byte-verified store), claim `gnf4.cold-engine.phase0-premise-refuted`. Receipts: `bench/phase3/flagship/RESULTS-flagship-phaseB.md`, `bench/phase3/flagship/RESULTS-flagship-offload.md`, [`RESULTS-k3-slice-roundtrip.md`](../RESULTS-k3-slice-roundtrip.md).
