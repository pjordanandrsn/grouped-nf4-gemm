# How do I verify that the quantized expert bytes I serve or train on are the released checkpoint bytes?
<!-- summary: file_tensor_sha256, provenance_table and verify_arena_matches hash safetensors byte ranges against the loaded and baked bytes, and verify_provenance re-derives a run's hash table. -->

Hash them at the byte-range level with `grouped-nf4-gemm`'s provenance primitives: `mxfp4_loader.file_tensor_sha256(path, name)` streams sha256 over a tensor's data-section bytes in a safetensors shard without loading it, `mxfp4_loader.tensor_sha256(t)` hashes the tensor actually placed in memory, and `mxfp4_loader.verify_arena_matches(path, loaded)` raises on any mismatch. The `verify_provenance` CLI re-derives a training run's recorded hash table from the public checkpoint; `nvme_arena.verify` closes the chain for a baked arena.

## Symptoms

- You serve a gpt-oss or Kimi-K3-class MXFP4 model and want a receipt that the expert bytes being multiplied are the release, not a requantized copy.
- You fine-tuned with LoRA over frozen 4-bit experts and need to show the frozen base is byte-identical after training.
- You relocated a checkpoint into an expert-major NVMe arena and want to prove the relocation kept every byte, from the vendor's shards to the row on disk.
- You quantized a bf16 checkpoint to NF4 at bake time and need to record which shipped bytes and which quantizer produced each row.

## Why it happens

A whole-file hash cannot answer these: a relocation reorders bytes, so `sha256(arena) != sha256(file)` even when nothing changed; a training run touches only adapters, so only the frozen tensors' ranges are the claim; a loader that dequantizes and re-packs produces plausible weights whose bytes are not the release. The honest unit is the tensor's byte range in the safetensors data section, hashed on the file side without torch and on the loaded side after every reshape ([`K3-PROVENANCE-CHAIN.md`](../K3-PROVENANCE-CHAIN.md)). `mxfp4_loader.to_kernel_shapes` is a contiguous view, which is what makes the loaded-side hash meaningful.

## Which project solves it

`grouped-nf4-gemm` owns the hashing primitives, the arena manifests and verifier, and the CLI. [experts4bit-qlora](https://github.com/pjordanandrsn/experts4bit-qlora) ([PyPI](https://pypi.org/project/experts4bit-qlora/)) records the hash table in a run artifact during training and serving. [`SECURITY.md`](../../SECURITY.md) puts a way to make verification report success on mismatched bytes in the highest-severity class; report one privately.

## Install

Kernel package (the only route this page needs):

```bash
pip install grouped-nf4-gemm
```

The provenance surface is pure torch plus the standard library and runs on any host, including boxes with no GPU stack; the CLI needs a local Hugging Face snapshot with `model.safetensors.index.json`. Kernels elsewhere in the package need Linux, an NVIDIA GPU (sm_80+), `triton>=3.4` and `torch>=2.8`; CI tests Python 3.11.

## Smallest correct example

CPU-only: write a tiny gpt-oss-shaped shard by hand, build the provenance table, verify the kernel-shaped views, and confirm one flipped bit is caught.

```python
# CPU-only (pure torch + stdlib; no GPU, no triton, no safetensors package needed)
import json, os, struct, tempfile
import torch
from mxfp4_loader import (layer_expert_names, provenance_table, to_kernel_shapes,
                          verify_arena_matches)

E, N, NB = 2, 8, 4                                   # layer 0, K = NB * 32
names = layer_expert_names(0)                        # the four expert-tensor names
tensors = {
    names["mlp.experts.gate_up_proj_blocks"]: torch.randint(0, 256, (E, N, NB, 16), dtype=torch.uint8),
    names["mlp.experts.gate_up_proj_scales"]: torch.randint(120, 135, (E, N, NB), dtype=torch.uint8),
    names["mlp.experts.down_proj_blocks"]: torch.randint(0, 256, (E, N, NB, 16), dtype=torch.uint8),
    names["mlp.experts.down_proj_scales"]: torch.randint(120, 135, (E, N, NB), dtype=torch.uint8),
}
hdr, blobs, off = {}, [], 0                          # a minimal safetensors writer
for name, t in tensors.items():
    b = t.numpy().tobytes()
    hdr[name] = {"dtype": "U8", "shape": list(t.shape), "data_offsets": [off, off + len(b)]}
    blobs.append(b); off += len(b)
hj = json.dumps(hdr).encode()
path = os.path.join(tempfile.mkdtemp(), "model.safetensors")
with open(path, "wb") as f:
    f.write(struct.pack("<Q", len(hj))); f.write(hj); f.write(b"".join(blobs))

table = provenance_table(path, layers=(0,))          # sha256 of each tensor's on-disk bytes
loaded = {}
for kind in ("gate_up", "down"):
    kb, ks = to_kernel_shapes(tensors[names[f"mlp.experts.{kind}_proj_blocks"]],
                              tensors[names[f"mlp.experts.{kind}_proj_scales"]])
    loaded[names[f"mlp.experts.{kind}_proj_blocks"]] = kb
    loaded[names[f"mlp.experts.{kind}_proj_scales"]] = ks
report = verify_arena_matches(path, loaded)          # raises ValueError on any mismatch
assert all(r["match"] for r in report.values())
assert table["hashes"] == {n: r["arena"] for n, r in report.items()}

bad = loaded[names["mlp.experts.down_proj_blocks"]].clone()
bad.view(-1)[5] ^= 0x01                              # one flipped bit
try:
    verify_arena_matches(path, {names["mlp.experts.down_proj_blocks"]: bad})
    raise SystemExit("tamper not detected")
except ValueError as e:
    assert "PROVENANCE FAIL" in str(e)
```

The CLI, against a real release and a run artifact (CPU-only; needs a local snapshot):

```bash
python -m verify_provenance --artifact run_artifact.json --model openai/gpt-oss-120b
python -m verify_provenance --artifact run_artifact.json --snapshot /path/to/snapshot --limit 8   # smoke
python -m nvme_arena verify --arena /nvme/model.arena --against-source /path/to/snapshot
```

## Expected result

The Python block completes: every kernel-shaped view hashes to its file range, the provenance table equals the arena-side hashes, and the flipped bit raises `ValueError("PROVENANCE FAIL ...")`. `verify_provenance` prints a line per checked tensor, then `PROVENANCE OK` with exit 0, or `PROVENANCE FAIL` with the differing names and exit 1; a run that recorded `pre_equals_post` reports the bytes unchanged by training. `nvme_arena verify` exits 0 only if every segment matches the manifest and, with `--against-source`, every recorded source range still hashes to what the bake saw.

## Supported scope

- `file_tensor_sha256(path, name)`: any safetensors shard, any dtype; sharded checkpoints resolve through the index (`provenance_table(..., weight_map=, snapshot=)`).
- `provenance_table(path, layers, prefix="model.layers")` covers the four gpt-oss expert tensors per layer; `verify_arena_matches(path, loaded)` takes `{name: uint8 tensor}` and returns a per-tensor report.
- The artifact `verify_provenance` reads is JSON with `model`, optional `snapshot`, and `provenance.file_hashes` (`{tensor_name: sha256}`), plus `pre_equals_post` and `n_file_tensors` as recorded.
- Arena manifests: relocation (`nvme_arena.bake`, `bake_expert_tensors`) records one source range and hash per segment; `nvme_bake_nf4` records `bake_mode: nf4-quantize` with the consumed source ranges, their hashes and the quantizer record, so `verify --against-source` checks a two-hop chain.
- Training: `mxfp4_qlora` keeps the packed storage `requires_grad=False` with no code path that writes it; `mxfp4_native_load.build_native_qlora_model` returns the file hashes it verified at load.

## Limitations

- Hash equality proves bytes, not numerics. Whether the kernel decodes those bytes correctly is a separate oracle: `mxfp4_pack_ref.dequant_mxfp4` against transformers' path and against compressed-tensors for K3 (claim `gnf4.k3.oracle-exact`). Two implementations agreeing rules out a convention mismatch, not a shared misreading of the spec.
- Scope is expert bytes. Non-expert tensors (attention, embeddings, norms) are outside the chain, as K3's own quantization config excludes them.
- The NF4 quantize-bake claim is "these rows came from exactly these shipped bytes by exactly this quantizer", weaker than bit identity; the manifest says which claim it makes.
- The serve-side receipt's per-shard check was a spot sample of a few tensors, not shard-level coverage (claim `gnf4.mxfp4.serve-tax-deleted`, notes).
- `--limit` is a smoke check, not a receipt. `resolve_snapshot` warns when several cached snapshots exist and picks the newest; pass `--snapshot` to pin.
- Parsing a crafted arena or checkpoint is the surface [`SECURITY.md`](../../SECURITY.md) names as in scope.

## Related

[`K3-PROVENANCE-CHAIN.md`](../K3-PROVENANCE-CHAIN.md) · [`RESULTS-nvme-determinism.md`](../RESULTS-nvme-determinism.md) · [`provenance/gptoss20b_expert_bytes.md`](../provenance/gptoss20b_expert_bytes.md) · [`STATUS.md`](../STATUS.md) · [`claims.json`](../claims.json) · [README](../../README.md) ("Provenance in four lines") · [`native-mxfp4-moe-inference.md`](native-mxfp4-moe-inference.md) · [`stream-moe-experts-from-host-or-nvme.md`](stream-moe-experts-from-host-or-nvme.md)

## Evidence

Register: [`claims.json`](../claims.json). Confirmed: claim `gnf4.mxfp4.train-9.82gb` (every expert-tensor hash identical across file, loaded and post-training on gpt-oss-120b QLoRA), claim `gnf4.mxfp4.serve-tax-deleted` (served on released bytes; provenance spot-checked per shard). Measured: claim `gnf4.k3.oracle-exact` (decode exact against K3's declared reference; arena round trip byte-identical with a byte-flip control), claim `gnf4.nvme.tier-batch-only` (its evidence includes [`RESULTS-nvme-determinism.md`](../RESULTS-nvme-determinism.md), a cross-host bake determinism run). Tests: `kernel/test_mxfp4_provenance.py`, `kernel/test_nvme_arena.py`, `kernel/test_arena_experts.py`.
