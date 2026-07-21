# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Native-MXFP4 loader + the provenance primitive (Phase 3).

The seam map's load-bearing fact: the checkpoint's `*_blocks [E, N, n_blk, 16]`
uint8 flattens to `[E, N, n_blk*16] == [E, N, K//2]` — the kernel's B-tensor
width — as a contiguous VIEW (zero copy, zero reorder), and `*_scales
[E, N, n_blk]` is already the kernel's scale shape. So the bytes the kernel
computes on ARE the checkpoint's bytes.

The provenance primitive makes that claim checkable, cheaply: for each expert
tensor, `sha256(bytes in the safetensors data section) ==
sha256(bytes as loaded into the arena)`. The first is read straight from the
file's byte range (no torch, no dequant — the actual on-disk bytes of OpenAI's
release); the second hashes the loaded arena view. Equality is the receipt:
*"the expert bytes being computed on are bit-identical to OpenAI's release —
here are the hashes."* One hashing pass; no model materialization needed for
the file side.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct

import torch

# expert-tensor suffixes on a gpt-oss MoE layer (verified live, Phase 0)
GATE_UP_BLOCKS = "mlp.experts.gate_up_proj_blocks"
GATE_UP_SCALES = "mlp.experts.gate_up_proj_scales"
DOWN_BLOCKS = "mlp.experts.down_proj_blocks"
DOWN_SCALES = "mlp.experts.down_proj_scales"
EXPERT_SUFFIXES = (GATE_UP_BLOCKS, GATE_UP_SCALES, DOWN_BLOCKS, DOWN_SCALES)


def _read_st_header(path: str):
    """Return (header_dict, data_start_offset) for a .safetensors file without
    loading any tensor. Format: u64 LE header length, then that many JSON bytes,
    then the contiguous data blob."""
    with open(path, "rb") as f:
        (n_hdr,) = struct.unpack("<Q", f.read(8))
        hdr = json.loads(f.read(n_hdr).decode("utf-8"))
    return hdr, 8 + n_hdr


def file_tensor_sha256(path: str, name: str, chunk: int = 1 << 22) -> str:
    """sha256 of a tensor's raw bytes IN THE FILE'S DATA SECTION — streamed from
    the byte range in the header, no torch load, no dequant. This is the hash of
    OpenAI's actual on-disk bytes for `name`.

    Example:
        >>> from mxfp4_loader import file_tensor_sha256, tensor_sha256
        >>> file_tensor_sha256("model.safetensors", "model.layers.0.mlp.experts.gate_up_proj_blocks")
        '9f2c…'   # == tensor_sha256(loaded_tensor); a flipped byte changes it
    """
    hdr, data_start = _read_st_header(path)
    if name not in hdr:
        raise KeyError(f"{name} not in {path}")
    begin, end = hdr[name]["data_offsets"]
    h = hashlib.sha256()
    remaining = end - begin
    with open(path, "rb") as f:
        f.seek(data_start + begin)
        while remaining > 0:
            buf = f.read(min(chunk, remaining))
            if not buf:
                raise EOFError(f"truncated data for {name}")
            h.update(buf)
            remaining -= len(buf)
    return h.hexdigest()


def tensor_sha256(t: torch.Tensor) -> str:
    """sha256 of a tensor's raw bytes AS LOADED (contiguous, native dtype).
    For a uint8 tensor this is the arena bytes; a contiguous reshape does not
    change these bytes, which the provenance test asserts."""
    return hashlib.sha256(t.detach().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def to_kernel_shapes(blocks: torch.Tensor, scales: torch.Tensor):
    """Native `blocks [E, N, n_blk, 16]` + `scales [E, N, n_blk]` (uint8) ->
    the kernel's `blocks [E, N, K//2]` (contiguous VIEW) + `scales [E, N, K//32]`
    (unchanged). K = n_blk*32; K//2 = n_blk*16."""
    assert blocks.dtype == torch.uint8 and scales.dtype == torch.uint8
    assert blocks.shape[-1] == 16 and blocks.shape[:-1] == scales.shape, \
        (blocks.shape, scales.shape)
    E, N, n_blk, _ = blocks.shape
    kb = blocks.reshape(E, N, n_blk * 16)     # view iff contiguous (it is)
    return kb, scales


def layer_expert_names(layer: int, prefix: str = "model.layers") -> dict:
    """The four expert-tensor full names for a decoder layer."""
    base = f"{prefix}.{layer}."
    return {s: base + s for s in EXPERT_SUFFIXES}


def _resolve_tensor_path(name, path, weight_map, snapshot):
    """Sharded checkpoints keep each tensor in its own shard: with a
    ``weight_map`` (the safetensors index), resolve per name; otherwise the
    single ``path`` serves the whole checkpoint."""
    if weight_map is None:
        return path
    if name not in weight_map:
        raise KeyError(f"{name} not in the checkpoint index")
    base = snapshot if snapshot is not None else os.path.dirname(path)
    return os.path.join(base, weight_map[name])


def provenance_table(path: str, layers, prefix: str = "model.layers", *,
                     weight_map: dict = None, snapshot: str = None) -> dict:
    """Build the stamped provenance artifact: for every expert tensor in
    `layers`, the file-side sha256 (OpenAI's on-disk bytes). This is the demo
    artifact; `verify_arena_matches` proves the loaded arena reproduces it.
    Pass ``weight_map`` (+ ``snapshot``) for sharded checkpoints."""
    table = {}
    for L in layers:
        for suffix, name in layer_expert_names(L, prefix).items():
            table[name] = file_tensor_sha256(
                _resolve_tensor_path(name, path, weight_map, snapshot), name)
    return {"algo": "sha256", "source": snapshot or path, "n_tensors": len(table), "hashes": table}


def verify_arena_matches(path: str, loaded: dict, *,
                         weight_map: dict = None, snapshot: str = None) -> dict:
    """`loaded` maps tensor-name -> the uint8 tensor placed in the arena. Assert
    each equals the file's data-section bytes. Returns a per-tensor
    match report; raises on any mismatch (a provenance failure is not a
    tolerance)."""
    report = {}
    for name, t in loaded.items():
        want = file_tensor_sha256(
            _resolve_tensor_path(name, path, weight_map, snapshot), name)
        got = tensor_sha256(t)
        report[name] = {"file": want, "arena": got, "match": want == got}
        if want != got:
            raise ValueError(f"PROVENANCE FAIL {name}: file {want[:16]} != arena {got[:16]}")
    return report
