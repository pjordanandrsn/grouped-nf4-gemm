# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Phase-3 gates: the loader is byte-preserving and the provenance primitive
detects change. Device-free (CPU) on a synthetic safetensors fixture shaped
like a gpt-oss layer; the real-checkpoint hash table is generated on the pod
(Phase 6) where the model is already resident."""
import struct

import pytest
import torch

pytest.importorskip("safetensors")
from safetensors.torch import save_file  # noqa: E402

from mxfp4_loader import (  # noqa: E402
    file_tensor_sha256, layer_expert_names, provenance_table, tensor_sha256,
    to_kernel_shapes, verify_arena_matches,
)


def _fixture(tmp_path, E=4, N_gu=64, N_dn=32, n_blk=4, layers=(0, 1)):
    """A tiny checkpoint with gpt-oss expert-tensor names/shapes/dtypes."""
    g = torch.Generator().manual_seed(0)
    tensors = {}
    for L in layers:
        nm = layer_expert_names(L)
        tensors[nm["mlp.experts.gate_up_proj_blocks"]] = torch.randint(
            0, 256, (E, N_gu, n_blk, 16), generator=g, dtype=torch.uint8)
        tensors[nm["mlp.experts.gate_up_proj_scales"]] = torch.randint(
            0, 256, (E, N_gu, n_blk), generator=g, dtype=torch.uint8)
        tensors[nm["mlp.experts.down_proj_blocks"]] = torch.randint(
            0, 256, (E, N_dn, n_blk, 16), generator=g, dtype=torch.uint8)
        tensors[nm["mlp.experts.down_proj_scales"]] = torch.randint(
            0, 256, (E, N_dn, n_blk), generator=g, dtype=torch.uint8)
    # a non-expert tensor, to prove the table only covers experts
    tensors["model.layers.0.self_attn.q_proj.weight"] = torch.randn(8, 8, generator=g)
    path = str(tmp_path / "fixture.safetensors")
    save_file(tensors, path)
    return path, tensors


def test_reshape_is_byte_preserving(tmp_path):
    """to_kernel_shapes must not reorder a single byte: the [E,N,nb,16]->
    [E,N,nb*16] view hashes identically to the source."""
    path, tensors = _fixture(tmp_path)
    nm = layer_expert_names(0)
    blocks = tensors[nm["mlp.experts.gate_up_proj_blocks"]]
    scales = tensors[nm["mlp.experts.gate_up_proj_scales"]]
    kb, ks = to_kernel_shapes(blocks, scales)
    assert kb.shape == (blocks.shape[0], blocks.shape[1], blocks.shape[2] * 16)
    assert tensor_sha256(kb) == tensor_sha256(blocks)      # zero reorder
    assert tensor_sha256(ks) == tensor_sha256(scales)


def test_arena_matches_file(tmp_path):
    """The provenance receipt: loaded arena bytes == file data-section bytes,
    for every expert tensor across layers."""
    path, tensors = _fixture(tmp_path)
    table = provenance_table(path, layers=(0, 1))
    assert table["n_tensors"] == 8          # 4 tensors x 2 layers, experts only
    # "load into the arena" = the kernel-shaped views of the same tensors
    loaded = {}
    for L in (0, 1):
        nm = layer_expert_names(L)
        gb, gs = to_kernel_shapes(tensors[nm["mlp.experts.gate_up_proj_blocks"]],
                                  tensors[nm["mlp.experts.gate_up_proj_scales"]])
        db, ds = to_kernel_shapes(tensors[nm["mlp.experts.down_proj_blocks"]],
                                  tensors[nm["mlp.experts.down_proj_scales"]])
        loaded[nm["mlp.experts.gate_up_proj_blocks"]] = gb
        loaded[nm["mlp.experts.gate_up_proj_scales"]] = gs
        loaded[nm["mlp.experts.down_proj_blocks"]] = db
        loaded[nm["mlp.experts.down_proj_scales"]] = ds
    report = verify_arena_matches(path, loaded)
    assert all(r["match"] for r in report.values())
    # the table's file hashes equal the arena hashes
    for name, h in table["hashes"].items():
        assert h == report[name]["arena"]


def test_provenance_detects_a_flipped_byte(tmp_path):
    """A single mutated byte must break the receipt (the hash is doing work)."""
    path, tensors = _fixture(tmp_path)
    nm = layer_expert_names(0)
    name = nm["mlp.experts.down_proj_blocks"]
    blocks = tensors[name].clone()
    blocks.view(-1)[123] ^= 0x01                    # flip one bit
    scales = tensors[nm["mlp.experts.down_proj_scales"]]
    kb, _ = to_kernel_shapes(blocks, scales)
    with pytest.raises(ValueError, match="PROVENANCE FAIL"):
        verify_arena_matches(path, {name: kb})


def test_file_sha_matches_torch_load(tmp_path):
    """The streamed file-range hash equals hashing the torch-loaded tensor —
    proves file_tensor_sha256 reads the right byte range."""
    from safetensors.torch import load_file
    path, tensors = _fixture(tmp_path)
    loaded = load_file(path)
    nm = layer_expert_names(1)
    for name in nm.values():
        assert file_tensor_sha256(path, name) == tensor_sha256(loaded[name])
