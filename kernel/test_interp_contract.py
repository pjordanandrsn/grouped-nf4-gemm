# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Device-free contract + fuzz suite (Triton interpreter mode, CPU).

Runs anywhere Python + torch + triton import — no GPU, no bitsandbytes — so
CI and any would-be porter can validate kernel SEMANTICS before touching
silicon. Set by conftest/CI: TRITON_INTERPRET=1.

What it guards (the things an external contributor or a new backend will
break first):
  - decode gemv matches dequant_ref of the SAME bytes (self-consistent)
  - the M-tile prefill path matches too (V0 loop; V1 needs tl.gather)
  - nibble order + LUT are exactly as the reference decodes them
  - boundary shapes: N not a multiple of BLOCK_N, single group, many groups,
    min K, wide N, adversarial absmax
The GPU property suite (test_nf4_grouped.py) still owns bnb bit-exactness and
tensor-core numerics; this owns portability of the contract.
"""
import os
os.environ.setdefault("TRITON_INTERPRET", "1")

import pytest
import torch

from nf4_grouped import (BLOCKSIZE, dequant_ref, gemm_4bit_grouped)
from nf4_pack_ref import make_stack


def _ref(B, A, sizes, ids, acts, N, K):
    out = torch.empty(sum(sizes), N, dtype=torch.float32)
    row = 0
    for g, (m, e) in enumerate(zip(sizes, ids)):
        w = dequant_ref(B[e], A[e], N, K).float()          # [N,K]
        for _ in range(m):
            out[row] = w @ acts[row].float()
            row += 1
    return out


def _run(E, N, K, sizes, ids, seed=0):
    B, A = make_stack(E, N, K, seed=seed)
    T = sum(sizes)
    acts = torch.randn(T, K, dtype=torch.bfloat16, generator=torch.Generator().manual_seed(seed + 1))
    ids_t = torch.tensor(ids, dtype=torch.int32)
    out = gemm_4bit_grouped(acts, B, A, sizes, ids_t, prefill_variant=0)
    ref = _ref(B, A, sizes, ids, acts, N, K)
    rel = (out.float() - ref).abs().max() / ref.abs().max().clamp_min(1e-4)
    return rel.item()


# --- decode (all groups size 1) ---
@pytest.mark.parametrize("N,K", [(32, 128), (128, 256), (48, 64), (2048, 128)])
def test_decode_matches_reference(N, K):
    E = 4
    assert _run(E, N, K, [1, 1, 1], [1, 3, 0]) < 1e-2


# --- prefill (M-tile path, groups > 1) ---
@pytest.mark.parametrize("m", [4, 16, 65])
def test_prefill_matches_reference(m):
    assert _run(4, 128, 128, [m, 1], [2, 0]) < 1e-2


# --- boundary / adversarial shapes (the fuzz surface) ---
def test_N_not_multiple_of_block():
    assert _run(3, 130, 128, [1, 1], [0, 2]) < 1e-2   # N=130, ragged n-tile


def test_single_group():
    assert _run(2, 64, 64, [1], [1]) < 1e-2


def test_many_groups_same_expert():
    assert _run(2, 96, 128, [1] * 12, [0] * 12) < 1e-2  # 12 tokens, one expert


def test_min_k():
    assert _run(2, 64, 64, [1, 1], [0, 1]) < 1e-2       # K == BLOCKSIZE


@pytest.mark.parametrize("seed", range(6))
def test_fuzz_random_shapes(seed):
    g = torch.Generator().manual_seed(seed)
    E = int(torch.randint(1, 6, (1,), generator=g))
    N = int(torch.randint(1, 40, (1,), generator=g)) * 8      # 8..312
    K = int(torch.randint(1, 5, (1,), generator=g)) * BLOCKSIZE  # 64..256
    G = int(torch.randint(1, 5, (1,), generator=g))
    sizes = [int(torch.randint(1, 4, (1,), generator=g)) for _ in range(G)]
    ids = [int(torch.randint(0, E, (1,), generator=g)) for _ in range(G)]
    assert _run(E, N, K, sizes, ids, seed=seed) < 1e-2, (E, N, K, sizes, ids)


def test_adversarial_absmax():
    # one block with a huge outlier, rest tiny — stresses per-block scaling
    B, A = make_stack(2, 64, 128, seed=7)
    A[0, :, 0] *= 1e3
    acts = torch.randn(1, 128, dtype=torch.bfloat16)
    out = gemm_4bit_grouped(acts, B, A, [1], torch.tensor([0], dtype=torch.int32),
                            prefill_variant=0)
    w = dequant_ref(B[0], A[0], 64, 128).float()
    ref = w @ acts[0].float()
    assert (out[0].float() - ref).abs().max() / ref.abs().max().clamp_min(1e-4) < 1e-2
