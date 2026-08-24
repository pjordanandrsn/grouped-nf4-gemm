# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""K2 (PREREG-k2-vectorized-nibbles): the vectorized dual-nibble
mainloop must be BITWISE-equal to the legacy per-element path at any
fixed config -- same nibbles, same lanes, same LUT gather. Interp-mode
tripwire (the on-box CUDA bitwise arm is the real gate; interp mode is
known to mask int-width bugs, so this test guards the ALGEBRA, not the
codegen)."""

import os

import pytest
import torch

pytest.importorskip("triton", reason="interp-mode kernels need triton")
os.environ.setdefault("TRITON_INTERPRET", "1")

import nf4_grouped  # noqa: E402


@pytest.mark.skipif(not nf4_grouped.HAS_TL_INTERLEAVE,
                    reason="tl.interleave absent on this triton")
@pytest.mark.parametrize("sk", [1, 4])
def test_vec_loads_bitwise_vs_legacy(sk, monkeypatch):
    torch.manual_seed(9)
    E, N, K, T = 4, 128, 256, 4
    packed = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8)
    absmax = torch.rand(E, N, K // 64) + 0.5
    a = torch.randn(T, K, dtype=torch.bfloat16)
    eids = torch.arange(E, dtype=torch.int32)[:T]
    sizes = [1] * T
    monkeypatch.setenv("GNF4_GEMV_SCALAR_LOADS", "1")
    legacy = nf4_grouped.gemm_4bit_grouped(
        a, packed, absmax, sizes, eids, decode_config=(64, 2), split_k=sk)
    monkeypatch.delenv("GNF4_GEMV_SCALAR_LOADS")
    vec = nf4_grouped.gemm_4bit_grouped(
        a, packed, absmax, sizes, eids, decode_config=(64, 2), split_k=sk)
    assert torch.equal(legacy, vec)
