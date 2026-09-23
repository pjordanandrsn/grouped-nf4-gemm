# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""CPU checks of lane B393's census instrument (kernel/b393_bitwise_census.py), so CI proves the ruler
before a GPU reads anything with it: the bf16 ULP distance is exact, the reference chains are the
ones experts4bit-qlora runs, and the sequential reading really is slot order."""
from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b393_bitwise_census as C  # noqa: E402


def _bf16(*vals):
    return torch.tensor(vals, dtype=torch.float32).to(torch.bfloat16)


def test_ulp_distance_counts_representable_values():
    one = _bf16(1.0)
    nxt = torch.nextafter(one.float(), torch.tensor(2.0)).to(torch.bfloat16)   # rounds to 1.0 in bf16
    step = _bf16(1.0 + 2 ** -7)                                                # the next bf16 above 1.0
    assert C.ulp_stats(one, one)["max_ulp"] == 0 and C.ulp_stats(one, one)["equal"]
    assert C.ulp_stats(nxt, one)["max_ulp"] == 0
    assert C.ulp_stats(step, one)["max_ulp"] == 1 and not C.ulp_stats(step, one)["equal"]
    assert C.ulp_stats(_bf16(1.0 + 2 ** -6), one)["max_ulp"] == 2


def test_ulp_distance_is_ordered_across_zero_and_sign():
    tiny = torch.tensor([1], dtype=torch.int16).view(torch.bfloat16)           # smallest positive subnormal
    ntiny = torch.tensor([-32767], dtype=torch.int16).view(torch.bfloat16)     # its negative (0x8001)
    pz, nz = _bf16(0.0), _bf16(-0.0)
    assert C.ulp_stats(pz, nz)["max_ulp"] == 0                                 # +0 and -0: no value between
    assert C.ulp_stats(tiny, pz)["max_ulp"] == 1
    assert C.ulp_stats(tiny, ntiny)["max_ulp"] == 2
    assert C.ulp_stats(_bf16(-1.0), _bf16(1.0))["max_ulp"] == 2 * int(C.bf16_keys(_bf16(1.0)))


def test_ulp_stats_counts_differing_elements_and_refuses_mismatches():
    a, b = _bf16(1.0, 2.0, 3.0), _bf16(1.0, 2.0 + 2 ** -6, 3.0)
    s = C.ulp_stats(a, b)
    assert (s["n"], s["n_diff"], s["max_ulp"], s["equal"]) == (3, 1, 1, False)
    with pytest.raises(ValueError):
        C.ulp_stats(a, a.float())


def test_combine_chain_is_the_unfused_path_verbatim():
    T, k, H = 3, 4, 5
    dn, w = C.combine_inputs(T, k, H, seed=0, heavy=False, dev="cpu")
    assert dn.dtype == torch.bfloat16 and w.dtype == torch.float32
    assert torch.allclose(w.view(T, k).sum(-1), torch.ones(T))                 # normalised per token
    want = (dn.to(torch.float32) * w[:, None]).view(T, k, H).sum(dim=1).to(torch.bfloat16)
    assert torch.equal(C.combine_chain(dn, w, T, k, H), want)


def test_sequential_readings_add_in_slot_order():
    # 2^24 + 1 + 1 in fp32 depends on the order: sequential keeps neither 1 (each add rounds to 2^24)
    big = torch.tensor([[2.0 ** 24], [1.0], [1.0]], dtype=torch.float32)       # sk=3, R=1, N=1
    seq = C.reduce_sequential(big, 3, 1, 1)
    assert float(seq) == 2.0 ** 24
    dn = torch.tensor([[2.0 ** 24], [1.0], [1.0]]).to(torch.bfloat16)
    w = torch.ones(3)
    assert float(C.combine_sequential(dn, w, 1, 3, 1)) == 2.0 ** 24


def test_heavy_variant_scales_a_few_entries_only():
    dn, _ = C.combine_inputs(2, 8, 2048, seed=1, heavy=True, dev="cpu")
    base, _ = C.combine_inputs(2, 8, 2048, seed=1, heavy=False, dev="cpu")
    changed = int((dn != base).sum())
    assert 1 <= changed <= dn.numel() // 500


def test_census_refuses_without_cuda(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert C.run(str(tmp_path / "x.json")) == 3
    assert not (tmp_path / "x.json").exists()
