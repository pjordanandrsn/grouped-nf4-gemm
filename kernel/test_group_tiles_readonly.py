"""`build_group_tiles` must not mutate its `sizes` argument.

Found by the first caller that ran two GEMMs off one sizes tensor (gate/up/down
in one MoE layer): the second call asserted `sum(sizes) == T` with sizes zeroed.
Cause was aliasing, not arithmetic — `enumerate` over a tensor yields 0-dim
views, so `left = m; left -= take` subtracted in place. A list caller never saw
it because `-=` on an int rebinds.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))


def _build():
    # import lazily: nf4_grouped imports triton, absent on the control box
    triton = pytest.importorskip("triton")  # noqa: F841
    from nf4_grouped import build_group_tiles
    return build_group_tiles


@pytest.mark.parametrize("sizes", [[4, 4], [1, 1, 1], [7, 2], [16]])
def test_tensor_sizes_survive_the_call(sizes):
    build_group_tiles = _build()
    t = torch.tensor(sizes, dtype=torch.int32)
    before = t.clone()
    build_group_tiles(t, 16, "cpu")
    assert torch.equal(t, before), f"sizes mutated: {before.tolist()} -> {t.tolist()}"


def test_two_calls_off_one_tensor_agree():
    """The actual failure shape: reuse must be idempotent."""
    build_group_tiles = _build()
    t = torch.tensor([4, 4], dtype=torch.int32)
    a = build_group_tiles(t, 16, "cpu")
    b = build_group_tiles(t, 16, "cpu")
    for x, y in zip(a, b):
        assert torch.equal(x, y)


def test_list_and_tensor_sizes_give_identical_tiles():
    """A tensor caller and a list caller must get the same tiling."""
    build_group_tiles = _build()
    for x, y in zip(build_group_tiles([5, 3], 2, "cpu"),
                    build_group_tiles(torch.tensor([5, 3], dtype=torch.int32), 2, "cpu")):
        assert torch.equal(x, y)


# --- the twin-build memo (one entry, list callers only) ----------------------

def test_list_twin_call_is_a_cache_hit():
    """gate_up and down run off the same grouping: the second build must
    return the SAME tensors, not equal copies."""
    build_group_tiles = _build()
    a = build_group_tiles([5, 3], 16, "cpu")
    b = build_group_tiles([5, 3], 16, "cpu")
    for x, y in zip(a, b):
        assert x is y


def test_memo_key_snapshots_the_sizes():
    """Mutating the caller's list after a build must not corrupt a later
    call: the key is a tuple snapshot, so the mutated list misses and
    rebuilds correctly."""
    build_group_tiles = _build()
    sizes = [5, 3]
    a = build_group_tiles(sizes, 16, "cpu")
    sizes[0] = 7
    b = build_group_tiles(sizes, 16, "cpu")
    ref = _build()([7, 3], 16, "cpu")
    for x, y in zip(b, ref):
        assert torch.equal(x, y)
    # and the original values still build the original tiles
    c = build_group_tiles([5, 3], 16, "cpu")
    for x, y in zip(a, c):
        assert torch.equal(x, y)


def test_memo_misses_on_block_m():
    build_group_tiles = _build()
    a = build_group_tiles([5, 3], 16, "cpu")
    b = build_group_tiles([5, 3], 4, "cpu")
    assert a[0].numel() != b[0].numel() or not torch.equal(a[0], b[0])


def test_tensor_sizes_bypass_the_memo():
    """Tensor callers keep pre-memo behavior: 0-dim views are
    identity-hashed, so a value key would need a hidden D2H."""
    build_group_tiles = _build()
    t = torch.tensor([5, 3], dtype=torch.int32)
    a = build_group_tiles(t, 16, "cpu")
    b = build_group_tiles(t, 16, "cpu")
    for x, y in zip(a, b):
        assert torch.equal(x, y)
        assert x is not y
