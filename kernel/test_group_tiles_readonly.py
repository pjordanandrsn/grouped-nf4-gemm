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
