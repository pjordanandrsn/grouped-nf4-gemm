# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""``nf4_grouped.build_group_tiles_split`` (e4b#846, opt-in): the two tile tables of one grouping, read exactly.

Pure tensor ops on CPU, like ``test_group_tiles_device.py``. Guards: the tables name every sorted row exactly once; a
big tile always holds ``big_m`` rows of one expert and a small tile at most 16; each expert gets ``c // big_m`` big tiles
and ``ceil((c % big_m) / 16)`` small ones; the padded rows equal the 16-row table's (the split moves no padding); the
budgets are static (``R // big_m`` and ``ceil(R / 16) + E``) with ``rows = 0`` past the live tiles; ``order`` and
``counts`` are ``build_group_tiles_device``'s; a host walk of the same rule gives the same integers; bad tile heights are
refused.
"""
import math

import pytest
import torch

from nf4_grouped import build_group_tiles_device, build_group_tiles_split


def _ids(counts, seed=0):
    out = []
    for e, c in enumerate(counts):
        out.extend([e] * int(c))
    t = torch.tensor(out, dtype=torch.int64)
    g = torch.Generator().manual_seed(seed)
    return t[torch.randperm(t.numel(), generator=g)]       # unsorted, as a router emits them


def _host(counts, big_m, small_m=16):
    """The rule walked on the host: per expert in order, full big tiles, then small remainder tiles."""
    big, small, row = [], [], 0
    for e, c in enumerate(counts):
        nb = c // big_m
        for i in range(nb):
            big.append((row + i * big_m, big_m, e))
        left, r0 = c - nb * big_m, row + nb * big_m
        while left > 0:
            take = min(small_m, left)
            small.append((r0, take, e))
            r0 += take
            left -= take
        row += c
    return big, small


def _live(t):
    row0, rows, grp = (x.tolist() for x in t)
    return [(a, b, c) for a, b, c in zip(row0, rows, grp) if b]


COUNTS = [
    [400] + [24] * 90 + [0] * 37,              # the census's prefill shape: one heavy expert, a light tail
    [1] * 8,
    [64, 0, 128, 63, 65, 0, 1, 127],           # exact multiples and one short of them
    [3, 0, 5, 0, 0, 1, 0, 2],
    [512],                                     # one expert takes every row
    [0, 0, 0, 7],
]


@pytest.mark.parametrize("big_m", [32, 64])
@pytest.mark.parametrize("counts", COUNTS, ids=[f"E{len(c)}_R{sum(c)}" for c in COUNTS])
def test_split_tables_are_the_rule_exactly(counts, big_m):
    E, R = len(counts), sum(counts)
    ids = _ids(counts)
    big, small, order, cnt = build_group_tiles_split(ids, E, big_m)
    hb, hs = _host(counts, big_m)
    assert _live(big) == hb and _live(small) == hs
    assert big[0].numel() == max(1, R // big_m) and small[0].numel() == -(-R // 16) + E    # static budgets
    for t in (big, small):
        assert all(x.dtype == torch.int32 for x in t)
        live = t[1] != 0
        assert int(t[0][~live].abs().sum()) == 0 and int(t[2][~live].abs().sum()) == 0      # padding slots are zero
    # every sorted row exactly once, each tile inside its own expert
    seen = torch.zeros(R, dtype=torch.int64)
    sorted_ids = ids[order]
    for row0, rows, grp in hb + hs:
        seen[row0:row0 + rows] += 1
        assert bool((sorted_ids[row0:row0 + rows] == grp).all())
    assert bool((seen == 1).all())
    # the padding the kernel sees is the 16-row table's
    padded = big_m * len(hb) + 16 * len(hs)
    assert padded == 16 * sum(math.ceil(c / 16) for c in counts)
    # the same order and counts as the chained builder
    _r0, _rw, _g, order16, counts16 = build_group_tiles_device(ids, E, 16)
    assert torch.equal(order, order16) and torch.equal(cnt, counts16)


def test_the_heavy_expert_moves_to_tall_tiles():
    counts = [400] + [24] * 90 + [0] * 37
    big, small, _o, _c = build_group_tiles_split(_ids(counts), len(counts), 64)
    assert _live(big) == [(64 * i, 64, 0) for i in range(6)]                   # 384 of its 400 rows
    assert sum(1 for t in _live(small) if t[2] == 0) == 1                       # its last 16 rows
    assert len(_live(big)) + len(_live(small)) == 6 + 1 + 90 * 2 < sum(math.ceil(c / 16) for c in counts)


@pytest.mark.parametrize("big_m,small_m", [(16, 16), (24, 16), (8, 16)])
def test_bad_tile_heights_are_refused(big_m, small_m):
    with pytest.raises(ValueError, match="big_m"):
        build_group_tiles_split(_ids([5, 5]), 2, big_m, small_m)
