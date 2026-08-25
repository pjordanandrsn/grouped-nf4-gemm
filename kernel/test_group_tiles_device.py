# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Gate 1 of e4b PREREG-s3-grouped-verify: the device tile builder must
reproduce the host builder exactly on identical routing.

Runs on CPU (pure tensor ops), so the grouping half of a captured T>1
MoE is verified before a box is rented. What CANNOT be checked here is
capture-legality itself -- that is an on-box gate -- but every op used
is in-stream by construction: argsort, scatter_add_, cumsum,
searchsorted, index_select, where. No .item(), no tolist(), no
data-dependent shape.
"""

import pytest
import torch

from nf4_grouped import build_group_tiles, build_group_tiles_device


def _host_reference(counts, block_m, device="cpu"):
    """The host builder, fed sizes in expert-major order."""
    return build_group_tiles(list(counts), block_m, device)


def _live(row0, rows, grp):
    """Drop the zero-row padding slots the device builder emits."""
    keep = rows != 0
    return row0[keep], rows[keep], grp[keep]


def _ids_from_counts(counts):
    out = []
    for e, c in enumerate(counts):
        out.extend([e] * int(c))
    return torch.tensor(out, dtype=torch.int64)


@pytest.mark.parametrize("block_m", [16, 32, 64])
@pytest.mark.parametrize("counts", [
    [1] * 8,                       # the T=1 decode shape
    [3, 0, 5, 0, 0, 1, 0, 2],      # zeros interleaved
    [0, 0, 0, 0],                  # every expert empty
    [17, 1, 48, 0, 33],            # M_e > BM, several tiles
    [128],                         # one expert, many tiles
    [64, 64],                      # exact multiples of BM at 64
])
def test_device_tiles_match_host(counts, block_m):
    ids = _ids_from_counts(counts)
    r0, rw, gp, order, dcounts = build_group_tiles_device(
        ids, len(counts), block_m)
    assert torch.equal(dcounts.cpu(),
                       torch.tensor(counts, dtype=torch.int64)), dcounts
    hr0, hrw, hgp = _host_reference(counts, block_m)
    lr0, lrw, lgp = _live(r0, rw, gp)
    assert torch.equal(lr0, hr0), (lr0, hr0)
    assert torch.equal(lrw, hrw), (lrw, hrw)
    assert torch.equal(lgp, hgp), (lgp, hgp)


def test_budget_always_suffices():
    """ceil(R/BM) + E must cover sum(ceil(c_e/BM)) for any split --
    if it ever did not, tiles would be silently DROPPED and rows would
    vanish from the product."""
    torch.manual_seed(0)
    for _ in range(200):
        e = int(torch.randint(1, 12, (1,)))
        counts = torch.randint(0, 40, (e,)).tolist()
        for block_m in (16, 32, 64):
            r = sum(counts)
            budget = -(-r // block_m) + e
            need = sum(-(-c // block_m) for c in counts)
            assert need <= budget, (counts, block_m, need, budget)


def test_order_is_expert_major_and_invertible():
    """The gather permutation must sort rows by expert and invert
    exactly -- a scatter-back that is not the inverse silently pairs
    each token with another token's expert output."""
    counts = [3, 0, 5, 1, 2]
    ids = _ids_from_counts(counts)
    perm = torch.randperm(ids.numel())
    shuffled = ids[perm]
    _r0, _rw, _gp, order, _c = build_group_tiles_device(
        shuffled, len(counts), 16)
    assert torch.equal(shuffled[order].sort().values, shuffled[order]), \
        "gathered ids are not expert-major"
    inv = torch.argsort(order)
    assert torch.equal(shuffled[order][inv], shuffled), "not invertible"


def test_padding_slots_are_inert():
    """Slots past the live tile count must carry rows=0 (the kernel's
    m_mask no-ops them) AND a legal row0/group, since the kernel still
    dereferences those pointers before masking."""
    counts = [1, 1]
    r0, rw, gp, _o, _c = build_group_tiles_device(
        _ids_from_counts(counts), 2, 16)
    pad = rw == 0
    assert pad.any(), "budget should leave padding slots here"
    assert (r0[pad] >= 0).all() and (gp[pad] >= 0).all()
    assert (gp[pad] < 2).all(), "padding group id out of range"


def test_no_host_sync_ops_in_the_builder():
    """The point of this function is capture-legality. Guard the source
    against the ops that would break it -- a future edit that reaches
    for .item()/tolist() would pass every numeric test above while
    making captured grouping impossible again."""
    import inspect

    src = inspect.getsource(build_group_tiles_device)
    body = src.split('"""')[-1]          # skip the docstring
    for banned in (".item()", ".tolist()", "unique_consecutive",
                   "int(", "if counts", "for e in"):
        assert banned not in body, f"{banned} would break capture"


def test_captured_wrapper_static_and_syncfree():
    """The captured wrapper's reason to exist is capture-legality:
    static launch shape, no host reads. Source guard, same rationale as
    the builder's."""
    import inspect

    from nf4_grouped import gemm_4bit_grouped_captured

    src = inspect.getsource(gemm_4bit_grouped_captured)
    body = src.split('"""')[-1]
    for banned in (".item()", ".tolist()", "unique_consecutive",
                   "max(sizes)", ".cpu()"):
        assert banned not in body, f"{banned} would break capture"
