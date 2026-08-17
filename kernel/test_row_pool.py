# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""N4 RowPool contract: append-only partitioned rows, zero-copy resident
runs, copy-on-demote with publish-after-drain, and the byte identity that
makes it a tier (what you wrote is what every view reads, wherever the row
currently lives)."""

import os
import sys

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, os.path.dirname(__file__))

from row_pool import RowPool  # noqa: E402

P, DEV_ROWS, HOST_ROWS, RB = 3, 4, 16, 256


def _fill(view, seed):
    g = torch.Generator().manual_seed(seed)
    view.copy_(torch.randint(0, 256, (view.numel(),), generator=g,
                             dtype=torch.uint8).view_as(view))


def _expect(seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (RB,), generator=g, dtype=torch.uint8)


@pytest.fixture(params=["cpu"] + (["cuda"] if torch.cuda.is_available()
                                  else []))
def pool(request):
    return RowPool(P, DEV_ROWS, HOST_ROWS, RB, device=request.param)


def test_append_write_read_roundtrip(pool):
    for p in range(P):
        for i in range(3):
            idx, view = pool.append(p)
            assert idx == i
            _fill(view, seed=100 * p + i)
    for p in range(P):
        for i in range(3):
            got = pool.row_view(p, i).cpu()
            assert torch.equal(got, _expect(100 * p + i))


def test_resident_run_is_a_zero_copy_view(pool):
    for i in range(3):
        _, v = pool.append(0)
        _fill(v, seed=i)
    run = pool.run_view(0, 0, 3)
    assert run.shape == (3, RB)
    assert run.data_ptr() == pool.dev[0].data_ptr(), \
        "unwrapped resident run must be a VIEW of the device pool"
    for i in range(3):
        assert torch.equal(run[i].cpu(), _expect(i))


def test_window_full_is_a_clean_error_not_an_overwrite(pool):
    for i in range(DEV_ROWS):
        _, v = pool.append(1)
        _fill(v, seed=i)
    with pytest.raises(RuntimeError, match="window full"):
        pool.append(1)
    # the resident bytes were not disturbed by the refused append
    for i in range(DEV_ROWS):
        assert torch.equal(pool.row_view(1, i).cpu(), _expect(i))


def test_demote_settle_migrates_source_of_truth(pool):
    side = (torch.cuda.Stream() if pool.device.type == "cuda" else None)
    for i in range(DEV_ROWS):
        _, v = pool.append(2)
        _fill(v, seed=1000 + i)
    n = pool.demote_head(2, 2, stream=side)
    assert n == 2
    # rows stay device-readable until settle observes the copy complete
    assert pool.resident_run(2) == (0, DEV_ROWS)
    if side is not None:
        side.synchronize()
    settled = pool.settle()
    assert settled == 2
    assert pool.resident_run(2) == (2, DEV_ROWS)
    host = pool.host_run(2, 0, 2)
    for i in range(2):
        assert torch.equal(host[i].cpu(), _expect(1000 + i)), \
            "host copy must be byte-identical to what was written"
    # the freed window admits new appends, and the ring wraps correctly
    for i in range(2):
        idx, v = pool.append(2)
        _fill(v, seed=2000 + i)
        assert idx == DEV_ROWS + i
    run = pool.run_view(2, 2, DEV_ROWS + 2)      # wraps: gathered copy
    assert run.shape == (DEV_ROWS, RB)
    assert torch.equal(run[-2].cpu(), _expect(2000))
    assert torch.equal(run[-1].cpu(), _expect(2001))


def test_reads_outside_contracts_are_refused(pool):
    pool.append(0)
    with pytest.raises(KeyError):
        pool.row_view(0, 5)
    with pytest.raises(KeyError):
        pool.run_view(0, 0, 2)
    with pytest.raises(KeyError):
        pool.host_run(0, 0, 1)                   # nothing settled
    p2 = RowPool(1, 2, 0, RB, device=str(pool.device))
    p2.append(0)
    with pytest.raises(RuntimeError, match="host_rows=0"):
        p2.demote_head(0, 1)


def test_stats_vocabulary(pool):
    _, v = pool.append(0)
    _fill(v, 1)
    s = pool.stats()
    for k in ("appends", "demotions", "settled", "host_reads",
              "host_read_bytes", "device_resident_rows"):
        assert k in s
    assert s["appends"] == 1 and s["device_resident_rows"] == 1
