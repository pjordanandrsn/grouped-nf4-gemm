# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The reader's queue depth: how the default is derived, and what it must not do.

``qd=4`` was measured optimal on a 12-core box sitting at load ~9.8, with 8 and
16 coming back worse. On an idle 32-vCPU L40S, against the same arena and the
same scattered pattern, that inverts: qd=8 and qd=16 read 12% and 15% faster.
So the constant was tuned to a CPU-starved regime, and the default now scales.

The property that matters most here is the one that keeps this from being a
regression: **below ~20 CPUs the new default must return exactly the old 4**,
because the starved-box measurement is the only evidence covering that region
and it says 4 wins there.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from nvme_arena import bake, load_index  # noqa: E402
from nvme_reader import ArenaReader, cpu_budget, default_qd  # noqa: E402
from nvme_residency import ColdTier  # noqa: E402
from test_nvme_arena import make_snapshot  # noqa: E402


@pytest.mark.parametrize("cpus,expect", [
    (1, 4), (2, 4), (4, 4), (6, 4), (8, 4), (12, 4), (16, 4),   # unchanged region
    (20, 5), (24, 6), (32, 8), (48, 12), (64, 16),
    (128, 16), (256, 16),                                        # clamped
])
def test_default_qd_curve(cpus, expect):
    assert default_qd(cpus) == expect


def test_default_qd_never_below_the_old_constant():
    """The floor is the whole safety argument: no host gets a SHALLOWER queue
    than it had before, so the change cannot regress one."""
    for cpus in range(1, 300):
        assert default_qd(cpus) >= 4


def test_default_qd_is_monotone_and_capped():
    """Monotone so a bigger box never gets a smaller queue, capped because the
    measurement stops at 16 -- extrapolating past the data is how the original
    constant got its reputation."""
    vals = [default_qd(c) for c in range(1, 300)]
    assert vals == sorted(vals)
    assert max(vals) == 16


def test_cpu_budget_is_positive():
    n = cpu_budget()
    assert isinstance(n, int) and n >= 1


def test_cpu_budget_prefers_cgroup_quota_over_host_cores(tmp_path, monkeypatch):
    """In a container with a CPU quota and no cpuset, sched_getaffinity reports
    the HOST's cores -- 256 on the RunPod L40S where the real budget was 27.2.
    Reading the quota is the difference between qd=16 and qd=6 there."""
    cg = tmp_path / "cpu.max"
    cg.write_text("2720000 100000\n")
    real_open = open

    def fake_open(path, *a, **k):
        if path == "/sys/fs/cgroup/cpu.max":
            return real_open(str(cg), *a, **k)
        return real_open(path, *a, **k)

    monkeypatch.setattr("builtins.open", fake_open)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(256)),
                        raising=False)
    assert cpu_budget() == 27          # not 256
    assert default_qd() == 6           # not 16


def test_cgroup_max_quota_falls_through_to_affinity(tmp_path, monkeypatch):
    """`cpu.max` reading "max <period>" means NO quota -- it must not be parsed
    as a number, which would raise and silently take the fallback anyway."""
    cg = tmp_path / "cpu.max"
    cg.write_text("max 100000\n")
    real_open = open

    def fake_open(path, *a, **k):
        if path.startswith("/sys/fs/cgroup/cpu"):
            if path == "/sys/fs/cgroup/cpu.max":
                return real_open(str(cg), *a, **k)
            raise OSError("no cgroup v1")
        return real_open(path, *a, **k)

    monkeypatch.setattr("builtins.open", fake_open)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(40)),
                        raising=False)
    assert cpu_budget() == 40
    assert default_qd() == 10


@pytest.fixture()
def arena(tmp_path):
    snap = tmp_path / "snap"
    make_snapshot(str(snap))
    out = tmp_path / "a.arena"
    bake(str(snap), str(out))
    return str(out)


def test_explicit_qd_still_wins(arena):
    """The default is a default. A caller that passes qd gets exactly it --
    otherwise tuning for a known host becomes impossible."""
    r = ArenaReader(arena, qd=3)
    try:
        assert r.qd == 3 and r._pool._max_workers == 3
    finally:
        r.close()


def test_reader_default_matches_helper(arena):
    r = ArenaReader(arena)
    try:
        assert r.qd == default_qd()
    finally:
        r.close()


def test_cold_tier_passes_qd_through(arena):
    t = ColdTier(arena, hot_rows=4, pinned=False, qd=7)
    assert t.reader.qd == 7


def test_cold_tier_default_is_the_scaled_one(arena):
    t = ColdTier(arena, hot_rows=4, pinned=False)
    assert t.reader.qd == default_qd()


def test_qd_zero_is_refused(arena):
    """A ThreadPoolExecutor(max_workers=0) raises far from the cause; refuse at
    the seam where the number came in."""
    with pytest.raises(ValueError, match="qd must be >= 1"):
        ArenaReader(arena, qd=0)
