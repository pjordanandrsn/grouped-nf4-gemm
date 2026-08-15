"""Arena permanence semantics under CUDA graph capture.

Bugbot on PR #88 caught that the capture-conditional repair left the arena
unable to rewind OR grow: `take` runs only under capture (rewind disabled
there), so after one capture `off` sat at its high-water mark forever and a
second capture could refuse spuriously. The fix is PERMANENCE — captured
slices are owned for the life of the process because a graph's H2D nodes
re-read their pinned host slices on every replay — with growth and `reserve()`
legal outside capture.

These tests pin the three properties that make permanence correct:

  1. SEQUENTIAL CAPTURES WORK — the Bugbot scenario. Two captures in one
     process both succeed, and the second's slices sit ABOVE the first's.
  2. REPLAY INTEGRITY ACROSS LATER ACTIVITY — the latent hazard permanence
     closes. A graph captured first still replays its ORIGINAL indices after
     later uncaptured calls and a second capture; under the old rewind design
     those bytes could be overwritten and replays would read garbage.
  3. UNCAPTURED CALLS OWN NOTHING — the pageable path never advances `off`,
     so a process that never captures pays the arena nothing but its one-time
     allocation.
"""
from __future__ import annotations

import pytest
import torch

import nf4_grouped as NG

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="capture semantics need CUDA")


def _fresh_arena(n=256):
    dev = torch.device("cuda")
    NG._ARENAS[str(dev)] = NG._PinnedIndexArena(dev, n)
    return NG._ARENAS[str(dev)]


def _capture(fn):
    """Uncaptured warm-up (the capture discipline), then capture fn once."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    torch.cuda.synchronize()
    return g, out


def test_sequential_captures_in_one_process():
    ar = _fresh_arena(256)
    dst = torch.zeros(8, dtype=torch.int32, device="cuda")

    def step_a():
        (t,) = NG.to_device_i32(([1, 2, 3, 4],), "cuda")
        dst[:4].copy_(t)
        return dst

    def step_b():
        (t,) = NG.to_device_i32(([9, 8, 7],), "cuda")
        dst[4:7].copy_(t)
        return dst

    g1, _ = _capture(step_a)
    off_after_first = ar.off
    assert off_after_first > 0, "capture must consume arena bytes"
    g2, _ = _capture(step_b)
    assert ar.off > off_after_first, (
        "second capture must take FRESH bytes above the first's watermark")
    g1.replay(); g2.replay()
    torch.cuda.synchronize()
    assert dst[:7].tolist() == [1, 2, 3, 4, 9, 8, 7]


def test_replay_reads_original_indices_after_later_activity():
    _fresh_arena(256)
    dst = torch.zeros(4, dtype=torch.int32, device="cuda")

    def step():
        (t,) = NG.to_device_i32(([11, 22, 33, 44],), "cuda")
        dst.copy_(t)
        return dst

    g, _ = _capture(step)
    # Later activity that under a rewinding design could reclaim the slice:
    for _ in range(5):
        NG.to_device_i32(([5, 6, 7, 8],), "cuda")     # uncaptured (pageable)
    dst2 = torch.zeros(2, dtype=torch.int32, device="cuda")

    def step2():
        (t,) = NG.to_device_i32(([70, 71],), "cuda")
        dst2.copy_(t)
        return dst2

    g2, _ = _capture(step2)
    dst.zero_()
    g.replay()
    torch.cuda.synchronize()
    assert dst.tolist() == [11, 22, 33, 44], (
        "replay must read the ORIGINAL captured indices — its pinned slice is "
        "permanent and later activity may not clobber it")


def test_uncaptured_calls_own_nothing_and_reserve_grows_outside_capture():
    ar = _fresh_arena(64)
    for _ in range(10):
        NG.to_device_i32(([1] * 32,), "cuda")
    assert ar.off == 0, "pageable path must not consume the arena"
    ar.reserve(1024)
    assert ar.host.numel() >= 1024
    (t,) = NG.to_device_i32((list(range(48)),), "cuda")
    assert t.tolist() == list(range(48))


def test_capture_larger_than_arena_refuses_by_name():
    _fresh_arena(64)
    big = list(range(128))
    err = None
    try:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            NG.to_device_i32((big,), "cuda")
    except RuntimeError as e:
        err = str(e)
    assert err is not None and "pinned index arena" in err, (
        f"expected the named refusal, got: {err!r}")
