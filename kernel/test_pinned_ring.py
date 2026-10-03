# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The opt-in non-capture pinned ring behind ``to_device_i32`` (GNF4_PINNED_RING=1).

It exists to remove the host syncs of the pageable index transfers in steps whose
queue is NOT idle at those points (the fused training step on a fast card), without
changing a single value the kernels read. So the contract is: identical values,
no synchronizing call, and slot reuse that can never overwrite a copy still in flight.
"""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the ring stages to a CUDA device")


@pytest.fixture(autouse=True)
def _drop_rings():
    yield
    import nf4_grouped
    nf4_grouped._RINGS.clear()                        # never leak a resized ring into another test


def _fresh(monkeypatch, on=True, slots=None, slot_ints=None):
    monkeypatch.setenv("GNF4_PINNED_RING", "1" if on else "0")
    if slots is not None:
        monkeypatch.setenv("GNF4_PINNED_RING_SLOTS", str(slots))
    if slot_ints is not None:
        monkeypatch.setenv("GNF4_PINNED_RING_SLOT_INTS", str(slot_ints))
    import nf4_grouped
    nf4_grouped._RINGS.clear()                        # the ring reads its size env at construction
    return nf4_grouped


def test_ring_values_equal_the_pageable_path(monkeypatch):
    seqs = ([3, 1, 4, 1, 5], [9, 2, 6], list(range(40)))
    g = _fresh(monkeypatch, on=False)
    want = [t.cpu().tolist() for t in g.to_device_i32(seqs, "cuda")]
    g = _fresh(monkeypatch, on=True)
    got = g.to_device_i32(seqs, "cuda")
    torch.cuda.synchronize()
    assert [t.dtype for t in got] == [torch.int32] * 3
    assert [t.cpu().tolist() for t in got] == want
    assert g._ring(torch.device("cuda")).staged == 1


def test_the_ring_does_not_synchronize_and_the_instrument_would_see_it(monkeypatch):
    seqs = ([7, 8, 9], [1, 2])
    g = _fresh(monkeypatch, on=False)
    g.to_device_i32(seqs, "cuda")                  # warm: allocator, arena, CUDA context
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        with pytest.raises(RuntimeError, match="synchroniz"):
            g.to_device_i32(seqs, "cuda")          # control: the pageable build DOES sync
    finally:
        torch.cuda.set_sync_debug_mode("default")
    g = _fresh(monkeypatch, on=True)
    g.to_device_i32(seqs, "cuda")                  # warm the ring (pinned allocation)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        out = g.to_device_i32(seqs, "cuda")
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
    assert [t.cpu().tolist() for t in out] == [[7, 8, 9], [1, 2]]


def test_slot_reuse_never_overwrites_a_copy_in_flight(monkeypatch):
    g = _fresh(monkeypatch, on=True, slots=2)
    torch.cuda.synchronize()
    torch.cuda._sleep(200_000_000)                 # keep the stream busy so every copy is still queued
    outs = [g.to_device_i32(([i, i + 1, i + 2],), "cuda")[0] for i in range(12)]
    torch.cuda.synchronize()
    assert [o.cpu().tolist() for o in outs] == [[i, i + 1, i + 2] for i in range(12)]
    ring = g._ring(torch.device("cuda"))
    assert ring.staged == 12 and ring.waits >= 1, (ring.staged, ring.waits)   # wrapped onto in-flight slots and waited


def test_a_call_larger_than_a_slot_falls_back_to_the_pageable_build(monkeypatch):
    g = _fresh(monkeypatch, on=True, slot_ints=8)
    out = g.to_device_i32((list(range(20)),), "cuda")[0]
    torch.cuda.synchronize()
    assert out.cpu().tolist() == list(range(20))
    assert g._ring(torch.device("cuda")).staged == 0


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("GNF4_PINNED_RING", raising=False)
    import nf4_grouped
    assert nf4_grouped._pinned_ring_enabled() is False
