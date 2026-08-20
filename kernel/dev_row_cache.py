# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""A device-resident cache of packed expert rows, shared across layers.

The residency engines already keep a device row buffer — ``slots64``, which
the gather copies wanted rows into and whose ``have`` address test skips a
row that is already there. But it is a POSITIONAL cache: row *i* holds
whatever expert routed to position *i* on the previous step. An expert that
is routed again at a different position misses, and its bytes cross PCIe a
second time while an identical copy sits on the device untouched.

This cache is keyed by ``(layer, expert)``. A row stays addressable as long
as it is resident, wherever the router puts it next, and it carries
:class:`~vram_slots.VramSlots`' reclaimable state so a row that has been
logically evicted is still a hit until something actually overwrites it.

It is shared across layers for the same reason :class:`ColdTier` is: one
device arena serving 92 engines, not 92 arenas. The key carries the layer
because slot addresses are reused across them, and an address alone has
never identified its contents — the lesson ``Mxfp4NvmeResidency._forget``
already exists to enforce.

**What it trades.** A hit that today would be skipped outright (same expert,
same position) is still skipped: the address the engine writes is stable, so
``have`` still matches. A hit at a NEW position becomes a device-to-device
copy instead of a PCIe read. A miss costs one extra device-side write —
host to cache, then cache to slot — against a PCIe transfer it does not
avoid. So the cache is a bet that re-routing to a new position is common
enough to pay for the miss overhead, which is a measurement, not a
guarantee, and ``stats()`` reports both sides of it.
"""

from __future__ import annotations

import torch

from vram_slots import VramSlots


class StepTag:
    """A CUDA event plus proof that it was recorded.

    ``torch.cuda.Event.query()`` returns True for an event that was NEVER
    recorded — there is nothing outstanding, so nothing is pending. That is
    the wrong answer here: a step that raised between :meth:`DevRowCache.want`
    and its gather would look *complete* and release slots whose readers had
    not run, which is precisely the "hand a live row to another expert" bug
    the slot state machine exists to prevent. The flag distinguishes
    "finished" from "never started".
    """

    __slots__ = ("ev", "recorded")

    def __init__(self, device=None):
        self.ev = (torch.cuda.Event() if torch.cuda.is_available()
                   and (device is None or torch.device(device).type == "cuda")
                   else None)
        self.recorded = False

    def record(self) -> None:
        if self.ev is not None:
            self.ev.record()
        self.recorded = True

    def sync(self) -> None:
        """Block until this step's readers are done. The stall path only."""
        if self.ev is not None and self.recorded:
            self.ev.synchronize()

    def done(self) -> bool:
        return self.recorded and (self.ev is None or self.ev.query())


class DevRowCache:
    """``rows`` device rows of ``row_stride`` bytes, keyed by (layer, expert).

    Args:
        rows: physical device rows. Must exceed the largest routed set the
            engines will ask for, and by enough headroom that the previous
            step's rows can still be retiring when the next step asks — see
            :meth:`want`'s stall path for what happens when it does not.
        row_stride: the arena's PADDED row size. Never ``row_bytes``: the
            tier's own buffer strides by ``row_stride``, and a cache that
            strided by the unpadded size would start reading mid-row on slot
            1 and never fail loudly.
        protected: rows the cache will not demote. Defaults to half, which
            makes the other half the reclaimable ghost set. ``rows`` (the
            :class:`VramSlots` default) would leave nothing demotable and
            deadlock the allocator, so this class does NOT inherit it.
    """

    def __init__(self, rows: int, row_stride: int, *, device="cuda",
                 protected: int | None = None):
        rows, row_stride = int(rows), int(row_stride)
        if rows < 2:
            raise ValueError("rows must be >= 2: with one row every request "
                             "evicts the row it is about to read")
        if protected is None:
            protected = max(1, rows // 2)
        if protected >= rows:
            raise ValueError(
                f"protected={protected} must be < rows={rows}. At rows the "
                f"set of demotable slots is empty, so the first request that "
                f"misses on a full cache has nowhere to land — VramSlots' own "
                f"default is deliberately not inherited here.")
        self.rows, self.row_stride = rows, row_stride
        self.buf = torch.empty(rows * row_stride, dtype=torch.uint8,
                               device=device)
        self.base = self.buf.data_ptr()
        self.slots = VramSlots(rows, protected=protected)
        self.stalls = 0          # want() had to block on a previous step
        self.filled = 0          # rows written host -> cache

    def rowview(self) -> torch.Tensor:
        return self.buf.view(self.rows, self.row_stride)

    def addr(self, slot: int) -> int:
        return self.base + slot * self.row_stride

    def want(self, layer: int, experts, tag: StepTag, prev: StepTag | None):
        """Resolve ``experts`` to cache slots for one step of one layer.

        Returns ``(assign, need)`` with ``assign`` keyed by the ORIGINAL
        expert ids, not the internal ``(layer, expert)`` keys.

        Two things happen before the resolve. Slots retiring under a settled
        event are released, and — if the allocator still cannot find room —
        the previous step is WAITED ON and the release retried. That stall is
        a real synchronization and is counted, because the alternative
        (raising, or overwriting a row whose reader may still be running) is
        either a crash or a correctness bug. A nonzero ``stalls`` means the
        arena is too small for the routed set plus one step of pipelining;
        it does not mean anything is wrong.
        """
        if prev is not None:
            self.slots.settle(lambda t: t.done())
        keys = [(layer, e) for e in experts]
        try:
            assign, need = self.slots.want(keys, event_tag=tag)
        except RuntimeError:
            if prev is None:
                raise
            prev.sync()
            self.stalls += 1
            self.slots.settle(lambda t: t.done())
            assign, need = self.slots.want(keys, event_tag=tag)
        return ({e: assign[(layer, e)] for e in experts},
                [k[1] for k in need])

    def note_filled(self, n: int) -> None:
        self.filled += int(n)

    def stats(self) -> dict:
        s = dict(self.slots.stats())
        s.update({"rows": self.rows, "row_stride": self.row_stride,
                  "stalls": self.stalls, "host_to_cache_rows": self.filled,
                  "bytes": self.rows * self.row_stride})
        return s
