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
        """Idempotent. A tag names ONE point -- the gather that read the rows
        it was issued for -- so recording it twice is never right. A second
        record() moves the event forward on the stream, and rows retiring
        under it then wait on a position that keeps receding; every repeat
        pushes it again (Bugbot, gnf4#131)."""
        if self.recorded:
            return
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
        routed: the largest routed set a single ``want`` will ask for --
            the engine's ``k``. Used only to pick ``protected``.
        protected: rows the cache will not demote. Defaults to
            ``rows - routed``: the demotable margin only has to absorb ONE
            request, and every row beyond that margin is retention the cache
            is paying VRAM for.

            The old default of ``rows // 2`` threw half of it away, and it
            was the single largest thing wrong with this class. Replayed
            against a captured OLMoE decode trace at 384 rows, the default
            made 49,708 fills where the same cache with a one-request margin
            made 35,401 and an ideal LRU made 26,613
            (`bench/cold-engine/routing-trace`). At the smallest size it was
            worse than not having the cache at all. ``rows`` itself (the
            :class:`VramSlots` default) leaves nothing demotable and
            deadlocks the allocator, which is why that one is not inherited
            either.
    """

    def __init__(self, rows: int, row_stride: int, *, device="cuda",
                 routed: int = 8, protected: int | None = None):
        rows, row_stride = int(rows), int(row_stride)
        if rows < 2:
            raise ValueError("rows must be >= 2: with one row every request "
                             "evicts the row it is about to read")
        if protected is None:
            protected = max(1, rows - int(routed))
        if protected >= rows:
            raise ValueError(
                f"protected={protected} must be < rows={rows}. At rows the "
                f"set of demotable slots is empty, so the first request that "
                f"misses on a full cache has nowhere to land — VramSlots' own "
                f"default is deliberately not inherited here.")
        # Surfaced on the cache, not reached for through .slots: the engine's
        # sizing guard needs it, and a guard that reaches into another
        # object's internals raises AttributeError at construction instead of
        # the message it was written to give.
        self.rows, self.row_stride, self.protected = rows, row_stride, protected
        self.buf = torch.empty(rows * row_stride, dtype=torch.uint8,
                               device=device)
        self.base = self.buf.data_ptr()
        self.slots = VramSlots(rows, protected=protected)
        self.stalls = 0          # want() had to block on a previous step
        self.filled = 0          # rows written host -> cache
        self.abandoned = 0       # steps that never reached their record()
        # Rows one decode STEP asks for, learned from what actually arrives:
        # every layer contributes its routed set, so this is layers x k. A
        # cache smaller than that cannot retain ANYTHING across steps -- it is
        # evicted before its own next request -- and then the extra
        # host->cache write per miss makes it worse than the positional cache
        # already in the engine.
        #
        # `rows >= per_step` is NECESSARY BUT NOT SUFFICIENT. Below one step
        # the cache has never won: 36 of 36 configurations across three models
        # and four prompts. AT exactly one step it wins on OLMoE (top-8) and
        # Granite (top-8) but LOSES on all four Qwen1.5-MoE prompts (top-4),
        # which crosses two to three rows higher and then plateaus at
        # per_step + top_k. An earlier version of this comment claimed the
        # rule separated helped from lost 24 of 24; that was two models, and
        # a preregistered third refuted it
        # (bench/cold-engine/routing-trace/RESULTS-third-model.md).
        #
        # So size ABOVE one step, not at it. `too_small_to_retain` below still
        # means what it says -- it flags the regime where retention is
        # impossible -- but its absence does not promise the cache wins.
        # Accumulated for the CURRENT step only. The engines walk layers in
        # ASCENDING order once per step, so a layer index that does not
        # increase is a new step -- which covers both a layer repeating and a
        # layer that was silent becoming active again with a lower index than
        # the last one seen. Watching only for repeats missed the second case
        # and folded two steps into one; summing each layer's last-seen count
        # instead kept a silent layer in the total forever. Both reported
        # historical demand as one step (Bugbot, gnf4#165, twice).
        #
        # The residency engine skips want() entirely for a layer with no cold
        # experts, so which layers appear varies step to step and neither a
        # fixed layer count nor a repeat test is safe.
        self._cur: dict = {}
        self._last_layer = None
        self._per_step = None            # last COMPLETED step
        self._per_step_max = 0
        # The PREVIOUS step is a property of the CACHE, not of any one engine.
        # Tracking it per-engine meant every engine after the first started
        # with prev=None, so it neither settled nor stall-waited, and a shared
        # arena could not evict another layer's working set -- the second
        # layer to touch a full cache just raised (Bugbot, gnf4#131).
        self._last = None

    def rowview(self) -> torch.Tensor:
        return self.buf.view(self.rows, self.row_stride)

    def addr(self, slot: int) -> int:
        return self.base + slot * self.row_stride

    def want(self, layer: int, experts, tag: StepTag):
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
        prev = self._last
        if prev is not None and not prev.recorded:
            # The step that owned it never reached its record() -- it raised.
            # Its gather has either already run or will never run, so nothing
            # is still reading those rows. Recording now marks the stream
            # position past that step's work, which is what settle() needs;
            # leaving it unrecorded would retire those rows permanently.
            prev.record()
            self.abandoned += 1
        if prev is not None:
            self.slots.settle(lambda t: t.done())
        if self._cur and self._last_layer is not None \
                and layer <= self._last_layer:     # a step just ended
            done = sum(self._cur.values())
            self._per_step = done
            self._per_step_max = max(self._per_step_max, done)
            self._cur = {}
        self._cur[layer] = len(set(experts))
        self._last_layer = layer
        keys = [(layer, e) for e in experts]
        try:
            assign, need = self.slots.want(keys, event_tag=tag)
        except RuntimeError:
            if prev is None:
                raise
            # Every DISTINCT pending tag, not just the most recent one: the
            # rows blocking this request may be retiring under an older step,
            # and syncing only `_last` leaves them stuck.
            for t in {id(t): t for t in self.slots.pending_tags()}.values():
                t.sync()
            prev.sync()
            self.stalls += 1
            self.slots.settle(lambda t: t.done())
            assign, need = self.slots.want(keys, event_tag=tag)
        self._last = tag
        return ({e: assign[(layer, e)] for e in experts},
                [k[1] for k in need])

    def discard(self, layer: int, experts) -> int:
        """Unpublish rows a failed fill left mapped but not written."""
        return self.slots.discard([(layer, e) for e in experts])

    def note_filled(self, n: int) -> None:
        self.filled += int(n)

    def stats(self) -> dict:
        s = dict(self.slots.stats())
        # The last COMPLETED step. None until one has completed, so a cache
        # driven for half a step does not report a ratio against a partial
        # count -- and never a ratio against zero.
        per_step = self._per_step
        s.update({"rows": self.rows, "row_stride": self.row_stride,
                  "stalls": self.stalls, "host_to_cache_rows": self.filled,
                  "abandoned_steps": self.abandoned,
                  "bytes": self.rows * self.row_stride,
                  "per_step_rows": per_step,
                  "per_step_rows_max": self._per_step_max or None,
                  "steps_held": (self.rows / per_step) if per_step else None,
                  # Judged against the WORST step seen, not the last one:
                  # capacity that cannot hold the heaviest step retains
                  # nothing across it, whatever the average does.
                  "too_small_to_retain": (self._per_step_max > self.rows)
                                         if self._per_step_max else None})
        return s
