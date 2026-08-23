# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""SegmentedRowPool: the elastic transient pool, built so shrink() is REAL.

The monolithic DevRowCache buffer cannot return VRAM: freeing rows from one
flat allocation frees nothing. This pool is a list of DevRowCache SEGMENTS,
each owning its own buffer, so `shrink()` can event-gate a whole segment and
free its bytes back to the allocator -- which is what the spec's
`shrink(bytes) -> bytes_freed` (SPEC-elastic-phase2 S6, I3) has meant since
#203 and what P2-G3 measures.

Division of labour, chosen to keep every allocator invariant where it
already lives:

* Each segment is an UNMODIFIED DevRowCache: slot states, event gating,
  the RETIRING path, and the I1 margin rule all stay inside the shipped
  artifact.
* The wrapper owns REPLACEMENT and PLACEMENT: a pool-wide LRU of keys,
  insert routing to the fill segment, and cross-segment eviction via the
  segment's own `discard()` (unpublish -> RECLAIMABLE -> the segment's
  `_claim` reuses the slot under the existing event order).
* GPU addressing stays uniform-stride WITHIN a segment: `views()` returns
  the per-segment (blocks, scales) as_strided pair, and callers group a
  step's resident rows by segment (one gemm launch per touched segment).

Not thread-safe; single-controller use, like DevRowCache itself.
"""
import torch

from dev_row_cache import DevRowCache, StepTag


class SegmentedRowPool:
    def __init__(self, segments: int, seg_rows: int, row_stride: int, *,
                 device="cuda", routed: int = 8):
        if segments < 1 or seg_rows < 1:
            raise ValueError("need >= 1 segment of >= 1 row")
        if seg_rows - max(1, seg_rows - int(routed)) < 1 and seg_rows <= routed:
            # DevRowCache enforces the real margin rule; this is only the
            # trivial-size guard so error messages point here, not at I1.
            raise ValueError("seg_rows must exceed the routed set")
        self.seg_rows, self.row_stride, self.routed = seg_rows, row_stride, routed
        self.device = device
        self._segs: list[DevRowCache | None] = [
            DevRowCache(seg_rows, row_stride, device=device, routed=routed)
            for _ in range(segments)
        ]
        self._where: dict = {}          # key -> (seg_idx, expert id); an
                                        # upper bound: internally-displaced
                                        # keys stay until their next touch
        self.fills = 0
        self.refills = 0                # internal displacements re-copied
        self.shrunk_segments = 0
        self.grown_segments = 0

    # ------------------------------------------------------------- sizing --
    def segments_alive(self) -> int:
        return sum(1 for s in self._segs if s is not None)

    def rows_capacity(self) -> int:
        return self.segments_alive() * self.seg_rows

    def rows_resident(self) -> int:
        return len(self._where)

    def seg_bytes(self) -> int:
        return self.seg_rows * self.row_stride

    # ------------------------------------------------------------- lookup --
    def views(self, seg_idx: int, shape_blocks, shape_scales, pb: int):
        """The uniform-stride (blocks, scales) stacks for one segment."""
        seg = self._segs[seg_idx]
        if seg is None:
            raise KeyError(f"segment {seg_idx} was shrunk away")
        buf = seg.buf
        rows = self.seg_rows
        blocks = torch.as_strided(buf, (rows, *shape_blocks),
                                  (self.row_stride, *self._strides(shape_blocks)))
        scales = torch.as_strided(buf, (rows, *shape_scales),
                                  (self.row_stride, *self._strides(shape_scales)),
                                  storage_offset=pb)
        return blocks, scales

    @staticmethod
    def _strides(shape):
        out, acc = [], 1
        for d in reversed(shape):
            out.append(acc)
            acc *= d
        return tuple(reversed(out))

    # -------------------------------------------------------------- want --
    def want(self, layer: int, experts, tag: StepTag, budget=None):
        """Resolve one layer's routed set. Returns (placed, need_fill,
        skipped): `placed` maps key -> (seg_idx, slot) for every row that is
        or will be resident; `need_fill` lists keys the caller must copy
        into placed[key]; `skipped` lists keys refused under `budget` (they
        execute cold, no state change -- the S5 SMOOTH_CAP path).

        Replacement lives INSIDE each segment (the shipped margin machinery:
        protected = seg_rows - routed, LRU-beyond-protected demotes to
        RECLAIMABLE and inserts claim it). The wrapper only places: hits go
        to their owning segment; a hit the segment internally displaced
        comes back in `need` and is handled as a re-fill in place -- the
        engine's own behavior, not an error. Fresh inserts rotate across
        alive segments so churn spreads. `_where` may hold entries for
        internally-displaced keys until their next touch corrects them, so
        `rows_resident()` is an upper bound (documented)."""
        placed, need_fill, skipped = {}, [], []
        by_seg_hits, misses = {}, []
        for e in experts:
            key = (layer, e)
            if key in self._where and self._segs[self._where[key][0]] is not None:
                by_seg_hits.setdefault(self._where[key][0], []).append(e)
            else:
                misses.append(e)
        for si, es in by_seg_hits.items():
            seg = self._segs[si]
            assign, need = seg.want(layer, es, tag)
            refill = set(need)
            for e in es:
                key = (layer, e)
                placed[key] = (si, assign[e])
                if e in refill:              # internally displaced: re-fill
                    need_fill.append(key)
                    seg.note_filled(1)
                    self.refills += 1
        remaining = budget if budget is not None else len(misses)
        for e in misses:
            key = (layer, e)
            if remaining <= 0:
                skipped.append(key)
                continue
            si = self._next_fill_segment()
            if si is None:
                skipped.append(key)          # pool fully shrunk away
                continue
            seg = self._segs[si]
            assign, need = seg.want(layer, [e], tag)
            placed[key] = (si, assign[e])
            if need:
                need_fill.append(key)
                seg.note_filled(1)
                self.fills += 1
            self._where[key] = (si, e)
            remaining -= 1
        return placed, need_fill, skipped

    def _next_fill_segment(self):
        # pure rotation across alive segments: spreads churn so shrink()'s
        # emptiest-first ordering stays meaningful, with no occupancy
        # bookkeeping to drift (DevRowCache.filled is cumulative, not
        # current, and the wrapper deliberately holds no second policy)
        alive = [si for si, s in enumerate(self._segs) if s is not None]
        if not alive:
            return None
        self._rr = getattr(self, "_rr", 0) + 1
        return alive[self._rr % len(alive)]

    # ------------------------------------------------------------ shrink --
    def shrink(self, n_segments: int) -> int:
        """Free whole segments, emptiest-first: event-gate their rows via the
        segment's own settle path, drop every key, release the buffer. Returns
        bytes freed. The event gate is the segment's existing machinery -- a
        row under an unsettled reader is exactly what StepTag.sync() waits
        out before the buffer goes away (I3: within one step)."""
        freed = 0
        pop = {si: 0 for si, s in enumerate(self._segs) if s is not None}
        for (si, _e) in self._where.values():
            if si in pop:
                pop[si] += 1
        order = sorted(pop, key=lambda si: pop[si])
        for si in order[:n_segments]:
            seg = self._segs[si]
            if seg._last is not None:
                if not seg._last.recorded:
                    seg._last.record()
                seg._last.sync()
            for key in [k for k, (s_, _) in self._where.items() if s_ == si]:
                del self._where[key]
            self._segs[si] = None
            freed += self.seg_bytes()
            self.shrunk_segments += 1
        return freed

    def grow(self, n_segments: int) -> int:
        """Re-allocate segments (recovery). Returns segments added."""
        added = 0
        for si, s in enumerate(self._segs):
            if added >= n_segments:
                break
            if s is None:
                self._segs[si] = DevRowCache(self.seg_rows, self.row_stride,
                                             device=self.device,
                                             routed=self.routed)
                self.grown_segments += 1
                added += 1
        for _ in range(n_segments - added):
            self._segs.append(DevRowCache(self.seg_rows, self.row_stride,
                                          device=self.device,
                                          routed=self.routed))
            self.grown_segments += 1
            added += 1
        return added

    def stats(self) -> dict:
        return {"segments_alive": self.segments_alive(),
                "rows_capacity": self.rows_capacity(),
                "rows_resident": self.rows_resident(),
                "fills": self.fills, "refills": self.refills,
                "shrunk_segments": self.shrunk_segments,
                "grown_segments": self.grown_segments}
