# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""N4 — RowPool: the weight tier abstraction, generalized to WRITABLE rows.

:class:`~nvme_residency.ColdTier` is the read-only specialization of one
idea — fixed-stride rows, keyed, living in a fast window over a bigger
store, with publish-only-when-bytes-are-whole discipline and one stats
vocabulary. Its rows come from a baked arena, so the arena is always the
source of truth. KV-cache blocks (hybrid Stage 2 Phase 6) have the same
row shape but the opposite lifecycle: rows are BORN in device memory,
written in place by the caller, and their source of truth MIGRATES when a
row demotes to the pinned-DRAM store. This class is that writable sibling
— same landing conventions (:func:`nvme_reader.alloc_landing`), same
counters, deliberately no second vocabulary (the Stage-2 directive's
invariant 8: KV reuses the weight tier abstraction rather than growing a
parallel system).

v1 contract, sized to the Phase-6 gate and stated plainly:

- **Partitioned, append-only, head-demoting.** A partition (one per model
  layer, in the KV use) appends rows at its tail and demotes rows from
  its head — exactly a sliding context window. Because appends are
  sequential, the resident run ``[head, tail)`` is physically contiguous
  in the device pool, so a consumer gets a zero-copy VIEW of its hot
  window — that contiguity is what makes Phase 6's "paged overhead ≤2%"
  clause a non-event at batch 1. Random free / reuse / NVMe park are the
  Phase-10 extension, not silently half-shipped here.
- **Copy-on-demote, publish-after-drain.** ``demote_head`` enqueues
  device→pinned copies on the caller's side stream and records one event;
  the device row stays the source of truth until :meth:`settle` observes
  the event complete (a non-blocking query) and flips ownership. Nothing
  ever reads a half-copied row, and nothing on the critical path waits.
- **Single-writer.** One thread appends/demotes/settles (the decode
  loop); the only concurrency is the side stream's DMA. This is the KV
  reality and it keeps the pool lock-free; the ColdTier-style concurrent
  contract can be added when a consumer needs it.

Rows are opaque bytes (uint8). Consumers view-cast to their dtype and
shape; the pool never interprets a row (which is also what lets Phase 7
swap FP8 block formats without touching tiering).
"""
from __future__ import annotations

import torch

from nvme_reader import alloc_landing


class RowPool:
    def __init__(self, partitions: int, device_rows: int, host_rows: int,
                 row_bytes: int, *, device: str = "cuda"):
        if partitions < 1 or device_rows < 1 or row_bytes < 1:
            raise ValueError("partitions, device_rows, row_bytes must be >= 1")
        self.P = partitions
        self.device_rows = device_rows
        self.host_rows = host_rows
        self.row_bytes = row_bytes
        self.device = torch.device(device)
        self._cuda = self.device.type == "cuda"
        if self._cuda and self.device.index is None:
            # Pin a CONCRETE index now: bare `cuda` has index None, and
            # current_stream(torch.device("cuda")) still resolves through
            # the thread's current device — so the device-scoped stream
            # query would be exactly as blind as the bare call it replaced.
            # The pool's own storage is allocated here, on this device.
            self.device = torch.device("cuda", torch.cuda.current_device())

        self.dev = torch.zeros(partitions, device_rows, row_bytes,
                               dtype=torch.uint8, device=self.device)
        if host_rows > 0:
            n = partitions * host_rows * row_bytes
            mv, keep = alloc_landing(n, pinned=self._cuda)
            self._host_keep = keep
            if self._cuda:
                # Slice the PINNED KEEPALIVE at the aligned offset (the
                # ColdTier.pinned_tensor pattern) — a frombuffer wrap of the
                # memoryview is a tensor torch does not KNOW is pinned, so
                # copy_(non_blocking=True) silently degrades to synchronous
                # and the publish-after-drain contract quietly dies (Bugbot).
                from nvme_reader import buffer_address
                pad = buffer_address(mv) - keep.data_ptr()
                self.host = keep[pad:pad + n].view(
                    partitions, host_rows, row_bytes)
                assert self.host.is_pinned()
            else:
                self.host = torch.frombuffer(mv, dtype=torch.uint8).view(
                    partitions, host_rows, row_bytes)
        else:
            self.host = None
            self._host_keep = None

        # per-partition ring state (absolute row indices, never wrapped)
        self.head = [0] * partitions          # first row still device-resident
        self.tail = [0] * partitions          # next row to append
        self.demoted = [0] * partitions       # rows [0, demoted) live in host
        self._frontier = [0] * partitions     # rows [0, frontier) enqueued
        self._pending: list[tuple[int, int, object]] = []  # (p, upto, event)

        self.appends = 0
        self.demotions = 0
        self.settled = 0
        self.host_reads = 0
        self.host_read_bytes = 0

    # ------------------------------------------------------------- helpers --
    def _dslot(self, p: int, idx: int) -> int:
        """Physical device slot for absolute row idx (ring)."""
        return idx % self.device_rows

    def resident_run(self, p: int):
        """(lo, hi) absolute rows currently device-resident in partition p."""
        return self.head[p], self.tail[p]

    # ----------------------------------------------------------------- API --
    def append(self, p: int):
        """Claim the next row of partition ``p``; returns (abs_idx, view).
        The view is the caller's to WRITE (uint8, ``row_bytes``)."""
        idx = self.tail[p]
        if idx - self.head[p] >= self.device_rows:
            raise RuntimeError(
                f"partition {p} device window full "
                f"({self.device_rows} rows) and head not demoted+settled — "
                f"demote_head/settle before appending, or size device_rows "
                f"to the hot window")
        self.tail[p] = idx + 1
        self.appends += 1
        return idx, self.dev[p, self._dslot(p, idx)]

    def row_view(self, p: int, idx: int):
        """Device view of an absolute row (must be device-resident)."""
        if not (self.head[p] <= idx < self.tail[p]):
            raise KeyError(f"row {idx} of partition {p} is not "
                           f"device-resident (run {self.resident_run(p)})")
        return self.dev[p, self._dslot(p, idx)]

    def run_view(self, p: int, lo: int, hi: int):
        """Zero-copy [hi-lo, row_bytes] device view when the run does not
        wrap the ring; a gathered COPY when it does (counted in stats via
        the caller — wrap only happens under demotion pressure, which is
        outside the ≤2% clause's regime)."""
        if not (self.head[p] <= lo and hi <= self.tail[p] and lo <= hi):
            raise KeyError(f"[{lo},{hi}) outside resident run "
                           f"{self.resident_run(p)} of partition {p}")
        a, b = self._dslot(p, lo), self._dslot(p, lo) + (hi - lo)
        if b <= self.device_rows:
            return self.dev[p].narrow(0, a, hi - lo)
        first = self.dev[p].narrow(0, a, self.device_rows - a)
        rest = self.dev[p].narrow(0, 0, b - self.device_rows)
        return torch.cat([first, rest])

    def demote_head(self, p: int, upto: int, stream=None):
        """Enqueue device→host copies for rows [demote-front, upto) of
        partition ``p`` on ``stream`` (side stream — the whole point is
        that these copies never ride the compute stream). Rows remain
        device-readable until :meth:`settle` flips them."""
        if self.host is None:
            raise RuntimeError("RowPool built with host_rows=0 cannot demote")
        # start from the ENQUEUED frontier, not the settled cursor: a second
        # demote before settle must never re-DMA rows an in-flight batch is
        # already copying (Bugbot — overlapping copies on different streams)
        lo = self._frontier[p]
        # only rows already appended, never past what host can hold
        upto = min(upto, self.tail[p])
        if upto <= lo:
            return 0
        if upto > self.host_rows:
            raise RuntimeError(
                f"host store of partition {p} full ({self.host_rows} rows) — "
                f"NVMe park is the Phase-10 extension")
        n = 0
        ctx = torch.cuda.stream(stream) if (self._cuda and stream is not None) \
            else _nullctx()
        with ctx:
            for idx in range(lo, upto):
                dst = self.host[p, idx]
                src = self.dev[p, self._dslot(p, idx)]
                dst.copy_(src, non_blocking=self._cuda)
                n += 1
        ev = None
        if self._cuda:
            ev = torch.cuda.Event()
            # current_stream(SELF.DEVICE), never bare — a bare call reads the
            # THREAD's current device, so on a multi-GPU host the event can
            # record on a stream that has nothing to do with this pool
            ev.record(stream if stream is not None
                      else torch.cuda.current_stream(self.device))
        self._frontier[p] = upto
        self._pending.append((p, upto, ev))
        self.demotions += n
        return n

    def settle(self):
        """Flip ownership for every demotion batch whose copy has landed
        (non-blocking event query — never a sync). Strictly FIFO per
        partition: a later batch's completed event must NOT advance the
        head past an earlier, still-pending batch — freeing those slots
        would let appends overwrite rows the earlier copy still reads
        (Bugbot). Returns rows settled."""
        done = 0
        keep = []
        blocked: set[int] = set()
        for p, upto, ev in self._pending:
            if p in blocked or (ev is not None and not ev.query()):
                blocked.add(p)
                keep.append((p, upto, ev))
                continue
            if upto > self.demoted[p]:
                done += upto - self.demoted[p]
                self.demoted[p] = upto
            self.head[p] = max(self.head[p], upto)
        self._pending = keep
        self.settled += done
        return done

    def host_run(self, p: int, lo: int, hi: int):
        """[hi-lo, row_bytes] pinned-host view of settled rows (source of
        truth = host). The caller streams it wherever it wants."""
        if not (0 <= lo and hi <= self.demoted[p] and lo <= hi):
            raise KeyError(f"[{lo},{hi}) not settled to host in partition {p} "
                           f"(settled up to {self.demoted[p]})")
        self.host_reads += 1
        self.host_read_bytes += (hi - lo) * self.row_bytes
        return self.host[p].narrow(0, lo, hi - lo)

    def stats(self) -> dict:
        return {
            "partitions": self.P,
            "device_rows": self.device_rows,
            "host_rows": self.host_rows,
            "row_bytes": self.row_bytes,
            "appends": self.appends,
            "demotions": self.demotions,
            "settled": self.settled,
            "pending_batches": len(self._pending),
            "host_reads": self.host_reads,
            "host_read_bytes": self.host_read_bytes,
            "device_resident_rows": sum(t - h for h, t
                                        in zip(self.head, self.tail)),
        }


class _nullctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
