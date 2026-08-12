# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""N3 — residency: which expert rows live in pinned DRAM, which come from NVMe.

N1 (:mod:`nvme_arena`) relocates a checkpoint into an expert-major arena, one
aligned row per (layer, expert). N2 (:mod:`nvme_reader`) reads those rows at 99%
of the device link. Neither decides *what stays resident* — and until something
does, a model whose experts exceed host RAM cannot run at all. That is this
module.

Why it matters concretely: Kimi K3's experts total **1.446 TB** (93 layers x 896
experts x ~17.5 MB, measured 2026-07-30), while a rented pod exposed **503 GB** of
host RAM — and that 503 GB proved to be a per-container ceiling, unchanged by
renting 4 GPUs instead of 1. The expert set is ~2.9x too large no matter how many
cards you pay for. But top-16-of-896 routing touches only **25.83 GB per token**,
so residency, not capacity, is the binding constraint.

**The pinned buffer IS the arena the gather kernel indexes.** :meth:`ColdTier.ensure`
returns slot indices into :attr:`ColdTier.buffer`, so a hit costs *zero* copies
(the GPU gathers straight from pinned DRAM over UVA — see
``host_gather.gather_expert_rows``) and a miss DMA-lands from NVMe directly into
the slot it will be gathered from. No staging tier, no shadow copy.

Eviction is LFU with an LRU tie-break: MoE routing is heavy-tailed, so a modest
hot set covers most picks and frequency is the signal that exploits it. Rows
placed for the *current* request are protected for its duration, so a request can
never evict its own freshly-read rows.

Usage::

    tier = ColdTier(arena, hot_rows=capacity_for_bytes(free_ram, row_stride))
    slots = tier.ensure(layer, routed_expert_ids)   # -> slots into tier.buffer
    # hand tier.pinned_tensor() + slots to the gather kernel
    print(tier.stats())        # hit_rate, disk bytes, evictions
"""
from __future__ import annotations

import threading
from collections import Counter

from nvme_reader import ArenaReader, alloc_landing, buffer_address


class ColdTier:
    """Pinned-DRAM residency over an NVMe expert arena.

    Args:
        arena_path: baked arena from :func:`nvme_arena.bake` or
            :func:`nvme_arena.bake_expert_tensors`.
        hot_rows: expert rows to keep pinned. Size from MEASURED free RAM via
            :func:`capacity_for_bytes`.
        pinned: True allocates a CUDA-pinned buffer (the engine path, gatherable
            over UVA); False uses plain mmap (tests, CPU-only benches).
        qd: in-flight NVMe reads; N0 measured the device link saturated at >= 4.
    """

    def __init__(self, arena_path: str, *, hot_rows: int, pinned: bool = True,
                 qd: int = 4, index=None, reader: ArenaReader | None = None):
        if hot_rows < 1:
            raise ValueError("hot_rows must be >= 1")
        self.reader = reader or ArenaReader(arena_path, index, qd=qd)
        self.row_stride = self.reader.row_stride
        self.row_bytes = self.reader.row_bytes
        self.hot_rows = hot_rows
        self.buffer, self._keepalive = alloc_landing(
            hot_rows * self.row_stride, pinned=pinned)
        self.pinned = pinned

        self._lock = threading.Lock()
        self._slot_of: dict[tuple[int, int], int] = {}
        self._key_of: list[tuple[int, int] | None] = [None] * hot_rows
        self._free: list[int] = list(range(hot_rows))
        self._freq: Counter = Counter()
        self._clock = 0
        self._last_use: dict[tuple[int, int], int] = {}

        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.requests = 0

    # ---------------------------------------------------------------- slots --
    def _slot_view(self, slot: int) -> memoryview:
        lo = slot * self.row_stride
        return self.buffer[lo:lo + self.row_stride]

    def _victim(self, protected: set) -> int:
        """LFU, LRU tie-break, never a slot this request already claimed."""
        best, best_key = None, None
        for slot, key in enumerate(self._key_of):
            if slot in protected or key is None:
                continue
            k = (self._freq[key], self._last_use.get(key, 0))
            if best_key is None or k < best_key:
                best, best_key = slot, k
        if best is None:
            raise RuntimeError(
                f"hot_rows={self.hot_rows} too small: every slot is protected "
                f"by the current request. Size hot_rows >= max routed experts "
                f"per layer.")
        return best

    # -------------------------------------------------------------- the API --
    def ensure(self, layer: int, experts) -> list:
        """Make every (layer, expert) resident; return slot indices in order.

        Hits are free. Misses are submitted concurrently (bounded by the
        reader's queue depth) and land directly in the slot they will be
        gathered from.

        Capacity is counted in UNIQUE rows: repeats in one request share a slot,
        so ``ensure(l, [7, 7, 7])`` needs one row, not three.

        A slot is *reserved* while its read is in flight and only PUBLISHED into
        the residency maps once the fill lands. Nothing can observe a slot as
        resident while it holds partial or stale bytes, and a failed batch drains
        every in-flight read before reclaiming slots, so no read can still be
        writing into a slot that has been handed to someone else.
        """
        experts = [int(e) for e in experts]
        keys = [(layer, e) for e in experts]
        uniq = list(dict.fromkeys(keys))          # order-preserving dedupe
        if len(uniq) > self.hot_rows:
            raise ValueError(
                f"request of {len(uniq)} unique rows exceeds "
                f"hot_rows={self.hot_rows}")
        with self._lock:
            self.requests += 1
            self._clock += 1
            now = self._clock
            for key in keys:                      # frequency counts every pick
                self._freq[key] += 1
                self._last_use[key] = now

            resolved: dict = {}
            reserved: list = []                   # (key, slot) awaiting fill
            protected = set()

            # Pass 1 — resolve hits and RESERVE slots for misses. Reserving
            # before any read means a miss cannot evict a row this same request
            # already hit or claimed.
            for key in uniq:
                slot = self._slot_of.get(key)
                if slot is not None:
                    self.hits += 1
                    resolved[key] = slot
                    protected.add(slot)
                    continue
                self.misses += 1
                if self._free:
                    slot = self._free.pop()
                else:
                    slot = self._victim(protected)
                    old = self._key_of[slot]
                    if old is not None:
                        del self._slot_of[old]
                        self._key_of[slot] = None   # unpublish before refilling
                        self.evictions += 1
                protected.add(slot)
                resolved[key] = slot
                reserved.append((key, slot))

            # Pass 2 — concurrent O_DIRECT fills, then publish. Every future is
            # drained even on failure: returning early could hand a slot back to
            # the free list while a read is still landing bytes in it.
            futures = [(self.reader.read_row(layer, k[1], self._slot_view(s)), k, s)
                       for k, s in reserved]
            first_err = None
            for fut, key, slot in futures:
                try:
                    fut.result()
                except Exception as exc:          # noqa: BLE001 - re-raised below
                    if first_err is None:
                        first_err = exc
                    continue
                self._key_of[slot] = key          # publish only now
                self._slot_of[key] = slot
            if first_err is not None:
                for key, slot in reserved:        # reclaim whatever never landed
                    if self._slot_of.get(key) != slot:
                        self._key_of[slot] = None
                        if slot not in self._free:
                            self._free.append(slot)
                raise first_err
            return [resolved[k] for k in keys]

    def row(self, layer: int, expert: int) -> memoryview:
        """A resident row's bytes (``row_bytes``, excluding alignment padding).

        Lock-guarded: the residency maps are only published after a fill lands, so
        taking the lock here is what guarantees a caller can never be handed a
        view of a slot whose read is still in flight.
        """
        with self._lock:
            slot = self._slot_of.get((layer, int(expert)))
            if slot is None:
                raise KeyError(f"(layer {layer}, expert {expert}) not resident")
            return self._slot_view(slot)[:self.row_bytes]

    def resident(self, layer: int, expert: int) -> bool:
        with self._lock:
            return (layer, int(expert)) in self._slot_of

    def pinned_tensor(self):
        """The pinned buffer as a ``[hot_rows, row_stride]`` uint8 tensor — what
        the gather kernel indexes by slot. Requires ``pinned=True``.

        Sliced from the keepalive at the SAME offset :func:`alloc_landing`
        aligned to, never viewed from its base: since that function over-allocates
        by one alignment unit and hands back an interior sub-view, a base view
        would describe different bytes than :attr:`buffer` (and, at the shapes
        used here, raise on numel). This tensor's ``data_ptr()`` is what an
        address-table engine adds slot offsets to, so a silent skew here would
        make every gather read one page early.
        """
        if not self.pinned:
            raise RuntimeError("pinned_tensor() requires pinned=True")
        n = self.hot_rows * self.row_stride
        pad = self.buffer_ptr - self._keepalive.data_ptr()
        return self._keepalive[pad:pad + n].view(self.hot_rows, self.row_stride)

    @property
    def buffer_ptr(self) -> int:
        """Address of slot 0. Slots are :attr:`row_stride` apart — *not*
        ``row_bytes``: the arena pads rows out to ``align`` for O_DIRECT, so an
        engine that strides its own ``row_bytes`` through this buffer starts
        reading mid-row on slot 1 and never fails loudly."""
        return buffer_address(self.buffer)

    # --------------------------------------------------------------- stats --
    def stats(self) -> dict:
        with self._lock:
            return self._stats_locked()

    def _stats_locked(self) -> dict:
        total = self.hits + self.misses
        t = self.reader.traffic()
        return {
            "requests": self.requests,
            "rows_requested": total,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": (self.hits / total) if total else 0.0,
            "evictions": self.evictions,
            "resident_rows": len(self._slot_of),
            "hot_rows": self.hot_rows,
            "disk_reads": t["reads"],
            "disk_bytes": t["bytes_read"],
            "io_mode": t["mode"],
        }

    def close(self):
        self.reader.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# safetensors dtype tag -> torch dtype. Only the tags the bake accepts.
_ST_TO_TORCH = {
    "U8": "uint8", "I8": "int8", "F16": "float16", "BF16": "bfloat16",
    "F32": "float32", "F64": "float64", "I16": "int16", "I32": "int32",
    "I64": "int64", "U16": "uint16", "U32": "uint32", "U64": "uint64",
}


def segment_geometry(index: dict, suffix: str):
    """``(torch_dtype, shape_per_expert, seg_off, length)`` for one segment.

    Pulled out of :func:`segment_tensor` so a caller can size a destination
    buffer *before* any row is resident — a staging path has to allocate its
    landing tensor once, at setup, not per read.
    """
    import torch

    geo = next((g for g in index["segments"] if g["suffix"] == suffix), None)
    if geo is None:
        raise KeyError(f"segment {suffix!r} not in this arena "
                       f"(have {[g['suffix'] for g in index['segments']]})")
    return (getattr(torch, _ST_TO_TORCH[geo["dtype"]]),
            tuple(geo["shape_per_expert"]), geo["seg_off"], geo["length"])


def segment_into(tier: "ColdTier", index: dict, layer: int, experts,
                 suffix: str, out, *, rows=None, non_blocking: bool = False):
    """Fill ``out`` with one segment's bytes for ``experts``, into storage the
    CALLER owns. Returns ``out``.

    Same bytes as :func:`segment_tensor`, and the same residency side effect —
    the difference is who owns the destination, which is what makes this usable
    on a staging path. ``segment_tensor`` allocates a fresh pageable tensor per
    call, and a pageable source silently downgrades ``non_blocking=True`` to a
    synchronous copy; a caller holding one reusable buffer (or copying straight
    to the device) avoids both.

    When the tier is pinned this is a genuinely zero-bounce path: ``ColdTier``
    already lands rows in pinned memory, so the segment is copied out of the
    pinned slot itself — disk -> pinned slot -> ``out``, with no intermediate
    host allocation. ``segment_tensor`` cannot do that: ``torch.frombuffer``
    needs a writable buffer, so it copies through a ``bytearray`` first.
    Unpinned tiers keep that fallback, correct but with the extra copy.

    ``out`` must be contiguous and shaped ``[R, *shape_per_expert]`` at the
    segment's own dtype, where ``R == len(experts)``; it may live on any device.
    ``rows`` restricts the write to those row indices of ``out`` (same length and
    order as ``experts``), which lets a caller fill only the routed rows of a
    full-shaped ``[E, ...]`` destination and leave the rest untouched.

    Bytes are moved as ``uint8``, so this is bit-preserving by construction —
    there is no dtype reinterpretation step that could disagree with
    ``segment_tensor``'s. Any deliberate cast is the caller's, applied after.
    """
    import torch

    dt, shape, off, ln = segment_geometry(index, suffix)
    experts = [int(e) for e in experts]
    dst_rows = list(range(len(experts))) if rows is None else [int(r) for r in rows]
    if len(dst_rows) != len(experts):
        raise ValueError(f"rows has {len(dst_rows)} entries for {len(experts)} experts")
    if out.dtype != dt:
        raise TypeError(f"out has dtype {out.dtype} but segment {suffix!r} is {dt}")
    if tuple(out.shape[1:]) != shape:
        raise ValueError(f"out is {tuple(out.shape)}; segment {suffix!r} needs "
                         f"[R, {', '.join(str(s) for s in shape)}]")
    if not out.is_contiguous():
        raise ValueError("out must be contiguous — rows are filled as flat byte runs")

    slots = tier.ensure(layer, experts)
    pinned = tier.pinned_tensor() if tier.pinned else None
    for r, e, slot in zip(dst_rows, experts, slots):
        # Reinterpret the destination row as bytes. `out` is contiguous, so each
        # row is a flat byte run of exactly `ln` bytes and the copy is a memcpy
        # (or one H2D) regardless of the segment's logical dtype.
        dv = out[r].reshape(-1).view(torch.uint8)
        if pinned is not None:
            dv.copy_(pinned[slot, off:off + ln], non_blocking=non_blocking)
        else:
            mv = tier.row(layer, e)[off:off + ln]
            dv.copy_(torch.frombuffer(bytearray(mv), dtype=torch.uint8))
    return out


def segment_tensor(tier: "ColdTier", index: dict, layer: int, experts,
                   suffix: str, *, cast=None):
    """Reconstruct one arena segment across `experts` as a ``[R, *shape]`` tensor.

    This is the seam a serving engine sits on: the engine wants expert-major
    tensors per projection, while the arena is expert-major with segments nested
    inside each row. Geometry (per-expert shape + safetensors dtype) comes from
    the bake's own index, so nothing here guesses a layout.

    **Bit-identity:** the bytes are the shipped bytes (relocation bakes are
    single-source and hash-preserving) and reinterpreting them at the recorded
    dtype is bit-preserving, so the result equals the tensor read from the
    original checkpoint exactly. ``cast`` applies a deliberate, documented
    transform afterwards — e.g. the engine holds NF4 absmax as float32
    (``mod.gate_up_absmax.view(...).float()``), so a caller replacing that stack
    must pass ``cast="float32"`` or it will hand the kernel raw scale bytes
    reinterpreted as floats: plausible shapes, garbage numerics.

    Rows are made resident first, so this is where NVMe reads happen.
    """
    import torch

    geo = next((g for g in index["segments"] if g["suffix"] == suffix), None)
    if geo is None:
        raise KeyError(f"segment {suffix!r} not in this arena "
                       f"(have {[g['suffix'] for g in index['segments']]})")
    dt = getattr(torch, _ST_TO_TORCH[geo["dtype"]])
    shape = tuple(geo["shape_per_expert"])
    off, ln = geo["seg_off"], geo["length"]

    experts = [int(e) for e in experts]
    tier.ensure(layer, experts)
    rows = []
    for e in experts:
        mv = tier.row(layer, e)[off:off + ln]
        # bytearray() copies: frombuffer needs a writable buffer, and the stack
        # below would copy regardless.
        rows.append(torch.frombuffer(bytearray(mv), dtype=dt).view(shape))
    out = torch.stack(rows)
    if cast is not None:
        out = out.to(getattr(torch, cast) if isinstance(cast, str) else cast)
    return out


def capacity_for_bytes(usable_bytes: int, row_stride: int) -> int:
    """How many rows fit in a byte budget.

    Use MEASURED free RAM, never a declared figure: a pod rented with 4 GPUs
    still exposed 503 GB, identical to a 1-GPU pod (2026-07-30).
    """
    return max(1, int(usable_bytes) // int(row_stride))
