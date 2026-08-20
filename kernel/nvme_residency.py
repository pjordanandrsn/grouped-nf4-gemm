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
never evict its own freshly-read rows — and the rows of the latest *demand*
request stay protected until the next demand request replaces them, so a
concurrent ``ensure(..., speculative=True)`` (a prefetcher warming predicted
rows) can never evict what the serving thread is about to read. Disk fills run
outside the tier lock (reserved slots + pending-key events keep them safe), so
speculative I/O overlaps demand I/O instead of convoying it.

Usage::

    # capacity_for_bytes budgets for what a PINNED row really costs (~1.9x the
    # stride); pass pinned=False for the mmap tier.
    tier = ColdTier(arena, hot_rows=capacity_for_bytes(free_ram, row_stride))
    slots = tier.ensure(layer, routed_expert_ids)   # -> slots into tier.buffer
    # hand tier.pinned_tensor() + slots to the gather kernel
    print(tier.stats())        # hit_rate, disk bytes, evictions
"""
from __future__ import annotations

import threading
import time
from collections import Counter
from concurrent.futures import as_completed

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
        qd: in-flight NVMe reads. ``None`` (default) sizes it from the host's
            CPU budget via :func:`nvme_reader.default_qd`. The old fixed 4 was
            measured on a loaded 12-core box; on an idle 32-vCPU host qd=8 and
            qd=16 read 12% and 15% faster at the same pattern.
        protected_rows: capacity-ownership budget, ``<= hot_rows``. Rows
            beyond it are RECLAIMABLE: still mapped, still readable, first in
            line to be overwritten. ``None`` (default) = ``hot_rows``, which
            leaves the reclaimable set permanently empty and every code path
            below identical to the pre-Stage-3 tier. See "Reclaimable
            residency" in the class docstring.

    **Reclaimable residency** (Stage 3). Three states, not two:

    - ACTIVE — inside the protected budget.
    - RECLAIMABLE — logical eviction has revoked capacity ownership, but the
      packed bytes are untouched and a request before physical overwrite is a
      **resurrection**: a hit costing no disk read.
    - absent — the slot was claimed and refilled; the bytes are gone.

    The distinction this adds is bookkeeping over a property the tier already
    had (eviction here never zeroed a row, so an unprotected mapped row was
    always a free hit). What did not exist was a *nameable* logical-eviction
    event, and therefore no way to measure the load-bearing probability
    ``P(reuse before overwrite | logical eviction)`` — reported by
    :meth:`stats` as ``reuse_before_overwrite``, over resolved evictions only
    (rows still sitting reclaimable are unresolved and count on neither side).

    Slot generations make a held reference checkable: ``_gen[slot]`` bumps on
    every claim, so a caller that snapshotted ``(slot, generation)`` can ask
    :meth:`validate` whether those bytes are still the expert it meant. An
    expert id alone never proves a slot's contents — the address-vs-contents
    lesson, made explicit rather than left to a caller's discipline.
    """

    def __init__(self, arena_path: str, *, hot_rows: int, pinned: bool = True,
                 qd: int | None = None, index=None, reader: ArenaReader | None = None,
                 protected_rows: int | None = None, landing=None):
        if hot_rows < 1:
            raise ValueError("hot_rows must be >= 1")
        self.reader = reader or ArenaReader(arena_path, index, qd=qd)
        self.row_stride = self.reader.row_stride
        self.row_bytes = self.reader.row_bytes
        self.hot_rows = hot_rows
        if protected_rows is None:
            protected_rows = hot_rows
        protected_rows = int(protected_rows)
        if not 1 <= protected_rows <= hot_rows:
            raise ValueError(
                f"protected_rows={protected_rows} must be in [1, "
                f"hot_rows={hot_rows}] — it is a budget WITHIN the pool, not "
                f"a second pool")
        self.protected_rows = protected_rows
        # EXTERNAL LANDING (Stage 3). When set, a fill scatters straight into
        # views this callable supplies for (layer, expert, slot) instead of
        # this tier's own row buffer — the consumer's kernel-shaped stacks,
        # so the bytes never make the intermediate stop. Residency, eviction,
        # reclaimable state and the concurrency contract are unchanged: the
        # slot is still the name, only the destination moves.
        self._landing = landing
        # With fills going elsewhere this tier's buffer holds nothing, so the
        # byte-serving API must REFUSE rather than hand back stale rows. Keep
        # it one page instead of hot_rows*row_stride: allocating a landing
        # nothing writes would be pure footprint.
        self.buffer, self._keepalive = alloc_landing(
            (hot_rows * self.row_stride) if landing is None else self.row_stride,
            pinned=pinned)
        self.pinned = pinned

        self._lock = threading.Lock()
        self._slot_of: dict[tuple[int, int], int] = {}
        self._key_of: list[tuple[int, int] | None] = [None] * hot_rows
        self._free: list[int] = list(range(hot_rows))
        self._freq: Counter = Counter()
        self._clock = 0
        self._last_use: dict[tuple[int, int], int] = {}
        # Concurrency state (see ensure's docstring for the contract):
        self._reserved: set[int] = set()   # slots whose fill is in flight
        self._pending: dict[tuple[int, int], threading.Event] = {}
        self._demand_protected: set[int] = set()
        # Reclaimable residency: key -> the clock tick at its logical
        # eviction. Membership IS the RECLAIMABLE state; the row stays in
        # _slot_of throughout (it is still readable — that is the point).
        self._reclaimable: dict[tuple[int, int], int] = {}
        self._gen: list[int] = [0] * hot_rows

        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.requests = 0
        self.demand_misses = 0
        self.spec_misses = 0
        self.spec_hits = 0
        self.demand_waits = 0
        # Where wall-clock goes, attributable per path: time a demand ensure
        # spends draining its own fills, time it spends waiting on fills it
        # found in flight elsewhere, and speculative fill time (overlapped,
        # so NOT critical-path — reported to make that checkable).
        self.demand_fill_ns = 0
        self.demand_wait_ns = 0
        self.spec_fill_ns = 0
        # Reclaimable-residency accounting. Every logical eviction resolves
        # exactly once, as a resurrection or as an overwrite; the two are the
        # numerator and the complement of `reuse_before_overwrite`.
        self.logical_evictions = 0
        self.resurrections = 0
        self.spec_resurrections = 0
        self.reclaimable_overwritten = 0
        self._reclaim_ticks = 0        # sum of eviction->overwrite lifetimes
        self._resurrect_ticks = 0      # sum of eviction->resurrection ones

    # ---------------------------------------------------------------- slots --
    def _slot_view(self, slot: int) -> memoryview:
        lo = slot * self.row_stride
        return self.buffer[lo:lo + self.row_stride]

    def _submit_fill(self, layer: int, key, slot: int):
        """Start this slot's fill and return its future.

        One line of policy: with an external landing the row scatters
        straight into the consumer's buffers, otherwise it lands in this
        tier's own slot. Everything around it — reservation, publish-after-
        fill, the failure drain — is identical, because the slot is the
        NAME of the residency, not the storage.
        """
        if self._landing is None:
            return self.reader.read_row(layer, key[1], self._slot_view(slot))
        views = self._landing(layer, key[1], slot)
        if views is None:
            raise RuntimeError(
                f"landing callback returned no views for (layer {layer}, "
                f"expert {key[1]}, slot {slot}) — a fill has nowhere to go. "
                f"A consumer whose geometry cannot scatter must construct "
                f"the tier WITHOUT landing= and copy instead.")
        return self.reader.read_row_scatter(layer, key[1], views)

    def _victim(self, excluded: set) -> int:
        """LFU, LRU tie-break, never a slot in ``excluded`` (this request's own
        claims, in-flight reservations, and the current demand window).

        RECLAIMABLE rows lose every allocation contest by construction — the
        leading rank term. That is what makes them free capacity rather than a
        second pool: they are handed over the moment anything needs a slot,
        and only until then do they preserve information. With no protected
        budget set the reclaimable set is empty and the ranking is exactly the
        pre-Stage-3 (freq, last_use) pair.
        """
        best, best_key = None, None
        for slot, key in enumerate(self._key_of):
            if slot in excluded or slot in self._reserved or key is None:
                continue
            k = (0 if key in self._reclaimable else 1,
                 self._freq[key], self._last_use.get(key, 0))
            if best_key is None or k < best_key:
                best, best_key = slot, k
        if best is None:
            raise RuntimeError(
                f"hot_rows={self.hot_rows} too small: every slot is claimed by "
                f"the current request, reserved by an in-flight fill, or "
                f"protected by the current demand window. Size hot_rows >= max "
                f"routed experts per layer (plus speculative headroom).")
        return best

    # -------------------------------------------------------------- the API --
    def ensure(self, layer: int, experts, *, speculative: bool = False):
        """Make every (layer, expert) resident; return slot indices in order
        (demand) or ``None`` (speculative — a warming call has no positional
        contract because it is allowed to fetch a subset).

        Hits are free. Misses are submitted concurrently (bounded by the
        reader's queue depth) and land directly in the slot they will be
        gathered from. Capacity is counted in UNIQUE rows: repeats in one
        request share a slot, so ``ensure(l, [7, 7, 7])`` needs one row.

        **Concurrency contract** (what lets a speculative prefetcher overlap a
        demand path without a convoy — measured before this structure existed:
        a prefetch thread serializing every demand fetch behind ~340 MB of its
        own disk time was a 6.6x end-to-end slowdown at 235B):

        - Slot state changes happen under the lock, but disk reads do NOT: the
          plan phase reserves slots and registers pending keys, the lock drops
          for the O_DIRECT fills, and a publish phase re-takes it. Reserved
          slots are never victims and nothing observes a slot as resident
          while it holds partial bytes.
        - A key another ensure is already filling is never fetched twice
          (two fills of one key would let one eviction delete the other's
          mapping — map corruption, not just a wasted read). Demand callers
          wait on the in-flight fill's event; speculative callers skip it.
        - The rows of the LATEST demand ensure form the *demand window*:
          they cannot be evicted by any concurrent ensure until the next
          demand ensure replaces the window. This is what makes the caller's
          ensure -> :meth:`row` read sequence safe with a concurrent
          prefetcher and no external locking (the 235B crash class:
          ``KeyError: not resident`` between a demand ensure and its reads).
          One demand thread is assumed — the serving forward pass.
        - Speculative ensures are best-effort: no free slot and no evictable
          victim means that key is skipped, never an error. Speculative reads
          land in whatever slots LFU would have reclaimed anyway.

        A failed batch drains every in-flight read before reclaiming slots
        (a read could still be landing bytes in a reclaimed slot), wakes any
        waiters, and re-raises; woken waiters whose key never published fetch
        it themselves.
        """
        experts = [int(e) for e in experts]
        keys = [(layer, e) for e in experts]
        uniq = list(dict.fromkeys(keys))          # order-preserving dedupe
        if not speculative and len(uniq) > self.hot_rows:
            raise ValueError(
                f"request of {len(uniq)} unique rows exceeds "
                f"hot_rows={self.hot_rows}")
        return self._ensure(layer, keys, uniq, speculative,
                            fresh_window=not speculative)

    def _ensure(self, layer, keys, uniq, speculative, fresh_window):
        resolved: dict = {}
        reserved: list = []                       # (key, slot) awaiting fill
        waits: list = []                          # (key, event) filled elsewhere
        own: set = set()
        with self._lock:
            self.requests += 1
            self._clock += 1
            now = self._clock
            for key in keys:                      # frequency counts every pick
                self._freq[key] += 1
                self._last_use[key] = now
            if fresh_window:
                # the new demand request REPLACES the protected window; the
                # previous layer's rows have been read and are fair game again
                self._demand_protected = set()

            # Plan — resolve hits, note keys already being filled, RESERVE
            # slots for the misses this call will fill itself.
            for key in uniq:
                slot = self._slot_of.get(key)
                if slot is not None:
                    self.hits += 1
                    if speculative:
                        self.spec_hits += 1
                    self._resurrect_locked(key, speculative)
                    resolved[key] = slot
                    own.add(slot)
                    if not speculative:
                        self._demand_protected.add(slot)
                    continue
                ev = self._pending.get(key)
                if ev is not None:
                    if speculative:
                        continue                  # already on its way
                    self.demand_waits += 1
                    waits.append((key, ev))
                    continue
                slot = self._claim_slot(own, speculative)
                if slot is None:                  # speculative + nothing evictable
                    break
                # _claim_slot may have dropped the lock waiting for in-flight
                # fills to publish; this key's state can have changed under us.
                # Reserving without re-checking could start a SECOND fill of a
                # key someone else now owns — the map-corruption case.
                cur = self._slot_of.get(key)
                if cur is not None:
                    self.hits += 1
                    if speculative:
                        self.spec_hits += 1
                    self._resurrect_locked(key, speculative)
                    resolved[key] = cur
                    own.add(cur)
                    if not speculative:
                        self._demand_protected.add(cur)
                    self._free.append(slot)
                    continue
                ev = self._pending.get(key)
                if ev is not None:
                    self._free.append(slot)
                    if speculative:
                        continue
                    self.demand_waits += 1
                    waits.append((key, ev))
                    continue
                self.misses += 1
                if speculative:
                    self.spec_misses += 1
                else:
                    self.demand_misses += 1
                    self._demand_protected.add(slot)
                self._reserved.add(slot)
                self._pending[key] = threading.Event()
                own.add(slot)
                resolved[key] = slot
                reserved.append((key, slot))

        # Fill — disk reads run OUTSIDE the lock. Reservations make the slots
        # unevictable and pending events make the keys unfetchable by anyone
        # else, so concurrent ensures proceed instead of queueing behind I/O.
        # Each row PUBLISHES THE MOMENT ITS OWN READ LANDS (as_completed, a
        # brief lock take per row): a waiter blocked on one key must wake at
        # that row's landing, not at this whole batch's tail — batch-granular
        # publish held demand waits hostage to speculative batch stragglers.
        first_err = None
        if reserved:
            t_fill = time.monotonic_ns()
            landed: set = set()
            # _submit_fill can raise SYNCHRONOUSLY -- a landing callback that
            # errors or returns no views, or a scatter whose iovecs fail the
            # O_DIRECT alignment check. Any key already reserved when that
            # happens would keep its slot reserved and its pending event
            # unsignalled, and the next ensure of that key would wait on an
            # event nothing will ever set (Bugbot, gnf4#118). Reclaim
            # exactly what was reserved and re-raise.
            fut_of = {}
            try:
                for k, s in reserved:
                    fut_of[self._submit_fill(layer, k, s)] = (k, s)
            except BaseException:
                with self._lock:
                    for k, s in reserved:
                        if (k, s) in {v for v in fut_of.values()}:
                            continue          # already in flight; drained below
                        self._key_of[s] = None
                        self._reserved.discard(s)
                        self._demand_protected.discard(s)
                        if s not in self._free:
                            self._free.append(s)
                        ev = self._pending.pop(k, None)
                        if ev is not None:
                            ev.set()          # wake waiters; they refetch
                raise
            for fut in as_completed(fut_of):
                key, slot = fut_of[fut]
                try:
                    fut.result()
                except Exception as exc:          # noqa: BLE001 - re-raised below
                    if first_err is None:
                        first_err = exc
                    continue
                landed.add((key, slot))
                with self._lock:
                    self._key_of[slot] = key
                    self._slot_of[key] = slot
                    self._reserved.discard(slot)
                    self._pending.pop(key).set()
            with self._lock:
                dt = time.monotonic_ns() - t_fill
                if speculative:
                    self.spec_fill_ns += dt
                else:
                    self.demand_fill_ns += dt
                if first_err is not None:
                    # Reclaim EXACTLY the rows this batch reserved that never
                    # landed — tracked explicitly, never inferred from the
                    # maps. (Inferring via `_slot_of.get(key) != slot` freed
                    # slots that a concurrent ensure had already evicted and
                    # legitimately re-reserved after a published sibling was
                    # unreserved: two fills into one "free" slot.) These
                    # slots are still in _reserved (only a publish removes
                    # them), so nobody else can be holding them.
                    for key, slot in reserved:
                        if (key, slot) in landed:
                            continue
                        self._key_of[slot] = None
                        self._reserved.discard(slot)
                        self._demand_protected.discard(slot)
                        if slot not in self._free:
                            self._free.append(slot)
                        ev = self._pending.pop(key, None)
                        if ev is not None:
                            ev.set()              # wake waiters; they refetch
        if first_err is not None:
            raise first_err
        if speculative:
            return None

        # Resolve keys someone else was filling. A set event does not prove
        # the fill landed (it may have failed, or LFU may have already evicted
        # the row) — re-check under the lock and refetch on a same-request
        # window (fresh_window=False keeps THIS request's rows protected).
        for key, ev in waits:
            t_wait = time.monotonic_ns()
            ev.wait()
            with self._lock:
                self.demand_wait_ns += time.monotonic_ns() - t_wait
                slot = self._slot_of.get(key)
                if slot is not None:
                    self.hits += 1
                    self._resurrect_locked(key, False)
                    resolved[key] = slot
                    self._demand_protected.add(slot)
                    continue
            resolved[key] = self._ensure(layer, [key], [key], False,
                                         fresh_window=False)[0]
        if fresh_window:
            # One logical-eviction pass per demand ensure, after this
            # request's window is protected — never inside the recursive
            # refetch above, which would demote rows the outer call is
            # still resolving.
            with self._lock:
                self._demote_locked()
        return [resolved[k] for k in keys]

    def _claim_slot(self, own: set, speculative: bool):
        """A free or victimizable slot, or ``None`` for a speculative caller
        with nothing evictable. A demand caller finding every candidate merely
        RESERVED (in-flight speculative fills) waits for a fill to publish and
        rescans rather than failing — reservations resolve at disk speed, and
        raising there would turn transient speculative pressure into a serving
        error the old serialized structure could never produce."""
        while True:
            if self._free:
                return self._free.pop()
            try:
                slot = self._victim(own | self._demand_protected)
            except RuntimeError:
                if speculative:
                    return None
                blockers = [ev for k, ev in self._pending.items()]
                if not blockers:
                    raise                         # genuinely oversubscribed
                self._lock.release()
                try:
                    blockers[0].wait(timeout=30.0)
                finally:
                    self._lock.acquire()
                continue
            old = self._key_of[slot]
            if old is not None:
                del self._slot_of[old]
                self._key_of[slot] = None         # unpublish before refilling
                self.evictions += 1
                born = self._reclaimable.pop(old, None)
                if born is not None:
                    # a logical eviction resolving the losing way: the bytes
                    # were never re-requested and the slot is now spent
                    self.reclaimable_overwritten += 1
                    self._reclaim_ticks += self._clock - born
            # The slot's contents change from here — bump BEFORE the fill, so
            # a generation snapshot can never straddle a refill. Bumped on the
            # free-list path too: "generation moved" must mean "may not be the
            # bytes you saw", never "was definitely evicted".
            self._gen[slot] += 1
            return slot

    def _resurrect_locked(self, key, speculative: bool) -> None:
        """Promote a RECLAIMABLE row back to ACTIVE. Metadata only — the bytes
        never moved, which is the whole claim. No-op for an ACTIVE row."""
        born = self._reclaimable.pop(key, None)
        if born is None:
            return
        self._resurrect_ticks += self._clock - born
        if speculative:
            self.spec_resurrections += 1
        else:
            self.resurrections += 1

    def _demote_locked(self) -> None:
        """Revoke capacity ownership from the worst-ranked ACTIVE rows until
        the active set fits ``protected_rows``.

        This is the *logical eviction* the directive names, and the event
        whose interval-until-overwrite is the thing under test. It touches no
        bytes and issues no I/O: a demoted row stays in ``_slot_of``, stays
        readable through :meth:`row`, and a request for it before some later
        claim overwrites its slot is a resurrection.

        Rows in the current demand window are never demoted — the caller is
        between its ``ensure`` and its reads, and revoking there would count
        a resurrection for a row that was never at risk.
        """
        if self.protected_rows >= self.hot_rows:
            return                    # nothing can be reclaimable: today's tier
        over = (len(self._slot_of) - len(self._reclaimable)) - self.protected_rows
        if over <= 0:
            return
        cands = [k for k, s in self._slot_of.items()
                 if k not in self._reclaimable
                 and s not in self._demand_protected
                 and s not in self._reserved]
        cands.sort(key=lambda k: (self._freq[k], self._last_use.get(k, 0)))
        for k in cands[:over]:
            self._reclaimable[k] = self._clock
            self.logical_evictions += 1

    def row(self, layer: int, expert: int) -> memoryview:
        """A resident row's bytes (``row_bytes``, excluding alignment padding).

        Lock-guarded: the residency maps are only published after a fill lands, so
        taking the lock here is what guarantees a caller can never be handed a
        view of a slot whose read is still in flight.
        """
        if self._landing is not None:
            raise RuntimeError(
                "this tier fills an EXTERNAL landing, so its own buffer never "
                "receives a row — row() would hand back uninitialized bytes. "
                "Read the consumer's stacks, or build the tier without "
                "landing=.")
        with self._lock:
            slot = self._slot_of.get((layer, int(expert)))
            if slot is None:
                raise KeyError(f"(layer {layer}, expert {expert}) not resident")
            return self._slot_view(slot)[:self.row_bytes]

    def resident(self, layer: int, expert: int) -> bool:
        with self._lock:
            return (layer, int(expert)) in self._slot_of

    def attach_landing(self, callback) -> None:
        """Set the external landing AFTER construction.

        The tier and its consumer are mutually referential — the tier needs
        the consumer's ``landing`` callback, the consumer needs the tier's
        ``hot_rows`` and geometry — so one of them has to be built first.
        This closes the loop without a two-phase constructor.

        Refused once any fill has happened: rows already in this tier's own
        buffer would become unreachable the moment the landing redirects
        (``row()`` starts refusing), and rows filled after it would be the
        only readable ones. A tier that served both would be handing out two
        different meanings of "resident".
        """
        if self.requests:
            raise RuntimeError(
                f"attach_landing() after {self.requests} request(s): rows "
                f"already filled into this tier's own buffer would become "
                f"unreachable when the landing redirects. Attach before the "
                f"first ensure().")
        self._landing = callback
        # Construction with landing= allocates ONE row instead of
        # hot_rows*row_stride; a late attach must do the same or it keeps a
        # full landing nothing will ever write, doubling the DRAM the
        # scatter path exists to free (Bugbot, gnf4#120).
        self.buffer, self._keepalive = alloc_landing(
            self.row_stride, pinned=self.pinned)

    def slot_of(self, layer: int, expert: int):
        """The slot holding this row, or None. Lets a consumer ask what the
        tier thinks BEFORE calling ensure — which is the only moment at
        which a row can be invalidated, since ensure protects what it
        resolves."""
        with self._lock:
            return self._slot_of.get((layer, int(expert)))

    def invalidate(self, layer: int, expert: int) -> bool:
        """Drop a row's residency so the next :meth:`ensure` refills it.

        Returns True if a mapping was dropped. Refuses a slot with a fill in
        flight (its bytes are still landing) and one inside the current
        demand window (a caller is between its ensure and its reads) — in
        both cases dropping the mapping would strand a reader.

        Exists for a consumer that can tell a row is not usable to IT even
        though the tier considers it resident: an external-landing view
        whose stacks were never written for that slot, because the row was
        filled before the landing was attached. Without this the view would
        either serve bytes it never landed or have to give up on any tier
        with prior residency.
        """
        key = (layer, int(expert))
        with self._lock:
            slot = self._slot_of.get(key)
            if slot is None:
                return False
            if slot in self._reserved or slot in self._demand_protected:
                return False
            del self._slot_of[key]
            self._key_of[slot] = None
            self._reclaimable.pop(key, None)
            self._gen[slot] += 1
            if slot not in self._free:
                self._free.append(slot)
            return True

    def reclaimable(self, layer: int, expert: int) -> bool:
        """True iff this row is mapped but has lost capacity ownership — a
        request for it now is a resurrection, not a read."""
        with self._lock:
            return (layer, int(expert)) in self._reclaimable

    def generations(self, slots) -> list[int]:
        """Generation stamps for ``slots``, to snapshot alongside an
        :meth:`ensure` result. Pair them and hand both to :meth:`validate`
        before trusting a slot reference that outlived its ensure."""
        with self._lock:
            return [self._gen[int(s)] for s in slots]

    def validate(self, layer: int, expert: int, slot: int, generation: int) -> bool:
        """Are (slot, generation)'s bytes still this expert's?

        False means the slot was claimed since the snapshot — the bytes may be
        another expert's, or half of one. A caller that skips this check and
        reads anyway gets a plausible tensor, which is the failure mode this
        method exists to make impossible to reach by accident.
        """
        slot = int(slot)
        with self._lock:
            if not 0 <= slot < self.hot_rows:
                return False
            return (self._gen[slot] == int(generation)
                    and self._key_of[slot] == (layer, int(expert)))

    def _refuse_if_external(self, what: str):
        if self._landing is not None:
            raise RuntimeError(
                f"{what} exposes this tier's own landing buffer, which an "
                f"external-landing tier never fills. The bytes live in the "
                f"consumer's stacks.")

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
        self._refuse_if_external("pinned_tensor()")
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
        self._refuse_if_external("buffer_ptr")
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
            # hits/misses count EVERY ensure, speculative included — a
            # prefetcher inflates both. The split below is what a stall
            # metric must read: demand_misses are synchronous fetches on
            # the caller's critical path; spec_misses are background warming.
            "demand_misses": self.demand_misses,
            "spec_misses": self.spec_misses,
            "spec_hits": self.spec_hits,
            "demand_waits": self.demand_waits,
            "demand_fill_ns": self.demand_fill_ns,
            "demand_wait_ns": self.demand_wait_ns,
            "spec_fill_ns": self.spec_fill_ns,
            "evictions": self.evictions,
            "resident_rows": len(self._slot_of),
            "hot_rows": self.hot_rows,
            # --- reclaimable residency ---------------------------------- #
            # `evictions` above counts PHYSICAL overwrites; `logical_evictions`
            # counts revoked ownership. R8 predicts these diverge, and that the
            # physical one is the operational metric.
            "protected_rows": self.protected_rows,
            "reclaimable_rows": len(self._reclaimable),
            "logical_evictions": self.logical_evictions,
            "resurrections": self.resurrections,
            "spec_resurrections": self.spec_resurrections,
            "reclaimable_overwritten": self.reclaimable_overwritten,
            # bytes NOT read from NVMe because a demand request found bytes a
            # logical eviction had left in place
            "resurrection_bytes_saved": self.resurrections * self.row_bytes,
            # P(reuse before overwrite | logical eviction), over RESOLVED
            # evictions only — rows still sitting reclaimable have not
            # resolved either way and belong in neither term. None until one
            # resolves, deliberately: a 0.0 from an empty denominator would
            # read as a measured refutation of R1.
            "reuse_before_overwrite": (
                (self.resurrections + self.spec_resurrections) / resolved
                if (resolved := (self.resurrections + self.spec_resurrections
                                 + self.reclaimable_overwritten)) else None),
            "mean_ticks_to_overwrite": (
                self._reclaim_ticks / self.reclaimable_overwritten
                if self.reclaimable_overwritten else None),
            "mean_ticks_to_resurrection": (
                self._resurrect_ticks / res
                if (res := self.resurrections + self.spec_resurrections)
                else None),
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
#
# The FP8 tags map to ``uint8`` on purpose: this is a byte-relocation tier, and
# these tags label BYTES it must hand back unchanged, not values to convert.
# `mxfp4_residency._PACKED_BYTE_DTYPES` and `nvme_bake_nf4._MXFP4_BYTE_DTYPES`
# already say the same thing -- DeepSeek-V4 labels its MXFP4 experts
# ``I8``/``F8_E8M0`` where Kimi K3 labels both ``U8``, same bytes either way --
# and this table was the one place that had not been told. The consequence was
# narrow and total: the bake could WRITE a V4 arena and `mxfp4_residency` could
# SERVE from it, while anything going through :func:`segment_geometry` (the
# training tier's geometry check) died with ``KeyError: 'F8_E8M0'``.
#
# ``float8_e8m0fnu`` would be the wrong target even where torch has it (>= 2.7).
# An e8m0 byte is an EXPONENT; materializing it as a float and later casting
# yields the value (2**-5 -> 0) instead of the exponent byte (122), so every
# block would be scaled by 2**-127. Reinterpreting as ``uint8`` is what every
# consumer of these bytes already does.
_ST_TO_TORCH = {
    "U8": "uint8", "I8": "int8", "F16": "float16", "BF16": "bfloat16",
    "F32": "float32", "F64": "float64", "I16": "int16", "I32": "int32",
    "I64": "int64", "U16": "uint16", "U32": "uint32", "U64": "uint64",
    "F8_E8M0": "uint8", "F8_E4M3": "uint8",
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


_WIDENING = None


def widening_casts():
    """``{(segment_dtype, destination_dtype)}`` a staging copy may convert.

    Deliberately a whitelist of WIDENING casts only, and only ones that are
    value-preserving in that direction: every bf16 and fp16 value is exactly a
    float32. So a segment stored narrower than the consumer wants costs a
    conversion and nothing else — which is what lets an arena store absmax as
    bf16 (bitwise lossless for a bf16 checkpoint, -5.6% of every row) while the
    kernel keeps being handed the fp32 absmax its contract specifies.

    NARROWING is not here and must not be added: it would silently round values
    on a staging path whose whole promise is that the bytes it serves are the
    bytes that were baked.

    Exported so consumers (e4b's ``check_arena_geometry``) test the same table
    rather than growing a second, drifting copy of the policy.
    """
    global _WIDENING
    if _WIDENING is None:
        import torch
        _WIDENING = frozenset({
            (torch.bfloat16, torch.float32),
            (torch.float16, torch.float32),
        })
    return _WIDENING


def segment_into(tier: "ColdTier", index: dict, layer: int, experts,
                 suffix: str, out, *, rows=None, non_blocking: bool = False,
                 ensure: bool = True):
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
    widen = out.dtype != dt and (dt, out.dtype) in widening_casts()
    if out.dtype != dt and not widen:
        raise TypeError(
            f"out has dtype {out.dtype} but segment {suffix!r} is {dt}. "
            f"Widening is allowed only for {sorted(str(a) + '->' + str(b) for a, b in widening_casts())}.")
    if tuple(out.shape[1:]) != shape:
        raise ValueError(f"out is {tuple(out.shape)}; segment {suffix!r} needs "
                         f"[R, {', '.join(str(s) for s in shape)}]")
    if not out.is_contiguous():
        raise ValueError("out must be contiguous — rows are filled as flat byte runs")

    # ``ensure=False`` for a caller that ALREADY made these rows resident.
    # A nested demand ensure is not free: it replaces the demand window and
    # runs the demotion pass, so a caller materializing a batch one expert
    # at a time logically evicts its own siblings mid-materialization --
    # inflating logical_evictions/resurrections, and dropping the window
    # protection that keeps a concurrent speculative ensure from overwriting
    # rows still being copied out (Bugbot, gnf4#112).
    slots = (tier.ensure(layer, experts) if ensure
             else [tier.slot_of(layer, e) for e in experts])
    if any(s is None for s in slots):
        missing = [e for e, s in zip(experts, slots) if s is None]
        raise KeyError(
            f"ensure=False but (layer {layer}) experts {missing[:8]} are not "
            f"resident — the caller must ensure them before asking for their "
            f"bytes")
    pinned = tier.pinned_tensor() if tier.pinned else None
    for r, e, slot in zip(dst_rows, experts, slots):
        if widen:
            # A narrower segment feeding a wider destination: the bytes cannot
            # be memcpy'd, they have to be READ at the segment's dtype and
            # converted. torch does the conversion inside copy_, so this stays
            # one H2D and the destination keeps whatever dtype the kernel wants
            # -- which is the point: bf16 absmax on disk, fp32 absmax in VRAM,
            # kernel contract untouched.
            dst = out[r].reshape(-1)
            if pinned is not None:
                src = pinned[slot, off:off + ln].view(dt)
            else:
                mv = tier.row(layer, e)[off:off + ln]
                src = torch.frombuffer(bytearray(mv), dtype=torch.uint8).view(dt)
            dst.copy_(src, non_blocking=non_blocking)
            continue
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


#: Host RAM a PINNED row actually costs, as a multiple of ``row_stride``.
#:
#: Measured 2026-08-13 by laddering a container's swap-inclusive memory cap until
#: the process died, with no model loaded — the tier alone. A ``ColdTier`` at
#: ``hot_rows=128`` needed (768, 896] MiB and at 512 needed (2304, 2560] MiB:
#: **3.84–4.89 MB per row against a 2.654 MB stride, i.e. 1.45–1.84×.**
#:
#: It is not this module's doing. The same effect reproduces on a bare
#: ``torch.empty(n).pin_memory()`` with no gnf4 code in the process — a 1 GiB
#: pinned buffer needs (2048, 2560] MiB where the identical *pageable* buffer
#: needs (1280, 1536] MiB. Allocating pinned directly
#: (``torch.empty(n, pin_memory=True)``) changes nothing, so it is not the
#: pageable-source copy either; page-locked pages simply cost the cgroup more
#: than their nominal size.
#:
#: 1.9 is the conservative end of the measured band, so this UNDER-promises
#: capacity. Measured on cgroup v1 + driver 575.64.05 + torch 2.8.0+cu128; treat
#: it as a starting point on other stacks, not a constant of nature.
PINNED_ROW_FACTOR = 1.9


def capacity_for_bytes(usable_bytes: int, row_stride: int, *,
                       pinned: bool = True,
                       factor: float | None = None) -> int:
    """How many rows fit in a byte budget.

    Use MEASURED free RAM, never a declared figure: a pod rented with 4 GPUs
    still exposed 503 GB, identical to a 1-GPU pod (2026-07-30).

    ``pinned`` defaults to True because :class:`ColdTier` does. A pinned row
    costs about :data:`PINNED_ROW_FACTOR` × ``row_stride`` of real host memory,
    so dividing a budget by the stride alone over-promises capacity by that
    factor and hands back a ``hot_rows`` that OOMs partway through the first
    step. Pass ``pinned=False`` for the mmap tier, where a row costs its stride.

    Args:
        usable_bytes: measured free host RAM to spend on the tier.
        row_stride: bytes per row, from the arena index.
        pinned: whether the tier will page-lock its buffer.
        factor: override the multiplier; measure your own with a cap ladder
            rather than guessing, and see :data:`PINNED_ROW_FACTOR` for how.
    """
    f = factor if factor is not None else (PINNED_ROW_FACTOR if pinned else 1.0)
    if f <= 0:
        raise ValueError("factor must be > 0")
    return max(1, int(int(usable_bytes) // (int(row_stride) * f)))
