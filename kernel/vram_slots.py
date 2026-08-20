# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Reclaimable residency for a runtime-owned VRAM slot arena (R2, R3).

The DRAM half of this shipped in `nvme_residency`: logical eviction revokes
capacity ownership, the bytes survive until something overwrites the slot,
and a request in between is a resurrection costing no I/O. This is the same
idea one tier up, where it is worth more — a VRAM resurrection avoids the
whole refill chain (NVMe read, host staging, H2D, gather), not just the
disk read.

**Why this is a separate object rather than a flag on the engine.** The
directive is explicit that VRAM resurrection must not be built on
`del tensor; torch.cuda.empty_cache()`: a freed allocation may be
repurposed without the runtime's knowledge, so "the bytes are probably still
there" is not a property you can hold. It requires an arena the runtime
owns, with slots it accounts for itself. This is that accounting, kept
separate from any one engine's gather so it can be tested without a GPU.

**Four states, and the third is the one that does not exist in DRAM.**

    ACTIVE       wanted now; protected
    RETIRING     logically evicted, but in-flight GPU work may still read it
    RECLAIMABLE  no reader can remain; bytes valid; first to be overwritten
    absent       refilled with another expert

`RETIRING` is the whole reason `RowPool` records a CUDA event on demotion
and flips ownership only once a NON-BLOCKING query says it completed. A
slot released while a kernel is still reading it is a correctness bug that
looks like a numerics bug, so the transition out of `RETIRING` is gated on
the event, never on a timer or a guess. This class holds the state machine;
the caller supplies the event's completion, because "has this event landed"
is a torch question and this file has no torch in it.

No torch import: the caller passes booleans. That keeps the state machine
testable on any box, which matters because its failure mode — serving one
expert's bytes as another's — is exactly what a GPU-only test suite tends
to catch late.
"""
from __future__ import annotations

ACTIVE, RETIRING, RECLAIMABLE, ABSENT = (
    "active", "retiring", "reclaimable", "absent")


class VramSlots:
    """Slot accounting for a runtime-owned device expert arena.

    Args:
        n_slots: physical slots in the arena.
        protected: capacity-ownership budget, ``<= n_slots``. Slots beyond
            it hold RECLAIMABLE rows. ``None`` = ``n_slots``, which makes
            the reclaimable set permanently empty and every path below
            identical to a plain slot map.
    """

    def __init__(self, n_slots: int, *, protected: int | None = None):
        if n_slots < 1:
            raise ValueError("n_slots must be >= 1")
        if protected is None:
            protected = n_slots
        if not 1 <= protected <= n_slots:
            raise ValueError(
                f"protected={protected} must be in [1, n_slots={n_slots}] — "
                f"it is a budget WITHIN the arena, not a second arena")
        self.n_slots = n_slots
        self.protected = protected
        self._holds: list = [None] * n_slots      # expert id or None
        self._state: list = [ABSENT] * n_slots
        self._gen: list = [0] * n_slots
        self._pending: dict = {}                  # slot -> caller's event tag
        self._clock = 0

        self.gathers = 0            # slots that needed a real refill
        self.active_hits = 0        # already ACTIVE, nothing to do
        self.resurrections = 0      # RECLAIMABLE reused before overwrite
        self.logical_evictions = 0
        self.overwritten = 0        # reclaimable lost before reuse
        self.blocked_by_retiring = 0

    # ----------------------------------------------------------- helpers --
    def state(self, slot: int) -> str:
        return self._state[slot]

    def slot_of(self, expert):
        for s, e in enumerate(self._holds):
            if e == expert and self._state[s] != ABSENT:
                return s
        return None

    def generation(self, slot: int) -> int:
        return self._gen[slot]

    # ------------------------------------------------------------ events --
    def settle(self, completed) -> int:
        """Flip RETIRING slots whose in-flight readers have finished.

        ``completed(tag) -> bool`` is the caller's non-blocking event query;
        this file cannot ask torch. A slot only becomes RECLAIMABLE — that
        is, eligible to be handed to another expert — once its last reader
        is provably done. Never a sync, never a timer.
        """
        n = 0
        for slot, tag in list(self._pending.items()):
            if not completed(tag):
                continue
            del self._pending[slot]
            # Only a slot that is STILL retiring may be released. A slot
            # resurrected by want() since the tag was recorded is live, and
            # writing RECLAIMABLE over it would make it _claim's FIRST pick
            # -- handing an in-use assignment's bytes to another expert
            # (Bugbot, gnf4#128). want() drops the pending entry on
            # resurrection; this is the second lock on the same door.
            if self._state[slot] != RETIRING:
                continue
            self._state[slot] = RECLAIMABLE
            n += 1
        return n

    # -------------------------------------------------------------- want --
    def want(self, experts, *, event_tag=None):
        """Resolve a routed set to slots.

        Returns ``(assignment, gather_needed)``: a dict expert -> slot, and
        the subset of experts whose slot must actually be filled. An expert
        found ACTIVE or RECLAIMABLE needs **no** gather — the RECLAIMABLE
        case is the resurrection this class exists for, and it avoids the
        entire NVMe → staging → H2D → gather chain rather than just a disk
        read.

        ``event_tag`` is attached to slots this call logically evicts, so
        the caller can later tell :meth:`settle` when their readers finished.
        """
        experts = list(dict.fromkeys(experts))
        if len(experts) > self.n_slots:
            raise ValueError(
                f"{len(experts)} experts requested into {self.n_slots} slots")
        self._clock += 1
        assign, need = {}, []

        for e in experts:                     # hits first: never evict a hit
            s = self.slot_of(e)
            if s is None:
                continue
            if self._state[s] == RETIRING:
                # A self-hit on a retiring slot: the SAME expert wants bytes
                # that are still its own, so the in-flight readers are no
                # threat and this is a legitimate resurrection. It stops
                # being retired, though -- leaving the pending entry behind
                # let settle() later mark a live assignment RECLAIMABLE.
                self.blocked_by_retiring += 1
                self._pending.pop(s, None)
            elif self._state[s] == RECLAIMABLE:
                self.resurrections += 1
            else:
                self.active_hits += 1
            self._state[s] = ACTIVE
            assign[e] = s

        for e in experts:
            if e in assign:
                continue
            s = self._claim(set(assign.values()), event_tag)
            self._holds[s] = e
            self._state[s] = ACTIVE
            self._gen[s] += 1
            assign[e] = s
            need.append(e)
            self.gathers += 1

        self._demote(set(assign.values()), event_tag)
        return assign, need

    # ------------------------------------------------------------ claim --
    def _claim(self, protected_now: set, event_tag=None) -> int:
        """A slot for a new expert. RECLAIMABLE first — they lose every
        allocation contest, which is what makes them free capacity rather
        than a second arena — then ABSENT, then ACTIVE. Never RETIRING: its
        bytes may still be under a running kernel.

        The same argument bars ACTIVE **while the caller is pipelining.** An
        ACTIVE slot outside this request is one that survived the last
        _demote as within-budget, so it is the recent working set and its
        readers may well still be running — and it carries no event tag, so
        nothing here can prove otherwise. Under tags the natural victims are
        all RETIRING and get skipped, which made a live slot the common
        pick rather than a rare one (Bugbot, gnf4#128).

        Omitting ``event_tag`` is the caller asserting quiescence — the same
        assertion that sends _demote's victims straight to RECLAIMABLE — so
        the ACTIVE fallback is honored there and refused here.
        """
        order = ((RECLAIMABLE, ABSENT) if event_tag is not None
                 else (RECLAIMABLE, ABSENT, ACTIVE))
        for want in order:
            for s in range(self.n_slots):
                if s in protected_now or self._state[s] == RETIRING:
                    continue
                if self._state[s] == want:
                    if want == RECLAIMABLE:
                        self.overwritten += 1
                    return s
        raise RuntimeError(
            "no slot available: every slot is either wanted by this request, "
            "RETIRING with readers still in flight, or ACTIVE with readers "
            "this call cannot prove are done (you supplied an event tag, so "
            "they may not be). settle() first, or size the arena to the "
            "routed set. Overwriting a live slot is not offered as a "
            "fallback: it corrupts the reader instead of failing the "
            "allocation.")

    def _demote(self, keep: set, event_tag) -> None:
        """Revoke ownership from ACTIVE slots beyond the protected budget.

        The rows this request is using are never demoted — the caller is
        between resolve and use. Demoted slots go to RETIRING when the
        caller supplied an event tag (readers may be in flight) and straight
        to RECLAIMABLE when it did not, which is the caller asserting there
        is nothing to wait for.
        """
        if self.protected >= self.n_slots:
            return
        active = [s for s in range(self.n_slots)
                  if self._state[s] == ACTIVE and s not in keep]
        over = (len(keep) + len(active)) - self.protected
        if over <= 0:
            return
        for s in active[:over]:
            if event_tag is not None:
                self._state[s] = RETIRING
                self._pending[s] = event_tag
            else:
                self._state[s] = RECLAIMABLE
            self.logical_evictions += 1

    # ------------------------------------------------------------- stats --
    def stats(self) -> dict:
        resolved = self.resurrections + self.overwritten
        return {
            "n_slots": self.n_slots, "protected": self.protected,
            "gathers": self.gathers, "active_hits": self.active_hits,
            "resurrections": self.resurrections,
            "logical_evictions": self.logical_evictions,
            "overwritten": self.overwritten,
            "blocked_by_retiring": self.blocked_by_retiring,
            "retiring_now": len(self._pending),
            "reclaimable_now": sum(1 for s in self._state if s == RECLAIMABLE),
            # R2/R3's load-bearing probability, over RESOLVED evictions only;
            # None until one resolves, so an empty denominator cannot read as
            # a measured refutation.
            "reuse_before_overwrite": (self.resurrections / resolved
                                       if resolved else None),
        }
