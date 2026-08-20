# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Stage 3 / workstream 1 — the CPU destination for cold experts.

:class:`~nvme_residency.ColdTier` lands an expert's packed bytes in *arena row*
layout: segments nested inside one fixed-stride row, scattered across slots.
The native CPU kernels (``cpu_grouped.gemv_nf4_grouped_cpu`` and friends) want
the opposite — one contiguous ``[E, N, K//2]`` stack per segment, indexed by a
group's ``expert_ids``. A cold expert therefore cannot reach the CPU engine
without something re-laying-out its bytes, and that something is this module.

**Slot-parallel by construction.** The view holds exactly ``tier.hot_rows``
rows and materializes slot *s* of the tier into row *s* of every stack, so the
tier's residency IS the view's residency: one eviction policy, one set of
counters, no second vocabulary (the rule that produced ``RowPool`` rather than
a parallel KV tier). ``ensure`` returns the tier's own slot indices, which are
handed to the kernel as ``expert_ids`` directly — the kernel never learns that
these are cache rows rather than a model's expert ids, which is what keeps the
scheduler above the format boundary.

**What a hit avoids.** Slot generations (``ColdTier.validate``) decide whether
a slot's materialization is still current. A resurrected tier row keeps its
generation — no bytes moved — so a resurrection is a hit *here* too: no disk
read and no re-layout. That is the reclaimable-residency win arriving at the
CPU engine intact, and :meth:`stats` reports both halves separately because
they are different costs.

**The price, stated plainly.** The view is a second copy of the resident set,
in kernel layout: ``hot_rows * row_bytes`` of DRAM on top of the tier's own.
It buys the kernel's contiguous-stack contract without touching the kernel or
the packed bytes. A landing path that scattered segments straight from NVMe
into these stacks (the ``preadv`` iovec trick ``ArenaExpertSource`` already
uses) would remove the copy and the duplicate footprint both; it is the named
follow-up, deliberately not half-shipped here.

No torch at module top (house rule: importable everywhere).
"""
from __future__ import annotations

from nvme_reader import alloc_landing, check_aligned
from nvme_residency import segment_geometry, segment_into


def scatter_layout(index: dict, segments):
    """``[(suffix|None, length)]`` covering one whole row in FILE order, or
    ``None`` when a scattering read would not be legal for this arena.

    ``None`` entries are inter-segment gaps and trailing padding that scratch
    must absorb, because ``preadv`` fills its iovec list sequentially and
    cannot skip. Mirrors ``arena_experts._scatter_layout`` deliberately —
    same refusal rules, same reason.

    Refused when any segment length, gap, or the row padding is not
    ``align``-aligned: O_DIRECT would EINVAL, or every following destination
    would be pushed off alignment. Returning ``None`` rather than guessing is
    the point; the caller falls back to the copy path and says so.
    """
    align = index["align"]
    segs = sorted(index["segments"], key=lambda g: g["seg_off"])
    row_stride = index["row_stride"]
    plan, cur, ok = [], 0, True
    for g in segs:
        gap = g["seg_off"] - cur
        if gap:
            plan.append((None, gap))
            ok &= gap % align == 0
        plan.append((g["suffix"], g["length"]))
        ok &= g["length"] % align == 0
        cur = g["seg_off"] + g["length"]
    pad = row_stride - cur
    if pad:
        plan.append((None, pad))
        ok &= pad % align == 0
    # A view that materializes only SOME segments still has to absorb the
    # others: the read covers a whole row either way, so an unwanted segment
    # becomes scratch rather than a hole.
    want = set(segments)
    plan = [(sfx if sfx in want else None, ln) for sfx, ln in plan]
    return plan if ok else None


class ColdCpuView:
    """Kernel-shaped, slot-parallel materialization of a cold tier's rows.

    Args:
        tier: the :class:`~nvme_residency.ColdTier` supplying rows. Its
            ``hot_rows`` is this view's capacity, exactly.
        index: the arena index (``nvme_arena.load_index``).
        segments: segment suffixes to materialize, e.g.
            ``("gate_up_blocks", "gate_up_scales")``. Only what the kernel
            reads — materializing an unused segment is pure waste.
        casts: optional ``{suffix: torch dtype}`` for segments whose kernel
            contract is wider than their stored dtype (the bf16-absmax case).
            Only the widening casts in
            :func:`~nvme_residency.widening_casts` are permitted, and the
            check lives there rather than being re-implemented here.
    """

    def __init__(self, tier, index: dict, segments, *, casts=None,
                 direct=False):
        import torch

        self.tier = tier
        self.index = index
        self.segments = tuple(segments)
        if not self.segments:
            raise ValueError("materialize at least one segment")
        casts = dict(casts or {})
        unknown = set(casts) - set(self.segments)
        if unknown:
            raise ValueError(f"casts name segments this view does not hold: "
                             f"{sorted(unknown)}")

        self.direct = bool(direct)
        # Cast first: it is a contradiction in the CALLER's request, so it
        # should be reported whatever the arena's geometry happens to be.
        if direct and casts:
            raise ValueError(
                "direct=True cannot cast: the kernel DMAs the segment's own "
                "bytes into the stack, so a widening conversion has nowhere "
                "to happen. Drop the cast or drop direct=.")
        self._layout = scatter_layout(index, self.segments) if direct else None
        if direct and self._layout is None:
            raise ValueError(
                "direct=True but this arena's geometry cannot scatter: some "
                "segment length, inter-segment gap, or row padding is not a "
                "multiple of align. Construct without direct= to use the copy "
                "path (correct, one host memcpy per segment per fill).")

        self.stacks: dict = {}
        self._keep = []
        for suffix in self.segments:
            dt, shape, _off, ln = segment_geometry(index, suffix)
            dt = casts.get(suffix, dt)
            if not direct:
                self.stacks[suffix] = torch.empty(
                    (tier.hot_rows, *shape), dtype=dt).contiguous()
                continue
            # Page-aligned, because every row of this stack becomes an
            # O_DIRECT iovec base. torch's allocator gives 64 B; alloc_landing
            # gives `align`, and a misaligned base is an EINVAL far from its
            # cause (the lesson alloc_landing's own docstring records).
            mv, keep = alloc_landing(tier.hot_rows * ln)
            self._keep.append(keep)
            self._mv = getattr(self, "_mv", {})
            self._mv[suffix] = mv
            t = torch.frombuffer(mv, dtype=torch.uint8)
            self.stacks[suffix] = t.view(dt).view(tier.hot_rows, *shape)
            check_aligned(mv, index["align"])

        # What each slot currently holds, as the tier's own identity pair. A
        # slot's ADDRESS never identifies its contents (the invalidation
        # lesson this repo learned on the device gather); (key, generation)
        # does, and the tier is the authority on both.
        self._held: list[tuple | None] = [None] * tier.hot_rows

        self.materializations = 0
        self.view_hits = 0
        self.rows_requested = 0

    # ------------------------------------------------------------- landing --
    def landing(self, layer: int, expert: int, slot: int):
        """Views for one row's scattering read, in FILE order.

        Hand this to ``ColdTier(landing=...)`` and the kernel DMAs each
        segment straight into the stack row the CPU kernels will index — no
        arena-row stop, no ``segment_into`` memcpy. The tier still owns
        residency; only the destination moves.

        Scratch absorbs gaps, padding, and any segment this view does not
        materialize, because ``preadv`` fills sequentially and cannot skip.
        """
        if not self.direct:
            raise RuntimeError("landing() requires direct=True")
        views = []
        for suffix, ln in self._layout:
            if suffix is None:
                views.append(memoryview(self._scratch(ln)))
            else:
                mv = self._mv[suffix]
                views.append(mv[slot * ln:(slot + 1) * ln])
        return views

    def _scratch(self, ln: int):
        """One aligned throwaway buffer per distinct gap/padding size, reused
        across rows. These bytes are read and discarded; nothing may depend on
        them, and two concurrent fills writing the same scratch is harmless
        for exactly that reason."""
        cache = getattr(self, "_scratch_cache", None)
        if cache is None:
            cache = self._scratch_cache = {}
        if ln not in cache:
            mv, keep = alloc_landing(ln)
            self._keep.append(keep)
            cache[ln] = mv
        return cache[ln]

    # ------------------------------------------------------------------ API --
    def stack(self, suffix: str):
        """The ``[hot_rows, *shape]`` tensor for one segment — the kernel's
        ``packed``/``absmax`` argument, handed over without a copy."""
        try:
            return self.stacks[suffix]
        except KeyError:
            raise KeyError(
                f"{suffix!r} is not materialized by this view "
                f"(have {list(self.segments)})") from None

    def ensure(self, layer: int, experts):
        """Make ``experts`` kernel-ready; return their **slot indices**, in
        request order, to pass straight through as ``expert_ids``.

        Residency is the tier's decision — this only re-materializes slots
        whose contents changed under it. A row the tier resurrected needs
        nothing at all.
        """
        experts = [int(e) for e in experts]
        slots = self.tier.ensure(layer, experts)
        self.rows_requested += len(experts)
        # Dedupe while preserving the caller's order: repeats in one request
        # share a slot (the tier's own contract), so re-laying-out per
        # occurrence would pay the copy several times for one row.
        seen: set[int] = set()
        for e, slot in zip(experts, slots):
            if slot in seen:
                continue
            seen.add(slot)
            key = (layer, e)
            gen = self.tier.generations([slot])[0]
            if self._held[slot] == (key, gen):
                self.view_hits += 1
                continue
            if not self.direct:
                for suffix in self.segments:
                    segment_into(self.tier, self.index, layer, [e], suffix,
                                 self.stacks[suffix], rows=[slot])
            # direct: tier.ensure() above already scattered this row into
            # these stacks through landing(), so there is nothing to copy.
            # The stamp below still runs — it is what makes a later hit
            # skippable.
            # Stamp only after every segment landed: a partially materialized
            # slot must never look current, exactly as the tier publishes a
            # row only once its whole read lands.
            self._held[slot] = (key, gen)
            self.materializations += 1
        return slots

    def holds(self, layer: int, expert: int) -> bool:
        """Is this expert currently materialized and current? Diagnostic — the
        scheduler asks the tier, not the view."""
        key = (layer, int(expert))
        for slot, held in enumerate(self._held):
            if held is not None and held[0] == key:
                return self.tier.validate(layer, int(expert), slot, held[1])
        return False

    def stats(self) -> dict:
        """View-local costs, beside the tier's. ``view_hits`` are requests that
        needed neither a disk read nor a re-layout; the tier's own
        ``resurrections`` say how many of those the reclaimable path saved."""
        total = self.view_hits + self.materializations
        return {
            "rows_requested": self.rows_requested,
            "view_hits": self.view_hits,
            "materializations": self.materializations,
            "view_hit_rate": (self.view_hits / total) if total else 0.0,
            "capacity_rows": self.tier.hot_rows,
            "segments": list(self.segments),
        }
