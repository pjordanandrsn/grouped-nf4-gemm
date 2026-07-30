# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""MXFP4 experts served from NVMe — the engine seam that lets a >host-RAM MoE run.

:class:`~mxfp4_pipelined.Mxfp4PipelinedGptOss` pins **all E rows** in host DRAM and
dispatches through a static address table. For Kimi K3 that arena is **1.446 TB**
(**92 routed** layers x 896 experts x 17,547,264 B, measured 2026-07-30 -- K3
has 93 layers in total but 92 carry a routed MoE block, and 92 x 896 = 82,432 rows
is exactly the baked arena) against a **503 GB**
per-container host-RAM ceiling that renting more GPUs does not raise. So the arena
is the thing to remove, not the thing to grow.

This module keeps the engine — same fused ``gemm_mxfp4_grouped``, same k-slot store,
same by-address gather — and replaces only *where a cold row's address comes from*:
:class:`~nvme_residency.ColdTier` reads it from NVMe into pinned DRAM and returns the
slot it landed in. The gather then reads that slot over UVA exactly as it read the
arena. No staging copy, no dequantization.

**Bit-identity.** MXFP4 bytes are served as MXFP4 bytes. Routing K3's experts
through the NF4 engine instead would mean MXFP4 -> bf16 -> NF4, a lossy round trip;
nothing here converts a format, so the numbers the kernel sees are the numbers the
release shipped.

Two things the static-table engine gets for free and this one must earn:

1. **A tier slot's address identifies the SLOT, not its contents.** The gather skips
   a slot when its source address is unchanged since the last fetch — sound under a
   full arena, where address <-> expert is a bijection. Under a tier, expert 9 can
   land in the very slot expert 5 just vacated, at the identical address, and the
   skip would serve expert 5's weights as expert 9's. :meth:`Mxfp4NvmeResidency._invalidate`
   forces a re-gather whenever the expert occupying a device slot changed, so the
   optimization survives and the correctness bug does not.
2. **Slots stride by ``row_stride``, not ``row_bytes``.** The arena pads rows out to
   ``align`` (4096) for O_DIRECT while the engine's own table strides by its
   8-aligned ``row_bytes``. Mixing them starts reading mid-row at slot 1 and
   produces plausible garbage, never an error.

Not CUDA-graph capturable: making a row resident is a host-side disk read, so
:meth:`_resolve_src` needs the wanted ids on the CPU. It raises inside capture
rather than baking stale addresses into a graph.
"""
from __future__ import annotations

import torch

from mxfp4_pipelined import Mxfp4PipelinedGptOss, _align8
from nvme_residency import ColdTier

# The order this engine READS in. No longer the order a bake must WRITE in.
#
# The engine reads `[gu_blocks | gu_scales | dn_blocks | dn_scales]`, so a row
# feeds it directly only when the two blocks segments are adjacent and likewise
# the two scales:
#
#     w1_packed | w3_packed | w1_scale | w3_scale | w2_packed | w2_scale
#     \___ engine's gu_blocks __/ \___ gu_scales __/ \_ dn_b _/ \_ dn_s _/
#
# `arena_experts.K3_KINDS` interleaves per projection instead
# (w1_packed, w1_scale, w3_packed, w3_scale, ...). That is correct for
# `ArenaExpertSource`, which slices by suffix out of the index, but it puts
# w1_scale in the middle of what this engine reads as gu_blocks.
#
# The released K3 arena on disk is in the INTERLEAVED order (checked 2026-07-30:
# 82,432 rows, 1.446 TB, baked before this was noticed). Re-baking it would cost
# 1.45 TB of IO and permuting it in place ~482 GB, so instead the gather reorders
# segments as it copies each row into its device slot -- the copy was happening
# anyway. See `_perm_gather_kernel` and `Mxfp4NvmeResidency._init_permutation`.
# Any bake order is now readable; this constant is what the engine WANTS, and a
# fresh bake may as well use it to take the identity fast path.
#
# Names are the released-K3 spelling, taken from `arena_experts.K3_KINDS` rather
# than guessed -- guessing tensor names is what silently returned n_experts=0
# from `moonshot_gather` on K3 (three renames since K2).
K3_RESIDENCY_KINDS = ("w1.weight_packed", "w3.weight_packed",
                      "w1.weight_scale", "w3.weight_scale",
                      "w2.weight_packed", "w2.weight_scale")


_PERM_KERNEL = None


def _perm_gather_kernel():
    """Per-slot gather that REORDERS segments as it copies.

    Same contract as ``mxfp4_pipelined._gather_kernel`` — per-slot absolute source
    address, skip when unchanged — except the row is copied piecewise: each program
    handles one precomputed BLOCK-sized chunk carrying its own (src, dst) word
    offsets. Because the row is already being copied into the device slot, landing
    the segments in a different ORDER costs no extra bandwidth. That is what makes
    a mis-ordered 1.45 TB arena readable without rewriting it.

    Still format-agnostic: it moves int64 words and cannot see MXFP4.
    """
    # `tl` must land in MODULE globals, not this function's locals: with
    # `from __future__ import annotations` the `BLOCK: tl.constexpr` annotation is
    # the STRING "tl.constexpr", which triton evaluates against the jitted
    # function's __globals__ at compile time. A local import gives
    # NameError('tl is not defined') from inside the triton compiler — on
    # triton 3.2; 3.4 happens to tolerate it, so this hides on newer boxes.
    global _PERM_KERNEL, tl
    if _PERM_KERNEL is None:
        import triton
        import triton.language as tl

        @triton.jit
        def _gather_rows_perm(dst_ptr, src_ptr, have_ptr, c_src, c_dst, c_len,
                              row_words, BLOCK: tl.constexpr):
            slot = tl.program_id(0)
            piece = tl.program_id(1)
            want = tl.load(src_ptr + slot)
            have = tl.load(have_ptr + slot)
            if want == have:
                return
            so = tl.load(c_src + piece)
            do = tl.load(c_dst + piece)
            n = tl.load(c_len + piece)
            offs = tl.arange(0, BLOCK)
            mask = offs < n
            src = tl.cast(want, tl.pointer_type(tl.int64))
            vals = tl.load(src + so + offs, mask=mask)
            tl.store(dst_ptr + slot.to(tl.int64) * row_words + do + offs,
                     vals, mask=mask)

        _PERM_KERNEL = _gather_rows_perm
    return _PERM_KERNEL


def _chunk_table(pieces, block_words: int):
    """Split (src_off, dst_off, n) byte-triples into BLOCK-sized word chunks.

    Precomputed on the host so every launched program does real work — a
    grid over (segment, chunk) would launch the widest segment's chunk count for
    the narrow scale segments too, which here is ~47% waste.
    """
    src, dst, ln = [], [], []
    for s_off, d_off, n in pieces:
        for bad, name in ((s_off, "src_off"), (d_off, "dst_off"), (n, "length")):
            if bad % 8:
                raise ValueError(
                    f"{name}={bad} is not 8-byte aligned; the gather moves int64 "
                    "words, so every segment offset and length must be. Re-bake "
                    "with 8-aligned segments.")
        s_w, d_w, n_w = s_off // 8, d_off // 8, n // 8
        for c in range(0, n_w, block_words):
            src.append(s_w + c)
            dst.append(d_w + c)
            ln.append(min(block_words, n_w - c))
    return src, dst, ln


def engine_segment_map(index: dict):
    """Map an arena's segments onto the engine's four, in engine order.

    Returns ``(groups, geometry)`` where ``groups`` is the engine's
    ``[gu_blocks, gu_scales, dn_blocks, dn_scales]``, each a list of source
    ``(seg_off, length)`` pieces to be concatenated.

    Two signals, neither of them tensor names — names are the obvious key and are
    exactly what has moved three times since K2:

    * **blocks vs scales** from the MXFP4 packing invariant. Blocks hold two
      values per byte (``k//2`` wide), scales one e8m0 byte per group of 32
      (``k//32``), so a projection's blocks width is always exactly 16x its
      scales width. Shape *equality* is NOT a usable signal: when hidden equals
      intermediate all three projections share both shapes.
    * **which projection** from SOURCE ORDER. Both ``arena_experts.K3_KINDS`` and
      :data:`K3_RESIDENCY_KINDS` place w1 before w3 before w2, so the blocks in
      source order are (gate, up, down) and likewise the scales. w1=gate is
      confirmed against the release by shape (``moonshot_gather.K3_SCHEME``).
    """
    segs = index["segments"]
    for g in segs:
        if g["dtype"] != "U8":
            raise ValueError(
                f"segment {g['suffix']!r} is {g['dtype']}, not U8 — native MXFP4 "
                "rows are packed nibbles + e8m0 scale bytes, both uint8. An F32 "
                "segment means this is an NF4 arena (fp32 absmax); serve it with "
                "experts4bit_qlora.nvme_experts instead.")
        if len(g["shape_per_expert"]) != 2:
            raise ValueError(f"segment {g['suffix']!r} shape "
                             f"{g['shape_per_expert']} is not [n, k]")
    if len(segs) not in (4, 6):
        raise ValueError(
            f"expected 4 segments (fused gate_up) or 6 (split w1/w3), got "
            f"{len(segs)}: {[g['suffix'] for g in segs]}")

    # BLOCKS vs SCALES from the MXFP4 packing invariant, not from shapes being
    # distinct and not from names: blocks hold 2 values per byte (k//2 wide) and
    # scales one e8m0 byte per group of 32 (k//32), so a projection's blocks width
    # is ALWAYS exactly 16x its scales width. Shape alone is not a discriminator —
    # when hidden == intermediate all three projections share both shapes.
    widths = {tuple(g["shape_per_expert"])[1] for g in segs}
    blocks, scales = [], []
    for g in segs:
        w = g["shape_per_expert"][1]
        is_scale = (w * 32 // 2) in widths          # w*16, spelled to keep ints
        is_block = (w % 16 == 0) and (w // 16) in widths
        if is_scale and not is_block:
            scales.append(g)
        elif is_block and not is_scale:
            blocks.append(g)
        else:
            raise ValueError(
                f"segment {g['suffix']!r} width {w} is ambiguous against the "
                f"widths present {sorted(widths)}: cannot tell packed blocks "
                f"(k//2) from e8m0 scales (k//32), which must differ by exactly "
                f"16x. Is this really an MXFP4 arena?")
    if len(blocks) != len(scales) or len(blocks) != len(segs) // 2:
        raise ValueError(
            f"expected {len(segs) // 2} blocks and {len(segs) // 2} scales, got "
            f"{[g['suffix'] for g in blocks]} / {[g['suffix'] for g in scales]}")

    # SOURCE ORDER assigns projections. Both arena_experts.K3_KINDS and
    # K3_RESIDENCY_KINDS place w1 before w3 before w2, so the blocks in source
    # order are (gate, up, down) and likewise the scales. w1=gate is confirmed
    # against the release by shape (see moonshot_gather.K3_SCHEME).
    n_gu = len(blocks) - 1                 # 2 for split w1/w3, 1 for fused
    gu_blocks, dn_blocks = blocks[:n_gu], blocks[n_gu:]
    gu_scales, dn_scales = scales[:n_gu], scales[n_gu:]

    def _one_shape(group, what):
        shapes = {tuple(g["shape_per_expert"]) for g in group}
        if len(shapes) != 1:
            raise ValueError(f"{what}: members disagree on shape {shapes} "
                             f"({[g['suffix'] for g in group]})")
        return shapes.pop()

    gu_b = _one_shape(gu_blocks, "gate_up blocks")
    gu_s = _one_shape(gu_scales, "gate_up scales")
    dn_b = _one_shape(dn_blocks, "down blocks")
    dn_s = _one_shape(dn_scales, "down scales")

    def pieces(group):
        return [(g["seg_off"], g["length"]) for g in group]

    groups = [pieces(gu_blocks), pieces(gu_scales),
              pieces(dn_blocks), pieces(dn_scales)]
    n1, n2 = gu_b[0] * len(gu_blocks), dn_b[0] * len(dn_blocks)
    if gu_s[0] * len(gu_scales) != n1 or dn_s[0] * len(dn_scales) != n2:
        raise ValueError(
            f"blocks/scales row counts disagree: gate_up {n1} vs "
            f"{gu_s[0] * len(gu_scales)}, down {n2} vs {dn_s[0] * len(dn_scales)}")
    for what, half, nb in (("gate_up", gu_b[1], gu_s[1]),
                           ("down", dn_b[1], dn_s[1])):
        if nb * 32 != half * 2:
            raise ValueError(
                f"{what}: {nb} scale groups do not tile {half * 2} columns at "
                f"MXFP4 group_size 32 (got {nb * 32})")
    return groups, (index["n_experts_per_layer"], n1, gu_b[1], gu_s[1],
                    n2, dn_b[1], dn_s[1])


def fuse_gate_up_segments(index: dict) -> dict:
    """Present a 6-segment (split w1/w3) arena as the engine's 4-segment layout.

    Returns a shallow copy of ``index`` with segments
    ``[gate_up_blocks, gate_up_scales, down_blocks, down_scales]``, where each
    fused segment spans its two source segments as ONE range.

    The fusion is only byte-exact if the pair is contiguous — the bake pads every
    segment to 8 bytes, so an odd-length w1 segment would leave a hole between w1
    and w3 that the engine's single ``[n1, half1]`` view would read straight
    through. That is checked, not assumed: K3's real dims happen to be 8-aligned
    (w1 blocks = 3072 x 1792 B, scales = 3072 x 112 B), which is exactly the kind
    of luck that should be asserted rather than relied on.
    """
    segs = index["segments"]
    if len(segs) == 4:
        return index
    if len(segs) != 6:
        raise ValueError(
            f"expected 4 fused or 6 split segments, got {len(segs)}: "
            f"{[g['suffix'] for g in segs]}")

    order_hint = (
        "This engine needs the two BLOCKS segments adjacent and the two SCALES "
        "segments adjacent, because it reads gate_up at a single computed offset:\n"
        f"    kinds={K3_RESIDENCY_KINDS}\n"
        "`arena_experts.K3_KINDS` interleaves per projection "
        "(w1_packed, w1_scale, w3_packed, w3_scale, ...) which is correct for "
        "ArenaExpertSource (it slices by suffix) but NOT for this engine.\n"
        f"This arena's order: {[g['suffix'] for g in segs]}")

    def _fuse(a, b, name):
        sa, sb = tuple(a["shape_per_expert"]), tuple(b["shape_per_expert"])
        if sa[1:] != sb[1:]:
            raise ValueError(
                f"{name}: cannot fuse {a['suffix']!r} + {b['suffix']!r} — their "
                f"trailing dims differ ({sa} vs {sb}), so these are not the same "
                f"KIND of segment. Almost certainly a blocks/scales pair, i.e. the "
                f"bake used the wrong kinds ORDER.\n{order_hint}")
        if a["seg_off"] + a["length"] != b["seg_off"]:
            raise ValueError(
                f"{name}: segments {a['suffix']!r} and {b['suffix']!r} are not "
                f"contiguous ({a['seg_off']}+{a['length']} != {b['seg_off']}) — "
                "the bake's 8-byte padding left a hole, so they cannot be read "
                "as one fused tensor. Re-bake with 8-aligned segment lengths.")
        if a["dtype"] != b["dtype"]:
            raise ValueError(f"{name}: dtype mismatch "
                             f"{a['dtype']} vs {b['dtype']}")
        return {"suffix": f"{a['suffix']}+{b['suffix']}", "seg_off": a["seg_off"],
                "length": a["length"] + b["length"], "dtype": a["dtype"],
                "shape_per_expert": [sa[0] + sb[0], *sa[1:]]}

    out = dict(index)
    out["segments"] = [_fuse(segs[0], segs[1], "gate_up blocks"),
                       _fuse(segs[2], segs[3], "gate_up scales"),
                       segs[4], segs[5]]
    return out


def mxfp4_geometry_from_arena(index: dict):
    """``(E, n1, half1, nb1, n2, half2, nb2)`` from the bake's own index.

    The arena is the authority on expert geometry, never the config: K3's routed
    experts are built at ``routed_expert_hidden_size`` (3584, the latent width),
    not ``hidden_size`` (7168), so a config-derived width is 2x too wide.

    Segment ORDER no longer has to match the engine's — see
    :func:`engine_segment_map` and
    :meth:`Mxfp4NvmeResidency._init_permutation`, which reorder on gather. This
    is the geometry half of that result, kept as its own name because callers
    sizing buffers want the shapes without the piece table.
    """
    return engine_segment_map(index)[1]


class Mxfp4NvmeResidency(Mxfp4PipelinedGptOss):
    """The pipelined MXFP4 engine with its all-E host arena replaced by a tier.

    Args:
        arena_path: arena baked by :func:`nvme_arena.bake_expert_tensors`.
        layer: which layer's rows this engine serves.
        hot_ids: experts to keep resident in VRAM (never read again after init).
        k_slots: experts routed per token — the device slot count.
        hot_rows: pinned-DRAM rows the tier may hold. Must be at least the number
            of DISTINCT cold experts one fetch can want; size it from measured
            free RAM via :func:`nvme_residency.capacity_for_bytes`.
        gate_up_bias / down_bias: ``None`` for K3 (bias-free experts).
        tier: a prebuilt :class:`~nvme_residency.ColdTier` to share across layers
            — the usual arrangement, since one pinned buffer should back all 92
            MoE layers rather than 92 buffers each sized for one.
        store: a prebuilt :class:`~mxfp4_pipelined.SlotStore` to share across layers,
            for the same reason one layer deeper. Slots are k x row_bytes of VRAM
            (281 MB at K3's geometry), and only one layer's are live at a time, so
            92 private stores would spend 25.8 GB of a 24 GB card on buffers that sit
            idle. Pass ``engines[0].store``.
    """

    def __init__(self, arena_path, layer, *, hot_ids=(), k_slots,
                 hot_rows=None, gate_up_bias=None, down_bias=None,
                 device="cuda", alpha=1.702, limit=7.0,
                 compute_dtype=torch.bfloat16, tier=None, index=None, qd=4,
                 store=None):
        from nvme_arena import load_index
        index = index if index is not None else load_index(arena_path)
        groups, geo = engine_segment_map(index)
        E, n1, half1, nb1, n2, half2, nb2 = geo
        self.layer = int(layer)
        self.index = index
        self._src_groups = groups

        if tier is None:
            if hot_rows is None:
                raise ValueError("pass hot_rows (or a prebuilt tier)")
            tier = ColdTier(arena_path, hot_rows=int(hot_rows), pinned=True,
                            qd=qd, index=index)
        if not tier.pinned:
            raise ValueError(
                "the tier must be pinned=True: the gather kernel reads its buffer "
                "over UVA by absolute address, and an mmap buffer is not mapped "
                "into the GPU's address space — the kernel would fault or read "
                "unrelated memory.")
        self.tier = tier
        self.arena = None                     # the 1.446 TB this class avoids

        self._init_geometry(E, n1, half1, nb1, n2, half2, nb2, k_slots=k_slots,
                            device=device, alpha=alpha, limit=limit,
                            compute_dtype=compute_dtype)
        if tier.row_bytes != self.row_bytes:
            raise ValueError(f"arena row_bytes {tier.row_bytes} != engine "
                             f"{self.row_bytes}")
        self.row_stride = tier.row_stride
        self.pinned = True
        self._init_permutation()
        self._build_source_from_arena(hot_ids)
        self._init_slots(store)
        self._init_bias(gate_up_bias, down_bias)
        self._init_tier_state()
        self._prime()

    def _build_source(self, *args, **kwargs):
        raise TypeError(
            "Mxfp4NvmeResidency never builds an all-E host arena — that arena is "
            "the 1.446 TB this class exists to avoid. Rows come from the tier; "
            "hot rows are read once by _build_source_from_arena.")

    _BLOCK_WORDS = 2048

    def _init_permutation(self):
        """Decide whether rows must be REORDERED as they are gathered.

        The engine reads ``[gu_blocks | gu_scales | dn_blocks | dn_scales]`` at
        offsets it computes. An arena baked in a different segment order holds the
        same bytes elsewhere in the row. Rather than refuse it — or rewrite 482 GB
        of a 1.45 TB arena — the gather lands each segment at the offset the engine
        expects. The copy was happening anyway, so this is free.

        The identity case keeps the original contiguous kernel untouched: no new
        code on the path #21 measured, and no perf risk for gpt-oss.
        """
        pieces, dst = [], 0
        for group, want_off in zip(self._src_groups, self.off):
            if dst != want_off:                # groups are laid out back to back
                dst = want_off
            for s_off, ln in group:
                pieces.append((s_off, dst, ln))
                dst += ln
        self._perm_pieces = pieces
        self.permuted = any(s != d for s, d, _ in pieces)
        if not self.permuted:
            self._c_src = self._c_dst = self._c_len = None
            return
        src, dstw, lnw = _chunk_table(pieces, self._BLOCK_WORDS)
        dev = self.device
        self._c_src = torch.tensor(src, dtype=torch.int64, device=dev)
        self._c_dst = torch.tensor(dstw, dtype=torch.int64, device=dev)
        self._c_len = torch.tensor(lnw, dtype=torch.int64, device=dev)
        covered = sum(n for _s, _d, n in pieces)
        if covered != sum(self.seg):
            raise ValueError(f"permutation covers {covered} B but the engine's "
                             f"segments total {sum(self.seg)} B")

    def _gather(self, src):
        if not self.permuted:
            return super()._gather(src)
        kern = _perm_gather_kernel()
        kern[(self.k, self._c_src.numel())](
            self.slots64, src, self.have, self._c_src, self._c_dst, self._c_len,
            self.row_words, BLOCK=self._BLOCK_WORDS, num_warps=4)

    def _build_source_from_arena(self, hot_ids):
        """Read the hot set out of the arena ONCE into VRAM; the tail stays cold.

        Read through the tier in chunks bounded by its capacity, so priming a hot
        set larger than the pinned buffer still works. Those rows keep a tier slot
        until LFU ages them out — harmless (they are in VRAM and will never be
        requested again), and not worth a special eviction path.
        """
        E = self.E
        ids = sorted({int(e) for e in (hot_ids or ())})
        if ids and (ids[0] < 0 or ids[-1] >= E):
            raise ValueError(f"hot_ids outside [0, {E}): "
                             f"{[e for e in ids if e < 0 or e >= E]}")
        stack = torch.empty(len(ids), self.row_bytes, dtype=torch.uint8)
        cap = self.tier.hot_rows
        for lo in range(0, len(ids), cap):
            chunk = ids[lo:lo + cap]
            self.tier.ensure(self.layer, chunk)
            for j, e in enumerate(chunk):
                mv = self.tier.row(self.layer, e)          # row_bytes, unpadded
                stack[lo + j] = torch.frombuffer(bytearray(mv), dtype=torch.uint8)
        self.hot_stack = stack.to(self.device)

        is_hot = torch.zeros(E, dtype=torch.bool, device=self.device)
        if ids:
            is_hot[torch.tensor(ids, device=self.device)] = True
        self.is_hot = is_hot
        # host mirrors: _resolve_src decides hot-vs-cold per slot on the CPU
        # (it must, to issue the reads), so it may not consult a device tensor.
        self._hot_host = [False] * E
        self._hot_row_host = [0] * E
        for j, e in enumerate(ids):
            self._hot_host[e], self._hot_row_host[e] = True, j
        self._hot_base = self.hot_stack.data_ptr()
        self._cold_base = self.tier.buffer_ptr

    def _init_tier_state(self):
        k = self.k
        self._want_eid = [-1] * k
        self._have_eid = [-1] * k
        self._have_addr = [-1] * k
        self._src_host = [0] * k
        # Deliberately NOT pinned + non_blocking: these are k*8 bytes, so the
        # copy costs microseconds against a millisecond-scale NVMe read, while an
        # async copy would let the next fetch overwrite the staging buffer before
        # the previous one drained.
        self._src_dev = torch.empty(k, dtype=torch.long, device=self.device)
        self._have_stage = torch.empty(k, dtype=torch.long)

    # ------------------------------------------------------------ the seam ----
    def _resolve_src(self):
        """Make every wanted row resident, then hand the gather its addresses.

        Hot rows resolve against the VRAM stack (stride ``row_bytes``); cold rows
        resolve to a tier slot (stride ``row_stride``). Only cold ids are passed
        to the tier: ensuring a hot expert would read it from disk into pinned
        DRAM to sit next to the copy already in VRAM.
        """
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Mxfp4NvmeResidency cannot be captured in a CUDA graph: "
                "residency is a host-side disk read, so the wanted expert ids "
                "must be read back to the CPU. Capture the fully-resident "
                "Mxfp4PipelinedGptOss instead, or run this engine eagerly.")
        ids = self.want_buf.tolist()
        cold = [e for e in ids if not self._hot_host[e]]
        slots = iter(self.tier.ensure(self.layer, cold)) if cold else iter(())
        addr = self._src_host
        for i, e in enumerate(ids):
            if self._hot_host[e]:
                addr[i] = self._hot_base + self._hot_row_host[e] * self.row_bytes
            else:
                addr[i] = self._cold_base + next(slots) * self.row_stride
        self._want_eid = ids
        self._src_dev.copy_(torch.tensor(addr, dtype=torch.long))
        return self._src_dev

    def _invalidate(self, src):
        """Poison ``have`` wherever a device slot's EXPERT changed.

        The gather's skip test is ``src == have``, an address comparison. Under a
        tier an unchanged address no longer implies unchanged contents, so without
        this a freshly-evicted-and-refilled slot is silently skipped and the
        previous expert's weights are used. Rewritten wholesale from the host
        mirror rather than read-modify-written on the device.
        """
        stage = self._have_stage
        for i, (w, h, a) in enumerate(zip(self._want_eid, self._have_eid,
                                          self._have_addr)):
            stage[i] = -1 if w != h else a
        self.have.copy_(stage)

    def _commit(self, src):
        super()._commit(src)
        self._have_eid = list(self._want_eid)
        self._have_addr = list(self._src_host)

    def _forget(self):
        """Drop this engine's residency mirror when another layer takes the slots.

        The base class can skip this because pinning every row makes address <->
        expert a bijection. Under a tier it is load-bearing twice over: the slots now
        hold another LAYER's rows, and — because the tier is shared and its slot
        addresses are reused across layers — the address this engine recorded for slot
        i can be exactly the address it wants next. ``_invalidate`` rebuilds ``have``
        from this mirror, so a stale mirror would reconstruct a ``have`` that says
        "already resident" about another layer's bytes.
        """
        self._have_eid = [-1] * self.k
        self._have_addr = [-1] * self.k

    def traffic(self):
        t = super().traffic()
        t["tier"] = self.tier.stats()
        t["slots"] = {"bytes": self.store.bytes, "users": self.store.users,
                      "claims": self.store.claims}
        return t


class Mxfp4NvmeResidencyK3(Mxfp4NvmeResidency):
    """K3's epilogue on the NVMe-backed engine.

    Two differences from gpt-oss, both sourced from the release's own
    ``modeling_kimi_linear.py`` (see :mod:`moonshot_gather`): gate and up are a
    CLEAN CONCAT (``cat([w1(x), w3(x)], -1)``) rather than interleaved columns,
    and the activation is SiTU, not clamped-GLU. ``alpha``/``limit`` are unused
    here; SiTU's bounds are its own ``beta``/``linear_beta``.
    """

    def _glu(self, gu):
        from moonshot_gather import apply_glu
        return apply_glu(gu, "situ").to(self.cd)
