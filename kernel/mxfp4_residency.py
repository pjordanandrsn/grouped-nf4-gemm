# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""MXFP4 experts served from NVMe — the engine seam that lets a >host-RAM MoE run.

:class:`~mxfp4_pipelined.Mxfp4PipelinedGptOss` pins **all E rows** in host DRAM and
dispatches through a static address table. For Kimi K3 that arena is **1.446 TB**
(93 layers x 896 experts x ~17.5 MB, measured 2026-07-30) against a **503 GB**
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

# K3 ships gate (w1) and up (w3) as SEPARATE per-expert tensors while the engine
# wants one fused gate_up. A bake whose `kinds` are ordered like this produces
# rows the engine reads directly, because `cat([w1, w3], dim=0)` is byte-wise w1's
# bytes then w3's -- see `fuse_gate_up_segments`, which checks that rather than
# assuming it. The KIND NAMES are deliberately absent: read them off the real
# checkpoint. Guessing tensor names is what silently returned n_experts=0 from
# `moonshot_gather` on K3 (three renames since K2).
K3_KIND_ORDER = ("<w1 blocks>", "<w3 blocks>", "<w1 scales>", "<w3 scales>",
                 "<w2 blocks>", "<w2 scales>")


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

    def _fuse(a, b, name):
        if a["seg_off"] + a["length"] != b["seg_off"]:
            raise ValueError(
                f"{name}: segments {a['suffix']!r} and {b['suffix']!r} are not "
                f"contiguous ({a['seg_off']}+{a['length']} != {b['seg_off']}) — "
                "the bake's 8-byte padding left a hole, so they cannot be read "
                "as one fused tensor. Re-bake with 8-aligned segment lengths.")
        if a["dtype"] != b["dtype"]:
            raise ValueError(f"{name}: dtype mismatch "
                             f"{a['dtype']} vs {b['dtype']}")
        sa, sb = tuple(a["shape_per_expert"]), tuple(b["shape_per_expert"])
        if sa[1:] != sb[1:]:
            raise ValueError(f"{name}: trailing dims differ {sa} vs {sb}")
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

    Also re-derives the engine's row layout from those shapes and requires it to
    equal the layout the bake actually wrote. ``test_mxfp4_arena_layout`` proves
    the two derivations agree in general; this is the same claim checked for THIS
    arena, before a single byte is served.
    """
    index = fuse_gate_up_segments(index)
    segs = index["segments"]
    if len(segs) != 4:
        raise ValueError(f"need 4 segments, got {len(segs)}")
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
    (n1, half1), (n1s, nb1), (n2, half2), (n2s, nb2) = (
        tuple(g["shape_per_expert"]) for g in segs)
    if n1s != n1 or n2s != n2:
        raise ValueError(f"blocks/scales row counts disagree: gate_up {n1} vs "
                         f"{n1s}, down {n2} vs {n2s}")
    for what, half, nb in (("gate_up", half1, nb1), ("down", half2, nb2)):
        if nb * 32 != half * 2:
            raise ValueError(
                f"{what}: {nb} scale groups do not tile {half * 2} columns at "
                f"MXFP4 group_size 32 (got {nb * 32})")

    # the runtime layout gate
    seg = [n1 * half1, n1 * nb1, n2 * half2, n2 * nb2]
    off = [0]
    for s in seg[:-1]:
        off.append(_align8(off[-1] + s))
    row_bytes = _align8(off[-1] + seg[-1])
    have_off = [g["seg_off"] for g in segs]
    have_len = [g["length"] for g in segs]
    if have_len != seg or have_off != off or index["row_bytes"] != row_bytes:
        raise ValueError(
            "arena row layout does not match the engine's: "
            f"offsets {have_off} vs {off}, lengths {have_len} vs {seg}, "
            f"row_bytes {index['row_bytes']} vs {row_bytes}. The engine reads "
            "each segment at a computed offset, so serving this arena would "
            "hand the kernel misaligned nibbles, not an error.")
    return index["n_experts_per_layer"], n1, half1, nb1, n2, half2, nb2


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
    """

    def __init__(self, arena_path, layer, *, hot_ids=(), k_slots,
                 hot_rows=None, gate_up_bias=None, down_bias=None,
                 device="cuda", alpha=1.702, limit=7.0,
                 compute_dtype=torch.bfloat16, tier=None, index=None, qd=4):
        from nvme_arena import load_index
        index = index if index is not None else load_index(arena_path)
        E, n1, half1, nb1, n2, half2, nb2 = mxfp4_geometry_from_arena(index)
        self.layer = int(layer)
        self.index = index

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
        self._build_source_from_arena(hot_ids)
        self._init_slots()
        self._init_bias(gate_up_bias, down_bias)
        self._init_tier_state()
        self._prime()

    def _build_source(self, *args, **kwargs):
        raise TypeError(
            "Mxfp4NvmeResidency never builds an all-E host arena — that arena is "
            "the 1.446 TB this class exists to avoid. Rows come from the tier; "
            "hot rows are read once by _build_source_from_arena.")

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

    def traffic(self):
        t = super().traffic()
        t["tier"] = self.tier.stats()
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
