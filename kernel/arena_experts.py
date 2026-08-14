"""Arena-backed routed-expert source — the link from a baked NVMe arena to a
forward pass.

`nvme_arena` relocates a checkpoint's per-expert tensors into an expert-major
arena (hash-preserving: every row segment is one whole source tensor range) and
`nvme_reader` reads a row back. Neither knows what the bytes *mean*, so until
now the arena was a storage result with no consumer: the gather/bake half of
the tier existed and the compute half did not.

This module closes that. It slices a row by the bake's own recorded segment
geometry and returns the fused `[E, N, K//2]` blocks and `[E, N, K//32]` e8m0
scales that :func:`mxfp4_grouped.gemm_mxfp4_grouped` already consumes.

The shapes line up exactly, which is the point worth stating: a
DeepSeek-V3-lineage MXFP4 release such as Kimi K3 ships each expert as
`weight_packed [N, K//2]` + `weight_scale [N, K//32]`, and that *is* the
kernel's input contract. So the bytes go disk -> arena -> kernel with **no
dequantize round trip and no requantization** — the shipped bytes are the ones
multiplied, which is what makes per-tensor hash provenance meaningful all the
way to the GEMM.

What this module does NOT do: routing, attention, or any model wiring. It
answers exactly one question — "give me these experts' bytes for this layer" —
and leaves the forward pass to the caller.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from nvme_arena import load_index
from nvme_reader import ArenaReader, alloc_landing

# Released-K3 spelling. `kinds` order here fixes the on-disk segment order and
# must match the order passed to `bake_expert_tensors` — the index records it,
# and `ArenaExpertSource` reads geometry from the index rather than assuming.
#
# NOT the order to bake in if the arena will also feed the PIPELINED engine.
# `ArenaExpertSource` slices each segment by suffix, so any order works here; but
# `mxfp4_residency.Mxfp4NvmeResidency` reads gate_up at one computed offset and
# therefore needs both blocks segments adjacent and both scales segments
# adjacent — see `mxfp4_residency.K3_RESIDENCY_KINDS`. That order works for BOTH
# consumers, so prefer it for any bake big enough that you would not want to
# repeat it.
K3_KINDS = ("w1.weight_packed", "w1.weight_scale",
            "w3.weight_packed", "w3.weight_scale",
            "w2.weight_packed", "w2.weight_scale")
K3_TEMPLATE = ("language_model.model.layers.{layer}"
               ".block_sparse_moe.experts.{expert}.{kind}")
# w1=gate, w3=up, w2=down — confirmed against the release by shape, not by
# convention (see moonshot_gather.K3_SCHEME).
K3_PROJ = {"gate": "w1", "up": "w3", "down": "w2"}

_DTYPES = {"U8": torch.uint8, "I8": torch.int8, "BF16": torch.bfloat16,
           "F16": torch.float16, "F32": torch.float32, "F8_E4M3": torch.uint8,
           "F8_E5M2": torch.uint8}


def _torch_dtype(name: str) -> torch.dtype:
    try:
        return _DTYPES[name]
    except KeyError:
        raise ValueError(
            f"arena segment dtype {name!r} has no torch mapping; add it to "
            f"arena_experts._DTYPES (known: {sorted(_DTYPES)})") from None


class ArenaExpertSource:
    """Read routed experts for one layer out of a baked arena.

    ``qd`` is the reader's max in-flight reads; a top-k fetch issues one read
    per expert and they overlap, so qd should be >= the routing top-k to keep
    the device busy. Reads land in page-aligned buffers (O_DIRECT where the
    platform has it), one per in-flight expert.
    """

    def __init__(self, arena: str, *, qd: int = 16,
                 device: str = "cpu", pin: bool = False):
        self.index = load_index(arena)
        self.reader = ArenaReader(arena, qd=qd)
        self.device = device
        self.row_stride = self.index["row_stride"]
        self.segments = {g["suffix"]: g for g in self.index["segments"]}
        # alloc_landing returns (memoryview, keepalive); the keepalive must
        # outlive the view or the mapping is collected under the reader.
        pairs = [alloc_landing(self.row_stride, pinned=pin) for _ in range(qd)]
        self._landing = [mv for mv, _k in pairs]
        self._keepalive = [k for _mv, k in pairs]
        self._qd = qd
        self._pin = pin
        # Per-segment host staging, allocated on first use and reused. Keyed by
        # expert count because routing is top-k and that count is stable, so in
        # steady state this allocates once. See `_staging`.
        self._stage: dict = {}

    # -- geometry ---------------------------------------------------------
    @property
    def layers(self) -> list:
        return sorted({l for l, _e, _o in self.index["rows"]})

    @property
    def n_experts(self) -> int:
        return 1 + max(e for _l, e, _o in self.index["rows"])

    def segment_shape(self, suffix: str):
        return tuple(self.segments[suffix]["shape_per_expert"])

    # -- reads ------------------------------------------------------------
    def _staging(self, n: int) -> dict:
        """``{suffix: [n, length] uint8}`` host staging, PINNED when a device is
        in play, allocated once per expert count and reused.

        This is the `nvme_residency.segment_into` shape applied here: a
        destination the reader's bytes are copied into ONCE, laid out so the
        transfer to the device is a single contiguous DMA per segment.
        """
        got = self._stage.get(n)
        if got is None:
            want_pin = str(self.device) != "cpu" and torch.cuda.is_available()
            align = self.index["align"]
            got, keep = {}, []
            for s, g in self.segments.items():
                ln = g["length"]
                # OVER-ALLOCATE and hand back an aligned sub-view. A pinned
                # tensor is NOT reliably page-aligned: torch's caching host
                # allocator suballocates and has been measured 1024 B off
                # (see nvme_reader.alloc_landing). The scatter path DMAs
                # O_DIRECT bytes straight into these rows, so an unaligned base
                # is an EINVAL -- and worse, it depends on what else has been
                # pinned, so it passes in a fresh process and fails in a real
                # one.
                t = torch.empty(n * ln + align, dtype=torch.uint8)
                if want_pin:
                    t = t.pin_memory()
                pad = (-t.data_ptr()) % align
                view = t[pad:pad + n * ln].view(n, ln)
                assert view.data_ptr() % align == 0, "aligned sub-view is not aligned"
                got[s] = view
                keep.append(t)          # the view aliases it; it must outlive
            self._stage[n] = got
            self._stage_keep = getattr(self, "_stage_keep", [])
            self._stage_keep.append(keep)
        return got

    def _scatter_ok(self, stage: dict, n: int) -> bool:
        """Is a scattering read legal for THIS staging, right now?

        The layout check is about the arena's geometry; this is about the
        addresses actually allocated. Both must hold, and a failure here must
        FALL BACK rather than raise -- an allocator that hands back an
        unaligned block is not a reason to fail a fetch.
        """
        if self._scatter_layout() is None:
            return False
        align = self.index["align"]
        for s, g in self.segments.items():
            t = stage[s]
            if t.data_ptr() % align or (g["length"] % align):
                return False
        return True

    def _scatter_layout(self):
        """``[(suffix|None, length)]`` covering a whole row in file order, or
        ``None`` when a scattering read would not be legal here.

        ``None`` entries are gaps or trailing padding that must be absorbed by
        scratch, because `preadv` fills its iovec sequentially and cannot skip.

        Refused, explicitly, when:
          * a segment length is not ``align``-aligned — O_DIRECT would EINVAL;
          * a gap or the padding is not ``align``-aligned, which would push
            every following destination off alignment.

        Both hold for K3 (all six lengths are multiples of 4096 and
        ``row_stride == row_bytes``) and NEITHER is guaranteed in general, so
        this returns None rather than guessing. A silent fallback would be a
        silent ~6x regression, so `fetch_raw` records which path it took.
        """
        if getattr(self, "_layout_cache", "unset") != "unset":
            return self._layout_cache
        align = self.index["align"]
        segs = sorted(self.segments.values(), key=lambda g: g["seg_off"])
        plan, cur, ok = [], 0, True
        for g in segs:
            gap = g["seg_off"] - cur
            if gap:
                plan.append((None, gap))
                ok &= gap % align == 0
            plan.append((g["suffix"], g["length"]))
            ok &= g["length"] % align == 0
            cur = g["seg_off"] + g["length"]
        pad = self.row_stride - cur
        if pad:
            plan.append((None, pad))
            ok &= pad % align == 0
        self._layout_cache = plan if ok else None
        return self._layout_cache

    def _scatter_views(self, stage: dict, row: int, scratch: dict):
        layout = self._scatter_layout()
        views = []
        for suffix, ln in layout:
            if suffix is None:
                views.append(memoryview(scratch[ln]))
            else:
                views.append(memoryview(stage[suffix][row].numpy()))
        return views

    def fetch_raw(self, layer: int, expert_ids: Sequence[int]) -> dict:
        """``{suffix: [E, *shape_per_expert]}`` for the given experts, stacked
        in the order given — the caller's routing order, not sorted.

        One host copy per segment per expert, straight from the landing buffer
        into pinned staging, then ONE transfer per segment. The previous form
        copied every byte three times before the device saw it —
        ``bytearray(mv[...])`` per segment (the landing buffer is reused, so a
        copy is required, but it was a Python-level one), then ``torch.stack``,
        then a pageable ``.to()``. Measured effect of that: **~0.72 GB/s
        regardless of the device**, identical on a 6.88 and a 22.71 GB/s NVMe,
        i.e. a host ceiling rather than a read limit (#73).

        The transfer is **synchronous on purpose.** Staging is reused across
        calls, and a `non_blocking=True` copy is not ordered against the *host*
        writes of the next call — the CPU could overwrite staging while the DMA
        is still reading it. Making it async needs an event recorded here and
        waited on before the next reuse; it is not free correctness.
        """
        # Routing hands ids as a device tensor; row_offset keys on plain ints,
        # and a 0-dim tensor key misses the map with an opaque KeyError.
        ids = [int(e) for e in expert_ids]
        n = len(ids)
        stage = self._staging(n)

        if self._scatter_ok(stage, n):
            # DMA straight into staging: the CPU never touches these bytes, so
            # neither the memcpy nor the dirty-page penalty on the H2D exists.
            scratch = self._scratch()
            futs = [self.reader.read_row_scatter(
                        layer, e, self._scatter_views(stage, r, scratch))
                    for r, e in enumerate(ids)]
            for f in futs:
                f.result()
            self.last_fetch_path = "scatter"
            return self._to_device(stage, n)

        self.last_fetch_path = "copy"
        for base in range(0, n, self._qd):
            chunk = ids[base:base + self._qd]
            futs = [self.reader.read_row(layer, e, self._landing[i])
                    for i, e in enumerate(chunk)]
            for i, f in enumerate(futs):
                f.result()
                # Aliases the landing buffer — no copy here. The copy_ below is
                # the single one, and it must happen before this landing slot is
                # reused by the next chunk, which the serial loop guarantees.
                src = torch.frombuffer(self._landing[i], dtype=torch.uint8)
                row = base + i
                for s, g in self.segments.items():
                    off, ln = g["seg_off"], g["length"]
                    stage[s][row].copy_(src[off:off + ln])
        return self._to_device(stage, n)

    def _scratch(self) -> dict:
        """One aligned throwaway buffer per distinct gap/padding size."""
        got = getattr(self, "_scratch_cache", None)
        if got is None:
            got = {}
            for suffix, ln in (self._scatter_layout() or []):
                if suffix is None and ln not in got:
                    mv, keep = alloc_landing(ln)
                    got[ln] = mv
                    self._keepalive.append(keep)
            self._scratch_cache = got
        return got

    def _to_device(self, stage: dict, n: int) -> dict:
        out = {}
        for s, g in self.segments.items():
            src = stage[s]
            t = src.to(self.device)
            # `.to()` is a NO-OP when the tensor is already on the target, so on
            # a CPU source it returns `src` itself and the result would ALIAS
            # the reused staging -- the next fetch of the same expert count
            # would rewrite a caller's earlier result in place. The old
            # `torch.stack` path always allocated fresh, so this would be a
            # silent regression, and `device="cpu"` is the DEFAULT. Detect the
            # alias by identity rather than by comparing device strings, which
            # get "cuda" vs "cuda:0" wrong.
            if t.data_ptr() == src.data_ptr():
                t = t.clone()
            dt = _torch_dtype(g["dtype"])
            if dt is not torch.uint8:
                t = t.view(dt)
            out[s] = t.reshape(n, *g["shape_per_expert"])
        return out

    def fused_stacks(self, layer: int, expert_ids: Sequence[int],
                     proj: str = "gate", *, raw: dict | None = None):
        """``(blocks [E, N, K//2] uint8, scales [E, N, K//32] uint8)`` for one
        projection — the exact input contract of ``gemm_mxfp4_grouped``.

        ``proj`` is canonical (``gate``/``up``/``down``); the released-K3
        spelling (w1/w3/w2) is applied here so callers never hold it.

        ``raw`` accepts an already-fetched :meth:`fetch_raw` dict and skips the
        read. A row holds ALL SIX segments, so fetching per projection reads the
        same row once per projection and discards two thirds of each read —
        measured at **842 MB where 281 MB is needed** on a K3 layer. Callers that
        want more than one projection should fetch once and pass it here.
        """
        w = K3_PROJ.get(proj, proj)
        if raw is None:
            raw = self.fetch_raw(layer, expert_ids)
        try:
            return raw[f"{w}.weight_packed"], raw[f"{w}.weight_scale"]
        except KeyError:
            raise KeyError(
                f"projection {proj!r} -> {w!r}: arena has segments "
                f"{sorted(self.segments)}; was it baked with K3_KINDS?"
            ) from None

    def traffic(self) -> dict:
        return self.reader.traffic()

    def close(self) -> None:
        self.reader.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def moe_layer_forward(src: "ArenaExpertSource", layer: int,
                      a_cat: torch.Tensor, sizes: torch.Tensor,
                      expert_ids: Sequence[int], *,
                      glu=torch.nn.functional.silu,
                      block_m: int = 64) -> torch.Tensor:
    """One routed-MoE layer, computed straight off the arena.

    ``a_cat [T, K]`` is group-sorted hidden state and ``sizes`` the per-group
    token counts, exactly as ``gemm_mxfp4_grouped`` expects; ``expert_ids``
    is the routing order those groups correspond to.

    This is the arena's first consumer that produces activations rather than
    bytes: fetch the routed experts' shipped bytes, run gate/up/down through
    the packed-MXFP4 kernel, never materialising a dequantized expert. The
    caller still owns routing, normalisation and the residual — this is the
    expert block only.
    """
    from mxfp4_grouped import gemm_mxfp4_grouped

    # ONE fetch for all three projections. A row carries all six segments, so a
    # fetch per projection read every row three times and threw two thirds of
    # each read away: measured 842 MB where 281 MB is needed on a K3 layer
    # (16 of 896 experts, 17.5 MB rows), and the reader's own byte counter
    # confirmed the 3x to the byte.
    raw = src.fetch_raw(layer, expert_ids)
    gb, gs = src.fused_stacks(layer, expert_ids, "gate", raw=raw)
    ub, us = src.fused_stacks(layer, expert_ids, "up", raw=raw)
    db, ds = src.fused_stacks(layer, expert_ids, "down", raw=raw)
    dev = a_cat.device
    mv = lambda t: t.to(dev, non_blocking=True)  # noqa: E731

    # The kernel's `expert_ids` indexes the STACK it is handed, not the model's
    # expert numbering. `fused_stacks` returns exactly the requested experts in
    # request order, so group g uses stack row g. Passing the model-global ids
    # here reads out of bounds whenever any id >= len(expert_ids) -- silently,
    # since the kernel cannot know the stack was subsetted.
    n_groups = len(list(expert_ids))
    stack_ids = torch.arange(n_groups, device=dev, dtype=torch.int32)

    gate = gemm_mxfp4_grouped(a_cat, mv(gb), mv(gs), sizes, stack_ids,
                              block_m=block_m)
    up = gemm_mxfp4_grouped(a_cat, mv(ub), mv(us), sizes, stack_ids,
                            block_m=block_m)
    h = glu(gate) * up
    return gemm_mxfp4_grouped(h, mv(db), mv(ds), sizes, stack_ids,
                              block_m=block_m)


def expert_bytes_per_token(index: dict, top_k: int) -> int:
    """Routed bytes a fully-cold token costs: `top_k` experts x every routed
    layer x the row's *useful* bytes (segments, not the padded stride).

    This is the number the tier is bounded by — at Kimi K3's 92 routed layers,
    896 experts and top-16 it is ~25.8 GB/token, which is why this is a batch
    tier and not an interactive one.
    """
    row_bytes = sum(g["length"] for g in index["segments"])
    n_layers = len({l for l, _e, _o in index["rows"]})
    return top_k * n_layers * row_bytes
