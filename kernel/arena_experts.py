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
    def _slice(self, mv: memoryview, suffix: str) -> torch.Tensor:
        g = self.segments[suffix]
        off, ln = g["seg_off"], g["length"]
        # copy out of the landing buffer: it is reused by the next expert, and
        # torch.frombuffer would alias it.
        t = torch.frombuffer(bytearray(mv[off:off + ln]), dtype=torch.uint8)
        dt = _torch_dtype(g["dtype"])
        if dt is not torch.uint8:
            t = t.view(dt)
        return t.reshape(*g["shape_per_expert"])

    def fetch_raw(self, layer: int, expert_ids: Sequence[int]) -> dict:
        """``{suffix: [E, *shape_per_expert]}`` for the given experts, stacked
        in the order given — the caller's routing order, not sorted."""
        # Routing hands ids as a device tensor; row_offset keys on plain ints,
        # and a 0-dim tensor key misses the map with an opaque KeyError.
        ids = [int(e) for e in expert_ids]
        out = {s: [] for s in self.segments}
        for base in range(0, len(ids), self._qd):
            chunk = ids[base:base + self._qd]
            futs = [self.reader.read_row(layer, e, self._landing[i])
                    for i, e in enumerate(chunk)]
            for i, f in enumerate(futs):
                f.result()
                for s in self.segments:
                    out[s].append(self._slice(self._landing[i], s))
        return {s: torch.stack(v).to(self.device) for s, v in out.items()}

    def fused_stacks(self, layer: int, expert_ids: Sequence[int],
                     proj: str = "gate"):
        """``(blocks [E, N, K//2] uint8, scales [E, N, K//32] uint8)`` for one
        projection — the exact input contract of ``gemm_mxfp4_grouped``.

        ``proj`` is canonical (``gate``/``up``/``down``); the released-K3
        spelling (w1/w3/w2) is applied here so callers never hold it.
        """
        w = K3_PROJ.get(proj, proj)
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

    gb, gs = src.fused_stacks(layer, expert_ids, "gate")
    ub, us = src.fused_stacks(layer, expert_ids, "up")
    db, ds = src.fused_stacks(layer, expert_ids, "down")
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
