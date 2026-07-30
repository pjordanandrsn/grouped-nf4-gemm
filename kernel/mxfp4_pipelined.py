# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Native-MXFP4 pipelined residency (Phase 4) — the fused mxfp4 kernel + the
address-table residency engine + the gpt-oss clamped-GLU epilogue, composed.

Self-contained in the unpublished implementation (rail: the shipped integration threads a
FORMAT-AGNOSTIC row_bytes/codebook param through e4b's public pipelined.py; the
word mxfp4 never enters a public file). The gather kernel here mirrors e4b
pipelined.py's format-agnostic `_gather_rows_addr` byte-for-byte — it copies a
per-slot row-block by absolute address and cannot see the format.

What differs from the NF4 engine (the seam map, realized in the arena):
  - row-block segments are native blocks (uint8) + e8m0 scales (uint8) for
    gate_up and down — NO fp32 absmax segment (scales are 4x smaller).
  - the fused call is gemm_mxfp4_grouped (flipped nibble, exp2 e8m0, BLOCK_K=32).
  - the epilogue is gpt-oss clamped-GLU + per-expert biases.
K (hot experts/layer resident) is a table rebuild, not a code path — identical
to the NF4 engine's contract. Decode (T==1) on CUDA; correctness-gated vs the
dequant reference at every K.
"""
from __future__ import annotations

import torch

from mxfp4_grouped import gemm_mxfp4_grouped


def _align8(n: int) -> int:
    return (n + 7) & ~7


_KERNEL = None


def _gather_kernel():
    """Per-slot absolute-address gather with have-skip — mirrors e4b
    pipelined.py::_gather_rows_addr (format-agnostic; copies bytes)."""
    # `tl` into MODULE globals, not locals: with `from __future__ import
    # annotations` the `BLOCK: tl.constexpr` annotation is the string
    # "tl.constexpr", which triton resolves against the jitted function's
    # __globals__. A local import raises NameError('tl is not defined') from
    # inside the triton compiler on 3.2. 3.4 tolerates it, so the bug is
    # invisible on a current box and fatal on an older one.
    global _KERNEL, tl
    if _KERNEL is None:
        import triton
        import triton.language as tl

        @triton.jit
        def _gather_rows_addr(dst_ptr, src_ptr, have_ptr, row_words, BLOCK: tl.constexpr):
            slot = tl.program_id(0)
            chunk = tl.program_id(1)
            want = tl.load(src_ptr + slot)
            have = tl.load(have_ptr + slot)
            if want == have:
                return
            offs = chunk * BLOCK + tl.arange(0, BLOCK)
            mask = offs < row_words
            src = tl.cast(want, tl.pointer_type(tl.int64))
            vals = tl.load(src + offs, mask=mask)
            tl.store(dst_ptr + slot.to(tl.int64) * row_words + offs, vals, mask=mask)

        _KERNEL = _gather_rows_addr
    return _KERNEL


class Mxfp4PipelinedGptOss:
    """Per-layer engine: pinned native-mxfp4 arena + resident hot stack + k-slot
    store, address-table dispatch, fused mxfp4 GEMM, gpt-oss GLU epilogue.

    gu_blocks [E, n1, k1//2] u8, gu_scales [E, n1, k1//32] u8   (k1 = hidden)
    dn_blocks [E, n2, k2//2] u8, dn_scales [E, n2, k2//32] u8   (k2 = inter)
    gate_up_bias [E, n1] bf16, down_bias [E, n2] bf16.
    """

    def __init__(self, gu_blocks, gu_scales, dn_blocks, dn_scales,
                 gate_up_bias, down_bias, hot_ids, k_slots, device="cuda",
                 alpha=1.702, limit=7.0, compute_dtype=torch.bfloat16):
        E, n1, half1 = gu_blocks.shape
        _, n2, half2 = dn_blocks.shape
        nb1, nb2 = gu_scales.shape[-1], dn_scales.shape[-1]
        assert nb1 == (half1 * 2) // 32 and nb2 == (half2 * 2) // 32
        self._init_geometry(E, n1, half1, nb1, n2, half2, nb2, k_slots=k_slots,
                            device=device, alpha=alpha, limit=limit,
                            compute_dtype=compute_dtype)
        self._build_source(gu_blocks, gu_scales, dn_blocks, dn_scales, hot_ids)
        self._init_slots()
        self._init_bias(gate_up_bias, down_bias)
        self._prime()

    # ------------------------------------------------------------- geometry --
    def _init_geometry(self, E, n1, half1, nb1, n2, half2, nb2, *, k_slots,
                       device, alpha, limit, compute_dtype):
        """Row layout only — no weights, so a subclass that gets its geometry
        from a baked arena's index rather than from tensors shares this exactly."""
        self.device = torch.device(device)
        self.k = int(k_slots)
        self.cd = compute_dtype
        self.alpha, self.limit = float(alpha), float(limit)
        self.E, self.n1, self.n2 = E, n1, n2
        self.half1, self.half2, self.nb1, self.nb2 = half1, half2, nb1, nb2
        self.k1, self.k2 = half1 * 2, half2 * 2

        seg = [n1 * half1, n1 * nb1, n2 * half2, n2 * nb2]
        off = [0]
        for s in seg[:-1]:
            off.append(_align8(off[-1] + s))
        row_bytes = _align8(off[-1] + seg[-1])
        self.seg = seg
        self.row_bytes, self.off, self.row_words = row_bytes, off, row_bytes // 8

    def _build_source(self, gu_blocks, gu_scales, dn_blocks, dn_scales, hot_ids):
        """Where rows come from: here, a pinned arena holding ALL E of them.

        That is the piece a >host-RAM model cannot afford — see
        :mod:`mxfp4_residency`, which REPLACES this step (and refuses this
        method outright) to keep only the hot rows and stream the tail from
        NVMe."""
        E, off, seg, row_bytes = self.E, self.off, self.seg, self.row_bytes
        # pinned arena [E, row_bytes]: native bytes, laid out, not converted
        arena = torch.zeros(E, row_bytes, dtype=torch.uint8)
        try:
            arena = arena.pin_memory()
            self.pinned = arena.is_pinned()
        except (RuntimeError, AssertionError):
            self.pinned = False
        arena[:, off[0]:off[0] + seg[0]] = gu_blocks.reshape(E, -1)
        arena[:, off[1]:off[1] + seg[1]] = gu_scales.reshape(E, -1)
        arena[:, off[2]:off[2] + seg[2]] = dn_blocks.reshape(E, -1)
        arena[:, off[3]:off[3] + seg[3]] = dn_scales.reshape(E, -1)
        self.arena = arena

        hot_ids = torch.as_tensor(hot_ids, dtype=torch.long).unique()
        if hot_ids.numel():
            self.hot_stack = arena.index_select(0, hot_ids).to(self.device)
        else:
            self.hot_stack = torch.empty(0, row_bytes, dtype=torch.uint8, device=self.device)
        is_hot = torch.zeros(E, dtype=torch.bool, device=self.device)
        is_hot[hot_ids.to(self.device)] = True
        self.is_hot = is_hot
        h_row = torch.zeros(E, dtype=torch.long, device=self.device)
        h_row[hot_ids.to(self.device)] = torch.arange(hot_ids.numel(), device=self.device)
        host_addr = self.arena.data_ptr() + torch.arange(E, device=self.device, dtype=torch.long) * row_bytes
        hot_addr = self.hot_stack.data_ptr() + h_row * row_bytes
        self.src_of_expert = torch.where(is_hot, hot_addr, host_addr)

    def _init_slots(self):
        k, off, row_bytes = self.k, self.off, self.row_bytes
        n1, half1, nb1 = self.n1, self.half1, self.nb1
        n2, half2, nb2 = self.n2, self.half2, self.nb2
        slots = torch.empty(k, row_bytes, dtype=torch.uint8, device=self.device)
        self.slots, self.slots64 = slots, slots.view(torch.int64)
        self.gu_p_v = torch.as_strided(slots, (k, n1, half1), (row_bytes, half1, 1), off[0])
        self.gu_a_v = torch.as_strided(slots, (k, n1, nb1), (row_bytes, nb1, 1), off[1])
        self.dn_p_v = torch.as_strided(slots, (k, n2, half2), (row_bytes, half2, 1), off[2])
        self.dn_a_v = torch.as_strided(slots, (k, n2, nb2), (row_bytes, nb2, 1), off[3])

        self.sizes = [1] * k
        self.slot_eids = torch.arange(k, dtype=torch.int32, device=self.device)
        self.have = torch.full((k,), -1, dtype=torch.long, device=self.device)
        self.want_buf = torch.zeros(k, dtype=torch.long, device=self.device)
        self.a_buf = None
        self.hot_d2d_bytes = torch.zeros((), dtype=torch.long, device=self.device)
        self.cold_pcie_bytes = torch.zeros((), dtype=torch.long, device=self.device)

    def _init_bias(self, gate_up_bias, down_bias):
        """``None`` means this family has no per-expert bias — SKIP the add, do
        not synthesize zeros. gpt-oss ships biases; Kimi K3's experts are three
        bias-free Linears, and an `[E, n]` zeros pair is real VRAM at E=896."""
        for name, b, n in (("gate_up_bias", gate_up_bias, self.n1),
                           ("down_bias", down_bias, self.n2)):
            if b is None:
                setattr(self, name, None)
                continue
            if tuple(b.shape) != (self.E, n):
                raise ValueError(f"{name} must be [E={self.E}, {n}], "
                                 f"got {tuple(b.shape)}")
            setattr(self, name, b.to(self.device))

    # -------------------------------------------------------------- fetch ----
    def _resolve_src(self):
        """Absolute byte address of the row each slot wants, from ``want_buf``.

        A static table suffices here because pinning every row makes
        address <-> expert a BIJECTION. A tiering subclass must override this
        together with :meth:`_invalidate`, because a tier slot's address
        identifies the SLOT, not what currently lives in it."""
        return self.src_of_expert.index_select(0, self.want_buf)

    def _invalidate(self, src):
        """Force a re-gather of device slots whose CONTENTS changed while their
        source address did not. Nothing to do under a bijection; overridden by
        the tiering subclass, where it is the difference between correct output
        and silently stale experts."""

    def _commit(self, src):
        self.have.copy_(src)

    def _gather(self, src):
        """Copy each wanted row into its device slot, verbatim.

        The single place the gather is launched, so a subclass can substitute a
        kernel that REORDERS segments while copying — see
        :meth:`mxfp4_residency.Mxfp4NvmeResidency._gather`, which reads an arena
        whose segments are laid out in a different order than this engine's."""
        kern = _gather_kernel()
        grid = (self.k, -(-self.row_words // 2048))
        kern[grid](self.slots64, src, self.have, self.row_words, BLOCK=2048, num_warps=4)

    def _prime(self):
        self.want_buf.zero_()
        src0 = self._resolve_src()
        self._gather(src0)
        self._commit(src0)

    def _fetch(self, want):
        self.want_buf.copy_(want)
        src = self._resolve_src()
        self._invalidate(src)                             # before the miss count
        miss = src != self.have
        hot = self.is_hot.index_select(0, self.want_buf)   # resident -> D2D, cold -> UVA
        self.cold_pcie_bytes += (miss & ~hot).sum() * self.row_bytes
        self.hot_d2d_bytes += (miss & hot).sum() * self.row_bytes
        self._gather(src)
        self._commit(src)

    def forward(self, hidden_states, router_indices, router_scores):
        in_dtype, in_dev = hidden_states.dtype, hidden_states.device
        x = hidden_states.to(device=self.device, dtype=self.cd)
        want = router_indices.reshape(-1).to(device=self.device, dtype=torch.long)
        # the decode engine takes exactly k VALID ids; transformers' padded
        # routing index (== num_experts) would index bias/expert data OOB.
        # Checked eagerly only — a sync inside CUDA-graph capture is illegal.
        if not torch.cuda.is_current_stream_capturing():
            if bool((want >= self.E).any()):
                raise ValueError(
                    "padded routing index (== num_experts) reached the pipelined "
                    "decode engine — drop padding upstream; this engine takes "
                    "exactly k valid expert ids per token")
        k = self.k
        self._fetch(want)
        if self.a_buf is None or self.a_buf.dtype != self.cd:
            self.a_buf = torch.empty(k, x.shape[-1], dtype=self.cd, device=self.device)
        self.a_buf.copy_(x.expand(k, -1))
        gu = gemm_mxfp4_grouped(self.a_buf, self.gu_p_v, self.gu_a_v, self.sizes, self.slot_eids)
        if self.gate_up_bias is not None:
            gu = gu + self.gate_up_bias.index_select(0, self.want_buf)
        h = self._glu(gu)
        dn = gemm_mxfp4_grouped(h.contiguous().to(self.cd), self.dn_p_v, self.dn_a_v,
                                self.sizes, self.slot_eids)
        dn = dn.to(torch.float32)
        if self.down_bias is not None:
            dn = dn + self.down_bias.index_select(0, self.want_buf).to(torch.float32)
        w = router_scores.reshape(-1).to(device=self.device, dtype=torch.float32)
        out = (dn * w[:, None]).sum(0, keepdim=True)
        return out.to(device=in_dev, dtype=in_dtype)

    def _glu(self, gu):
        """gpt-oss epilogue: gate/up are INTERLEAVED in the output columns (not
        two halves), then clamped-GLU. Other families differ in BOTH respects —
        Kimi K3 concatenates (``cat([w1(x), w3(x)])``) and activates with SiTU —
        so this is a hook, not inlined arithmetic."""
        gate, up = gu[..., ::2], gu[..., 1::2]     # gpt-oss INTERLEAVED, not chunk(2)
        gate = gate.clamp(max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        return (up + 1) * (gate * torch.sigmoid(gate * self.alpha))

    def traffic(self):
        return {"hot_d2d_bytes": int(self.hot_d2d_bytes.item()),
                "cold_pcie_bytes": int(self.cold_pcie_bytes.item())}
