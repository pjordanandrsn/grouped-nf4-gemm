# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Native-MXFP4 pipelined residency (Phase 4) — the fused mxfp4 kernel + the
address-table residency engine + the gpt-oss clamped-GLU epilogue, composed.

Self-contained in the private fork (rail: the shipped integration threads a
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
    global _KERNEL
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
        self.device = torch.device(device)
        self.k = int(k_slots)
        self.cd = compute_dtype
        self.alpha, self.limit = float(alpha), float(limit)
        E, n1, half1 = gu_blocks.shape
        _, n2, half2 = dn_blocks.shape
        self.E, self.n1, self.n2 = E, n1, n2
        self.k1, self.k2 = half1 * 2, half2 * 2
        nb1, nb2 = gu_scales.shape[-1], dn_scales.shape[-1]
        assert nb1 == self.k1 // 32 and nb2 == self.k2 // 32

        seg = [n1 * half1, n1 * nb1, n2 * half2, n2 * nb2]
        off = [0]
        for s in seg[:-1]:
            off.append(_align8(off[-1] + s))
        row_bytes = _align8(off[-1] + seg[-1])
        self.row_bytes, self.off, self.row_words = row_bytes, off, row_bytes // 8

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

        k = self.k
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
        self.gate_up_bias = gate_up_bias.to(self.device)
        self.down_bias = down_bias.to(self.device)
        self.hot_d2d_bytes = torch.zeros((), dtype=torch.long, device=self.device)
        self.cold_pcie_bytes = torch.zeros((), dtype=torch.long, device=self.device)
        self._prime()

    def _prime(self):
        kern = _gather_kernel()
        src0 = self.src_of_expert[0].expand(self.k).contiguous()
        grid = (self.k, -(-self.row_words // 2048))
        kern[grid](self.slots64, src0, self.have, self.row_words, BLOCK=2048, num_warps=4)
        self.have.copy_(src0)

    def _fetch(self, want):
        self.want_buf.copy_(want)
        src = self.src_of_expert.index_select(0, self.want_buf)
        miss = src != self.have
        hot = self.is_hot.index_select(0, self.want_buf)   # resident -> D2D, cold -> UVA
        self.cold_pcie_bytes += (miss & ~hot).sum() * self.row_bytes
        self.hot_d2d_bytes += (miss & hot).sum() * self.row_bytes
        kern = _gather_kernel()
        grid = (self.k, -(-self.row_words // 2048))
        kern[grid](self.slots64, src, self.have, self.row_words, BLOCK=2048, num_warps=4)
        self.have.copy_(src)

    def forward(self, hidden_states, router_indices, router_scores):
        in_dtype, in_dev = hidden_states.dtype, hidden_states.device
        x = hidden_states.to(device=self.device, dtype=self.cd)
        want = router_indices.reshape(-1).to(device=self.device, dtype=torch.long)
        # the decode engine takes exactly k VALID ids; transformers' padded
        # routing index (== num_experts) would index bias/expert data OOB.
        # Checked eagerly only — a sync inside CUDA-graph capture is illegal.
        if not torch.cuda.is_current_stream_capturing():
            if bool((want >= self.gate_up_bias.shape[0]).any()):
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
        gu = gu + self.gate_up_bias.index_select(0, self.want_buf)
        gate, up = gu[..., ::2], gu[..., 1::2]     # gpt-oss INTERLEAVED, not chunk(2)
        gate = gate.clamp(max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        h = (up + 1) * (gate * torch.sigmoid(gate * self.alpha))
        dn = gemm_mxfp4_grouped(h.contiguous().to(self.cd), self.dn_p_v, self.dn_a_v,
                                self.sizes, self.slot_eids)
        dn = dn.to(torch.float32) + self.down_bias.index_select(0, self.want_buf).to(torch.float32)
        w = router_scores.reshape(-1).to(device=self.device, dtype=torch.float32)
        out = (dn * w[:, None]).sum(0, keepdim=True)
        return out.to(device=in_dev, dtype=in_dtype)

    def traffic(self):
        return {"hot_d2d_bytes": int(self.hot_d2d_bytes.item()),
                "cold_pcie_bytes": int(self.cold_pcie_bytes.item())}
