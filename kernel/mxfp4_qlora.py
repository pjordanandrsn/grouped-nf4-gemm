# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""QLoRA training over NATIVE MXFP4 expert stacks (Phase 7 / plan §Phase 2.5).

The training-lane thesis, realized: LoRA adapters train over the FROZEN native
gpt-oss expert bytes — the packed e2m1 blocks + e8m0 scales the checkpoint
ships — with the dequantized weight never stored between forward and backward.
Backward re-decodes on demand (recompute-in-backward, mirroring e4b's
``_FrozenLinearRecomputeBackward`` in ``experts4bit_qlora/_vendor/experts.py``),
so training memory is independent of the number of experts held between forward
and backward, and the provenance claim is structural: the uint8 storage is
``requires_grad=False`` and no code path writes it — ``sha256(bytes)`` before
training == after training == the shipped checkpoint's file ranges
(``mxfp4_loader.file_tensor_sha256``).

Anchors (R1):
  - decode primitive = ``mxfp4_pack_ref.dequant_mxfp4`` (Phase-1
    oracle-adjudicated vs transformers ``_convert_moe_packed_tensors``,
    bit-exact; NIBBLE_LOW_FIRST locked). All e2m1 codebook values and e8m0
    power-of-two scalings are exactly representable in bf16, so the bf16 decode
    equals the fp32 decode downcast — gated in test_mxfp4_qlora.py.
  - reference numerics = transformers v5 ``GptOssExperts.forward`` (verified
    from the installed 5.14.0 source): expert-mask loop, per-expert
    ``x @ W + b``, INTERLEAVED clamped GLU (``[..., ::2]``/``[..., 1::2]``,
    alpha 1.702, limit 7.0 — the Phase-4 gotcha), routing weight applied after
    down, accumulation in the INPUT dtype via ``index_add_`` (transformers
    accumulates bf16 here, unlike e4b's fp32 loop — we mirror transformers,
    the A/B reference arm).
  - fused lane = ``gemm_mxfp4_grouped`` (Phase-2 kernel, 12 gates) behind a
    custom autograd.Function whose backward re-decodes per hit expert.

The LoRA shape mirrors e4b's ``ExpertsLoRA``: per-expert low-rank
``scaling * (x @ A[e].T) @ B[e].T`` on both projections, ``B`` zero-init so the
adapted module is bit-identical to the frozen base at step 0.
"""
from __future__ import annotations

import hashlib
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mxfp4_pack_ref import MX_BLOCK, dequant_mxfp4

GPTOSS_ALPHA = 1.702
GPTOSS_LIMIT = 7.0


def apply_gate_gptoss(gate_up: torch.Tensor, alpha: float = GPTOSS_ALPHA,
                      limit: float = GPTOSS_LIMIT) -> torch.Tensor:
    """gpt-oss clamped GLU, verbatim transformers ``GptOssExperts._apply_gate``:
    INTERLEAVED gate/up split (never ``chunk(2)`` — Phase-4 lock)."""
    gate, up = gate_up[..., ::2], gate_up[..., 1::2]
    gate = gate.clamp(min=None, max=limit)
    up = up.clamp(min=-limit, max=limit)
    glu = gate * torch.sigmoid(gate * alpha)
    return (up + 1) * glu


class _FrozenLinearRecomputeBackward(torch.autograd.Function):
    """``F.linear`` against a frozen decoded weight, re-decoding it in backward.

    Mirrors e4b ``_vendor/experts.py`` (same name, same contract): the weight
    produced by ``dequant_fn`` is an intermediate, not a Parameter; a plain
    ``F.linear`` would stash it as a saved activation for the whole
    forward-to-backward window — one dense expert weight per projection per
    layer. The base is frozen, so backward needs only ``grad_output @ weight``;
    the weight is dropped after the forward matmul and re-decoded on demand.
    Numerically identical to decode-then-linear by construction — recomputation
    changes what is SAVED, never what is computed.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, dequant_fn) -> torch.Tensor:
        ctx.dequant_fn = dequant_fn
        return F.linear(x, dequant_fn())

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = grad_output @ ctx.dequant_fn()
        return grad_x, None


class _FusedGroupedMxfp4(torch.autograd.Function):
    """Grouped forward through the Phase-2 fused kernel; recompute-decode
    backward. ``a_cat`` is group-sorted ``[T_cat, K]``; blocks/scales are the
    kernel-shaped native views ``[E, N, K//2]``/``[E, N, K//32]`` (uint8,
    non-differentiable constants — stashed on ctx, not saved_for_backward).
    Backward: ``grad_a[rows_g] = grad_out[rows_g] @ decode(e_g)`` per group,
    one decoded expert live at a time (same recompute guarantee as the loop
    path)."""

    @staticmethod
    def forward(ctx, a_cat, blocks, scales, sizes, expert_ids):
        from mxfp4_grouped import gemm_mxfp4_grouped
        out = gemm_mxfp4_grouped(a_cat, blocks, scales, sizes, expert_ids)
        ctx.blocks, ctx.scales = blocks, scales
        ctx.sizes = sizes
        ctx.expert_ids = [int(e) for e in expert_ids]
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        grad_a = None
        if ctx.needs_input_grad[0]:
            E, N, half = ctx.blocks.shape
            K = half * 2
            grad_a = torch.empty(grad_out.shape[0], K, dtype=grad_out.dtype,
                                 device=grad_out.device)
            row = 0
            for g, e in enumerate(ctx.expert_ids):
                n = ctx.sizes[g]
                w = dequant_mxfp4(
                    ctx.blocks[e].reshape(N, K // MX_BLOCK, 16).to(grad_out.device),
                    ctx.scales[e].to(grad_out.device),
                    dtype=grad_out.dtype)                      # [N, K], recomputed
                grad_a[row:row + n] = grad_out[row:row + n] @ w
                row += n
        return grad_a, None, None, None, None


class ExpertsMxfp4(nn.Module):
    """Frozen NATIVE-MXFP4 storage for a gpt-oss fused expert stack.

    Holds the checkpoint's bytes in kernel shape (``mxfp4_loader.to_kernel_shapes``
    flattens ``[E, N, n_blk, 16] -> [E, N, K//2]`` as a contiguous view):

      * ``gate_up_blocks [E, 2I, H//2]`` u8, ``gate_up_scales [E, 2I, H//32]`` u8
      * ``down_blocks    [E, H, I//2]``  u8, ``down_scales    [E, H, I//32]``  u8
      * ``gate_up_bias [E, 2I]``, ``down_bias [E, H]`` (checkpoint dtype, frozen)

    Storage may live on CPU (optionally pinned) while activations are CUDA: the
    decode closure stages the PACKED bytes host-to-device per visit (~4x less
    PCIe than a dense bf16 weight) and decodes on the compute device. Packed
    tensors are ``nn.Parameter(requires_grad=False)`` (e4b convention) so they
    serialize via state_dict; scales are buffers.
    """

    def __init__(self, gate_up_blocks, gate_up_scales, down_blocks, down_scales,
                 gate_up_bias, down_bias, alpha: float = GPTOSS_ALPHA,
                 limit: float = GPTOSS_LIMIT,
                 compute_dtype: Optional[torch.dtype] = None):
        super().__init__()
        for t in (gate_up_blocks, gate_up_scales, down_blocks, down_scales):
            if t.dtype != torch.uint8:
                raise ValueError("native mxfp4 storage must be uint8")
        E, n1, half1 = gate_up_blocks.shape
        E2, n2, half2 = down_blocks.shape
        if E != E2:
            raise ValueError(f"expert-count mismatch {E} vs {E2}")
        self.num_experts, self.n1, self.n2 = E, n1, n2
        self.k1, self.k2 = half1 * 2, half2 * 2
        if gate_up_scales.shape != (E, n1, self.k1 // MX_BLOCK):
            raise ValueError(f"gate_up_scales shape {tuple(gate_up_scales.shape)}")
        if down_scales.shape != (E, n2, self.k2 // MX_BLOCK):
            raise ValueError(f"down_scales shape {tuple(down_scales.shape)}")
        # gpt-oss geometry: n1 = 2*intermediate (interleaved gate/up), k1 = hidden,
        # n2 = hidden, k2 = intermediate.
        self.hidden_dim, self.intermediate_dim = self.k1, self.k2
        self.alpha, self.limit = float(alpha), float(limit)
        self.compute_dtype = compute_dtype

        self.gate_up_blocks = nn.Parameter(gate_up_blocks, requires_grad=False)
        self.down_blocks = nn.Parameter(down_blocks, requires_grad=False)
        self.register_buffer("gate_up_scales", gate_up_scales)
        self.register_buffer("down_scales", down_scales)
        self.gate_up_bias = nn.Parameter(gate_up_bias, requires_grad=False)
        self.down_bias = nn.Parameter(down_bias, requires_grad=False)

    # -- decode ------------------------------------------------------------
    def _dequantize_expert(self, blocks, scales, n, k, expert_idx, device, dtype):
        """One expert's dense ``[n, k]`` weight, decoded from the native bytes.
        Stages the packed bytes to ``device`` first (cheap: uint8), then
        decodes there. The recompute closure over exactly this call is what
        backward re-runs."""
        blk = blocks[expert_idx]
        scl = scales[expert_idx]
        if blk.device != device:
            blk = blk.to(device, non_blocking=True)
            scl = scl.to(device, non_blocking=True)
        return dequant_mxfp4(blk.reshape(n, k // MX_BLOCK, 16), scl, dtype=dtype)

    def _project(self, which: str, expert_idx, x: torch.Tensor) -> torch.Tensor:
        """One expert projection (no bias): decode + ``linear``, re-decoding in
        backward. ``which`` is ``"gate_up"`` or ``"down"``."""
        if which == "gate_up":
            blocks, scales, n, k = self.gate_up_blocks, self.gate_up_scales, self.n1, self.k1
        else:
            blocks, scales, n, k = self.down_blocks, self.down_scales, self.n2, self.k2

        def dequant_fn(blocks=blocks, scales=scales, n=n, k=k, e=expert_idx,
                       device=x.device, dtype=x.dtype):
            return self._dequantize_expert(blocks, scales, n, k, e, device, dtype)

        return _FrozenLinearRecomputeBackward.apply(x, dequant_fn)

    # -- provenance ----------------------------------------------------------
    def expert_bytes_sha256(self) -> dict:
        """sha256 of every frozen tensor's raw bytes, in a stable order — the
        pre/post training hash table. For blocks/scales these are the
        checkpoint's native bytes (loader gate proves == file ranges); biases
        are hashed as loaded (bit-identity across training is the claim)."""
        out = {}
        for name in ("gate_up_blocks", "gate_up_scales", "down_blocks",
                     "down_scales", "gate_up_bias", "down_bias"):
            t = getattr(self, name).detach().contiguous().cpu()
            out[name] = hashlib.sha256(
                t.view(torch.uint8).numpy().tobytes()).hexdigest()
        return out


class ExpertsMxfp4LoRA(nn.Module):
    """Per-expert LoRA adapters over a frozen :class:`ExpertsMxfp4` base — a
    drop-in for transformers v5 ``GptOssExperts`` (same forward signature, same
    expert-mask loop, same input-dtype accumulation), with the base projections
    running decode+linear under recompute-in-backward.

    ``mode`` selects the base-projection engine:
      * ``"loop"``  — per-expert decode + ``F.linear`` (the A4-oracle-consistent
        reference path; dev default).
      * ``"fused"`` — group-sorted single-launch ``gemm_mxfp4_grouped`` for each
        projection (requires storage on the compute device; CUDA+triton).
    """

    def __init__(self, base: ExpertsMxfp4, r: int = 8, alpha: int = 16,
                 dtype: torch.dtype = torch.float32, mode: str = "loop"):
        super().__init__()
        if mode not in ("loop", "fused"):
            raise ValueError(f"mode must be loop|fused, got {mode!r}")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r, self.scaling, self.mode = r, alpha / r, mode
        E = base.num_experts
        self.gate_up_lora_A = nn.Parameter(torch.empty(E, r, base.hidden_dim, dtype=dtype))
        self.gate_up_lora_B = nn.Parameter(torch.zeros(E, base.n1, r, dtype=dtype))
        self.down_lora_A = nn.Parameter(torch.empty(E, r, base.intermediate_dim, dtype=dtype))
        self.down_lora_B = nn.Parameter(torch.zeros(E, base.n2, r, dtype=dtype))
        nn.init.normal_(self.gate_up_lora_A, std=1.0 / r)
        nn.init.normal_(self.down_lora_A, std=1.0 / r)

    def _lora(self, x, A, B):
        # e4b ExpertsLoRA._lora verbatim: run the low-rank path in the adapter
        # dtype (typically fp32) and cast the delta back to the compute dtype.
        return (self.scaling * F.linear(F.linear(x.to(A.dtype), A), B)).to(x.dtype)

    def forward(self, hidden_states: torch.Tensor, router_indices=None,
                routing_weights=None) -> torch.Tensor:
        if self.mode == "fused":
            if self.base.gate_up_blocks.device != hidden_states.device:
                raise RuntimeError(
                    "fused mode needs the native storage on the compute device "
                    f"(storage {self.base.gate_up_blocks.device}, activations "
                    f"{hidden_states.device}); use mode='loop' for host-side storage")
            return self._forward_fused(hidden_states, router_indices, routing_weights)
        return self._forward_loop(hidden_states, router_indices, routing_weights)

    # -- reference loop (transformers GptOssExperts numerics) ---------------
    def _forward_loop(self, hidden_states, router_indices, routing_weights):
        base = self.base
        next_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(router_indices, num_classes=base.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == base.num_experts:
                continue  # transformers' padded-routing masking index
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]

            gate_up = (base._project("gate_up", expert_idx, current_state)
                       + base.gate_up_bias[expert_idx].to(current_state.dtype)
                       + self._lora(current_state,
                                    self.gate_up_lora_A[expert_idx],
                                    self.gate_up_lora_B[expert_idx]))
            gated = apply_gate_gptoss(gate_up, base.alpha, base.limit)
            out = (base._project("down", expert_idx, gated)
                   + base.down_bias[expert_idx].to(gated.dtype)
                   + self._lora(gated,
                                self.down_lora_A[expert_idx],
                                self.down_lora_B[expert_idx]))
            weighted = out * routing_weights[token_idx, top_k_pos, None]
            next_states.index_add_(0, token_idx, weighted.to(hidden_states.dtype))
        return next_states

    # -- fused grouped path (Phase-2 kernel fwd, recompute bwd) -------------
    def _forward_fused(self, hidden_states, router_indices, routing_weights):
        base = self.base
        T, k = router_indices.shape
        flat_eids = router_indices.reshape(-1)
        # transformers pads ragged routing with index == num_experts; the loop
        # path skips it — drop those pairs here too or blocks[e] reads OOB
        valid = flat_eids < base.num_experts
        if not bool(valid.all()):
            keep = valid.nonzero(as_tuple=True)[0]
            order = keep[torch.argsort(flat_eids[keep], stable=True)]
        else:
            order = torch.argsort(flat_eids, stable=True)
        eids_sorted = flat_eids[order]
        tok_of_pair = torch.div(order, k, rounding_mode="floor")
        pos_of_pair = order - tok_of_pair * k
        uniq, counts = torch.unique_consecutive(eids_sorted, return_counts=True)
        sizes = [int(c) for c in counts]
        uniq_dev = uniq.to(torch.int32)

        a_cat = hidden_states[tok_of_pair]                       # [T*k, H] sorted by expert
        gu = _FusedGroupedMxfp4.apply(a_cat, base.gate_up_blocks,
                                      base.gate_up_scales, sizes, uniq_dev)
        gu = gu + base.gate_up_bias[eids_sorted].to(gu.dtype)
        gu = gu + self._lora_rows(a_cat, eids_sorted,
                                  self.gate_up_lora_A, self.gate_up_lora_B)
        h = apply_gate_gptoss(gu, base.alpha, base.limit).to(hidden_states.dtype)
        h = h.contiguous()
        dn = _FusedGroupedMxfp4.apply(h, base.down_blocks,
                                      base.down_scales, sizes, uniq_dev)
        dn = dn + base.down_bias[eids_sorted].to(dn.dtype)
        dn = dn + self._lora_rows(h, eids_sorted, self.down_lora_A, self.down_lora_B)

        w = routing_weights[tok_of_pair, pos_of_pair, None]
        next_states = torch.zeros_like(hidden_states)
        next_states.index_add_(0, tok_of_pair, (dn * w).to(hidden_states.dtype))
        return next_states

    def _lora_rows(self, x, eids_sorted, A, B):
        """Row-wise LoRA where row i uses expert ``eids_sorted[i]``'s adapter:
        ``scaling * ((x_i @ A[e_i].T) @ B[e_i].T)`` via batched gathers."""
        xa = torch.einsum("tk,trk->tr", x.to(A.dtype), A[eids_sorted])
        delta = torch.einsum("tr,tnr->tn", xa, B[eids_sorted])
        return (self.scaling * delta).to(x.dtype)


def lora_parameters(module: nn.Module):
    """The trainable (adapter) parameters — what the optimizer sees."""
    return [p for p in module.parameters() if p.requires_grad]


def adapter_parameters(module: "ExpertsMxfp4LoRA"):
    """The four adapter tensors BY NAME, independent of requires_grad state.
    Use this to (re-)enable training after a freeze-all sweep —
    ``lora_parameters`` filters by requires_grad and returns [] there (the
    empty-optimizer bug, smoke 4)."""
    return [module.gate_up_lora_A, module.gate_up_lora_B,
            module.down_lora_A, module.down_lora_B]
