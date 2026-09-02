# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Calibrated int4 packing (GPTQ-style) for the existing int4-b32 grid.

The serving attention projections were refused twice on quality: uniform
int4 cost +0.0558 ppl and fp8 +0.0489, both against a 0.05 gate. Both
attempts quantised by ROUNDING each weight to the nearest grid point,
which minimises *weight* error. That is the wrong objective, and the fp8
lane proved it directly: e4m3 carried 4.6x lower mean weight error than
int4 and bought only ~12% less perplexity cost.

What matters is the error in the layer's OUTPUT, weighted by how the
calibration activations actually excite each input channel. GPTQ
(Frantar et al.) minimises exactly that: with ``H = 2 X X^T + lambda I``
over calibration activations, it quantises column by column and pushes
each column's rounding error into the not-yet-quantised columns via the
Cholesky factor of ``H^-1``. Same grid, same bytes, same kernels -- only
the choice of grid point changes, so nothing downstream needs to know.

Reference implementations (bits=4, group_size=128, symmetric) match what
the comparison checkpoint ships; this module keeps the shipped int4-b32
grid (symmetric, block 32) so the existing GEMV and packing format are
unchanged.
"""
from __future__ import annotations

import torch

from int4_pack_ref import BLOCK


def _quantise_block(w_col: torch.Tensor, scale: torch.Tensor):
    """Round one column onto the symmetric int4 grid for its block."""
    q = (w_col / scale).round().clamp(-8, 7)
    return q, q * scale


def gptq_pack_int4_b32(w: torch.Tensor, hessian: torch.Tensor,
                       damp: float = 0.01, blocksize: int = 128):
    """``w [N, K]`` + ``hessian [K, K]`` -> the SAME (packed, scales) pair
    :func:`int4_pack_ref.pack_int4_b32` returns, but with grid points
    chosen to minimise activation-weighted output error.

    ``hessian`` is ``2 X X^T`` accumulated over calibration activations
    for this layer's input. Error from each quantised column is
    propagated to the remaining columns through ``chol(H^-1)``, so a
    column the calibration barely excites absorbs error a heavily
    excited column cannot.

    Two details matter and both were wrong in the first draft:

    * **Scales are found per block on the COMPENSATED weights**, at the
      moment that block is reached -- not once upfront on the source.
      Compensation pushes error into later columns, which can carry a
      value past the source block's absmax; scaling to the source would
      clamp it at the grid edge and throw away exactly the correction
      that was just computed.
    * **The packed bytes are emitted with THESE scales.** Handing the
      dequantised result back to the uncalibrated packer re-derives a
      scale from ``amax/7``, and unless a block happens to reach the
      grid edge that rescales every integer and silently discards the
      calibration.
    """
    N, K = w.shape
    if K % BLOCK:
        raise ValueError(f"K={K} must be a multiple of {BLOCK}")
    W = w.detach().float().clone()
    H = hessian.detach().float().clone()

    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0
    H[range(K), range(K)] += damp * torch.mean(torch.diag(H))

    Hinv = torch.linalg.cholesky(
        torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)

    # buffers follow the weight's device, as pack_int4_b32's do: a caller
    # packing on the GPU must not have a CPU scale meet a CUDA column
    Qi = torch.zeros(N, K, dtype=torch.int8, device=W.device)   # grid ints
    scales = torch.zeros(N, K // BLOCK, dtype=torch.float32, device=W.device)

    for i0 in range(0, K, blocksize):
        i1 = min(i0 + blocksize, K)
        W1 = W[:, i0:i1].clone()
        E1 = torch.zeros_like(W1)
        Hb = Hinv[i0:i1, i0:i1]
        for j in range(i1 - i0):
            c = i0 + j
            if c % BLOCK == 0:                        # entering a new block
                blk = W1[:, j:j + BLOCK]
                s = blk.abs().amax(-1).clamp_min(1e-12) / 7.0
                scales[:, c // BLOCK] = s
            s = scales[:, c // BLOCK]
            col = W1[:, j]
            q = (col / s).round().clamp(-8, 7)
            Qi[:, c] = q.to(torch.int8)
            deq = q * s
            err = (col - deq) / Hb[j, j]
            W1[:, j + 1:] -= err.unsqueeze(1) * Hb[j, j + 1:].unsqueeze(0)
            E1[:, j] = err
        W[:, i1:] -= E1 @ Hinv[i0:i1, i1:]

    qq = (Qi + 8).to(torch.uint8)
    packed = (qq[:, 0::2] | (qq[:, 1::2] << 4)).contiguous()
    return packed, scales.to(torch.float16).contiguous()


class HessianAccumulator:
    """Collects ``2 X X^T`` for one layer over calibration batches."""

    def __init__(self, in_features: int, device=None):
        self.H = torch.zeros(in_features, in_features, dtype=torch.float32,
                             device=device)
        self.n = 0

    def add(self, x: torch.Tensor) -> None:
        """``x [..., in_features]`` activations seen by this layer."""
        rows = x.reshape(-1, x.shape[-1]).float()
        b = rows.shape[0]
        # running mean so batches of different sizes weight correctly
        self.H *= self.n / (self.n + b)
        self.n += b
        self.H += (2.0 / self.n) * (rows.t() @ rows)
