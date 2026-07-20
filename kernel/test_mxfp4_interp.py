# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Phase-2 gate (a): interpreter parity — the mxfp4 kernel decode == the
mxfp4_pack_ref reference on the SAME bytes, device-free (TRITON_INTERPRET=1).
Run in its OWN pytest process: interpreter mode is process-global and cannot
co-exist with compiled-GPU tests (documented lesson). Import order sets the
env before triton is touched."""
import os
os.environ["TRITON_INTERPRET"] = "1"

import torch  # noqa: E402

from mxfp4_pack_ref import MX_BLOCK, dequant_mxfp4, quantize_pack_mxfp4  # noqa: E402
import mxfp4_grouped  # noqa: E402


def _ref(a_cat, blocks, scales, sizes, eids):
    outs, r = [], 0
    for g, m in enumerate(sizes):
        blk = blocks[eids[g]]
        nb = scales[eids[g]].shape[-1]
        W = dequant_mxfp4(blk.reshape(blk.shape[0], nb, 16), scales[eids[g]])
        outs.append(a_cat[r:r + m].float() @ W.t())
        r += m
    return torch.cat(outs, 0)


def test_interpreter_parity():
    E, N, K = 3, 64, 128
    g = torch.Generator().manual_seed(0)
    w = torch.randn(E, N, K, generator=g) * 0.3
    B = torch.empty(E, N, K // 2, dtype=torch.uint8)
    S = torch.empty(E, N, K // MX_BLOCK, dtype=torch.uint8)
    for e in range(E):
        b, s = quantize_pack_mxfp4(w[e])
        B[e], S[e] = b.reshape(N, K // 2), s
    a = torch.randn(5, K, dtype=torch.bfloat16)
    sizes, eids = [2, 2, 1], [0, 2, 1]
    got = mxfp4_grouped.gemm_mxfp4_grouped(a, B, S, sizes, eids)
    ref = _ref(a, B, S, sizes, eids)
    assert ((got.float() - ref).abs().max() / ref.abs().max()).item() < 2e-2
