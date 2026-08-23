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
import pytest  # noqa: E402

# Interpreter mode IS a triton feature -- with no triton there is nothing to
# interpret. triton is a Linux-only dependency here (see `_triton_shim`), so on
# a platform without it this gate SKIPS rather than failing on a stub attribute.
pytest.importorskip("triton", reason="interpreter mode needs triton (Linux-only dependency)")

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

def test_decode_expert_offset_past_2gib():
    """Regression: eid * stride_be must be int64 -- a slot id whose byte
    offset passes 2^31 faulted (illegal memory access) or wrapped before the
    cast. Exercises the exact G1 transient-pool shape: an as_strided view over
    one flat buffer, row stride 8.8 MB, slot 250 -> offset 2.20e9 > 2^31."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    N, K, rows = 5760, 2880, 260
    pb, sb = N * (K // 2), N * (K // 32)
    stride = pb + sb
    buf = torch.zeros(rows * stride, dtype=torch.uint8, device="cuda")
    blocks = torch.as_strided(buf, (rows, N, K // 2), (stride, K // 2, 1))
    scales = torch.as_strided(buf, (rows, N, K // 32), (stride, K // 32, 1),
                              storage_offset=pb)
    g = torch.Generator().manual_seed(3)
    blk = torch.randint(0, 256, (N, K // 2), generator=g, dtype=torch.uint8)
    scl = torch.randint(100, 140, (N, K // 32), generator=g, dtype=torch.uint8)
    slot = 250                       # 250 * stride = 2.20e9 -- past int32
    blocks[slot].copy_(blk)
    scales[slot].copy_(scl)
    a = torch.randn(1, K, generator=g, dtype=torch.float32).to(torch.bfloat16)
    got = mxfp4_grouped.gemm_mxfp4_grouped(a.cuda(), blocks, scales, [1], [slot])
    torch.cuda.synchronize()
    W = dequant_mxfp4(blk.reshape(N, K // 32, 16).cuda(), scl.cuda())
    ref = a.float().cuda() @ W.t()
    rel = ((got.float() - ref).abs().max() / ref.abs().max()).item()
    assert rel < 2e-2, rel
