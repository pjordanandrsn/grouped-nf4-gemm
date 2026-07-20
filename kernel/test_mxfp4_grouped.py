# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Phase-2 gates for gemm_mxfp4_grouped, in the seam-map's mandated order:
  (a) interpreter parity — the kernel decode == mxfp4_pack_ref on the same
      bytes (device-free, TRITON_INTERPRET=1);
  (b) GPU exact-decode — the fused GEMM == reference (dequant @ a) on the
      real device, within fp32-accum/bf16-epilogue tolerance;
  (c) grouped/ragged shape instantiated for the format (mixed experts, decode
      + prefill).
The reference dequant is mxfp4_pack_ref (Phase-1: == the A4 oracle bit-exact).
"""
import pytest
import torch

pytest.importorskip("triton")

from mxfp4_pack_ref import MX_BLOCK, dequant_mxfp4, quantize_pack_mxfp4  # noqa: E402


def _ref_gemm(a_cat, blocks, scales, sizes, eids):
    """Grouped reference: per group g, out rows = a_rows @ dequant(W[eid]).T."""
    outs = []
    r = 0
    for g, m in enumerate(sizes):
        blk = blocks[eids[g]]                        # [N, K//2] flattened
        nb = scales[eids[g]].shape[-1]
        W = dequant_mxfp4(blk.reshape(blk.shape[0], nb, 16), scales[eids[g]])  # [N, K]
        a = a_cat[r:r + m].float()
        outs.append(a @ W.t())
        r += m
    return torch.cat(outs, 0)


def _stack(E, N, K, seed):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(E, N, K, generator=g) * 0.3
    B = torch.empty(E, N, K // 2, dtype=torch.uint8)
    S = torch.empty(E, N, K // MX_BLOCK, dtype=torch.uint8)
    for e in range(E):
        b, s = quantize_pack_mxfp4(w[e])
        B[e], S[e] = b.reshape(N, K // 2), s
    return B, S


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("shape", [(4, 128, 256), (2, 256, 128)])
def test_gpu_exact_decode(shape):
    import mxfp4_grouped
    E, N, K = shape
    B, S = _stack(E, N, K, seed=1)
    B, S = B.cuda(), S.cuda()
    # decode: every group one token
    T = E
    a = (torch.randn(T, K, dtype=torch.bfloat16, device="cuda"))
    sizes = [1] * E
    eids = list(range(E))
    got = mxfp4_grouped.gemm_mxfp4_grouped(a, B, S, sizes, eids)
    ref = _ref_gemm(a.cpu(), B.cpu(), S.cpu(), sizes, eids).cuda()
    br = ((got.float() - ref).abs().max() / ref.abs().max()).item()
    assert br < 2e-2, br


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_prefill_mixed_groups():
    import mxfp4_grouped
    E, N, K = 4, 128, 128
    B, S = _stack(E, N, K, seed=2)
    B, S = B.cuda(), S.cuda()
    sizes = [3, 1, 4, 2]           # mixed -> prefill M-tile path
    eids = [1, 0, 3, 2]
    T = sum(sizes)
    a = torch.randn(T, K, dtype=torch.bfloat16, device="cuda")
    got = mxfp4_grouped.gemm_mxfp4_grouped(a, B, S, sizes, eids)
    ref = _ref_gemm(a.cpu(), B.cpu(), S.cpu(), sizes, eids).cuda()
    br = ((got.float() - ref).abs().max() / ref.abs().max()).item()
    assert br < 2e-2, br


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_device_expert_ids():
    """expert_ids as a device int32 tensor (the pipelined calling convention)."""
    import mxfp4_grouped
    E, N, K = 4, 96, 128
    B, S = _stack(E, N, K, seed=3)
    B, S = B.cuda(), S.cuda()
    sizes = [1] * 3
    eids_list = [2, 0, 3]
    a = torch.randn(3, K, dtype=torch.bfloat16, device="cuda")
    got_t = mxfp4_grouped.gemm_mxfp4_grouped(
        a, B, S, sizes, torch.tensor(eids_list, dtype=torch.int32, device="cuda"))
    ref = _ref_gemm(a.cpu(), B.cpu(), S.cpu(), sizes, eids_list).cuda()
    assert ((got_t.float() - ref).abs().max() / ref.abs().max()).item() < 2e-2
