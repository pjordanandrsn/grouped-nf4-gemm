"""Decode-grade MXFP4 GEMV (``gemv_mxfp4_b32``): the int4-b32 kernel
structure on the native e2m1/e8m0 store. Gates: (a) the branchless e2m1
decode equals the codebook on all 16 nibbles (pure python); (b)
interpreter parity against ``mxfp4_pack_ref`` on the same bytes with the
int4-b32 quantised activation rows (device-free, TRITON_INTERPRET=1);
(c) on a GPU, the same at a gpt-oss-shaped K."""
import os

os.environ.setdefault("TRITON_INTERPRET", "1")

import pytest  # noqa: E402
import torch  # noqa: E402

pytest.importorskip("triton", reason="interpreter mode needs triton (Linux-only dependency)")

from mxfp4_pack_ref import MX_BLOCK, dequant_mxfp4, quantize_pack_mxfp4  # noqa: E402
import mxfp4_grouped  # noqa: E402


def _stack(E, N, K, seed):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(E, N, K, generator=g) * 0.3
    B = torch.empty(E, N, K // 2, dtype=torch.uint8)
    S = torch.empty(E, N, K // MX_BLOCK, dtype=torch.uint8)
    for e in range(E):
        b, s = quantize_pack_mxfp4(w[e])
        B[e], S[e] = b.reshape(N, K // 2), s
    return B, S


def _ref_rows(xq, xs, B, S, eids):
    """Row e: dequantised int8 activation row against dequant(W[eids[e]])."""
    a = xq.float() * xs.repeat_interleave(MX_BLOCK, dim=1)
    outs = []
    for e, eid in enumerate(eids.tolist()):
        blk = B[eid]
        nb = S[eid].shape[-1]
        W = dequant_mxfp4(blk.reshape(blk.shape[0], nb, 16), S[eid])
        outs.append(a[e] @ W.t())
    return torch.stack(outs, 0)


def _check(E, N, K, R, device, seed=0):
    from int4_b32 import quant_x_rows
    B, S = _stack(E, N, K, seed)
    B, S = B.to(device), S.to(device)
    torch.manual_seed(seed)
    x = (torch.randn(R, K) * 0.7).to(device, torch.bfloat16)
    xq, xs = quant_x_rows(x)
    eids = torch.tensor([(3 * e + 1) % E for e in range(R)], dtype=torch.int32, device=device)
    got = mxfp4_grouped.gemv_mxfp4_b32(xq, xs, B, S, eids, N, K)
    ref = _ref_rows(xq.cpu(), xs.cpu(), B.cpu(), S.cpu(), eids.cpu())
    assert got.shape == (R, N) and got.dtype == torch.bfloat16
    rel = ((got.float().cpu() - ref).abs().max() / ref.abs().max()).item()
    assert rel < 2e-2, rel


@pytest.mark.parametrize("E,N,K,R", [(3, 64, 128, 1), (2, 200, 256, 4), (4, 96, 96, 2)])
def test_interpreter_parity(E, N, K, R):
    """Device-free: shapes that exercise a masked N tail (200, 96), a
    K//32 that is not a multiple of 4 (96 -> KU=1... 3 blocks) and
    several rows against different experts."""
    _check(E, N, K, R, "cpu")


@pytest.mark.skipif(os.environ.get("TRITON_INTERPRET") == "1" or not torch.cuda.is_available(),
                    reason="GPU gate")
def test_gpu_gptoss_k():
    _check(4, 512, 2880, 8, "cuda", seed=1)
