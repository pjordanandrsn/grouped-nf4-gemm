"""The 2^31-byte offset boundary in the grouped kernels.

CONFIRMED shipped bug (0.13.0/0.13.1, found on an A6000 sweeping
Qwen3.8-2.4T-A95B): `gemm_4bit_grouped` raises an illegal memory access
whenever the packed stack `B` exceeds 2^31 bytes. Signed-int32 offset
arithmetic — `eid * stride_be` — so a stack of exactly 2 GiB is the last one
that works. The boundary was PREDICTED from the stride math and then hit
exactly on two different shapes (256 x 8 MiB passes, 257 fails; 128 x 16 MiB
likewise). The gather is fine; it is the GEMM, on both the decode and M-tile
paths, and dgrad shares the same pattern.

The fixture builds packed bytes and absmax DIRECTLY as random tensors — no bnb
quantize pass, because 258 experts of it would take minutes and prove nothing
extra: `dequant_ref` is the oracle either way, and it decodes whatever bytes it
is given. Experts are sampled on BOTH sides of the boundary (the one whose base
offset is the first past 2^31 is the one that used to read garbage or fault).

~2.3 GiB device memory for the big stack; skipped below 6 GiB free.
"""
from __future__ import annotations

import pytest
import torch

import nf4_grouped as NG
import mxfp4_grouped as MX
from mxfp4_pack_ref import dequant_mxfp4

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="offset boundary needs CUDA")

# down_proj-class experts: 8 MiB packed each -> expert 256's base offset is
# exactly 2^31. E=258 puts two full experts past the boundary.
N, K = 8192, 2048
E_BIG, E_SMALL = 258, 8
PER_EXPERT_PACKED = N * (K // 2)                      # 8 MiB
SAMPLES = (0, 128, 255, 256, 257)                     # both sides + boundary


def _mem_ok():
    free, _ = torch.cuda.mem_get_info()
    return free >= 6 * 2**30


def _stack(E, seed=11):
    g = torch.Generator(device="cpu").manual_seed(seed)
    B = torch.randint(0, 256, (E, N, K // 2), generator=g,
                      dtype=torch.uint8).cuda()
    A = (torch.rand(E, N, K // NG.BLOCKSIZE, generator=g) * 0.02 + 1e-3
         ).float().cuda()
    return B, A


def _check_expert(B, A, e, tag):
    """One-token gemv + a 4-row M-tile call + dgrad against dequant_ref, for
    expert e alone — the offset under test is e * stride_be."""
    torch.manual_seed(e)
    w_ref = NG.dequant_ref(B[e].cpu(), A[e].cpu(), N, K).cuda()

    a1 = (torch.randn(1, K, device="cuda") * 0.5).bfloat16()
    got = NG.gemm_4bit_grouped(a1, B, A, [1], [e])
    want = (a1.float() @ w_ref.T).bfloat16()
    rel = (got.float() - want.float()).norm() / want.float().norm()
    assert rel < 5e-2, f"{tag} gemv expert {e}: rel {rel:.3e}"

    a4 = (torch.randn(4, K, device="cuda") * 0.5).bfloat16()
    got = NG.gemm_4bit_grouped(a4, B, A, [4], [e])
    want = (a4.float() @ w_ref.T).bfloat16()
    rel = (got.float() - want.float()).norm() / want.float().norm()
    assert rel < 5e-2, f"{tag} m-tile expert {e}: rel {rel:.3e}"

    go = (torch.randn(4, N, device="cuda") * 0.1).bfloat16()
    got = NG.dgrad_4bit_grouped(go, B, A, [4], [e])
    want = (go.float() @ w_ref).bfloat16()
    rel = (got.float() - want.float()).norm() / want.float().norm()
    assert rel < 5e-2, f"{tag} dgrad expert {e}: rel {rel:.3e}"
    del w_ref
    torch.cuda.empty_cache()


def test_small_stack_control():
    """Positive control for the instrument: the same checks pass on a stack
    far below the boundary, so a big-stack failure is the OFFSET, not the
    fixture or tolerances."""
    B, A = _stack(E_SMALL)
    for e in (0, E_SMALL - 1):
        _check_expert(B, A, e, "small")
    del B, A
    torch.cuda.empty_cache()


def test_experts_across_the_2gib_boundary():
    """The bug: experts whose byte offsets sit past 2^31 must decode the SAME
    values dequant_ref reads there. Pre-fix this faults (illegal memory access)
    or reads the wrong expert; post-fix it must agree everywhere sampled."""
    if not _mem_ok():
        pytest.skip("needs ~6 GiB free device memory for the 2.3 GiB stack")
    B, A = _stack(E_BIG)
    assert B.numel() > 2**31, "fixture must actually cross the boundary"
    for e in SAMPLES:
        _check_expert(B, A, e, "big")
    # And one grouped call touching BOTH sides in a single launch.
    a = (torch.randn(2, K, device="cuda") * 0.5).bfloat16()
    got = NG.gemm_4bit_grouped(a, B, A, [1, 1], [0, 257])
    for row, e in ((0, 0), (1, 257)):
        w_ref = NG.dequant_ref(B[e].cpu(), A[e].cpu(), N, K).cuda()
        want = (a[row:row + 1].float() @ w_ref.T).bfloat16()
        rel = (got[row:row + 1].float() - want.float()).norm() / want.float().norm()
        assert rel < 5e-2, f"grouped both-sides expert {e}: rel {rel:.3e}"
        del w_ref
    del B, A
    torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# MXFP4: the port dropped NF4's int64 cast (found live by the P2-G1 box run:
# illegal memory access at transient-pool slot 244, 244 x 8,812,800 > 2^31).
# Same boundary, same fix, so the same test structure.

MX_N, MX_K = 8192, 2048              # 8 MiB packed/expert: expert 256 = 2^31


def _mx_stack(E, seed=13):
    g = torch.Generator(device="cpu").manual_seed(seed)
    B = torch.randint(0, 256, (E, MX_N, MX_K // 2), generator=g,
                      dtype=torch.uint8).cuda()
    S = torch.randint(100, 140, (E, MX_N, MX_K // 32), generator=g,
                      dtype=torch.uint8).cuda()
    return B, S


def _mx_check(B, S, e, tag):
    """Decode (sizes==1) + one M-tile call for expert e alone — the offset
    under test is e * stride_be, shared by both kernels."""
    w_ref = dequant_mxfp4(B[e].reshape(MX_N, MX_K // 32, 16), S[e]).float()
    a1 = (torch.randn(1, MX_K, device="cuda") * 0.5).bfloat16()
    got = MX.gemm_mxfp4_grouped(a1, B, S, [1], [e])
    want = a1.float() @ w_ref.t()
    rel = ((got.float() - want).abs().max() / want.abs().max()).item()
    assert rel < 2e-2, f"{tag} decode expert {e}: rel {rel:.3e}"
    a4 = (torch.randn(4, MX_K, device="cuda") * 0.5).bfloat16()
    got = MX.gemm_mxfp4_grouped(a4, B, S, [4], [e])
    want = a4.float() @ w_ref.t()
    rel = ((got.float() - want).abs().max() / want.abs().max()).item()
    assert rel < 2e-2, f"{tag} m-tile expert {e}: rel {rel:.3e}"
    del w_ref
    torch.cuda.empty_cache()


def test_mxfp4_small_stack_control():
    B, S = _mx_stack(E_SMALL)
    for e in (0, E_SMALL - 1):
        _mx_check(B, S, e, "mx-small")
    del B, S
    torch.cuda.empty_cache()


def test_mxfp4_experts_across_the_2gib_boundary():
    if not _mem_ok():
        pytest.skip("needs ~6 GiB free device memory for the 2.3 GiB stack")
    B, S = _mx_stack(E_BIG)
    assert B.numel() > 2**31, "fixture must actually cross the boundary"
    for e in SAMPLES:
        _mx_check(B, S, e, "mx-big")
    a = (torch.randn(2, MX_K, device="cuda") * 0.5).bfloat16()
    got = MX.gemm_mxfp4_grouped(a, B, S, [1, 1], [0, 257])
    for row, e in ((0, 0), (1, 257)):
        w_ref = dequant_mxfp4(B[e].reshape(MX_N, MX_K // 32, 16), S[e]).float()
        want = a[row:row + 1].float() @ w_ref.t()
        rel = ((got[row:row + 1].float() - want).abs().max()
               / want.abs().max()).item()
        assert rel < 2e-2, f"mx grouped both-sides expert {e}: rel {rel:.3e}"
        del w_ref
    del B, S
    torch.cuda.empty_cache()


def test_mxfp4_pool_stride_past_2gib():
    """The exact configuration that found the bug: an as_strided transient
    pool over ONE flat buffer, row stride = packed + scales (8.8 MB), so the
    stride exceeds the row payload and slot 250's byte offset is 2.20e9."""
    if not _mem_ok():
        pytest.skip("needs ~6 GiB free device memory for the 2.3 GiB pool")
    N, K, rows = 5760, 2880, 260
    pb = N * (K // 2)
    stride = pb + N * (K // 32)
    buf = torch.zeros(rows * stride, dtype=torch.uint8, device="cuda")
    blocks = torch.as_strided(buf, (rows, N, K // 2), (stride, K // 2, 1))
    scales = torch.as_strided(buf, (rows, N, K // 32), (stride, K // 32, 1),
                              storage_offset=pb)
    g = torch.Generator(device="cpu").manual_seed(3)
    blk = torch.randint(0, 256, (N, K // 2), generator=g, dtype=torch.uint8)
    scl = torch.randint(100, 140, (N, K // 32), generator=g, dtype=torch.uint8)
    slot = 250
    blocks[slot].copy_(blk)
    scales[slot].copy_(scl)
    a = (torch.randn(1, K, generator=g).float() * 0.5).bfloat16()
    got = MX.gemm_mxfp4_grouped(a.cuda(), blocks, scales, [1], [slot])
    torch.cuda.synchronize()
    w_ref = dequant_mxfp4(blk.reshape(N, K // 32, 16).cuda(), scl.cuda()).float()
    want = a.float().cuda() @ w_ref.t()
    rel = ((got.float() - want).abs().max() / want.abs().max()).item()
    assert rel < 2e-2, rel
    del buf, w_ref
    torch.cuda.empty_cache()
