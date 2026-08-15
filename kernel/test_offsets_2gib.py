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
