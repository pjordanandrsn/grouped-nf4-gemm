# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The 2^31 expert-offset boundary (gnf4#87), straddled, in every kernel that
scales an expert id or a block-table row by a stride.

The bug as filed: an expert id loaded from an int32 tensor and multiplied by
``stride_be`` in int32 wraps once ``max(expert_ids) * stride_be >= 2**31``,
so the highest experts of a large stack read the wrong bytes or fault. The
cast to int64 at the load shipped in 0.13.2 (NF4) and 0.14.0 (MXFP4), and
``test_offsets_2gib.py`` keeps the original 258 x 8 MiB reproduction -- but
that fixture never engages split-K (K=2048 is below the split floor), never
runs the dot-pad or wide-load decode variants, and covers neither the
int4-b32 kernels nor the fp8 paged-attention readers, whose block-table row
times ``k_row_bytes`` is the same pattern. The issue itself flagged split-K
and dgrad "by inspection, not observation". This file observes.

Geometry (state the sizes): small K and N with a large expert count, so the
stack is barely past 2 GiB and the boundary lands on a whole expert:

  N=256, K=1024  ->  128 KiB packed per expert (2^17 B)
  expert 16383's base offset = 2^31 - 2^17   (the last below the boundary)
  expert 16384's base offset = 2^31          (the first at/above it)
  E = 16386 experts: 2.00 GiB packed + scales (NF4 absmax 256 MiB fp32;
  MXFP4 e8m0 128 MiB; int4-b32 fp16 256 MiB)

  dot-pad path (only at its census shape): N=1536, K=2048 -> 1.5 MiB/expert,
  expert 1365 = 2^31 - 1 MiB below, expert 1366 = 2^31 + 0.5 MiB above,
  E = 1368: 2.00 GiB packed + 269 MiB absmax

  fp8 paged attention: head_dim 128, 8 kv heads, 16 tokens/block, K rows
  padded to a 1 MiB stride (``k_row_bytes``), 2050 rows = 2.00 GiB K pool
  (+ 35 MiB V pool at its natural row); the 32-token sequence's blocks sit
  at rows 2047 (base 2^31 - 2^20) and 2048 (base 2^31)

About 2.3 GiB of device memory per case; skipped below 4 GiB free. Every
ABOVE-boundary case runs in its own subprocess, because an illegal access
poisons the CUDA context and would make every later result in the process
meaningless (the misdiagnosis the issue reports). The below-boundary
controls run in-process and prove the instrument: same code, same checks,
a stack far below the boundary.

The check is against the pure-torch references (``dequant_ref``,
``dequant_mxfp4``, ``dequant_int4_ref``, ``paged_attn_ref``), so a wrapped
offset that lands inside another valid allocation -- wrong numbers instead
of a fault, the case the issue could not rule out -- fails here too.

    cd kernel && python -m pytest test_expert_offset_boundary.py -q
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="the offset boundary needs a CUDA device")

BOUNDARY = 2 ** 31
MIN_FREE = 4 * 2 ** 30
DEV = "cuda"


def _mem_ok() -> bool:
    free, _ = torch.cuda.mem_get_info()
    return free >= MIN_FREE


def straddle(stride: int) -> tuple[int, int]:
    """(last expert below 2^31, first expert at or above it) for a per-expert
    byte stride."""
    first_above = -(-BOUNDARY // stride)
    last_below = first_above - 1
    assert first_above * stride >= BOUNDARY > last_below * stride
    return last_below, first_above


def _rel(got, want) -> float:
    return ((got.float() - want.float()).norm() / want.float().norm().clamp_min(1e-6)).item()


# ------------------------------------------------------------------- NF4 --
NF4_N, NF4_K = 256, 1024
NF4_STRIDE = NF4_N * (NF4_K // 2)                     # 131072 = 2^17
NF4_BELOW, NF4_ABOVE = straddle(NF4_STRIDE)            # 16383, 16384
NF4_E = NF4_ABOVE + 2                                  # 16386


def _nf4_stack(E, N, K, seed=11):
    import nf4_grouped as NG
    g = torch.Generator(device=DEV).manual_seed(seed)
    B = torch.randint(0, 256, (E, N, K // 2), generator=g, dtype=torch.uint8, device=DEV)
    A = (torch.rand(E, N, K // NG.BLOCKSIZE, generator=g, device=DEV) * 0.02 + 1e-3)
    return B, A


def _nf4_check_expert(B, A, N, K, e, tag, *, dotpad_shape=False):
    """Decode (scalar / split-K / wide / vec, or dot-pad / dot-pad split-K at
    the census shape), the M-tile forward and dgrad, for expert e alone --
    the offset under test is e * stride_be. Each launch's ROUTE is asserted
    through the dispatch tally, so a knob that quietly took another path
    cannot pass as coverage."""
    import nf4_grouped as NG
    torch.manual_seed(e)
    w_ref = NG.dequant_ref(B[e], A[e], N, K)          # [N, K] fp32 on device
    a1 = (torch.randn(1, K, device=DEV) * 0.5).bfloat16()
    want1 = (a1.float() @ w_ref.T)
    a4 = (torch.randn(4, K, device=DEV) * 0.5).bfloat16()
    want4 = (a4.float() @ w_ref.T)
    go = (torch.randn(4, N, device=DEV) * 0.1).bfloat16()
    want_dg = (go.float() @ w_ref)

    def decode(label, **kw):
        env = kw.pop("env", {})
        saved = {k: os.environ.get(k) for k in ("GNF4_GEMV_WIDE_LOADS", "GNF4_GEMV_VEC_LOADS",
                                                "GNF4_GEMV_DOTPAD", "GNF4_GEMV_SPLITK")}
        for k in saved:
            os.environ.pop(k, None)
        os.environ.update(env)
        try:
            before = NG.dispatch_counts()
            got = NG.gemm_4bit_grouped(a1, B, A, [1], [e], **kw)
            torch.cuda.synchronize()
            after = NG.dispatch_counts()
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        route = [k for k in after if after[k] != before[k]]
        assert route == [label.split("/")[0]], f"{tag} {label}: routed {route}"
        r = _rel(got, want1)
        assert r < 5e-2, f"{tag} decode {label} expert {e}: rel {r:.3e}"

    if dotpad_shape:
        decode("dotpad")                                        # the sm_120 default here
        decode("dotpad_splitk", split_k=4)
        decode("scalar/off", env={"GNF4_GEMV_DOTPAD": "0"})
    else:
        decode("scalar", decode_config=(64, 2), split_k=1)
        decode("scalar_splitk", decode_config=(64, 2), split_k=4)
        decode("scalar/wide", decode_config=(64, 2), split_k=1,
               env={"GNF4_GEMV_WIDE_LOADS": "1"})
        decode("scalar_splitk/wide", decode_config=(64, 2), split_k=4,
               env={"GNF4_GEMV_WIDE_LOADS": "1"})
        if NG.HAS_TL_INTERLEAVE:
            decode("scalar/vec", decode_config=(64, 2), split_k=1,
                   env={"GNF4_GEMV_VEC_LOADS": "1"})

    got = NG.gemm_4bit_grouped(a4, B, A, [4], [e])
    torch.cuda.synchronize()
    r = _rel(got, want4)
    assert r < 5e-2, f"{tag} m-tile expert {e}: rel {r:.3e}"
    got = NG.dgrad_4bit_grouped(go, B, A, [4], [e])
    torch.cuda.synchronize()
    r = _rel(got, want_dg)
    assert r < 5e-2, f"{tag} dgrad expert {e}: rel {r:.3e}"
    del w_ref


def case_nf4(big: bool):
    E = NF4_E if big else 8
    B, A = _nf4_stack(E, NF4_N, NF4_K)
    if big:
        assert (E - 1) * NF4_STRIDE >= BOUNDARY and B.numel() > BOUNDARY
        experts = (0, NF4_BELOW, NF4_ABOVE, NF4_ABOVE + 1)
    else:
        experts = (0, E - 1)
    for e in experts:
        _nf4_check_expert(B, A, NF4_N, NF4_K, e, "nf4")
    # one grouped launch touching both sides of the boundary
    import nf4_grouped as NG
    lo, hi = experts[0], experts[-1]
    a = (torch.randn(2, NF4_K, device=DEV) * 0.5).bfloat16()
    got = NG.gemm_4bit_grouped(a, B, A, [1, 1], [lo, hi], decode_config=(64, 2), split_k=1)
    torch.cuda.synchronize()
    for row, e in ((0, lo), (1, hi)):
        want = a[row:row + 1].float() @ NG.dequant_ref(B[e], A[e], NF4_N, NF4_K).T
        r = _rel(got[row:row + 1], want)
        assert r < 5e-2, f"nf4 grouped both-sides expert {e}: rel {r:.3e}"


DP_N, DP_K = 1536, 2048                                 # the gate_up census shape
DP_STRIDE = DP_N * (DP_K // 2)                          # 1572864
DP_BELOW, DP_ABOVE = straddle(DP_STRIDE)                # 1365, 1366
DP_E = DP_ABOVE + 2                                     # 1368


def case_nf4_dotpad(big: bool):
    """The dot-pad decode GEMV and its split-K form engage only at their
    census shapes on >= 160-SM parts; elsewhere this case still runs the
    scalar route at that shape (the tally says which)."""
    import nf4_grouped as NG
    if NG._sm_count(torch.device(DEV)) < 160:
        print("SKIP dot-pad: fewer than 160 SMs, the dot-pad route is off here")
        return
    E = DP_E if big else 4
    B, A = _nf4_stack(E, DP_N, DP_K, seed=17)
    if big:
        assert (E - 1) * DP_STRIDE >= BOUNDARY
        experts = (0, DP_BELOW, DP_ABOVE, DP_ABOVE + 1)
    else:
        experts = (0, E - 1)
    for e in experts:
        _nf4_check_expert(B, A, DP_N, DP_K, e, "nf4-dotpad", dotpad_shape=True)


# ----------------------------------------------------------------- MXFP4 --
MX_N, MX_K = 256, 1024
MX_STRIDE = MX_N * (MX_K // 2)                          # 2^17
MX_BELOW, MX_ABOVE = straddle(MX_STRIDE)
MX_E = MX_ABOVE + 2


def _mx_stack(E, seed=13):
    g = torch.Generator(device=DEV).manual_seed(seed)
    B = torch.randint(0, 256, (E, MX_N, MX_K // 2), generator=g, dtype=torch.uint8, device=DEV)
    S = torch.randint(100, 140, (E, MX_N, MX_K // 32), generator=g, dtype=torch.uint8, device=DEV)
    return B, S


def _mx_check_expert(B, S, e, tag):
    import mxfp4_grouped as MX
    from int4_b32 import quant_x_rows
    from mxfp4_pack_ref import dequant_mxfp4
    torch.manual_seed(e)
    w_ref = dequant_mxfp4(B[e].reshape(MX_N, MX_K // 32, 16), S[e]).float()
    a1 = (torch.randn(1, MX_K, device=DEV) * 0.5).bfloat16()
    got = MX.gemm_mxfp4_grouped(a1, B, S, [1], [e])
    torch.cuda.synchronize()
    want = a1.float() @ w_ref.T
    r = _rel(got, want)
    assert r < 2e-2, f"{tag} decode expert {e}: rel {r:.3e}"
    a4 = (torch.randn(4, MX_K, device=DEV) * 0.5).bfloat16()
    got = MX.gemm_mxfp4_grouped(a4, B, S, [4], [e])
    torch.cuda.synchronize()
    r = _rel(got, a4.float() @ w_ref.T)
    assert r < 2e-2, f"{tag} m-tile expert {e}: rel {r:.3e}"
    # the b32 decode GEMV: int8 activations, exact int32 e2m1 dot
    xq, xs = quant_x_rows(a1)
    got = MX.gemv_mxfp4_b32(xq, xs, B, S, torch.tensor([e], dtype=torch.int32, device=DEV),
                            MX_N, MX_K)
    torch.cuda.synchronize()
    x_deq = xq.float() * xs.repeat_interleave(32, dim=1)
    r = _rel(got, x_deq @ w_ref.T)
    assert r < 2e-2, f"{tag} gemv_b32 expert {e}: rel {r:.3e}"
    del w_ref


def case_mxfp4(big: bool):
    E = MX_E if big else 8
    B, S = _mx_stack(E)
    if big:
        assert (E - 1) * MX_STRIDE >= BOUNDARY
        experts = (0, MX_BELOW, MX_ABOVE, MX_ABOVE + 1)
    else:
        experts = (0, E - 1)
    for e in experts:
        _mx_check_expert(B, S, e, "mxfp4")
    import mxfp4_grouped as MX
    from mxfp4_pack_ref import dequant_mxfp4
    lo, hi = experts[0], experts[-1]
    a = (torch.randn(2, MX_K, device=DEV) * 0.5).bfloat16()
    got = MX.gemm_mxfp4_grouped(a, B, S, [1, 1], [lo, hi])
    torch.cuda.synchronize()
    for row, e in ((0, lo), (1, hi)):
        w_ref = dequant_mxfp4(B[e].reshape(MX_N, MX_K // 32, 16), S[e]).float()
        r = _rel(got[row:row + 1], a[row:row + 1].float() @ w_ref.T)
        assert r < 2e-2, f"mxfp4 grouped both-sides expert {e}: rel {r:.3e}"


# -------------------------------------------------------------- int4-b32 --
I4_N, I4_K = 256, 1024
I4_STRIDE = I4_N * (I4_K // 2)
I4_BELOW, I4_ABOVE = straddle(I4_STRIDE)
I4_E = I4_ABOVE + 2


def _i4_stack(E, seed=19):
    g = torch.Generator(device=DEV).manual_seed(seed)
    P = torch.randint(0, 256, (E, I4_N, I4_K // 2), generator=g, dtype=torch.uint8, device=DEV)
    S = (torch.rand(E, I4_N, I4_K // 32, generator=g, device=DEV) * 0.05 + 1e-3).half()
    return P, S


def _i4_check_experts(P, S, experts, tag):
    """The decode GEMV over all sampled experts in one launch, then the
    M-tile grouped GEMM over prebuilt device tiles (row0 and eid int64)."""
    from int4_b32 import gemm_int4_b32_grouped_captured, gemv_int4_b32, quant_x_rows
    from int4_pack_ref import dequant_int4_ref
    from nf4_grouped import build_group_tiles_device
    E = P.shape[0]
    R = len(experts)
    torch.manual_seed(R)
    x = (torch.randn(R, I4_K, device=DEV) * 0.5).bfloat16()
    xq, xs = quant_x_rows(x)
    x_deq = xq.float() * xs.repeat_interleave(32, dim=1)
    eids = torch.tensor(list(experts), dtype=torch.int32, device=DEV)
    got = gemv_int4_b32(xq, xs, P, S, eids, I4_N, I4_K)
    torch.cuda.synchronize()
    for r_i, e in enumerate(experts):
        want = x_deq[r_i:r_i + 1] @ dequant_int4_ref(P[e], S[e], I4_N, I4_K).T
        r = _rel(got[r_i:r_i + 1], want)
        assert r < 1e-2, f"{tag} gemv expert {e}: rel {r:.3e}"
    # M-tile: 4 rows per sampled expert, expert-major order, device tiles
    rows_per = 4
    flat = eids.repeat_interleave(rows_per)
    xm = (torch.randn(R * rows_per, I4_K, device=DEV) * 0.5).bfloat16()
    xq_m, xs_m = quant_x_rows(xm)
    t_row0, t_rows, t_group, order, _counts = build_group_tiles_device(flat, E, 16)
    got = gemm_int4_b32_grouped_captured(xq_m.index_select(0, order),
                                         xs_m.index_select(0, order), P, S,
                                         t_row0, t_rows, t_group, block_m=16)
    torch.cuda.synchronize()
    xm_deq = (xq_m.float() * xs_m.repeat_interleave(32, dim=1)).index_select(0, order)
    flat_sorted = flat.index_select(0, order)
    for r_i in range(R * rows_per):
        e = int(flat_sorted[r_i])
        want = xm_deq[r_i:r_i + 1] @ dequant_int4_ref(P[e], S[e], I4_N, I4_K).T
        r = _rel(got[r_i:r_i + 1], want)
        assert r < 1e-2, f"{tag} m-tile row {r_i} expert {e}: rel {r:.3e}"


def case_int4_b32(big: bool):
    E = I4_E if big else 8
    P, S = _i4_stack(E)
    if big:
        assert (E - 1) * I4_STRIDE >= BOUNDARY
        experts = (0, I4_BELOW, I4_ABOVE, I4_ABOVE + 1)
    else:
        experts = (0, E - 1)
    _i4_check_experts(P, S, experts, "int4-b32")


# ------------------------------------------------------ fp8 paged attention --
FA_D, FA_HKV, FA_G, FA_BT, FA_KG = 128, 8, 2, 16, 4
FA_KROW = 2 ** 20                                       # padded K row stride
FA_ROWS = 2050                                          # rows 2047 | 2048 straddle
FA_BELOW, FA_ABOVE = straddle(FA_KROW)                  # 2047, 2048


def _fa_rows(seed=23):
    """Two packed 16-token blocks (K and V) plus the query, built on CPU
    through the real pack path."""
    from fp8_kv import kv_block_bytes, pack_kv_block, quantize_kv_fp8
    g = torch.Generator().manual_seed(seed)
    k_nat = kv_block_bytes(FA_BT, FA_HKV, FA_D) + FA_BT * FA_HKV * 4 * (FA_KG - 1)
    v_nat = kv_block_bytes(FA_BT, FA_HKV, FA_D)
    kt = torch.randn(2 * FA_BT, FA_HKV, FA_D, generator=g) * 1.5
    vt = torch.randn(2 * FA_BT, FA_HKV, FA_D, generator=g)
    k_rows, v_rows = [], []
    for i in range(2):
        qk, sk = quantize_kv_fp8(kt[i * FA_BT:(i + 1) * FA_BT], group=FA_D // FA_KG)
        row = torch.zeros(k_nat, dtype=torch.uint8)
        pack_kv_block(qk, sk, row)
        k_rows.append(row)
        qv, sv = quantize_kv_fp8(vt[i * FA_BT:(i + 1) * FA_BT])
        row = torch.zeros(v_nat, dtype=torch.uint8)
        pack_kv_block(qv, sv, row)
        v_rows.append(row)
    q = (torch.randn(1, FA_HKV * FA_G, FA_D, generator=g)).bfloat16()
    return q, k_rows, v_rows, k_nat, v_nat


def _fa_modes():
    modes = [("split-f32", {"compute": "f32"}),
             ("packed-f32", {"compute": "f32", "pack_heads": True})]
    if torch.cuda.get_device_capability() >= (8, 9):
        modes += [("f8dot", {"compute": "fp8"}),
                  ("pf8", {"compute": "fp8", "pack_heads": True})]
    return modes


def case_fp8_attn(big: bool):
    """The reader's block-table row times ``k_row_bytes``: the sequence's two
    blocks sit at the rows whose byte offsets straddle 2^31 (K pool padded
    to a 1 MiB row). Instrument: the same kernel over a SMALL pool holding
    the same two rows at indices 0 and 1 must be bitwise-equal (same
    config, same block-table width, so the same n_split and reduction
    order) -- that pins the offset independently of #319's f32 tolerance;
    the fp8 modes are also checked against ``paged_attn_ref``."""
    from fp8_paged_attn import fp8_paged_decode_attention, paged_attn_ref
    q, k_rows, v_rows, k_nat, v_nat = _fa_rows()
    rows = (FA_BELOW, FA_ABOVE) if big else (2, 3)
    n_rows = FA_ROWS if big else 8
    if big:
        assert FA_ABOVE * FA_KROW >= BOUNDARY > FA_BELOW * FA_KROW
    k_pool = torch.zeros(n_rows * FA_KROW, dtype=torch.uint8, device=DEV)
    v_pool = torch.zeros(n_rows * v_nat, dtype=torch.uint8, device=DEV)
    for i, r in enumerate(rows):
        k_pool[r * FA_KROW:r * FA_KROW + k_nat].copy_(k_rows[i])
        v_pool[r * v_nat:(r + 1) * v_nat].copy_(v_rows[i])
    tab = torch.tensor([list(rows)], dtype=torch.int32, device=DEV)
    # the small control pool: identical rows at 0 and 1, natural strides
    ks = torch.cat(k_rows).to(DEV)
    vs = torch.cat(v_rows).to(DEV)
    tab_s = torch.tensor([[0, 1]], dtype=torch.int32, device=DEV)
    lens = torch.tensor([2 * FA_BT], dtype=torch.int32, device=DEV)
    qd = q.to(DEV)
    kw = dict(n_kv_heads=FA_HKV, head_dim=FA_D, k_groups=FA_KG)
    want = paged_attn_ref(q, ks.cpu(), vs.cpu(), tab_s.cpu(), lens.cpu(),
                          n_kv_heads=FA_HKV, head_dim=FA_D, k_groups=FA_KG).float()
    for label, mkw in _fa_modes():
        got = fp8_paged_decode_attention(qd, k_pool, v_pool, tab, lens,
                                         k_row_bytes=FA_KROW, **kw, **mkw)
        torch.cuda.synchronize()
        ctl = fp8_paged_decode_attention(qd, ks, vs, tab_s, lens, **kw, **mkw)
        torch.cuda.synchronize()
        assert torch.equal(got, ctl), \
            f"fp8-attn {label}: rows {rows} differ from the same rows at 0/1 " \
            f"(max |d| {(got.float() - ctl.float()).abs().max().item():.3e})"
        if mkw["compute"] == "fp8":
            torch.testing.assert_close(got.cpu().float(), want, rtol=1.5e-1, atol=1.5e-1)
        else:
            assert torch.isfinite(got.float()).all(), f"fp8-attn {label}: non-finite"


CASES = {
    "nf4": case_nf4,
    "nf4_dotpad": case_nf4_dotpad,
    "mxfp4": case_mxfp4,
    "int4_b32": case_int4_b32,
    "fp8_attn": case_fp8_attn,
}


# ------------------------------------------------------------------ tests --
@pytest.mark.parametrize("case", sorted(CASES))
def test_below_boundary_control(case):
    """Positive control for the instrument: the identical checks on a stack
    far below the boundary, in-process (it cannot fault)."""
    pytest.importorskip("triton")
    CASES[case](big=False)
    torch.cuda.empty_cache()


@pytest.mark.parametrize("case", sorted(CASES))
def test_above_boundary_in_own_process(case):
    """The boundary: experts (or pool rows) whose base offsets sit just below
    and just above 2^31, compared with the references, in a fresh process
    per case so a fault cannot poison the next case's context."""
    pytest.importorskip("triton")
    if not _mem_ok():
        pytest.skip(f"needs >= {MIN_FREE >> 30} GiB free device memory for a 2.3 GiB stack")
    r = subprocess.run([sys.executable, os.path.abspath(__file__), case],
                       capture_output=True, text=True, cwd=_HERE, timeout=1800)
    assert r.returncode == 0, (
        f"case {case!r} failed above the boundary (exit {r.returncode})\n"
        f"--- stdout ---\n{r.stdout[-4000:]}\n--- stderr ---\n{r.stderr[-6000:]}")
    assert f"OK {case}" in r.stdout, r.stdout[-2000:]


if __name__ == "__main__":
    name = sys.argv[1]
    CASES[name](big=True)
    torch.cuda.synchronize()
    print(f"OK {name}")
