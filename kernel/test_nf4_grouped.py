# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).

"""TOLERANCE_CONTRACT property suite for the grouped NF4 kernel (GPU).

The load-bearing assertions, in contract order: (0) our decode == bnb's
dequantize_4bit EXACTLY (pins codebook values AND nibble order — the
exhaustiveness clause); (1) census shapes x {M=1, p50, p95}; (2) adversarial
absmax; (3) expert-boundary cases; (4) P-fid + B-rel vs the fp64 reference on
every correctness run."""

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="kernel is GPU-only")
try:
    import triton  # noqa: F401
except Exception:  # pragma: no cover
    pytestmark = pytest.mark.skip(reason="triton unavailable")

from nf4_grouped import (  # noqa: E402
    BLOCKSIZE,
    dequant_ref,
    gemm_4bit_grouped,
    repack_from_bnb,
)


def make_stack(E, N, K, device="cuda", seed=0, scale=0.02):
    from bitsandbytes import functional as F

    g = torch.Generator(device="cpu").manual_seed(seed)
    w = (torch.randn(E, N, K, generator=g) * scale).to(device, torch.bfloat16)
    packed, states = [], []
    for e in range(E):
        q, st = F.quantize_4bit(w[e], blocksize=BLOCKSIZE, quant_type="nf4")
        packed.append(q)
        states.append(st)
    B, A = repack_from_bnb(packed, states, N, K)
    return B, A, packed, states


def groups_for(E, k_active, m, K, device="cuda", seed=1):
    g = torch.Generator(device="cpu").manual_seed(seed)
    ids = list(range(0, E, max(1, E // k_active)))[:k_active]
    sizes = [m] * len(ids)
    a = (torch.randn(sum(sizes), K, generator=g) * 0.5).to(device, torch.bfloat16)
    return a, sizes, ids


def err_vs_fp64(out, a_cat, sizes, ids, B, A, N, K):
    num = den = 0.0
    row = 0
    for m, e in zip(sizes, ids):
        w64 = dequant_ref(B[e], A[e], N, K).to(torch.float64)
        ref = a_cat[row : row + m].to(torch.float64) @ w64.t()
        num += (out[row : row + m].to(torch.float64) - ref).norm().item() ** 2
        den += ref.norm().item() ** 2
        row += m
    return (num**0.5) / max(den**0.5, 1e-30)


def dequant_path_err(a_cat, sizes, ids, packed, states, B, A, N, K):
    """The comparator: bnb dequant to bf16 + bf16 mm, scored vs the same fp64 ref."""
    from bitsandbytes import functional as F

    num = den = 0.0
    row = 0
    for m, e in zip(sizes, ids):
        wb = F.dequantize_4bit(packed[e], states[e])
        out = a_cat[row : row + m] @ wb.t()
        w64 = dequant_ref(B[e], A[e], N, K).to(torch.float64)
        ref = a_cat[row : row + m].to(torch.float64) @ w64.t()
        num += (out.to(torch.float64) - ref).norm().item() ** 2
        den += ref.norm().item() ** 2
        row += m
    return (num**0.5) / max(den**0.5, 1e-30)


@cuda
class TestDecodeExactness:
    def test_decode_matches_bnb_exactly(self):
        """Values + nibble order, all 16 codes in both positions, real quantized
        data — torch.equal, not allclose."""
        from bitsandbytes import functional as F

        B, A, packed, states = make_stack(2, 128, 128)
        for e in range(2):
            ours = dequant_ref(B[e], A[e], 128, 128).to(torch.bfloat16)
            theirs = F.dequantize_4bit(packed[e], states[e])
            assert torch.equal(ours, theirs)

    def test_all_codes_both_nibble_positions(self):
        """Craft packed bytes covering every (code, position) pair; kernel output
        via one-hot activations must equal the reference decode exactly."""
        N, K, E = 16, 64, 1
        dev = "cuda"
        B = torch.empty(E, N, K // 2, dtype=torch.uint8, device=dev)
        for c in range(16):
            B[0, c, :] = (c << 4) | (15 - c)  # code c high, 15-c low, whole row
        A = torch.ones(E, N, K // BLOCKSIZE, dtype=torch.float32, device=dev)
        eye = torch.eye(K, dtype=torch.bfloat16, device=dev)  # one-hot rows
        out = gemm_4bit_grouped(eye, B, A, [K], [0])
        ref = dequant_ref(B[0], A[0], N, K)  # [N, K]
        # out[j] = W @ e_j -> out.T == W, one product per sum. tl.dot runs TF32
        # (~1e-3 rel rounding, measured 1.6e-3), which sits below bf16's 2^-8
        # resolution — so at the kernel's actual output precision the equality
        # is EXACT, and any wrong code/nibble-order still lands on a different
        # codebook value entirely.
        assert torch.equal(out.t(), ref.to(torch.bfloat16))


@cuda
class TestCensusShapes:
    SHAPES = [  # (name, N, K, E, k)
        ("olmoe_gu", 2048, 2048, 64, 8),
        ("olmoe_dn", 2048, 1024, 64, 8),
        ("qwen_gu", 1536, 2048, 128, 8),
        ("qwen_dn", 2048, 768, 128, 8),
        ("gemma_gu", 1408, 2816, 128, 8),
        ("gemma_dn", 2816, 704, 128, 8),
        ("gptoss_gu", 5760, 2880, 128, 4),
        ("gptoss_dn", 2880, 2880, 128, 4),
    ]

    @pytest.mark.parametrize("name,N,K,E,k", SHAPES)
    @pytest.mark.parametrize("m", [1, 128, 290])  # M=1 / ~p50 / ~p95
    def test_pfid_and_brel(self, name, N, K, E, k, m):
        B, A, packed, states = make_stack(E, N, K)
        a, sizes, ids = groups_for(E, k, m, K)
        out = gemm_4bit_grouped(a, B, A, sizes, ids)
        e_f = err_vs_fp64(out, a, sizes, ids, B, A, N, K)
        e_d = dequant_path_err(a, sizes, ids, packed, states, B, A, N, K)
        assert e_f <= 2.0 * e_d, f"B-rel: fused {e_f:.2e} > 2x dequant {e_d:.2e}"
        assert e_f <= 1e-2, f"B-abs: {e_f:.2e}"
        # P-fid is a median-over-shapes claim; record per-shape for the receipt
        print(
            f"PFID {name} m={m}: fused {e_f:.3e} dequant {e_d:.3e} ratio {e_f / e_d:.2f}"
        )


@cuda
class TestAdversarialAbsmax:
    @pytest.mark.parametrize("kind", ["tiny", "huge", "mixed", "denormal_adj"])
    def test_absmax_extremes(self, kind):
        N, K, E = 64, 128, 2
        dev = "cuda"
        torch.manual_seed(3)
        B = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, device=dev)
        val = {"tiny": 1e-30, "huge": 1e30, "denormal_adj": 1e-38}.get(kind)
        A = torch.full(
            (E, N, K // BLOCKSIZE), val or 1.0, dtype=torch.float32, device=dev
        )
        if kind == "mixed":
            A[:, ::2, :] = 1e30
            A[:, 1::2, :] = 1e-30
        a = (torch.randn(8, K) * 0.5).to(dev, torch.bfloat16)
        out = gemm_4bit_grouped(a, B, A, [4, 4], [0, 1])
        # reference in fp64 on the same decode; relative comparison scale-free
        e = err_vs_fp64(out, a, [4, 4], [0, 1], B, A, N, K)
        assert e <= 1e-2, f"{kind}: rel err {e:.2e}"
        assert torch.isfinite(out.to(torch.float32)).all()


@cuda
class TestBoundaries:
    def _stack(self):
        return make_stack(8, 128, 128)

    def test_single_token_groups(self):
        B, A, *_ = self._stack()
        a = torch.randn(3, 128, device="cuda", dtype=torch.bfloat16)
        out = gemm_4bit_grouped(a, B, A, [1, 1, 1], [2, 5, 7])
        assert out.shape == (3, 128)
        assert err_vs_fp64(out, a, [1, 1, 1], [2, 5, 7], B, A, 128, 128) < 1e-2

    def test_all_tokens_one_expert(self):
        B, A, *_ = self._stack()
        a = torch.randn(200, 128, device="cuda", dtype=torch.bfloat16)
        out = gemm_4bit_grouped(a, B, A, [200], [3])
        assert err_vs_fp64(out, a, [200], [3], B, A, 128, 128) < 1e-2

    def test_g_less_than_e_noncontiguous(self):
        B, A, *_ = self._stack()
        a = torch.randn(30, 128, device="cuda", dtype=torch.bfloat16)
        sizes, ids = [10, 5, 15], [6, 1, 4]  # sparse, unsorted expert ids
        out = gemm_4bit_grouped(a, B, A, sizes, ids)
        assert err_vs_fp64(out, a, sizes, ids, B, A, 128, 128) < 1e-2

    def test_k_equals_blocksize(self):
        B, A, packed, states = make_stack(2, 64, BLOCKSIZE)
        a = torch.randn(5, BLOCKSIZE, device="cuda", dtype=torch.bfloat16)
        out = gemm_4bit_grouped(a, B, A, [5], [1])
        assert err_vs_fp64(out, a, [5], [1], B, A, 64, BLOCKSIZE) < 1e-2

    def test_ragged_tail_tiles(self):
        B, A, *_ = self._stack()
        a = torch.randn(65 + 17, 128, device="cuda", dtype=torch.bfloat16)
        out = gemm_4bit_grouped(a, B, A, [65, 17], [0, 7], block_m=64)
        assert err_vs_fp64(out, a, [65, 17], [0, 7], B, A, 128, 128) < 1e-2


@cuda
class TestSplitK:
    """The split-K decode path (v3): occupancy-starved grids (top_k=1 class)
    take a (g, n-tile, k-split) grid with fp32 partials host-reduced. Same
    decode math; these tests pin exactness, fidelity ordering, plan logic,
    and agreement with the single-pass path."""

    def test_plan_thresholds(self):
        from nf4_grouped import _decode_plan

        # universal constant (v3: the SM-conditional premise didn't reproduce)
        for sm in (26, 64):
            bn, w, _ = _decode_plan(2048, 768, 8, sm)
            assert (bn, w) == (64, 2)
        # Scout-down-like starved cell splits (128 blocks / sk4 = 32 >= floor)
        bn, w, sk = _decode_plan(5120, 8192, 1, 64)
        assert (bn, w) == (64, 2) and sk == 4
        # census-class cells never split on either device class
        for N, K in ((2048, 2048), (1408, 2816), (1536, 2048), (2880, 2880)):
            for sm in (26, 64):
                *_, sk = _decode_plan(N, K, 8, sm)
                assert sk == 1, (N, K, sm)
        # per-split work floor: tiny-K cells never split even when starved
        *_, sk = _decode_plan(64, 64, 1, 64)
        assert sk == 1
        *_, sk = _decode_plan(6144, 768, 1, 64)  # Switch gu: 12 blocks
        assert sk == 1
        # floor caps the want: 128 blocks -> sk4 (32/split), not sk8 (16/split)
        *_, sk = _decode_plan(64, 8192, 1, 64)
        assert sk == 4

    def test_dispatch_floor(self):
        from nf4_grouped import decode_dispatch

        # Switch-Base cells (v3: 0.24-0.35x, 4-7x energy) route to dequant
        assert decode_dispatch(6144, 768, 1, 64) == ("dequant",)
        assert decode_dispatch(768, 3072, 1, 64) == ("dequant",)
        # granite down (3.5 MB, measured winner) stays fused
        d = decode_dispatch(1536, 512, 8, 64)
        assert d[0] == "fused" and d[1:3] == (64, 2)
        # census cells stay fused with no split
        assert decode_dispatch(2048, 2048, 8, 64) == ("fused", 64, 2, 1)
        # Scout down: fused with sk4
        assert decode_dispatch(5120, 8192, 1, 64) == ("fused", 64, 2, 4)

    def test_forced_split_matches_single_pass(self):
        B, A, packed, states = make_stack(8, 512, 2048)
        a, sizes, ids = groups_for(8, 2, 1, 2048)
        one = gemm_4bit_grouped(a, B, A, sizes, ids, split_k=1)
        for sk in (2, 4, 8):
            sp = gemm_4bit_grouped(a, B, A, sizes, ids, split_k=sk)
            # fp32 partial sums reassociate vs the single-pass K-loop; at bf16
            # output precision the results should round to (near-)identical.
            assert (sp.to(torch.float32) - one.to(torch.float32)).abs().max().item() <= 2 * 2**-8 * one.to(torch.float32).abs().max().item()

    def test_split_exactness_one_hot(self):
        # one-hot activation selects a single weight column: exact through
        # any split (only one k contributes per output).
        B, A, packed, states = make_stack(2, 128, 512)
        a = torch.zeros(1, 512, device="cuda", dtype=torch.bfloat16)
        a[0, 100] = 1.0
        ref = dequant_ref(B[1], A[1], 128, 512)[:, 100]
        out = gemm_4bit_grouped(a, B, A, [1], [1], split_k=4)
        assert torch.equal(out[0], ref.to(torch.bfloat16))

    def test_split_pfid_ordering(self):
        # Scout-down-like starved shape (scaled): fused-with-split error must
        # stay at-or-below the dequant path's vs fp64 (the P-fid claim).
        E, N, K = 4, 1280, 2048
        B, A, packed, states = make_stack(E, N, K)
        a, sizes, ids = groups_for(E, 1, 1, K)
        out = gemm_4bit_grouped(a, B, A, sizes, ids, split_k=4)
        fused = err_vs_fp64(out, a, sizes, ids, B, A, N, K)
        deq = dequant_path_err(a, sizes, ids, packed, states, B, A, N, K)
        assert fused <= deq * 1.05  # ordering, small slack for reassociation

    def test_split_span_not_divisible(self):
        # 45 absmax blocks (K=2880) over split 4 -> spans 12/12/12/9; and an
        # empty tail split (split 8 over 9 blocks) must contribute zeros.
        B, A, *_ = make_stack(2, 128, 2880)
        a, sizes, ids = groups_for(2, 1, 1, 2880)
        one = gemm_4bit_grouped(a, B, A, sizes, ids, split_k=1)
        for sk in (4, 8):
            sp = gemm_4bit_grouped(a, B, A, sizes, ids, split_k=sk)
            assert (sp.to(torch.float32) - one.to(torch.float32)).abs().max().item() <= 2 * 2**-8 * one.to(torch.float32).abs().max().item()

    def test_auto_plan_takes_split_and_is_correct(self):
        # Starved cell on the real device: the auto plan must split (on any
        # sm_86 card >=26 SM this cell wants sk>1) and stay correct vs fp64.
        from nf4_grouped import _decode_plan, _sm_count

        E, N, K = 2, 1024, 4096
        sm = _sm_count("cuda")
        *_, sk = _decode_plan(N, K, 1, sm)
        assert sk > 1
        B, A, *_ = make_stack(E, N, K)
        a, sizes, ids = groups_for(E, 1, 1, K)
        out = gemm_4bit_grouped(a, B, A, sizes, ids)  # auto plan
        assert err_vs_fp64(out, a, sizes, ids, B, A, N, K) < 1e-2
