# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""K33 (kernel/PREREG-k33-nf4-decode-gemv-bw.md): ``_gemv_nf4_bw`` compiled, on a CUDA card. Correctness only -- no
timing here (K33's microbench owns speed, on the registered card; an RTX A2000 run is parity evidence, never speed).

1. **prmt32 is the tree, bitwise**, at every plan K33 may select and at the families' expert shapes (Qwen3-30B-A3B,
   Granite-3.1-3b-a800m, OLMoE-1B-7B): the same fp32 codebook values enter the same reduction.
2. **One-hot readback on the card**, both decodes: activation e_k reads ``dequant_ref`` column k rounded to bf16 by the
   compiled store, for every k of four blocks (every nibble position of every word).
3. **The tolerance contract** against the certified scalar route (``GNF4_GEMV_DOTPAD=0``): the error against the fp32
   reference no more than 5 % above the scalar route's, at the family shapes, eight rows each.
4. **The compiled kernel is what it claims**: the prmt32 build's PTX carries ``prmt.b32`` and ``lop3.b32``; the tree
   build's does not; spills and vector-load use are reported.
5. **PDL** (sm_90+ only; inert below): the preamble's ``griddepcontrol.wait`` precedes the first global load in the PTX,
   and PDL on is bitwise PDL off.
"""
import os

import pytest
import torch

if os.environ.get("TRITON_INTERPRET") == "1":
    pytest.skip("the compiled checks; the interpreter contract is test_nf4_gemv_bw_interp.py's", allow_module_level=True)

pytest.importorskip("triton", reason="the compiled kernel needs triton")
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="the compiled kernel needs a CUDA device")

import nf4_grouped  # noqa: E402
from nf4_grouped import dequant_ref, gemm_4bit_grouped  # noqa: E402
from nf4_pack_ref import make_stack  # noqa: E402

DEV = "cuda"
FAMILY_SHAPES = [(1536, 2048), (2048, 768), (1024, 1536), (1536, 512), (2048, 2048), (2048, 1024)]
PLANS = [(16, 256, 4, 1), (32, 256, 4, 1), (16, 512, 8, 1), (16, 1024, 4, 1), (16, 128, 2, 2), (32, 512, 4, 2)]


def _run(monkeypatch, B, A, acts, ids, *, decode=None, bw="1", plan=None, dotpad=None):
    monkeypatch.setenv("GNF4_GEMV_BW", bw)
    if decode is None:
        monkeypatch.delenv("GNF4_GEMV_BW_DECODE", raising=False)
    else:
        monkeypatch.setenv("GNF4_GEMV_BW_DECODE", decode)
    if dotpad is not None:
        monkeypatch.setenv("GNF4_GEMV_DOTPAD", dotpad)
    out = gemm_4bit_grouped(acts, B, A, [1] * len(ids), torch.tensor(ids, dtype=torch.int32, device=DEV), bw_config=plan)
    torch.cuda.synchronize()
    return out


def _stack(N, K, E=4, seed=0):
    B, A = make_stack(E, N, K, seed=seed, device=DEV)
    return B, A


@needs_cuda
@pytest.mark.parametrize("plan", PLANS)
@pytest.mark.parametrize("N,K", FAMILY_SHAPES[:2])
def test_prmt32_is_bitwise_the_tree(monkeypatch, plan, N, K):
    B, A = _stack(N, K)
    acts = torch.randn(8, K, dtype=torch.bfloat16, device=DEV, generator=torch.Generator(DEV).manual_seed(1))
    ids = [3, 0, 2, 2, 1, 0, 3, 1]
    nf4_grouped.reset_dispatch_counts()
    tree = _run(monkeypatch, B, A, acts, ids, decode="tree", plan=plan)
    prmt = _run(monkeypatch, B, A, acts, ids, decode="prmt32", plan=plan)
    assert torch.equal(tree, prmt)
    tally = nf4_grouped.dispatch_counts()
    assert tally["bw_tree"] == 1 and tally["bw_prmt32"] == 1 and tally["dotpad"] == tally["scalar"] == 0, tally


@needs_cuda
@pytest.mark.parametrize("N,K", FAMILY_SHAPES[2:])
def test_prmt32_is_bitwise_the_tree_at_the_family_shapes(monkeypatch, N, K):
    B, A = _stack(N, K, seed=2)
    acts = torch.randn(8, K, dtype=torch.bfloat16, device=DEV, generator=torch.Generator(DEV).manual_seed(3))
    ids = [1, 2, 3, 0, 0, 1, 2, 3]
    assert torch.equal(_run(monkeypatch, B, A, acts, ids, decode="tree"),
                       _run(monkeypatch, B, A, acts, ids, decode="prmt32"))


@needs_cuda
@pytest.mark.parametrize("decode", ["tree", "prmt32"])
def test_one_hot_readback_on_the_card(monkeypatch, decode):
    N, K = 64, 256
    B, A = _stack(N, K, E=1, seed=11)
    acts = torch.eye(K, dtype=torch.bfloat16, device=DEV)                 # row k = e_k
    out = _run(monkeypatch, B, A, acts, [0] * K, decode=decode)
    w = dequant_ref(B[0], A[0], N, K)
    assert torch.equal(out, w.t().to(torch.bfloat16))


@needs_cuda
@pytest.mark.parametrize("N,K", FAMILY_SHAPES)
def test_the_tolerance_contract_against_the_scalar_route(monkeypatch, N, K):
    B, A = _stack(N, K, seed=4)
    acts = torch.randn(8, K, dtype=torch.bfloat16, device=DEV, generator=torch.Generator(DEV).manual_seed(5))
    ids = [0, 1, 2, 3, 3, 2, 1, 0]
    ref = torch.stack([dequant_ref(B[e], A[e], N, K).float() @ acts[r].float() for r, e in enumerate(ids)])
    scale = ref.abs().max().clamp_min(1e-6)
    bw = _run(monkeypatch, B, A, acts, ids)
    scalar = _run(monkeypatch, B, A, acts, ids, bw="0", dotpad="0")
    e_bw = ((bw.float() - ref).abs() / scale).max().item()
    e_sc = ((scalar.float() - ref).abs() / scale).max().item()
    assert e_bw <= max(1.05 * e_sc, 2**-9), (e_bw, e_sc)


def _compile(monkeypatch, decode, pdl=False, N=256, K=512):
    B, A = _stack(N, K, E=1)
    acts = torch.randn(1, K, dtype=torch.bfloat16, device=DEV)
    bw = B.view(torch.int32)
    out = torch.empty(1, N, dtype=torch.bfloat16, device=DEV)
    eids = torch.zeros(1, dtype=torch.int32, device=DEV)
    kw = {"PDL": True, "launch_pdl": True} if pdl else {}
    k = nf4_grouped._gemv_nf4_bw[(1, N // 16, 1)](
        acts, bw, A, out, nf4_grouped._lut(DEV), nf4_grouped._bw_tables(DEV), eids, K, N, 1, K // 64,
        bw.stride(0), bw.stride(1), A.stride(0), A.stride(1), BLOCK_N=16, KC=256,
        DECODE=nf4_grouped._BW_DECODES[decode], SPLITK=False, num_warps=4, num_stages=2, **kw)
    torch.cuda.synchronize()
    return k


@needs_cuda
def test_the_compiled_kernel_is_what_it_claims(monkeypatch):
    kp = _compile(monkeypatch, "prmt32")
    kt = _compile(monkeypatch, "tree")
    ptx_p, ptx_t = kp.asm["ptx"], kt.asm["ptx"]
    assert "prmt.b32" in ptx_p and "lop3.b32" in ptx_p
    assert "prmt.b32" not in ptx_t
    print(f"K33 PTX: prmt32 v4 loads={'ld.global.v4' in ptx_p} spills={getattr(kp, 'n_spills', None)} "
          f"regs={getattr(kp, 'n_regs', None)}; tree spills={getattr(kt, 'n_spills', None)} "
          f"regs={getattr(kt, 'n_regs', None)}")


def _sm90():
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (9, 0)


@needs_cuda
@pytest.mark.skipif(not _sm90(), reason="griddepcontrol needs sm_90+ (the switch is inert below)")
def test_the_pdl_preamble_precedes_every_global_load(monkeypatch):
    ptx = _compile(monkeypatch, "prmt32", pdl=True).asm["ptx"]
    assert ptx.index("griddepcontrol.wait") < ptx.index("ld.global")


@needs_cuda
@pytest.mark.skipif(not _sm90(), reason="griddepcontrol needs sm_90+ (the switch is inert below)")
def test_pdl_on_is_bitwise_pdl_off(monkeypatch):
    import int4_b32
    N, K = 1536, 2048
    B, A = _stack(N, K)
    acts = torch.randn(8, K, dtype=torch.bfloat16, device=DEV)
    ids = list(range(4)) * 2
    outs = []
    for v in ("0", "1"):
        monkeypatch.setenv("GNF4_PDL", v)
        int4_b32.pdl_refresh()
        outs.append(_run(monkeypatch, B, A, acts, ids))
    monkeypatch.delenv("GNF4_PDL")
    int4_b32.pdl_refresh()
    assert torch.equal(outs[0], outs[1])


@needs_cuda
@pytest.mark.parametrize("decode", ["tree", "prmt32"])
@pytest.mark.parametrize("plan", [None, (16, 256, 4, 2)])
@pytest.mark.parametrize("N,K", FAMILY_SHAPES[:2])
def test_gather_div_is_bitwise_the_expanded_rows_on_the_card(monkeypatch, decode, plan, N, K):
    """``gather_div=8`` on the token rows is bitwise the call on their (token, slot) expansion, compiled, at Qwen3's
    gate_up and down shapes, both decodes, one pass and split-K (experts4bit-qlora#1313, P127 item b2). The token
    rows head a larger buffer, so a kernel reading row g rather than g // 8 reads wrong values in bounds."""
    B, A = _stack(N, K, E=8, seed=13)
    tokens = torch.randn(16, K, device=DEV, dtype=torch.bfloat16)[:2]
    ids = [7, 0, 3, 5, 1, 1, 6, 2, 4, 0, 2, 7, 5, 3, 6, 1]
    rows = tokens.index_select(0, torch.arange(16, device=DEV) // 8)
    want = _run(monkeypatch, B, A, rows, ids, decode=decode, plan=plan)
    got = gemm_4bit_grouped(tokens, B, A, [1] * 16, torch.tensor(ids, dtype=torch.int32, device=DEV), bw_config=plan,
                            gather_div=8)
    torch.cuda.synchronize()
    assert torch.equal(got, want)
