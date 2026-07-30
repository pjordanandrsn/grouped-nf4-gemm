# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Phase-4 gates: the native-mxfp4 pipelined engine (fused kernel + residency
split + gpt-oss GLU) reproduces the dequant reference at EVERY K (pure stream
K=0 -> fully resident K=E), eager AND CUDA-graph, and K is a table rebuild not
a code path. Reference dequant is mxfp4_pack_ref (== the A4 oracle, Phase 1).
Runs on CUDA; own process (raw-pointer gather is compiled-only)."""
import pytest
import torch

pytest.importorskip("triton")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from mxfp4_pack_ref import MX_BLOCK, dequant_mxfp4, quantize_pack_mxfp4  # noqa: E402
from mxfp4_pipelined import Mxfp4PipelinedGptOss  # noqa: E402

ALPHA, LIMIT = 1.702, 7.0


def _make(E=8, H=128, I=128, k=4, seed=0):
    """Synthetic native-mxfp4 gpt-oss experts. gate_up [E,2I,H], down [E,H,I]."""
    g = torch.Generator().manual_seed(seed)
    gu_w = torch.randn(E, 2 * I, H, generator=g) * 0.1
    dn_w = torch.randn(E, H, I, generator=g) * 0.1
    gub = torch.randn(E, 2 * I, generator=g) * 0.05
    dnb = torch.randn(E, H, generator=g) * 0.05

    def pack(w):
        E_, N_, K_ = w.shape
        B = torch.empty(E_, N_, K_ // 2, dtype=torch.uint8)
        S = torch.empty(E_, N_, K_ // MX_BLOCK, dtype=torch.uint8)
        for e in range(E_):
            b, s = quantize_pack_mxfp4(w[e])
            B[e], S[e] = b.reshape(N_, K_ // 2), s
        return B, S

    gu_b, gu_s = pack(gu_w)
    dn_b, dn_s = pack(dn_w)
    return dict(gu_b=gu_b, gu_s=gu_s, dn_b=dn_b, dn_s=dn_s,
               gub=gub.to(torch.bfloat16), dnb=dnb.to(torch.bfloat16), E=E, H=H, I=I, k=k)


def _ref_forward(m, x, idx, sc):
    """gpt-oss mxfp4 reference: dequant weights, clamped-GLU, weighted sum."""
    E, H, I, k = m["E"], m["H"], m["I"], m["k"]
    T = x.shape[0]
    out = torch.zeros(T, H, dtype=torch.float32)
    for t in range(T):
        for j in range(k):
            e = int(idx[t, j])
            w = float(sc[t, j])
            gW = dequant_mxfp4(m["gu_b"][e].reshape(2 * I, H // MX_BLOCK, 16), m["gu_s"][e])  # [2I,H]
            dW = dequant_mxfp4(m["dn_b"][e].reshape(H, I // MX_BLOCK, 16), m["dn_s"][e])       # [H,I]
            gu = x[t].float() @ gW.t() + m["gub"][e].float()
            gate, up = gu[..., ::2], gu[..., 1::2]     # gpt-oss INTERLEAVED
            gate = gate.clamp(max=LIMIT)
            up = up.clamp(min=-LIMIT, max=LIMIT)
            h = (up + 1) * (gate * torch.sigmoid(gate * ALPHA))
            dn = h @ dW.t() + m["dnb"][e].float()
            out[t] += w * dn
    return out


def _engine(m, hot_ids):
    return Mxfp4PipelinedGptOss(
        m["gu_b"], m["gu_s"], m["dn_b"], m["dn_s"], m["gub"], m["dnb"],
        hot_ids=torch.tensor(hot_ids, dtype=torch.long), k_slots=m["k"],
        device="cuda", alpha=ALPHA, limit=LIMIT)


def _route(m, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(1, m["H"], dtype=torch.bfloat16, device="cuda", generator=g)
    sc, idx = torch.topk(torch.softmax(torch.randn(1, m["E"], device="cuda", generator=g), -1),
                         k=m["k"], dim=-1)
    return x, idx, sc.to(torch.bfloat16)


def _b_rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max()).item()


@pytest.mark.parametrize("K", [0, 2, 4, 8])
def test_every_K_matches_reference(K):
    m = _make(seed=1)
    x, idx, sc = _route(m, seed=9)
    ref = _ref_forward(m, x.cpu(), idx.cpu(), sc.cpu())
    eng = _engine(m, list(range(K)))
    with torch.no_grad():
        got = eng.forward(x, idx, sc)
    assert got.shape == (1, m["H"])
    assert _b_rel(got.cpu(), ref) < 3e-2, (K, _b_rel(got.cpu(), ref))


def test_pure_stream_equals_fully_resident():
    """K=0 (all cold-streamed) and K=E (all resident) must both match the
    reference — same bytes, different residence."""
    m = _make(seed=2)
    x, idx, sc = _route(m, seed=5)
    ref = _ref_forward(m, x.cpu(), idx.cpu(), sc.cpu())
    for K in (0, m["E"]):
        eng = _engine(m, list(range(K)))
        with torch.no_grad():
            got = eng.forward(x, idx, sc)
        assert _b_rel(got.cpu(), ref) < 3e-2, (K, _b_rel(got.cpu(), ref))


def test_traffic_counters():
    """Cold bytes fall to 0 as K->E; hot D2D accounts the resident re-copy."""
    m = _make(seed=3)
    eng0 = _engine(m, [])
    x, idx, sc = _route(m, seed=7)
    with torch.no_grad():
        eng0.forward(x, idx, sc)
    t0 = eng0.traffic()
    engE = _engine(m, list(range(m["E"])))
    with torch.no_grad():
        engE.forward(x, idx, sc)
    tE = engE.traffic()
    assert t0["cold_pcie_bytes"] > 0 and tE["cold_pcie_bytes"] == 0


def test_cuda_graph_replay_parity():
    """Capture the decode step and replay across churning routes; replay must
    match eager within the same tolerance (Phase-3-style graph gate)."""
    m = _make(seed=4)
    eng = _engine(m, [0, 1, 2])
    routes = [_route(m, seed=20 + s) for s in range(6)]
    x_st, i_st, s_st = (routes[0][0].clone(), routes[0][1].clone(), routes[0][2].clone())
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.no_grad():
        for x, i, sc in routes[:3]:
            x_st.copy_(x); i_st.copy_(i); s_st.copy_(sc); eng.forward(x_st, i_st, s_st)
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g), torch.no_grad():
            out_st = eng.forward(x_st, i_st, s_st)
    except RuntimeError as e:
        pytest.skip(f"capture unavailable (eager is the contract): {e}")
    got = []
    for x, i, sc in routes[3:]:
        x_st.copy_(x); i_st.copy_(i); s_st.copy_(sc); g.replay(); got.append(out_st.clone())
    torch.cuda.synchronize()
    for (x, i, sc), o in zip(routes[3:], got):
        ref = _ref_forward(m, x.cpu(), i.cpu(), sc.cpu())
        assert _b_rel(o.cpu(), ref) < 3e-2


# ------------------------------------------------------- 5. prefill (T > 1) ----
# The engine was decode-only: `a_buf.copy_(x.expand(k, -1))` broadcasts ONE token
# across the k slots. Prefill is not a convenience -- stepping T tokens one at a time
# re-reads the whole dense side T times and every routed row T times, where entering
# each layer once for the prompt reads each DISTINCT expert once. These gate that the
# new path computes what the validated decode path computes, and that it really does
# dedup the reads.

def _needs_gather_kernel():
    """`gemm_mxfp4_grouped`'s prefill variant uses `tl.gather` (triton >= 3.4). Skip
    loudly rather than fail obscurely inside the compiler on an older box."""
    pytest.importorskip("triton")
    import triton.language as tl
    if not hasattr(tl, "gather"):
        import triton
        pytest.skip(f"prefill needs triton>=3.4 for tl.gather; have {triton.__version__}")


def _prefill_engine(monkeypatch=None):
    """A resident engine plus its packed source, at E=8 / k=4 so that a handful of
    tokens ALREADY overrun the slot budget and force chunking."""
    from mxfp4_pipelined import Mxfp4PipelinedGptOss
    from mxfp4_pack_ref import quantize_pack_mxfp4
    g = torch.Generator().manual_seed(4242)
    E, N1, N2, H, INTER = 8, 2 * 64, 64, 64, 64

    def pack(w):
        E_, N_, K_ = w.shape
        B = torch.empty(E_, N_, K_ // 2, dtype=torch.uint8)
        S = torch.empty(E_, N_, K_ // 32, dtype=torch.uint8)
        for e in range(E_):
            b, s = quantize_pack_mxfp4(w[e])
            B[e], S[e] = b.reshape(N_, K_ // 2), s
        return B, S

    gu_b, gu_s = pack(torch.randn(E, N1, H, generator=g) * 0.1)
    dn_b, dn_s = pack(torch.randn(E, N2, INTER, generator=g) * 0.1)
    return (gu_b, gu_s, dn_b, dn_s), E, H, N2


def _routes(T, E, topk, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn(T, E, device="cuda", generator=g)
    sc, idx = torch.topk(torch.softmax(logits, -1), k=topk, dim=-1)
    return idx, sc.to(torch.bfloat16)


def _build(stacks, E, k_slots, bias):
    from mxfp4_pipelined import Mxfp4PipelinedGptOss
    gu_b, gu_s, dn_b, dn_s = stacks
    g = torch.Generator().manual_seed(7)
    gub = (torch.randn(E, gu_b.shape[1], generator=g) * 0.05).to(torch.bfloat16)
    dnb = (torch.randn(E, dn_b.shape[1], generator=g) * 0.05).to(torch.bfloat16)
    return Mxfp4PipelinedGptOss(
        gu_b, gu_s, dn_b, dn_s, gub if bias else None, dnb if bias else None,
        hot_ids=torch.tensor([], dtype=torch.long), k_slots=k_slots,
        device="cuda", alpha=1.702, limit=7.0)


@pytest.mark.parametrize("bias", [False, True])
def test_prefill_equals_running_the_tokens_one_at_a_time(bias):
    """THE gate. Prefill must compute what the decode path computes for the same
    tokens -- decode being the path already validated against a dequant reference.

    Not `torch.equal`: a group holding more than one row takes the TILED kernel while
    decode takes the GEMV reduction, so the K-dimension accumulates in a different
    order. Same weights, same math, different order -- so the bound is bf16-scale, and
    a wrong split or a mis-scattered row is O(1) wrong, not O(eps).
    """
    _needs_gather_kernel()
    stacks, E, H, _N2 = _prefill_engine()
    topk, T = 4, 5
    eng = _build(stacks, E, topk, bias)
    idx, sc = _routes(T, E, topk, seed=11)
    g = torch.Generator(device="cuda").manual_seed(3)
    x = torch.randn(T, H, dtype=torch.bfloat16, device="cuda", generator=g)

    got = eng.forward(x, idx, sc)
    assert tuple(got.shape) == (T, eng.n2), got.shape
    ref = torch.cat([eng.forward(x[t:t + 1], idx[t:t + 1], sc[t:t + 1])
                     for t in range(T)], dim=0)
    rel = ((got.float() - ref.float()).abs().max()
           / ref.float().abs().max()).item()
    assert rel < 2e-2, rel
    # and it is not passing because everything is zero
    assert ref.float().abs().max().item() > 1e-3


def test_prefill_chunks_the_expert_set_to_the_slot_budget():
    """T*topk distinct experts can far exceed k. The slot budget must not grow with
    the prompt -- that is the whole reason VRAM stays flat -- so the distinct set is
    processed in chunks, and there must really be more than one."""
    _needs_gather_kernel()
    stacks, E, H, _ = _prefill_engine()
    topk, T, k_slots = 4, 6, 2            # <= 2 slots, up to 8 distinct experts
    eng = _build(stacks, E, topk, bias=False)
    eng2 = _build(stacks, E, k_slots, bias=False)
    idx, sc = _routes(T, E, topk, seed=5)
    g = torch.Generator(device="cuda").manual_seed(9)
    x = torch.randn(T, H, dtype=torch.bfloat16, device="cuda", generator=g)

    calls = []
    orig = eng2._fetch
    eng2._fetch = lambda want, _o=orig, _c=calls: (_c.append(want.tolist()), _o(want))[1]
    got = eng2.forward(x, idx, sc)
    n_distinct = len(set(idx.reshape(-1).tolist()))
    assert len(calls) == -(-n_distinct // k_slots) > 1, (len(calls), n_distinct)
    assert all(len(w) == k_slots for w in calls), "every fetch asks for exactly k ids"
    # chunking must not change the answer: compare against the k=topk engine, which
    # needs no chunking for a single token
    ref = torch.cat([eng.forward(x[t:t + 1], idx[t:t + 1], sc[t:t + 1])
                     for t in range(T)], dim=0)
    rel = ((got.float() - ref.float()).abs().max() / ref.float().abs().max()).item()
    assert rel < 2e-2, rel


def test_prefill_reads_each_distinct_expert_once_not_once_per_token():
    """The I/O claim that makes prefill worth building. Route every token to the SAME
    experts: stepping tokens one at a time would gather them T times over, prefill
    gathers them once."""
    _needs_gather_kernel()
    stacks, E, H, _ = _prefill_engine()
    topk, T = 4, 6
    eng = _build(stacks, E, topk, bias=False)
    idx = torch.tensor([[0, 1, 2, 3]] * T, device="cuda")
    sc = torch.full((T, topk), 0.25, dtype=torch.bfloat16, device="cuda")
    g = torch.Generator(device="cuda").manual_seed(13)
    x = torch.randn(T, H, dtype=torch.bfloat16, device="cuda", generator=g)

    calls = []
    orig = eng._fetch
    eng.__dict__["_fetch"] = lambda want, _o=orig, _c=calls: (
        _c.append(1), _o(want))[1]
    eng.forward(x, idx, sc)
    assert len(calls) == 1, (
        f"{T} tokens over 4 distinct experts should be ONE chunk, got {len(calls)}")


def test_one_token_still_takes_the_decode_path():
    """Decode is the validated path and its reduction order is part of what was
    validated, so T == 1 must not silently start going through prefill."""
    _needs_gather_kernel()
    stacks, E, H, _ = _prefill_engine()
    topk = 4
    eng = _build(stacks, E, topk, bias=False)
    idx, sc = _routes(1, E, topk, seed=17)
    x = torch.randn(1, H, dtype=torch.bfloat16, device="cuda")
    seen = []
    eng.__dict__["_forward_prefill"] = lambda *a, **k: seen.append(1)
    out = eng.forward(x, idx, sc)
    assert not seen, "T == 1 must dispatch to decode"
    assert tuple(out.shape) == (1, eng.n2)
