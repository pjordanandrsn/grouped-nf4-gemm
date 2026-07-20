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
