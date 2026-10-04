"""GNF4_TRAIN_GEMM=grouped_mm (nf4_route.py): the dequant kernel is bit-equal to dequant_ref; the route's forward and dgrad match the
fused kernels within bf16 noise through FusedGroupedNf4 (torch._grouped_mm stubbed by a per-group loop where the card has none);
off sm_90 the real route refuses with the reason; the default is the fused route."""
import pytest
import torch

import nf4_grouped as NG
import nf4_qlora
import nf4_route

CUDA = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA, reason="the route's kernels are CUDA-only")


def _stack(E, N, K, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    B = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, device="cuda", generator=g)
    am = torch.rand(E, N, K // 64, device="cuda", generator=g) * 0.05 + 0.01
    return B, am


@pytest.mark.parametrize("N,K", [(1536, 2048), (2048, 768), (96, 128), (130, 192)])
def test_dequant_groups_is_bit_equal_to_dequant_ref(N, K):
    E = 8
    B, am = _stack(E, N, K, seed=N + K)
    eids = [6, 1, 3]
    out = nf4_route.dequant_groups(B, am, torch.tensor(eids, dtype=torch.int32, device="cuda"), N, K)
    assert out.shape == (3, N, K) and out.dtype == torch.bfloat16
    for g, e in enumerate(eids):
        assert torch.equal(out[g], NG.dequant_ref(B[e], am[e], N, K).to(torch.bfloat16)), (g, e)


def _loop_grouped_mm(a, b, offs=None, **kw):
    """torch._grouped_mm's 2D x 3D semantics: rows [offs[g-1], offs[g]) of `a` times b[g]."""
    out, start = [], 0
    for g, end in enumerate(offs.tolist()):
        out.append(a[start:end].float() @ b[g].float()); start = end
    return torch.cat(out).to(torch.bfloat16)


def _run(route, monkeypatch, sizes, eids, N, K, seed=0):
    monkeypatch.setenv("GNF4_TRAIN_GEMM", route)
    B, am = _stack(8, N, K, seed=seed)
    g = torch.Generator(device="cuda").manual_seed(seed + 1)
    a = (torch.randn(sum(sizes), K, device="cuda", generator=g) * 0.5).to(torch.bfloat16).requires_grad_(True)
    out = nf4_qlora.gemm_4bit_grouped_train(a, B, am, sizes, eids)
    go = (torch.randn(out.shape, device="cuda", generator=g) * 0.5).to(torch.bfloat16)
    out.backward(go)
    return out.detach().float(), a.grad.float()


@pytest.mark.parametrize("N,K", [(1536, 2048), (2048, 768)])
@pytest.mark.parametrize("sizes,eids", [([40, 3, 17, 9], [0, 2, 5, 7]), ([64], [4]), ([1, 1, 120, 2, 5], [1, 2, 3, 6, 7])])
def test_route_matches_the_fused_kernels(monkeypatch, N, K, sizes, eids):
    monkeypatch.setattr(nf4_route, "_refuse_unless_supported", lambda dev: None)
    monkeypatch.setattr(torch, "_grouped_mm", _loop_grouped_mm, raising=False)
    nf4_route.ROUTE_STATS.update(fwd=0, dgrad=0)
    fo, fg = _run("fused", monkeypatch, sizes, eids, N, K)
    assert nf4_route.ROUTE_STATS == {"fwd": 0, "dgrad": 0}
    ro, rg = _run("grouped_mm", monkeypatch, sizes, eids, N, K)
    assert nf4_route.ROUTE_STATS == {"fwd": 1, "dgrad": 1}
    for name, x, y in (("out", fo, ro), ("grad_a", fg, rg)):
        rel = ((x - y).norm() / x.norm()).item()
        assert rel < 5e-3, (name, rel)


def test_refuses_off_sm90(monkeypatch):
    if torch.cuda.get_device_capability() == (9, 0):
        pytest.skip("this card runs the route")
    monkeypatch.setenv("GNF4_TRAIN_GEMM", "grouped_mm")
    B, am = _stack(4, 96, 128)
    a = torch.randn(10, 128, device="cuda").to(torch.bfloat16)
    with pytest.raises(RuntimeError, match="compute capability 9.0|torch._grouped_mm"):
        nf4_qlora.gemm_4bit_grouped_train(a, B, am, [10], [2])


def test_route_env(monkeypatch):
    monkeypatch.delenv("GNF4_TRAIN_GEMM", raising=False)
    assert nf4_route.train_gemm_route() == "fused"
    monkeypatch.setenv("GNF4_TRAIN_GEMM", "grouped_mm")
    assert nf4_route.train_gemm_route() == "grouped_mm"
    monkeypatch.setenv("GNF4_TRAIN_GEMM", "cublas")
    with pytest.raises(ValueError):
        nf4_route.train_gemm_route()
