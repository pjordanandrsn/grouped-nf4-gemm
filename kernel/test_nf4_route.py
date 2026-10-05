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


@pytest.mark.parametrize("N,K", [(1536, 2048), (2048, 768), (96, 128), (130, 192), (17, 640), (33, 64)])
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
    nf4_route.ROUTE_STATS.update(fwd=0, dgrad=0, dense_fwd=0, dense_dgrad=0, decoded_fwd=0, decoded_dgrad=0)
    fo, fg = _run("fused", monkeypatch, sizes, eids, N, K)
    assert nf4_route.ROUTE_STATS == {"fwd": 0, "dgrad": 0, "dense_fwd": 0, "dense_dgrad": 0, "decoded_fwd": 0, "decoded_dgrad": 0}
    ro, rg = _run("grouped_mm", monkeypatch, sizes, eids, N, K)
    assert nf4_route.ROUTE_STATS == {"fwd": 1, "dgrad": 1, "dense_fwd": 0, "dense_dgrad": 0, "decoded_fwd": 0, "decoded_dgrad": 0}
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
    sm90 = hasattr(torch, "_grouped_mm") and torch.cuda.get_device_capability() == (9, 0)
    for unset in (None, "auto", " AUTO "):
        if unset is None:
            monkeypatch.delenv("GNF4_TRAIN_GEMM", raising=False)
        else:
            monkeypatch.setenv("GNF4_TRAIN_GEMM", unset)
        assert nf4_route.train_gemm_route() == ("grouped_mm" if sm90 else "fused")
        assert nf4_route.train_gemm_route(torch.device("cpu")) == "fused"
    monkeypatch.setenv("GNF4_TRAIN_GEMM", "fused")
    assert nf4_route.train_gemm_route() == "fused"
    monkeypatch.setenv("GNF4_TRAIN_GEMM", "grouped_mm")
    assert nf4_route.train_gemm_route() == "grouped_mm"
    monkeypatch.setenv("GNF4_TRAIN_GEMM", "dense")
    assert nf4_route.train_gemm_route() == "dense"
    # auto off sm_90: dense for a call with 1..DENSE_AUTO_MAX_GROUPS present groups, fused above (or without a count); sm_90 stays grouped_mm
    monkeypatch.delenv("GNF4_TRAIN_GEMM", raising=False)
    cap = nf4_route.DENSE_AUTO_MAX_GROUPS
    if not sm90:
        assert nf4_route.train_gemm_route(None, 8) == "dense" and nf4_route.train_gemm_route(None, cap) == "dense"
        assert nf4_route.train_gemm_route(None, cap + 1) == "fused" and nf4_route.train_gemm_route(None, 128) == "fused"
        assert nf4_route.train_gemm_route(None, 0) == "fused" and nf4_route.train_gemm_route() == "fused"
    else:
        assert nf4_route.train_gemm_route(None, 8) == "grouped_mm"
    assert nf4_route.train_gemm_route(torch.device("cpu"), 8) == "fused"
    monkeypatch.setenv("GNF4_TRAIN_GEMM", "fused")
    assert nf4_route.train_gemm_route(None, 8) == "fused"
    monkeypatch.setenv("GNF4_TRAIN_GEMM", "cublas")
    with pytest.raises(ValueError):
        nf4_route.train_gemm_route()


@pytest.mark.parametrize("N,K", [(1536, 2048), (2048, 768), (130, 192)])
@pytest.mark.parametrize("sizes,eids", [([40, 3, 17, 9], [0, 2, 5, 7]), ([64], [4]), ([1, 0, 120, 2, 5], [1, 2, 3, 6, 7])])
def test_dense_route_matches_the_fused_kernels_on_any_card(monkeypatch, N, K, sizes, eids):
    """GNF4_TRAIN_GEMM=dense: the per-expert dequant + torch.mm loop gives the fused kernels' forward and dgrad within bf16 noise,
    an empty group included, and counts itself (never the grouped_mm route's counters)."""
    nf4_route.ROUTE_STATS.update(fwd=0, dgrad=0, dense_fwd=0, dense_dgrad=0, decoded_fwd=0, decoded_dgrad=0)
    fo, fg = _run("fused", monkeypatch, sizes, eids, N, K)
    do, dg = _run("dense", monkeypatch, sizes, eids, N, K)
    assert nf4_route.ROUTE_STATS == {"fwd": 0, "dgrad": 0, "dense_fwd": 1, "dense_dgrad": 1, "decoded_fwd": 0, "decoded_dgrad": 0}
    for name, x, y in (("out", fo, do), ("grad_a", fg, dg)):
        rel = ((x - y).norm() / x.norm()).item()
        assert rel < 5e-3, (name, rel)


def test_dense_route_takes_device_sizes_and_ids():
    """The loop reads sizes once on the host; device-side sizes and ids give the same output as host lists."""
    B, am = _stack(8, 256, 512, seed=3)
    sizes, eids = [5, 0, 33, 7], [7, 1, 2, 4]
    a = (torch.randn(sum(sizes), 512, device="cuda") * 0.5).to(torch.bfloat16)
    host = nf4_route.dense_forward(a, B, am, sizes, eids)
    dev = nf4_route.dense_forward(a, B, am, torch.tensor(sizes, device="cuda"), torch.tensor(eids, device="cuda", dtype=torch.int32))
    assert torch.equal(host, dev)
    want = torch.cat([a[r0:r0 + n].float() @ NG.dequant_ref(B[e], am[e], 256, 512).to(torch.bfloat16).float().t()
                      for r0, n, e in zip([0, 5, 5, 38], sizes, eids) if n]).to(torch.bfloat16)
    rel = ((host.float() - want.float()).norm() / want.float().norm()).item()
    assert rel < 5e-3, rel



def test_auto_takes_dense_for_few_groups_off_sm90(monkeypatch):
    """auto, per call, through FusedGroupedNf4: a call with <= DENSE_AUTO_MAX_GROUPS present groups runs the dense route off sm_90 (both
    forward and dgrad), and a call with more stays on the fused kernels; on sm_90 auto is the grouped_mm route either way."""
    if torch.cuda.get_device_capability() == (9, 0):
        pytest.skip("auto is the grouped_mm route on sm_90")
    nf4_route.ROUTE_STATS.update(fwd=0, dgrad=0, dense_fwd=0, dense_dgrad=0)
    few = ([40, 3, 17, 9], [0, 2, 5, 7])
    _run("auto", monkeypatch, few[0], few[1], 256, 512)
    assert nf4_route.ROUTE_STATS["dense_fwd"] == 1 and nf4_route.ROUTE_STATS["dense_dgrad"] == 1
    many = [2] * (nf4_route.DENSE_AUTO_MAX_GROUPS + 1)
    B_E = len(many)
    monkeypatch.setenv("GNF4_TRAIN_GEMM", "auto")
    B, am = _stack(B_E, 256, 512)
    a = (torch.randn(sum(many), 512, device="cuda") * 0.5).to(torch.bfloat16).requires_grad_(True)
    out = nf4_qlora.gemm_4bit_grouped_train(a, B, am, many, list(range(B_E)))
    out.float().sum().backward()
    assert nf4_route.ROUTE_STATS["dense_fwd"] == 1 and nf4_route.ROUTE_STATS["dense_dgrad"] == 1, nf4_route.ROUTE_STATS


# ------------------------------------------------------------------------------------------- GNF4_TRAIN_GEMM=decoded (opt-in)
GATE_X = 2.0       # experts4bit-qlora RD1's correctness gate: rel. error against the fp32 reference <= 2x dense's, every call


@pytest.mark.parametrize("N,K", [(1536, 2048), (2048, 768), (130, 192)])
@pytest.mark.parametrize("sizes,eids", [([40, 3, 17, 9], [0, 2, 5, 7]), ([64], [4]), ([1, 0, 120, 2, 5], [1, 2, 3, 6, 7])])
def test_decoded_route_matches_the_fused_kernels_on_any_card(monkeypatch, N, K, sizes, eids):
    """GNF4_TRAIN_GEMM=decoded through FusedGroupedNf4: the fused kernels' forward and dgrad within bf16 noise, an empty group
    included; it counts itself, and its backward is the decoded dgrad (never the loop, never another route's)."""
    monkeypatch.delenv("GNF4_DECODED_MAX_BYTES", raising=False)
    nf4_route.ROUTE_STATS.update(fwd=0, dgrad=0, dense_fwd=0, dense_dgrad=0, decoded_fwd=0, decoded_dgrad=0)
    fo, fg = _run("fused", monkeypatch, sizes, eids, N, K)
    before = nf4_qlora.DGRAD_STATS["decoded"]
    do, dg = _run("decoded", monkeypatch, sizes, eids, N, K)
    assert nf4_route.ROUTE_STATS == {"fwd": 0, "dgrad": 0, "dense_fwd": 0, "dense_dgrad": 0, "decoded_fwd": 1, "decoded_dgrad": 1}
    assert nf4_qlora.DGRAD_STATS["decoded"] == before + 1
    for name, x, y in (("out", fo, do), ("grad_a", fg, dg)):
        rel = ((x - y).norm() / x.norm()).item()
        assert rel < 5e-3, (name, rel)


def _ref32(x, B, am, sizes, eids, N, K, mode):
    out = torch.zeros(x.shape[0], N if mode == "fwd" else K, device="cuda", dtype=torch.float32)
    r0 = 0
    for e, n in zip(eids, sizes):
        w = NG.dequant_ref(B[e], am[e], N, K).float()
        out[r0:r0 + n] = x[r0:r0 + n].float() @ (w.t() if mode == "fwd" else w)
        r0 += n
    return out


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


# expert shapes and router-like ragged groupings from RD1's grid (OLMoE / Granite-H / Qwen3 down, Nemotron-H's non-gated up),
# with few-row, one-row and empty groups, and both of the route's tilings (mean rows per group below and above the split)
GATE_CASES = [
    (2048, 2048, [70, 3, 1, 0, 120, 33, 9, 64]),
    (1024, 1536, [12, 40, 5, 2, 30, 17, 1, 8]),
    (2048, 768, [300, 150, 90, 260, 1, 0, 180, 40]),
    (1856, 2688, [9, 1, 22, 4, 16, 7, 0, 3]),
]


@pytest.mark.parametrize("mode", ["fwd", "dgrad"])
@pytest.mark.parametrize("N,K,sizes", GATE_CASES)
def test_decoded_route_passes_the_rd1_gate_against_the_dense_route(monkeypatch, mode, N, K, sizes):
    """The compiled route against an fp32 reference: at most 2x the error of the real `dense` route (cuBLAS, bf16 operands)
    on every call; a call capped to two groups per chunk is bit-identical to the uncapped call; device-form sizes and ids give
    the list form's output."""
    monkeypatch.delenv("GNF4_DECODED_MAX_BYTES", raising=False)
    eids = list(range(len(sizes)))
    B, am = _stack(len(sizes), N, K, seed=N + K)
    g = torch.Generator(device="cuda").manual_seed(N * K)
    x = (torch.randn(sum(sizes), K if mode == "fwd" else N, device="cuda", generator=g) * 0.5).to(torch.bfloat16)
    route = nf4_route.decoded_forward if mode == "fwd" else nf4_route.decoded_dgrad
    dense = nf4_route.dense_forward if mode == "fwd" else nf4_route.dense_dgrad
    y = route(x, B, am, sizes, eids)
    ref = _ref32(x, B, am, sizes, eids, N, K, mode)
    err, dense_err = _rel(y, ref), _rel(dense(x, B, am, sizes, eids), ref)
    assert err <= GATE_X * dense_err, (mode, N, K, err, dense_err)
    monkeypatch.setenv("GNF4_DECODED_MAX_BYTES", str(2 * N * K * 2))
    assert len(nf4_route.decoded_chunks(len(sizes), N, K, nf4_route.decoded_max_bytes())) == 4
    assert torch.equal(route(x, B, am, sizes, eids), y)
    monkeypatch.delenv("GNF4_DECODED_MAX_BYTES")
    dev = route(x, B, am, torch.tensor(sizes, device="cuda"), torch.tensor(eids, device="cuda", dtype=torch.int32))
    assert torch.equal(dev, y)


def test_the_cap_bounds_the_decode_transient(monkeypatch):
    """GNF4_DECODED_MAX_BYTES bounds what the route allocates above its inputs and output: capped to two experts per chunk, the
    peak stays within the cap (plus the chunk plan's small tensors); uncapped, all eight experts are decoded at once."""
    N, K, sizes = 2048, 2048, [40, 3, 17, 9, 64, 1, 30, 12]
    eids = list(range(len(sizes)))
    B, am = _stack(len(sizes), N, K, seed=11)
    x = (torch.randn(sum(sizes), K, device="cuda") * 0.5).to(torch.bfloat16)
    expert = N * K * 2
    out_bytes = sum(sizes) * N * 2

    def peak(cap):
        monkeypatch.setenv("GNF4_DECODED_MAX_BYTES", str(cap))
        nf4_route.decoded_forward(x, B, am, sizes, eids)              # the plan and any first-launch state, outside the reading
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        nf4_route.decoded_forward(x, B, am, sizes, eids)
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() - base

    capped = peak(2 * expert)
    assert capped <= out_bytes + 2 * expert + 2**20, (capped, out_bytes, expert)
    assert peak(64 * expert) >= out_bytes + 8 * expert

