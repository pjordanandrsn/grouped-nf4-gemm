"""GNF4_TRITON_PREBIND (on by default; =0 turns it off): the training GEMMs' launches without Triton's per-call binding, and the host work
around them remembered by value, give the SAME values, bit for bit.

The prebound launch (``_triton_shim.prebind``) launches the very compiled kernel Triton's own lookup returns, across odd shapes and
misaligned pointers, and takes Triton's own path on a Triton release it was not written for, under a launch hook (or, under triton
3.7, a stages hook), or with a callable grid. ``_ValueMemo`` -- behind the faster ``to_device_i32`` hit, ``_prefill_block_m_cost`` and the grouped_mm route's
``_plan`` -- only ever answers with what the full path would build: a changed list, context or element type is a miss. The bar is
``torch.equal`` against the flag off for the fused forward, the dgrad and the route's dequant.
"""
import pytest
import torch

import _triton_shim as shim
import nf4_grouped as NG
import nf4_route as NR

CUDA = torch.cuda.is_available()
SUPPORTED = shim.HAS_TRITON and shim._triton_version() in shim.PREBIND_TRITON
gpu = pytest.mark.skipif(not (CUDA and SUPPORTED), reason="needs CUDA and a Triton release the prebound path supports")


def test_on_unless_turned_off(monkeypatch):
    sentinel = object()
    monkeypatch.delenv("GNF4_TRITON_PREBIND", raising=False)
    assert shim.prebind_requested()                  # the default (experts4bit-qlora TC1 amendments 26 / 30)
    monkeypatch.setenv("GNF4_TRITON_PREBIND", "0")
    assert not shim.prebind_requested() and shim.prebind(sentinel) is sentinel
    monkeypatch.setenv("GNF4_TRITON_PREBIND", "1")
    assert shim.prebind_requested()
    assert shim.prebind(sentinel) is sentinel        # not a Triton kernel (or no Triton at all): returned as is
    assert NG._gemm_nf4_grouped_launch is NG._gemm_nf4_grouped or NG._PREBIND
    assert NR._dequant_groups_launch is NR._dequant_groups_kernel or NR._PREBIND


@pytest.mark.skipif(not shim.HAS_TRITON, reason="needs triton")
def test_unsupported_triton_version_keeps_tritons_launch(monkeypatch):
    k = NG._dgrad_nf4_grouped
    if SUPPORTED and type(k).__name__ == "JITFunction":
        assert isinstance(shim.prebind(k, force=True), shim.Prebound)
    for v in ((3, 3), (3, 5), (3, 8), (4, 0), None):
        monkeypatch.setattr(shim, "_triton_version", lambda v=v: v)
        assert shim.prebind(k, force=True) is k


def test_value_memo_answers_only_what_the_full_path_would_build():
    m = NG._ValueMemo(2)
    sizes, ids = [3, 5, 7], [0, 4, 9]
    m.put((sizes, ids), "ctx", "plan")
    assert m.get(([3, 5, 7], [0, 4, 9]), "ctx") == "plan"
    assert m.get(([3, 5, 7], [0, 4, 9]), "other stream") is None
    assert m.get(([3, 5, 8], [0, 4, 9]), "ctx") is None
    assert m.get(([3, 5, 7.0], [0, 4, 9]), "ctx") == "plan"            # 7.0 == 7: int(7.0) is what the full path uploads
    assert m.get(([3, 5, 7.5], [0, 4, 9]), "ctx") is None
    sizes[0] = 4                                                         # the snapshot is a copy: the caller's later edit is a miss
    assert m.get((sizes, ids), "ctx") is None
    m.put(([1.5, 2], [0, 1]), "ctx", "floats")                           # only all-int snapshots are stored
    assert m.get(([1.5, 2], [0, 1]), "ctx") is None
    m.put(([1], [2]), "a", 1)
    m.put(([1], [2]), "b", 2)
    assert m.get(([3, 5, 7], [0, 4, 9]), "ctx") is None                  # bounded: the oldest entry aged out


def test_block_m_cost_memo_is_value_identical(monkeypatch):
    g = torch.Generator().manual_seed(0)
    cases = [[1], [16, 17], [3, 300, 7, 64], [128] * 64] + [torch.randint(1, 900, (n,), generator=g).tolist() for n in (8, 128, 128)]
    monkeypatch.setattr(NG, "_PREBIND", False)
    want = [NG._prefill_block_m_cost(s) for s in cases] + [NG._prefill_block_m_cost(s, 40.0) for s in cases]
    monkeypatch.setattr(NG, "_PREBIND", True)
    for _ in range(2):
        got = [NG._prefill_block_m_cost(s) for s in cases] + [NG._prefill_block_m_cost(s, 40.0) for s in cases]
        assert got == want
    s = list(cases[-1])
    NG._prefill_block_m_cost(s)
    s[:] = [1] * len(s)                                                  # mutated in place after a call
    monkeypatch.setattr(NG, "_PREBIND", False)
    assert NG._prefill_block_m_cost(list(s)) == NG._block_m_cost(s, NG._TILE_D_DEFAULT)
    monkeypatch.setattr(NG, "_PREBIND", True)
    assert NG._prefill_block_m_cost(s) == NG._block_m_cost(s, NG._TILE_D_DEFAULT)


def test_capability_is_read_once_per_indexed_device(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d=None: (calls.append(d), (9, 0))[1])
    monkeypatch.setattr(NR, "_CAPABILITY", {})
    monkeypatch.setattr(NR, "_PREBIND", False)
    d0 = torch.device("cuda", 0)
    assert NR._capability(d0) == (9, 0) and NR._capability(d0) == (9, 0) and len(calls) == 2
    monkeypatch.setattr(NR, "_PREBIND", True)
    assert [NR._capability(d0) for _ in range(3)] == [(9, 0)] * 3 and len(calls) == 3
    NR._capability(torch.device("cuda"))                                 # no index: "the current device" may change, so not kept
    NR._capability(torch.device("cuda"))
    assert len(calls) == 5


def _flag(monkeypatch, on):
    monkeypatch.setattr(NG, "_PREBIND", on)
    monkeypatch.setattr(NR, "_PREBIND", on)
    for mod, name, k in ((NG, "_gemm_nf4_grouped_launch", NG._gemm_nf4_grouped), (NG, "_dgrad_nf4_grouped_launch", NG._dgrad_nf4_grouped),
                         (NR, "_dequant_groups_launch", NR._dequant_groups_kernel)):
        monkeypatch.setattr(mod, name, shim.prebind(k, force=True) if on else k)


def _stack(E, N, K, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    B = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, device="cuda", generator=g)
    am = torch.rand(E, N, K // 64, device="cuda", generator=g) * 0.05 + 0.01
    return B, am


# (N, K, sizes, expert ids, elements of offset the activations start at): odd N, K at and off 128, many and few groups
CASES = [
    (96, 128, [3, 3, 3, 3], [0, 1, 2, 3], 0),
    (130, 192, [5, 1, 7], [2, 0, 3], 1),
    (17, 640, [9, 2, 4, 1], [5, 1, 4, 0], 3),
    (1536, 2048, [40, 1, 77, 16], [6, 1, 3, 7], 0),
    (2048, 768, [64] * 8, list(range(8)), 1),
]


@gpu
@pytest.mark.parametrize("N,K,sizes,eids,skip", CASES)
def test_fused_forward_dgrad_and_dequant_are_bit_identical(monkeypatch, N, K, sizes, eids, skip):
    E = 8
    B, am = _stack(E, N, K, seed=N + K)
    T = sum(sizes)
    g = torch.Generator(device="cuda").manual_seed(T)
    a = torch.randn(T * K + skip, device="cuda", generator=g).to(torch.bfloat16)[skip:].view(T, K)   # skip > 0: misaligned
    go = torch.randn(T, N, device="cuda", generator=g).to(torch.bfloat16)
    e_dev = torch.tensor(eids, dtype=torch.int32, device="cuda")
    outs = {}
    for on in (False, True):
        _flag(monkeypatch, on)
        for rep in range(2):                                             # the first call of a key fills, the second is the hit
            outs[on, rep] = (NG.gemm_4bit_grouped(a, B, am, list(sizes), list(eids)),
                             NG.dgrad_4bit_grouped(go, B, am, list(sizes), list(eids)),
                             NR.dequant_groups(B, am, e_dev, N, K))
    for rep in range(2):
        for x, y in zip(outs[False, rep], outs[True, rep]):
            assert torch.equal(x, y)


@gpu
def test_route_plan_is_value_identical_and_never_stale(monkeypatch):
    dev = torch.device("cuda", torch.cuda.current_device())
    sizes, eids = [5, 1, 7, 2], [6, 0, 3, 1]
    _flag(monkeypatch, False)
    want = [t.clone() for t in NR._plan(sizes, eids, dev)]
    _flag(monkeypatch, True)
    for _ in range(3):
        assert all(torch.equal(x, y) for x, y in zip(NR._plan(sizes, eids, dev), want))
    sizes[1] = 9                                                         # the same list object, new values: a new plan
    got = NR._plan(sizes, eids, dev)
    assert got[1].tolist() == [5, 14, 21, 23] and got[0].tolist() == eids
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):                                        # another stream: its own upload
        other = NR._plan(sizes, eids, dev)
    torch.cuda.synchronize()
    assert other[1] is not got[1] and torch.equal(other[1], got[1])


@gpu
def test_upload_fast_path_is_value_identical(monkeypatch):
    dev = torch.device("cuda", torch.cuda.current_device())
    _flag(monkeypatch, True)
    a, b = [3, 1, 4, 1, 5], [9, 2, 6]
    first = NG.to_device_i32((a, b), dev)
    again = NG.to_device_i32((list(a), list(b)), dev)
    assert [t.tolist() for t in again] == [a, b] and all(x is y for x, y in zip(first, again))
    a[0] = 7
    assert NG.to_device_i32((a, b), dev)[0].tolist() == a
    assert NG.to_device_i32(([3, 1], torch.tensor([2, 2])), dev)[1].tolist() == [2, 2]   # not all lists: the full path


@gpu
def test_prebound_launches_tritons_own_compiled_kernel():
    from triton import knobs
    from triton.runtime.driver import driver
    p = shim.Prebound(NR._dequant_groups_kernel)
    for N, K in ((96, 128), (17, 640), (1536, 2048)):
        B, am = _stack(4, N, K, seed=N)
        for G in (1, 3):
            e = torch.arange(G, dtype=torch.int32, device="cuda")
            out = torch.empty(G, N, K, dtype=torch.bfloat16, device="cuda")
            args = (B, am, e, NG._lut(B.device), out, N, K // 2, B.stride(0), B.stride(1), am.stride(0), am.stride(1), out.stride(0),
                    out.stride(1))
            kw = dict(BLOCK_N=16, BLOCK_KB=256, QB=32, num_warps=8)
            grid = (G, -(-N // 16), -(-(K // 2) // 256))
            p[grid](*args, **kw)
            k = p[grid](*args, **kw)
            kw2 = dict(kw, debug=kw.get("debug", NR._dequant_groups_kernel.debug) or knobs.runtime.debug)
            if shim._triton_version() >= (3, 6):
                kw2["instrumentation_mode"] = knobs.compilation.instrumentation_mode
            caches = NR._dequant_groups_kernel.device_caches[driver.active.get_current_device()]
            _, spec, opts = caches[-1](*args, **kw2)
            if len(caches) == 5:
                from triton.runtime.jit import compute_cache_key
                theirs = caches[0].get(compute_cache_key(caches[1], spec, opts))
            else:
                theirs = caches[0].get(str(spec) + str(opts))
            assert k is not None and k is theirs, (N, K, G)


@gpu
def test_launch_hook_and_callable_grid_take_tritons_path(monkeypatch):
    from triton import knobs
    p = shim.Prebound(NR._dequant_groups_kernel)
    B, am = _stack(2, 32, 128, seed=1)
    e = torch.arange(2, dtype=torch.int32, device="cuda")
    out = torch.empty(2, 32, 128, dtype=torch.bfloat16, device="cuda")
    args = (B, am, e, NG._lut(B.device), out, 32, 64, B.stride(0), B.stride(1), am.stride(0), am.stride(1), out.stride(0), out.stride(1))
    kw = dict(BLOCK_N=16, BLOCK_KB=256, QB=32, num_warps=8)
    p[(2, 2, 1)](*args, **kw)
    seen = []
    chain = knobs.runtime.launch_enter_hook
    if hasattr(chain, "add"):                                            # 3.6: a HookChain
        chain.add(seen.append)
    else:
        knobs.runtime.launch_enter_hook = seen.append
    try:
        before = shim.PREBIND_STATS["triton"]
        p[(2, 2, 1)](*args, **kw)
        torch.cuda.synchronize()
        assert seen and shim.PREBIND_STATS["triton"] == before + 1
    finally:
        if hasattr(chain, "remove"):
            chain.remove(seen.append)
        else:
            knobs.runtime.launch_enter_hook = None
    before = dict(shim.PREBIND_STATS)
    p[lambda meta: (2, 2, 1)](*args, **kw)
    assert shim.PREBIND_STATS == before                                  # a callable grid is Triton's own launch
    for g in range(2):
        assert torch.equal(out[g], NG.dequant_ref(B[g], am[g], 32, 128).to(torch.bfloat16))


def _tritons_kernel(fn, args, kw):
    """The compiled kernel Triton's own ``run`` would launch for these arguments (its binder, key and per-device cache)."""
    from triton import knobs
    from triton.runtime.driver import driver
    kw = dict(kw, debug=kw.get("debug", fn.debug) or knobs.runtime.debug)
    if shim._triton_version() >= (3, 6):
        kw["instrumentation_mode"] = knobs.compilation.instrumentation_mode
    caches = fn.device_caches[driver.active.get_current_device()]
    _, spec, opts = caches[-1](*args, **kw)
    if len(caches) == 5:
        from triton.runtime.jit import compute_cache_key
        return caches[0].get(compute_cache_key(caches[1], spec, opts))
    return caches[0].get(str(spec) + str(opts))


@gpu
def test_a_stages_hook_takes_tritons_path_under_triton_37():
    """Triton 3.7 adds a registered compiler-stages hook's pipeline hash to its kernel key (3.4 and 3.6 do not), so under 3.7 a launch
    with one registered takes Triton's path."""
    if shim._triton_version() < (3, 7):
        pytest.skip("this Triton does not key a launch on the stages hook")
    from triton import knobs
    p = shim.Prebound(NR._dequant_groups_kernel)
    B, am = _stack(2, 32, 128, seed=2)
    e = torch.arange(2, dtype=torch.int32, device="cuda")
    out = torch.empty(2, 32, 128, dtype=torch.bfloat16, device="cuda")
    args = (B, am, e, NG._lut(B.device), out, 32, 64, B.stride(0), B.stride(1), am.stride(0), am.stride(1), out.stride(0), out.stride(1))
    kw = dict(BLOCK_N=16, BLOCK_KB=256, QB=32, num_warps=8)
    for _ in range(2):
        p[(2, 2, 1)](*args, **kw)

    def hook(*a):                                                        # bare: the key's (key, hash); with the stages at compile
        return ("prebind-test", "0") if not a else None

    saved, knobs.runtime.add_stages_inspection_hook = knobs.runtime.add_stages_inspection_hook, hook
    try:
        before = dict(shim.PREBIND_STATS)
        out.zero_()
        p[(2, 2, 1)](*args, **kw)
        assert shim.PREBIND_STATS == {"prebound": before["prebound"], "triton": before["triton"] + 1}
    finally:
        knobs.runtime.add_stages_inspection_hook = saved
    for g in range(2):
        assert torch.equal(out[g], NG.dequant_ref(B[g], am[g], 32, 128).to(torch.bfloat16))


if shim.HAS_TRITON:
    tl = shim.tl

    @shim.triton.jit
    def _plus_one(X, Y, n, BLOCK: tl.constexpr):
        i = tl.arange(0, BLOCK)
        m = i < n
        tl.store(Y + i, tl.load(X + i, mask=m) + 1, mask=m)


@gpu
def test_async_compile_keeps_only_tritons_compiled_kernel():
    """Under AsyncCompileMode the launch that compiles a key returns a FutureKernel proxy in triton 3.7 (3.6 resolves it first): what
    is kept is the CompiledKernel Triton's own lookup returns, never the proxy."""
    ac = pytest.importorskip("triton.runtime._async_compile")
    from concurrent.futures import ThreadPoolExecutor
    p = shim.Prebound(_plus_one)                                         # a kernel no other test compiles
    x = torch.arange(32, device="cuda", dtype=torch.float32)
    ys = [torch.empty_like(x) for _ in range(3)]
    with ThreadPoolExecutor(1) as pool, ac.AsyncCompileMode(pool):
        got = [p[(1,)](x, y, 32, BLOCK=32) for y in ys]
    ((kept, *_),) = p.kernels.values()
    assert not hasattr(kept, "result") and kept is _tritons_kernel(_plus_one, (x, ys[0], 32), dict(BLOCK=32)) and got[-1] is kept
    assert all(torch.equal(y, x + 1) for y in ys)
