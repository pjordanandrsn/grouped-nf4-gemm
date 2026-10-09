"""Lane K28 (``PREREG-k28-pdl-decode-chain.md``): programmatic dependent launch (PDL) for the decode-row kernels of
``int4_b32``, behind ``GNF4_PDL``: on by default since experts4bit-qlora's lane P113 read CAP_DEFAULT, with
``GNF4_PDL_MAX_ROWS`` defaulting to 8 (``GNF4_PDL=0`` turns it off; ``GNF4_PDL_MAX_ROWS=0`` removes the cap).

The contract, before any timing:

1. **Off is the old path.** With ``GNF4_PDL=0`` every launch passes exactly the keywords it passed before the switch
   existed, so the kernels compile and launch as before. On a device that cannot do PDL (CPU, the interpreter, ROCm, or a card
   below sm_90) the switch is inert even when set.
2. **Ordering is untouched.** Every decode-row kernel's FIRST statement is the preamble, which waits for the previous
   kernel on the stream to complete (``griddepcontrol.wait``) before it lets the next one launch
   (``griddepcontrol.launch_dependents``).
   Nothing in a kernel reads or writes memory before its wait, so PDL can only hide launch latency, never reorder.
3. **On is bitwise off.** On an sm_90+ card, every wrapper and a captured chain of all of them return ``torch.equal``
   outputs with the switch on and off.

1 and 2 run on CPU (CI's no-device step; Triton is installed there, no kernel is launched). 3 needs an sm_90+ card
and runs compiled (``TRITON_INTERPRET`` unset): it is lane K28's premise on its RTX 5090.
"""
import ast
import os
import pathlib

import pytest

pytest.importorskip("triton")
import torch  # noqa: E402

import int4_b32 as m  # noqa: E402

SRC = pathlib.Path(m.__file__).read_text()
KERNELS = ("_quant_x_rows", "_gemv_int4_b32", "_reduce_partials", "_quant_x_rows_gathered", "_swiglu_rows",
           "_combine_rows", "_rmsnorm_rows", "_rmsnorm_resid_rows", "_scaled_resid_add_rows", "_rope_norm_heads",
           "_rope_heads", "_router_epilogue", "_rope_norm_qk")
INTERP = os.environ.get("TRITON_INTERPRET", "0") == "1"
CUDA = torch.cuda.is_available() and not INTERP and not torch.version.hip
CC = torch.cuda.get_device_capability() if CUDA else (0, 0)


@pytest.fixture
def fresh(monkeypatch):
    """Clean capability and switch caches, restored afterwards, so a mocked capability or a switch read never leaks
    into another test (the switch is read once and cached; each test here sets the environment first)."""
    monkeypatch.setattr(m, "_CC_CACHE", {})
    monkeypatch.setattr(m, "_PDL_SWITCH", [])
    monkeypatch.setattr(m, "_PDL_CAP", [m.PDL_MAX_ROWS_DEFAULT])
    monkeypatch.delenv(m.PDL_MAX_ROWS_ENV, raising=False)
    return monkeypatch


# ---------------------------------------------------------------- 1. off is today, and inert where PDL cannot run --
def test_the_switch_is_on_by_default_and_parses(monkeypatch):
    monkeypatch.delenv(m.PDL_ENV, raising=False)
    assert m.PDL_ENV == "GNF4_PDL" and m.PDL_DEFAULT is True and m.pdl_default() is True     # P113: CAP_DEFAULT
    for v, want in (("1", True), ("on", True), ("TRUE", True), ("0", False), ("off", False), ("", True),
                    ("2", True), ("yes", True)):
        monkeypatch.setenv(m.PDL_ENV, v)
        assert m.pdl_default() is want, v


def test_the_switch_is_read_once_and_refreshed_on_request(fresh):
    fresh.setattr(torch.cuda, "get_device_capability", lambda d=None: (12, 0))
    fresh.setattr(torch.version, "hip", None)
    fresh.setattr(m, "_LAUNCH_PDL", [True])
    fresh.delenv("TRITON_INTERPRET", raising=False)
    fresh.setenv(m.PDL_ENV, "1")
    assert m.pdl_active("cuda") is True
    fresh.setenv(m.PDL_ENV, "0")
    assert m.pdl_active("cuda") is True, "read once: a later environment change needs pdl_refresh()"
    assert m.pdl_refresh() is False and m.pdl_active("cuda") is False and m._pdl_kw("cuda", 1) == {}


def test_off_passes_no_keywords_anywhere(fresh):
    fresh.setenv(m.PDL_ENV, "0")
    fresh.setattr(torch.cuda, "get_device_capability", lambda d=None: (12, 0))
    fresh.setattr(torch.version, "hip", None)
    assert m._pdl_kw("cuda", 1) == {} and m._pdl_kw("cpu", 1) == {}


def test_inert_where_pdl_cannot_run(fresh):
    fresh.setenv(m.PDL_ENV, "1")
    fresh.setattr(torch.version, "hip", None)
    fresh.setattr(torch.cuda, "get_device_capability", lambda d=None: (12, 0))
    fresh.delenv("TRITON_INTERPRET", raising=False)
    assert m._pdl_kw("cpu", 1) == {}, "CPU"
    fresh.setenv("TRITON_INTERPRET", "1")
    assert m._pdl_kw("cuda", 1) == {}, "the interpreter"
    fresh.delenv("TRITON_INTERPRET", raising=False)
    fresh.setattr(torch.version, "hip", "6.2")
    assert m._pdl_kw("cuda", 1) == {}, "ROCm reports its own capability numbers (gfx942 is (9, 4))"
    fresh.setattr(torch.version, "hip", None)
    for cc in ((8, 0), (8, 6), (8, 9)):
        fresh.setattr(m, "_CC_CACHE", {})
        fresh.setattr(torch.cuda, "get_device_capability", lambda d=None, cc=cc: cc)
        assert m._pdl_kw("cuda", 1) == {}, cc
    fresh.setattr(m, "_LAUNCH_PDL", [False])
    fresh.setattr(m, "_CC_CACHE", {})
    fresh.setattr(torch.cuda, "get_device_capability", lambda d=None: (12, 0))
    assert m._pdl_kw("cuda", 1) == {}, "a Triton that cannot launch with PDL"


def test_on_where_pdl_can_run(fresh):
    fresh.setenv(m.PDL_ENV, "1")
    fresh.delenv("TRITON_INTERPRET", raising=False)
    fresh.setattr(torch.version, "hip", None)
    fresh.setattr(m, "_LAUNCH_PDL", [True])
    for cc in ((9, 0), (10, 0), (12, 0)):
        fresh.setattr(m, "_CC_CACHE", {})
        fresh.setattr(torch.cuda, "get_device_capability", lambda d=None, cc=cc: cc)
        assert m._pdl_kw("cuda", 1) == {"PDL": True, "launch_pdl": True}, cc


def test_the_row_cap_parses(monkeypatch):
    assert m.PDL_MAX_ROWS_ENV == "GNF4_PDL_MAX_ROWS" and m.PDL_MAX_ROWS_DEFAULT == 8                 # P113: CAP_DEFAULT
    monkeypatch.delenv(m.PDL_MAX_ROWS_ENV, raising=False)
    assert m.pdl_max_rows() == 8
    for v, want in (("", 8), ("0", 0), ("8", 8), (" 16 ", 16), ("-4", 8), ("x", 8), ("2.5", 8)):
        monkeypatch.setenv(m.PDL_MAX_ROWS_ENV, v)
        assert m.pdl_max_rows() == want, v


def test_the_row_cap_keeps_pdl_for_small_launches_only(fresh):
    fresh.setenv(m.PDL_ENV, "1")
    fresh.delenv("TRITON_INTERPRET", raising=False)
    fresh.setattr(torch.version, "hip", None)
    fresh.setattr(m, "_LAUNCH_PDL", [True])
    fresh.setattr(torch.cuda, "get_device_capability", lambda d=None: (12, 0))
    on = {"PDL": True, "launch_pdl": True}
    fresh.setenv(m.PDL_MAX_ROWS_ENV, "0")
    assert m.pdl_refresh() is True and [m._pdl_kw("cuda", r) for r in (1, 8, 16, 128)] == [on] * 4, "no cap: every launch"
    fresh.delenv(m.PDL_MAX_ROWS_ENV)
    assert m._pdl_kw("cuda", 16) == on, "the cap is read with the switch, once"
    m.pdl_refresh()
    assert [m._pdl_kw("cuda", r) for r in (1, 8, 9, 16, 128)] == [on, on, {}, {}, {}], "the default cap, 8"
    fresh.setenv(m.PDL_MAX_ROWS_ENV, "16")
    m.pdl_refresh()
    assert [m._pdl_kw("cuda", r) for r in (8, 16, 17)] == [on, on, {}]
    fresh.setenv(m.PDL_ENV, "0")
    m.pdl_refresh()
    assert m._pdl_kw("cuda", 1) == {}, "the cap never turns PDL on"


def test_triton_can_launch_with_pdl():
    """pyproject pins triton>=3.4, the release that added ``launch_pdl``; a regression here would silently disable the
    switch."""
    from triton.backends.nvidia.compiler import CUDAOptions
    assert "launch_pdl" in CUDAOptions.__dataclass_fields__
    assert m._launch_pdl_available() is True


# ------------------------------------------------------------------- 2. the preamble comes first, in every kernel --
def _defs():
    tree = ast.parse(SRC)
    return {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}


def test_the_preamble_waits_before_it_releases_the_next_kernel():
    body = _defs()["_pdl_enter"].body
    body = body[1:] if isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) else body
    assert len(body) == 1 and isinstance(body[0], ast.If) and ast.unparse(body[0].test) == "PDL"
    calls = [s.value for s in body[0].body]
    assert [ast.unparse(c.func) for c in calls] == ["tl.inline_asm_elementwise"] * 2
    asm = [c.args[0].value for c in calls]
    assert asm[0].startswith("griddepcontrol.wait;") and asm[1].startswith("griddepcontrol.launch_dependents;"), asm
    for c in calls:                       # not pure: the compiler may neither drop nor hoist it
        assert {k.arg: ast.unparse(k.value) for k in c.keywords}["is_pure"] == "False"


@pytest.mark.parametrize("name", KERNELS)
def test_every_decode_row_kernel_enters_through_the_preamble(name):
    fn = _defs()[name]
    assert any(ast.unparse(d) == "triton.jit" for d in fn.decorator_list), name
    last = fn.args.args[-1]
    assert last.arg == "PDL" and ast.unparse(last.annotation) == "tl.constexpr", name
    assert ast.unparse(fn.args.defaults[-1]) == "False", f"{name}: PDL must default off"
    body = fn.body[1:] if isinstance(fn.body[0], ast.Expr) and isinstance(fn.body[0].value, ast.Constant) else fn.body
    assert ast.unparse(body[0]) == "_pdl_enter(PDL)", f"{name}: the preamble must be the first statement"


def _launches():
    """Every ``kernel[grid](...)`` launch of a decode-row kernel in the module, with its keyword names."""
    out = []
    for node in ast.walk(ast.parse(SRC)):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript)
                and isinstance(node.func.value, ast.Name) and node.func.value.id in KERNELS):
            out.append((node.func.value.id, [ast.unparse(k.value) if k.arg is None else k.arg for k in node.keywords]))
    return out


def test_every_launch_takes_the_switch_and_nothing_else_names_it():
    launches = _launches()
    assert {n for n, _ in launches} == set(KERNELS), sorted({n for n, _ in launches} ^ set(KERNELS))
    for name, kws in launches:
        star = [k for k in kws if k.startswith("_pdl_kw(")]
        assert len(star) == 1, (name, kws)
        call = ast.parse(star[0], mode="eval").body
        assert len(call.args) == 2 and not call.keywords, f"{name}: every launch passes its device and its rows ({star[0]})"
        assert "PDL" not in kws and "launch_pdl" not in kws, (name, kws)
    assert len(launches) == 14 and SRC.count("**_pdl_kw(") == 14


# ------------------------------------------------------------------------------- 3. on is bitwise off, on the card --
needs_pdl = pytest.mark.skipif(not CUDA or CC < (9, 0), reason="PDL needs a compiled sm_90+ CUDA device")


def _inputs(dev, seed=0):
    from int4_pack_ref import pack_int4_b32
    g = torch.Generator().manual_seed(seed)
    H, I, E, TOP, D, QH, KH = 256, 128, 4, 2, 64, 4, 2
    N_QKV = (QH + 2 * KH) * D

    def store(e, n, k):
        w = torch.randn(e, n, k, generator=g) / (k ** 0.5)
        pk, sc = zip(*(pack_int4_b32(w[i].float()) for i in range(e)))
        return (torch.stack([p.reshape(n, k // 2) for p in pk]).contiguous().to(dev),
                torch.stack([s.reshape(n, k // 32) for s in sc]).contiguous().to(dev))
    t = {"H": H, "I": I, "E": E, "TOP": TOP, "D": D, "QH": QH, "KH": KH, "N_QKV": N_QKV,
         "x": torch.randn(1, H, generator=g).to(torch.bfloat16).to(dev),
         "ln1": (1 + 0.1 * torch.randn(H, generator=g)).to(torch.bfloat16).to(dev),
         "ln2": (1 + 0.1 * torch.randn(H, generator=g)).to(torch.bfloat16).to(dev),
         "qn": (1 + 0.1 * torch.randn(D, generator=g)).to(torch.bfloat16).to(dev),
         "cos": torch.randn(1, D, generator=g).to(torch.bfloat16).to(dev),
         "sin": torch.randn(1, D, generator=g).to(torch.bfloat16).to(dev),
         "logits": torch.randn(1, E, generator=g).to(dev),
         "order": torch.arange(TOP, device=dev), "row_token": torch.zeros(TOP, dtype=torch.long, device=dev),
         "qkv": store(1, N_QKV, H), "o": store(1, H, QH * D), "gu": store(E, 2 * I, H), "dn": store(E, H, I)}
    t["e0"] = torch.zeros(1, dtype=torch.int32, device=dev)
    return t


def _chain(t):
    """Every decode-row wrapper once, in a served layer's order, each reading the one before (so a stale read shows)."""
    out = {}

    def trio(x, st, eids, n, k):
        xq, xs = m.quant_x_rows(x)
        return m.gemv_int4_b32(xq, xs, st[0], st[1], eids, n, k)
    out["h"] = h = m.rmsnorm_rows(t["x"], t["ln1"], 1e-6)
    out["qkv"] = qkv = trio(h, t["qkv"], t["e0"], t["N_QKV"], t["H"])
    qd = t["QH"] * t["D"]
    out["q"] = q = m.rope_norm_heads(qkv[:, :qd].view(1, t["QH"], t["D"]), t["qn"], t["cos"], t["sin"], 1e-6)
    out["k"] = m.rope_heads(qkv[:, qd:qd + t["KH"] * t["D"]].view(1, t["KH"], t["D"]), t["cos"], t["sin"])
    out["o"] = o = trio(q.view(1, qd), t["o"], t["e0"], t["H"], qd)
    out["normed"], out["resid"] = normed, resid = m.rmsnorm_resid_rows(o, t["x"], t["ln2"], 1e-6)
    out["first"], out["w"], out["idx"] = _first, w, idx = m.router_epilogue(t["logits"], t["TOP"], True)
    xr = normed.index_select(0, t["row_token"])
    xq, xs = m.quant_x_rows_gathered(xr, t["order"])
    out["gu"] = gu = m.gemv_int4_b32(xq, xs, t["gu"][0], t["gu"][1], idx.view(-1), 2 * t["I"], t["H"])
    out["sw"] = sw = m.swiglu_rows(gu)
    xq2, xs2 = m.quant_x_rows_gathered(sw, t["order"])
    out["dn"] = dn = m.gemv_int4_b32(xq2, xs2, t["dn"][0], t["dn"][1], idx.view(-1), t["H"], t["I"])
    out["moe"] = moe = m.combine_rows(dn, w, t["TOP"])
    out["x_next"] = m.scaled_resid_add_rows(moe, resid, 1.0)
    out["gu_fused"] = m.gemv_int4_b32(xq, xs, t["gu"][0], t["gu"][1], idx.view(-1), 2 * t["I"], t["H"],
                                      fused_reduce=True)
    return out


def _run(monkeypatch, on, graph):
    monkeypatch.setenv(m.PDL_ENV, "1" if on else "0")
    m.pdl_refresh()
    assert m.pdl_active("cuda") is on
    t = _inputs("cuda")
    if not graph:
        out = _chain(t)
        torch.cuda.synchronize()
        return out
    _chain(t)                                   # compile outside the capture
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = _chain(t)
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    return out


@needs_pdl
@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
def test_on_is_bitwise_off(monkeypatch, graph):
    off = _run(monkeypatch, False, graph)
    on = _run(monkeypatch, True, graph)
    assert off.keys() == on.keys()
    bad = [k for k in off if not torch.equal(off[k], on[k])]
    assert not bad, f"PDL changed {bad}"
    assert torch.equal(off["gu"], off["gu_fused"])


# The engagement probe K28's bench registers, in miniature. A 20 us spin releases its dependents at once and stamps
# the clock last; a dependent stamps the clock before and after its wait. With PDL the dependent starts during the spin
# and leaves its wait no earlier than the spin's last stamp; without PDL it starts after the spin.
SPIN_NS = 20_000

import triton  # noqa: E402
import triton.language as tl  # noqa: E402


@triton.jit
def _probe_spin(out_ptr, NS: tl.constexpr, PDL: tl.constexpr):
    t0 = tl.extra.cuda.globaltimer()
    if PDL:
        tl.inline_asm_elementwise("griddepcontrol.launch_dependents; // dummy $0", "=r", [], dtype=tl.int32,
                                  is_pure=False, pack=1)
    t = t0
    while t - t0 < NS:
        t = tl.extra.cuda.globaltimer()
    tl.store(out_ptr, t)


@triton.jit
def _probe_after(out_ptr, PDL: tl.constexpr):
    t_pre = tl.extra.cuda.globaltimer()
    if PDL:
        tl.inline_asm_elementwise("griddepcontrol.wait; // dummy $0", "=r", [], dtype=tl.int32, is_pure=False, pack=1)
    t_post = tl.extra.cuda.globaltimer()
    tl.store(out_ptr + 1, t_pre)
    tl.store(out_ptr + 2, t_post)


def _probe(pdl):
    buf = torch.zeros(3, dtype=torch.int64, device="cuda")
    kw = {"launch_pdl": True} if pdl else {}

    def step():
        _probe_spin[(1,)](buf, NS=SPIN_NS, PDL=pdl)
        _probe_after[(1,)](buf, PDL=pdl, **kw)
    step()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        step()
    rows = []
    for _ in range(5):
        g.replay()
        torch.cuda.synchronize()
        end, pre, post = buf.tolist()
        rows.append({"early_ns": end - pre, "wait_ok": post >= end})
    return rows


@needs_pdl
def test_a_dependent_launches_before_its_primary_ends_and_waits_for_it():
    on, off = _probe(True), _probe(False)
    assert all(r["early_ns"] >= SPIN_NS // 2 and r["wait_ok"] for r in on), on
    assert all(r["early_ns"] <= 0 and r["wait_ok"] for r in off), off
