"""NF4_QLORA_PAD_BUCKETS=1: the lean padded LoRA delta, padded per bucket of similar-sized groups.

Against the single block (the flag off) the bar is ROUNDING, not ``torch.equal``: every real row gets the same
arithmetic, but the ``bmm``s run at other shapes, and a BLAS may pick its kernel, or split a reduction, by shape. Each
tensor is held to a few units in the last place of its own largest entry: ``2**-16 * max|ref|`` for an fp32 tensor, two
bf16 ulps at the top (``2**-6 * max|ref|``) for a bf16 one. A misplaced, dropped or doubled row moves a tensor by a
large fraction of its max, far outside either. Against the per-expert loop the bar is the existing tests'
(``rtol=1e-4, atol=1e-5``, fp32). With the flag unset or ``0`` the bar is ``torch.equal`` and the same aten op sequence
as main's single-block body, kept below verbatim as the oracle.
"""
import heapq
import math
import random

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

import nf4_grouped as NG
import nf4_qlora
from nf4_qlora import _lora_delta_grouped_loop, lora_delta_grouped

CUDA = torch.cuda.is_available()
DEVICES = ["cpu"] + (["cuda"] if CUDA else [])

CASES = [
    ([3, 3, 3, 3], [0, 1, 2, 3]),                                    # even groups: one bucket
    ([5, 0, 1, 7], [2, 0, 3, 1]),                                    # an empty group, skew, permuted ids
    ([1], [3]),                                                      # a single group
    ([60, 2, 2, 2, 2, 2, 2, 2], [0, 1, 2, 3, 4, 5, 6, 7]),           # one hot expert
    ([2, 1, 0, 90, 3, 0, 1, 4, 75, 2, 5, 1], [9, 4, 0, 1, 7, 2, 11, 3, 6, 10, 8, 5]),   # two hot experts, empties
    ([4, 6, 2], [7, 7, 1]),                                          # a repeated id: the adapters' backward accumulates
    ([40, 3, 0, 9, 3, 17, 1], [2, 5, 0, 2, 5, 3, 2]),               # repeated ids across buckets
]
DTYPES = [(torch.bfloat16, torch.bfloat16), (torch.bfloat16, torch.float32), (torch.float32, torch.float32)]


def _fresh():
    NG._UPLOAD_MEMO.clear()
    NG._UPLOAD_FAST.entries.clear()   # the value-keyed upload memo (CUDA only): a hit there skips the upload's own ops
    nf4_qlora._PLAN_MEMO.clear()


def _inputs(dev, sizes, eids, act_dtype, ad_dtype, seed=0, K=128, N=96, R=8):
    E = max(eids) + 1
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(sum(sizes), K, generator=g).to(act_dtype).to(dev)
    A = (torch.randn(E, R, K, generator=g) * 0.05).to(ad_dtype).to(dev)
    B = (torch.randn(E, N, R, generator=g) * 0.05).to(ad_dtype).to(dev)
    gout = torch.randn(sum(sizes), N, generator=g).to(ad_dtype).to(dev)
    return a, A, B, gout


def _run(fn, inputs, sizes, eids, scaling):
    a, A, B, gout = (t.detach().clone() for t in inputs)
    a, A, B = (t.requires_grad_(True) for t in (a, A, B))
    out = fn(a, A, B, sizes, eids, scaling)
    out.backward(gout)
    return out.detach(), a.grad, A.grad, B.grad


def _rounding_close(got, ref, name):
    """``got`` within rounding of ``ref``, scaled to ``ref``'s largest entry (see the module docstring)."""
    assert got.dtype == ref.dtype and got.shape == ref.shape, name
    tol = (2.0 ** -6 if ref.dtype == torch.bfloat16 else 2.0 ** -16) * ref.float().abs().max().item()
    diff = (got.float() - ref.float()).abs().max().item()
    assert diff <= tol, f"{name}: max |diff| {diff:.3e} > {tol:.3e}"


def _rule(rows):
    """The bucket rule, restated independently: sorted ascending, each bucket every group up to 2x its narrowest."""
    rest, out = sorted(rows), []
    while rest:
        n = sum(1 for r in rest if r <= 2 * rest[0])
        out.append((n, rest[n - 1]))
        rest = rest[n:]
    return out


# --- the plan (host only) ---

@pytest.mark.parametrize("rows", [[3, 3, 3, 3], [5, 1, 7], [1], [60, 2, 2, 2, 2, 2, 2, 2],
                                  [2, 1, 90, 3, 1, 4, 75, 2, 5, 1], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                                  [7, 7, 7, 14, 15, 28, 29, 1000, 2000, 2001]])
def test_bucket_plan_follows_the_rule(rows):
    order, buckets, pstart = nf4_qlora._pad_buckets(rows)
    assert buckets == _rule(rows)
    assert sorted(order) == list(range(len(rows))) and [rows[g] for g in order] == sorted(rows)
    assert order == sorted(range(len(rows)), key=lambda g: (rows[g], g))        # stable: ties keep the caller's order
    total_padded = sum(g * w for g, w in buckets)
    assert total_padded <= 2 * sum(rows)
    # Every group's block is [pstart, pstart + W_b), inside its bucket, and the blocks tile [0, total_padded) exactly.
    width = {}
    i = 0
    for g_b, w_b in buckets:
        lo = rows[order[i]]
        assert all(lo <= rows[g] <= w_b <= 2 * lo for g in order[i:i + g_b])
        for g in order[i:i + g_b]:
            width[g] = w_b
        i += g_b
    spans = sorted((pstart[g], pstart[g] + width[g]) for g in range(len(rows)))
    assert spans[0][0] == 0 and spans[-1][1] == total_padded
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))


# --- values against the single block and the loop ---

@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("sizes,eids", CASES)
@pytest.mark.parametrize("act_dtype,ad_dtype", DTYPES)
@pytest.mark.parametrize("scaling", [1.0, 2.0])
def test_bucketed_matches_the_single_block(monkeypatch, dev, sizes, eids, act_dtype, ad_dtype, scaling):
    monkeypatch.setenv("NF4_QLORA_LORA_PATH", "padded")
    inputs = _inputs(dev, sizes, eids, act_dtype, ad_dtype)
    monkeypatch.setenv("NF4_QLORA_PAD_BUCKETS", "0")
    _fresh()
    single = _run(lora_delta_grouped, inputs, sizes, eids, scaling)
    monkeypatch.setenv("NF4_QLORA_PAD_BUCKETS", "1")
    _fresh()
    before = dict(nf4_qlora.LORA_PATH_STATS)
    bucketed = _run(lora_delta_grouped, inputs, sizes, eids, scaling)
    assert nf4_qlora.LORA_PATH_STATS["padded_bucketed"] == before["padded_bucketed"] + 1
    assert nf4_qlora.LORA_PATH_STATS["padded"] == before["padded"]
    rows = [s for s in sizes if s]
    assert nf4_qlora.LORA_PAD_WASTE["last_rows_single"] == len(rows) * max(rows)
    assert nf4_qlora.LORA_PAD_WASTE["last_rows_bucketed"] == sum(g * w for g, w in _rule(rows))
    assert nf4_qlora.LORA_PAD_WASTE["last_buckets"] == len(_rule(rows))
    for name, b, s in zip(("out", "d_a", "d_A", "d_B"), bucketed, single):
        _rounding_close(b, s, name)


@pytest.mark.parametrize("sizes,eids", CASES)
def test_bucketed_matches_the_per_expert_loop(monkeypatch, sizes, eids):
    monkeypatch.setenv("NF4_QLORA_LORA_PATH", "padded")
    monkeypatch.setenv("NF4_QLORA_PAD_BUCKETS", "1")
    inputs = _inputs("cpu", sizes, eids, torch.float32, torch.float32, seed=4)
    ref = _run(_lora_delta_grouped_loop, inputs, sizes, eids, 2.0)
    got = _run(lora_delta_grouped, inputs, sizes, eids, 2.0)
    for name, g, r in zip(("out", "d_a", "d_A", "d_B"), got, ref):
        assert torch.allclose(g, r, rtol=1e-4, atol=1e-5), f"{name} diverged: max {(g - r).abs().max():.3e}"


def _zipf_top8_sizes(tokens=4096, experts=128, k=8, s=1.0, seed=0):
    """Per-expert row counts of ``tokens`` tokens each routed to ``k`` DISTINCT experts drawn with Zipf(s) weights (the
    expert ids shuffled, so the hot experts are not the first groups). Weighted sampling without replacement by the
    exponential-key method on Python's own generator, so the counts do not depend on the torch version."""
    rng = random.Random(seed)
    w = [1.0 / (r + 1) ** s for r in range(experts)]
    rng.shuffle(w)
    sizes = [0] * experts
    for _ in range(tokens):
        for e in heapq.nlargest(k, range(experts), key=lambda e: math.log(1.0 - rng.random()) / w[e]):
            sizes[e] += 1
    return sizes


def test_zipf_top8_padded_rows_shrink_as_the_rule_predicts(monkeypatch):
    """4,096 tokens x top-8 of 128 experts, Zipf(1): the field shape's skew (one hot expert takes most tokens). The
    single block pads every group to the hottest; the buckets pad to at most twice the real rows."""
    sizes = _zipf_top8_sizes()
    eids = list(range(len(sizes)))
    rows = [s for s in sizes if s]
    total, single = sum(rows), len(rows) * max(rows)
    predicted = sum(g * w for g, w in _rule(rows))
    assert total == 4096 * 8
    assert predicted <= 2 * total
    ratio = single / predicted
    assert ratio >= 4.0, f"single {single} / bucketed {predicted} = {ratio:.2f}"
    monkeypatch.setenv("NF4_QLORA_LORA_PATH", "padded")
    inputs = _inputs("cpu", sizes, eids, torch.float32, torch.float32, seed=5, K=32, N=24, R=8)
    monkeypatch.setenv("NF4_QLORA_PAD_BUCKETS", "0")
    want = _run(lora_delta_grouped, inputs, sizes, eids, 2.0)
    monkeypatch.setenv("NF4_QLORA_PAD_BUCKETS", "1")
    got = _run(lora_delta_grouped, inputs, sizes, eids, 2.0)
    assert nf4_qlora.LORA_PAD_WASTE["last_rows_single"] == single
    assert nf4_qlora.LORA_PAD_WASTE["last_rows_bucketed"] == predicted
    for name, g, w in zip(("out", "d_a", "d_A", "d_B"), got, want):
        _rounding_close(g, w, name)
    print(f"\nZipf(1) top-8: {total} real rows; single block {single} ({single / total:.2f}x real); "
          f"bucketed {predicted} ({predicted / total:.3f}x real) in {len(_rule(rows))} buckets; {ratio:.2f}x fewer")


# --- unset / 0 is main's single block, op for op ---

def _main_single_block(a_cat, lora_A, lora_B, sizes, expert_ids, scaling):
    """main's lean padded path for host-list ids off CUDA (no plan memo), verbatim but for the dispatch around it."""
    from nf4_grouped import to_device_i32
    nz = [g for g in range(len(sizes)) if int(sizes[g]) > 0]
    rows = [int(sizes[g]) for g in nz]
    total, widest = sum(rows), max(rows)
    dev = a_cat.device
    host_ids = [int(expert_ids[g]) for g in nz]
    sz_i32, eid_i32 = to_device_i32((rows, host_ids), dev)
    eid = eid_i32.to(torch.int64)
    sz = sz_i32.to(torch.int64)
    G = len(rows)
    shift = torch.repeat_interleave(
        torch.arange(G, device=dev) * widest - (torch.cumsum(sz, 0) - sz), sz,
        output_size=total)
    flat = torch.arange(total, device=dev) + shift
    A, B = lora_A[eid], lora_B[eid]
    x = torch.zeros(G * widest, a_cat.shape[1], dtype=A.dtype, device=dev)
    x.index_copy_(0, flat, a_cat.to(A.dtype))
    d = torch.bmm(torch.bmm(x.view(G, widest, -1), A.transpose(1, 2)),
                  B.transpose(1, 2))
    return nf4_qlora._scaled(nf4_qlora._GatherRows.apply(d.view(G * widest, -1), flat), scaling)


class _OpLog(TorchDispatchMode):
    """Every aten op, with its tensor arguments' shapes and dtypes, forward and backward."""

    def __init__(self):
        super().__init__()
        self.ops = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops.append((str(func), tuple((tuple(t.shape), t.dtype) for t in args if isinstance(t, torch.Tensor))))
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize("sizes,eids", CASES)
@pytest.mark.parametrize("act_dtype,ad_dtype", DTYPES)
def test_unset_and_zero_are_main_op_for_op(monkeypatch, sizes, eids, act_dtype, ad_dtype):
    monkeypatch.setenv("NF4_QLORA_LORA_PATH", "padded")
    inputs = _inputs("cpu", sizes, eids, act_dtype, ad_dtype)
    sentinel = {"last_rows_single": -1, "last_rows_bucketed": -2, "last_buckets": -3}
    seen = {}
    for flag in (None, "0", "main"):
        if flag is None:
            monkeypatch.delenv("NF4_QLORA_PAD_BUCKETS", raising=False)
        else:
            monkeypatch.setenv("NF4_QLORA_PAD_BUCKETS", "0")
        nf4_qlora.LORA_PAD_WASTE.update(sentinel)
        before = dict(nf4_qlora.LORA_PATH_STATS)
        fn = _main_single_block if flag == "main" else lora_delta_grouped
        with _OpLog() as log:
            res = _run(fn, inputs, sizes, eids, 2.0)
        if flag != "main":
            assert nf4_qlora.LORA_PATH_STATS["padded"] == before["padded"] + 1
            assert nf4_qlora.LORA_PATH_STATS["padded_bucketed"] == before["padded_bucketed"]
            assert {k: nf4_qlora.LORA_PAD_WASTE[k] for k in sentinel} == sentinel
        seen[flag] = (log.ops, res)
    for flag in (None, "0"):
        assert seen[flag][0] == seen["main"][0], f"flag {flag!r}: the op sequence differs from main's"
        for name, x, y in zip(("out", "d_a", "d_A", "d_B"), seen[flag][1], seen["main"][1]):
            assert x.dtype == y.dtype and torch.equal(x, y), f"flag {flag!r}: {name} differs"


def test_auto_is_the_default(monkeypatch):
    """Unset is ``auto`` (experts4bit-qlora TC1 amendment 50): a call under the row gate keeps the single block, one at or over it
    buckets. ``0`` is the single block everywhere, ``1`` buckets every call."""
    monkeypatch.delenv("NF4_QLORA_PAD_BUCKETS", raising=False)
    monkeypatch.delenv("NF4_QLORA_PAD_BUCKETS_MIN_ROWS", raising=False)
    assert nf4_qlora._pad_buckets_mode() == "auto"
    assert nf4_qlora._pad_buckets_enabled(9040) is False and nf4_qlora._pad_buckets_enabled(32768) is True
    monkeypatch.setenv("NF4_QLORA_PAD_BUCKETS", "0")
    assert nf4_qlora._pad_buckets_mode() == "0" and nf4_qlora._pad_buckets_enabled(32768) is False
    monkeypatch.setenv("NF4_QLORA_PAD_BUCKETS", "1")
    assert nf4_qlora._pad_buckets_enabled() is True and nf4_qlora._pad_buckets_enabled(1) is True


def test_auto_gates_on_routed_rows(monkeypatch):
    """``auto`` buckets a call only when it carries at least ``_pad_buckets_min_rows()`` routed rows (default 16,384: TC1's field
    recipe peaked at 9,040 a call, packed 4,096-token rows carry 32,768); ``NF4_QLORA_PAD_BUCKETS_MIN_ROWS`` overrides it; any other
    value is the single block."""
    monkeypatch.delenv("NF4_QLORA_PAD_BUCKETS_MIN_ROWS", raising=False)
    monkeypatch.setenv("NF4_QLORA_PAD_BUCKETS", "auto")
    assert nf4_qlora._pad_buckets_min_rows() == nf4_qlora._PAD_BUCKETS_AUTO_MIN_ROWS == 16384
    assert nf4_qlora._pad_buckets_enabled(9040) is False and nf4_qlora._pad_buckets_enabled(32768) is True
    assert nf4_qlora._pad_buckets_enabled(16384) is True and nf4_qlora._pad_buckets_enabled(16383) is False
    assert nf4_qlora._pad_buckets_enabled() is False                      # no row count: never bucketed under auto
    monkeypatch.setenv("NF4_QLORA_PAD_BUCKETS_MIN_ROWS", "50")
    assert nf4_qlora._pad_buckets_enabled(49) is False and nf4_qlora._pad_buckets_enabled(50) is True
    for v in ("yes", "2", " AUTO "):
        monkeypatch.setenv("NF4_QLORA_PAD_BUCKETS", v)
        assert nf4_qlora._pad_buckets_mode() == ("auto" if v.strip().lower() == "auto" else "0"), v


@pytest.mark.parametrize("dev", DEVICES)
def test_auto_routes_each_call_by_its_rows(monkeypatch, dev):
    """End to end under ``auto``: a call under the gate is the single block (counted ``padded``, the same bytes as unset), a call at
    or over it is bucketed (counted ``padded_bucketed``, equal to the single block to rounding)."""
    monkeypatch.setenv("NF4_QLORA_LORA_PATH", "padded")
    small, big = ([3, 1, 0, 2, 4], [0, 1, 2, 3, 4]), ([60, 2, 0, 2, 9, 1, 5], [4, 0, 2, 1, 3, 5, 6])
    monkeypatch.setenv("NF4_QLORA_PAD_BUCKETS_MIN_ROWS", str(sum(big[0])))
    for sizes, eids, want in ((small[0], small[1], "padded"), (big[0], big[1], "padded_bucketed")):
        inputs = _inputs(dev, sizes, eids, torch.bfloat16, torch.float32)
        monkeypatch.setenv("NF4_QLORA_PAD_BUCKETS", "0")
        _fresh()
        ref = _run(lora_delta_grouped, inputs, sizes, eids, 2.0)
        monkeypatch.setenv("NF4_QLORA_PAD_BUCKETS", "auto")
        _fresh()
        before = dict(nf4_qlora.LORA_PATH_STATS)
        got = _run(lora_delta_grouped, inputs, sizes, eids, 2.0)
        assert nf4_qlora.LORA_PATH_STATS[want] == before[want] + 1, (sizes, want)
        for name, x, y in zip(("out", "d_a", "d_A", "d_B"), got, ref):
            if want == "padded":
                assert torch.equal(x, y), name                             # under the gate: the single block itself
            else:
                _rounding_close(x, y, name)


# --- what the flag does not touch, and what wins when it meets the compact delta ---

def test_lean_delta_off_ignores_buckets(monkeypatch):
    """NF4_QLORA_LEAN_DELTA=0's previous body is not bucketed: same bytes and same count as without the flag."""
    monkeypatch.setenv("NF4_QLORA_LORA_PATH", "padded")
    monkeypatch.setenv("NF4_QLORA_LEAN_DELTA", "0")
    sizes, eids = [60, 2, 0, 2, 9], [4, 0, 2, 1, 3]
    inputs = _inputs("cpu", sizes, eids, torch.bfloat16, torch.float32)
    monkeypatch.setenv("NF4_QLORA_PAD_BUCKETS", "0")
    off = _run(lora_delta_grouped, inputs, sizes, eids, 2.0)
    monkeypatch.setenv("NF4_QLORA_PAD_BUCKETS", "1")
    before = dict(nf4_qlora.LORA_PATH_STATS)
    on = _run(lora_delta_grouped, inputs, sizes, eids, 2.0)
    assert nf4_qlora.LORA_PATH_STATS["padded"] == before["padded"] + 1
    assert nf4_qlora.LORA_PATH_STATS["padded_bucketed"] == before["padded_bucketed"]
    for x, y in zip(on, off):
        assert torch.equal(x, y)


@pytest.mark.parametrize("dev", DEVICES)
def test_with_compact_delta_buckets_win(monkeypatch, dev):
    """Both flags set: the bucketed path, op for op what it is without NF4_QLORA_COMPACT_DELTA."""
    monkeypatch.setenv("NF4_QLORA_LORA_PATH", "padded")
    monkeypatch.setenv("NF4_QLORA_PAD_BUCKETS", "1")
    sizes, eids = [60, 2, 0, 2, 9], [4, 0, 2, 1, 3]
    inputs = _inputs(dev, sizes, eids, torch.bfloat16, torch.float32)
    runs = {}
    for compact in ("0", "1"):
        monkeypatch.setenv("NF4_QLORA_COMPACT_DELTA", compact)
        _fresh()
        before = dict(nf4_qlora.LORA_PATH_STATS)
        with _OpLog() as log:
            res = _run(lora_delta_grouped, inputs, sizes, eids, 2.0)
        assert nf4_qlora.LORA_PATH_STATS["padded_bucketed"] == before["padded_bucketed"] + 1
        runs[compact] = (log.ops, res)
    assert runs["1"][0] == runs["0"][0]
    for x, y in zip(runs["1"][1], runs["0"][1]):
        assert torch.equal(x, y)


# --- CUDA only: device ids, the extended plan memo, and the allocator peak ---

@pytest.mark.skipif(not CUDA, reason="device-tensor expert ids need CUDA")
def test_device_expert_ids_take_the_same_values(monkeypatch):
    monkeypatch.setenv("NF4_QLORA_LORA_PATH", "padded")
    monkeypatch.setenv("NF4_QLORA_PAD_BUCKETS", "1")
    sizes, eids = [2, 1, 0, 90, 3, 0, 1, 4, 75, 2, 5, 1], [9, 4, 0, 1, 7, 2, 11, 3, 6, 10, 8, 5]
    inputs = _inputs("cuda", sizes, eids, torch.bfloat16, torch.float32)
    _fresh()
    host = _run(lora_delta_grouped, inputs, sizes, eids, 2.0)
    _fresh()
    devd = _run(lora_delta_grouped, inputs, sizes, torch.tensor(eids, device="cuda", dtype=torch.int32), 2.0)
    for name, h, d in zip(("out", "d_a", "d_A", "d_B"), host, devd):
        _rounding_close(d, h, name)


def _twin(sizes, eids, ad_dtype, seed=0):
    """gate_up then down off one grouping, both through lora_delta_grouped, then backward through both."""
    E, K1, N1, K2, N2, R = max(eids) + 1, 128, 96, 48, 128, 8
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(sum(sizes), K1, generator=g).to(torch.bfloat16).cuda().requires_grad_(True)
    def mk(*s):
        return (torch.randn(*s, generator=g) * 0.05).to(ad_dtype).cuda().requires_grad_(True)
    A1, B1, A2, B2 = mk(E, R, K1), mk(E, N1, R), mk(E, R, K2), mk(E, N2, R)
    d1 = lora_delta_grouped(a, A1, B1, sizes, eids, 2.0)
    h = (d1[:, :K2].float().sigmoid() * 2).to(torch.bfloat16)
    d2 = lora_delta_grouped(h, A2, B2, sizes, eids, 2.0)
    gout = torch.randn(d2.shape, generator=g).to(d2.dtype).cuda()
    (d2 * gout).float().sum().backward()
    return [t.detach().clone() for t in (d1, d2, a.grad, A1.grad, B1.grad, A2.grad, B2.grad)]


@pytest.mark.skipif(not CUDA, reason="the plan memo is CUDA-only")
@pytest.mark.parametrize("sizes,eids", [CASES[4], CASES[6]])
@pytest.mark.parametrize("ad_dtype", [torch.bfloat16, torch.float32])
def test_bucketed_plan_memo_hits_on_the_twin(monkeypatch, sizes, eids, ad_dtype):
    """GNF4_HOST_REUSE on: the down call reuses the gate_up call's BUCKETED plan (one miss, one hit), never a single-block
    plan, and the values are the single block's to rounding."""
    monkeypatch.setenv("NF4_QLORA_LORA_PATH", "padded")
    monkeypatch.setenv("GNF4_HOST_REUSE", "1")
    monkeypatch.setenv("NF4_QLORA_PAD_BUCKETS", "0")
    _fresh()
    single = _twin(sizes, eids, ad_dtype)
    monkeypatch.setenv("NF4_QLORA_PAD_BUCKETS", "1")
    _fresh()
    for k in NG.HOST_REUSE_STATS:
        NG.HOST_REUSE_STATS[k] = 0
    bucketed = _twin(sizes, eids, ad_dtype)
    assert NG.HOST_REUSE_STATS["plan_misses"] == 1 and NG.HOST_REUSE_STATS["plan_hits"] == 1
    (key,) = nf4_qlora._PLAN_MEMO
    assert key[-1] == "buckets" and len(nf4_qlora._PLAN_MEMO[key]) == 4
    for name, b, s in zip(("d1", "d2", "d_a", "d_A1", "d_B1", "d_A2", "d_B2"), bucketed, single):
        _rounding_close(b, s, name)


@pytest.mark.skipif(not CUDA, reason="max_memory_allocated is a CUDA allocator statistic")
@pytest.mark.parametrize("ad_dtype", [torch.bfloat16, torch.float32])
def test_bucketed_peak_below_the_single_block(monkeypatch, ad_dtype):
    """A hot expert (8 x 60 padded rows for 74 real ones, against 74 bucketed): the forward and backward peaks of one
    call, above what was allocated before it, are below the single block's."""
    monkeypatch.setenv("NF4_QLORA_LORA_PATH", "padded")
    sizes, eids = [60, 2, 2, 2, 2, 2, 2, 2], list(range(8))
    inputs = _inputs("cuda", sizes, eids, torch.bfloat16, ad_dtype, K=512, N=384, R=16)
    peaks = {}
    for flag in ("0", "1"):
        monkeypatch.setenv("NF4_QLORA_PAD_BUCKETS", flag)
        _fresh()
        a, A, B, gout = (t.detach().clone() for t in inputs)
        a, A, B = (t.requires_grad_(True) for t in (a, A, B))
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        out = lora_delta_grouped(a, A, B, sizes, eids, 1.0)
        torch.cuda.synchronize()
        fwd = torch.cuda.max_memory_allocated() - base
        torch.cuda.reset_peak_memory_stats()
        out.backward(gout)
        torch.cuda.synchronize()
        peaks[flag] = (fwd, torch.cuda.max_memory_allocated() - base)
    assert peaks["1"][0] < peaks["0"][0] and peaks["1"][1] < peaks["0"][1], peaks
