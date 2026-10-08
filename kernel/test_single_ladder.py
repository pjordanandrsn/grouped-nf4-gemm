"""NF4_QLORA_SINGLE_LADDER=1: the single padded block's group count and width on ``_ladder_up``'s rungs.

Against the flag off the bar is ROUNDING (``test_lora_delta_pad_buckets._rounding_close``): every real row gets the same
arithmetic, but the ``bmm``s run at other shapes. Unset and ``0`` must be the single block op for op and bit for bit. The
point of the flag, that the products' shapes repeat across routings, is checked on the Zipf top-8 routings the bucket tests
use, at the field recipe's token count.
"""
import pytest
import torch

import nf4_qlora
from nf4_qlora import lora_delta_grouped
from test_lora_delta_pad_buckets import (CASES, CUDA, DEVICES, DTYPES, _OpLog, _fresh, _inputs, _rounding_close, _run,
                                         _zipf_top8_sizes)


def _single(monkeypatch, inputs, sizes, eids, flag, scaling=2.0, extra=None):
    monkeypatch.setenv("NF4_QLORA_LORA_PATH", "padded")
    monkeypatch.setenv("NF4_QLORA_PAD_BUCKETS", "0")
    for k, v in (extra or {}).items():
        monkeypatch.setenv(k, v)
    if flag is None:
        monkeypatch.delenv("NF4_QLORA_SINGLE_LADDER", raising=False)
    else:
        monkeypatch.setenv("NF4_QLORA_SINGLE_LADDER", flag)
    _fresh()
    before = nf4_qlora.SINGLE_LADDER_STATS["calls"]
    res = _run(lora_delta_grouped, inputs, sizes, eids, scaling)
    return res, nf4_qlora.SINGLE_LADDER_STATS["calls"] - before


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("sizes,eids", CASES)
@pytest.mark.parametrize("act_dtype,ad_dtype", DTYPES)
def test_ladder_matches_the_single_block(monkeypatch, dev, sizes, eids, act_dtype, ad_dtype):
    """Values and every gradient equal the single block's to rounding, unique and repeated ids alike; the call is counted with
    its laddered rows."""
    inputs = _inputs(dev, sizes, eids, act_dtype, ad_dtype)
    ref, n0 = _single(monkeypatch, inputs, sizes, eids, "0")
    rows0 = nf4_qlora.SINGLE_LADDER_STATS["rows_laddered"]
    got, n1 = _single(monkeypatch, inputs, sizes, eids, "1")
    rows = [s for s in sizes if s]
    up = nf4_qlora._ladder_up
    assert (n0, n1) == (0, 1)
    assert nf4_qlora.SINGLE_LADDER_STATS["rows_laddered"] - rows0 == up(len(rows)) * up(max(rows))
    for name, x, y in zip(("out", "d_a", "d_A", "d_B"), got, ref):
        _rounding_close(x, y, name)


@pytest.mark.parametrize("sizes,eids", CASES[:3])
def test_unset_and_zero_are_the_single_block_op_for_op(monkeypatch, sizes, eids):
    inputs = _inputs("cpu", sizes, eids, torch.bfloat16, torch.float32)
    seen = {}
    for flag in (None, "0", "1"):
        with _OpLog() as log:
            res, n = _single(monkeypatch, inputs, sizes, eids, flag)
        seen[flag] = (log.ops, res, n)
    assert seen[None][2] == seen["0"][2] == 0 and seen["1"][2] == 1
    assert seen[None][0] == seen["0"][0]
    for name, x, y in zip(("out", "d_a", "d_A", "d_B"), seen[None][1], seen["0"][1]):
        assert torch.equal(x, y), name


def test_ladder_repeats_shapes_across_routings():
    """Over 40 Zipf(1) top-8 routings of 512 tokens, the single block's distinct (G, widest) shapes against the laddered
    (G', width) ones. 512 is about one forward of TC1's field recipe: experts4bit-qlora's tc1-5090-137 read a median of 1,552
    real and 586 padded tokens a step, over four forwards."""
    up = nf4_qlora._ladder_up
    plain, laddered = set(), set()
    for seed in range(40):
        rows = [s for s in _zipf_top8_sizes(tokens=512, seed=seed) if s]
        plain.add((len(rows), max(rows)))
        laddered.add((up(len(rows)), up(max(rows))))
    print(f"\ndistinct single-block shapes over 40 routings: {len(plain)} plain, {len(laddered)} laddered")
    assert len(laddered) * 3 <= len(plain)


def test_the_flag_leaves_buckets_compact_and_the_previous_body_alone(monkeypatch):
    """Bucketed calls, the compact single-block node and NF4_QLORA_LEAN_DELTA=0's body never take the ladder."""
    sizes, eids = CASES[4]
    inputs = _inputs("cpu", sizes, eids, torch.bfloat16, torch.float32)
    for extra in ({"NF4_QLORA_PAD_BUCKETS": "1"}, {"NF4_QLORA_COMPACT_DELTA": "1"}, {"NF4_QLORA_LEAN_DELTA": "0"}):
        monkeypatch.setenv("NF4_QLORA_LORA_PATH", "padded")
        monkeypatch.setenv("NF4_QLORA_SINGLE_LADDER", "1")
        for k, v in extra.items():
            monkeypatch.setenv(k, v)
        _fresh()
        before = nf4_qlora.SINGLE_LADDER_STATS["calls"]
        _run(lora_delta_grouped, inputs, sizes, eids, 1.0)
        assert nf4_qlora.SINGLE_LADDER_STATS["calls"] == before, extra
        for k in extra:
            monkeypatch.delenv(k)


@pytest.mark.skipif(not CUDA, reason="the plan memo is CUDA-only")
def test_ladder_plan_memo_is_keyed_apart(monkeypatch):
    """A laddered single-block plan (its `flat` built at the rung's width) and an unladdered one never share a memo entry."""
    sizes, eids = CASES[4]
    monkeypatch.setenv("GNF4_HOST_REUSE", "1")
    inputs = _inputs("cuda", sizes, eids, torch.bfloat16, torch.float32)
    _single(monkeypatch, inputs, sizes, eids, "0")
    (k0,) = nf4_qlora._PLAN_MEMO
    monkeypatch.setenv("NF4_QLORA_SINGLE_LADDER", "1")
    _run(lora_delta_grouped, inputs, sizes, eids, 1.0)
    (k1,) = nf4_qlora._PLAN_MEMO
    assert k0[-1] != "single-ladder" and k1[-1] == "single-ladder"
