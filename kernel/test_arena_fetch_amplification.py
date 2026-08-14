# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""One layer forward must read each routed row ONCE.

Lives in its own file, apart from `test_arena_experts.py`, for one reason: that
file is on the packaging guard's `_NOT_IN_CI` allowlist as "needs CUDA", so
nothing in it runs on the CI runner. This gate needs no CUDA — the GEMM is
stubbed, because the question is how many times the BYTES are read — so it
belongs somewhere CI actually invokes. A gate in a file CI never runs is not a
gate.

What it catches: a row carries all six segments, so fetching per projection read
every row THREE times and threw two thirds of each read away — 842 MB where
281 MB is needed on a real K3 layer, confirmed to the byte by the reader's own
counter (gnf4#73). Every other test calls `fused_stacks` for a SINGLE
projection, where the behaviour is identical, which is why nothing caught it.
"""
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from arena_experts import ArenaExpertSource  # noqa: E402
from test_arena_experts import K, baked  # noqa: E402,F401  (fixture reuse)


def test_moe_layer_forward_reads_each_row_ONCE(baked, monkeypatch):  # noqa: F811
    torch = pytest.importorskip("torch")

    fake = types.ModuleType("mxfp4_grouped")

    def _stub(a_cat, blocks, scales, sizes, expert_ids, **kw):
        return torch.zeros(a_cat.shape[0], blocks.shape[1], dtype=a_cat.dtype)

    fake.gemm_mxfp4_grouped = _stub
    monkeypatch.setitem(sys.modules, "mxfp4_grouped", fake)
    from arena_experts import moe_layer_forward

    arena, _ = baked
    ids = [0, 2, 3]
    with ArenaExpertSource(arena) as src:
        calls = {"n": 0}
        real = src.fetch_raw

        def counting(layer, expert_ids):
            calls["n"] += 1
            return real(layer, expert_ids)

        monkeypatch.setattr(src, "fetch_raw", counting)
        before = src.reader.reads
        a_cat = torch.zeros(len(ids), K, dtype=torch.bfloat16)
        sizes = torch.ones(len(ids), dtype=torch.int32)
        moe_layer_forward(src, 1, a_cat, sizes, ids)

        assert calls["n"] == 1, (
            f"fetch_raw called {calls['n']}x for one layer -- each call re-reads "
            "every row; pass raw= to fused_stacks instead")
        assert src.reader.reads - before == len(ids), (
            f"read {src.reader.reads - before} rows for {len(ids)} experts; "
            "one row read per routed expert is the whole point")


def test_a_fetch_result_survives_the_next_fetch(baked):
    """Staging is reused, so a returned tensor must not alias it.

    `.to(device)` is a no-op when the tensor is already on the target, so on the
    DEFAULT `device="cpu"` the result would hand back the staging buffer itself
    and the next fetch of the same expert count would rewrite a caller's earlier
    result in place. The `torch.stack` path this replaced always allocated
    fresh.

    The existing reuse tests fetch the SAME rows twice and compare, which passes
    either way -- this fetches DIFFERENT rows, which is what makes it a gate.
    """
    torch = pytest.importorskip("torch")
    arena, _ = baked
    with ArenaExpertSource(arena) as src:
        first = src.fetch_raw(1, [0, 2])
        kept = {k: v.clone() for k, v in first.items()}
        src.fetch_raw(1, [1, 3])                      # same count, different rows
        for k, v in first.items():
            assert torch.equal(v, kept[k]), (
                f"{k}: the first fetch's tensor changed when the second ran -- "
                "it aliases the reused staging buffer")
