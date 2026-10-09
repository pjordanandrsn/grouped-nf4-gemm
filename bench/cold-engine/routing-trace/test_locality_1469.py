"""Tests for locality_1469's window census on traces whose answers are known by hand."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "kernel"))
sys.path.insert(0, HERE)

from locality_1469 import churn, row_bytes, windows_of  # noqa: E402


def _trace(steps):
    """recs as score_policies.load returns them: {'step': i, 'routed': {layer: [ids]}}."""
    return [{"step": i, "routed": {str(L): ids for L, ids in enumerate(layers)}} for i, layers in enumerate(steps)]


def test_a_repeated_topk_has_a_window_union_of_k_and_no_churn():
    recs = _trace([[[1, 2], [3, 4]]] * 32)
    for w in (1, 16, 32):
        per_layer = windows_of(recs, 2, w)
        assert all(u == 2 for v in per_layer.values() for u in v)
    assert churn(recs, 2, 2) == 0.0


def test_disjoint_topk_every_token_unions_to_w_times_k_and_full_churn():
    recs = _trace([[[2 * i, 2 * i + 1]] for i in range(32)])
    assert windows_of(recs, 1, 16) == {0: [32, 32]}             # 16 tokens x 2 distinct experts, two windows
    assert windows_of(recs, 1, 1) == {0: [2] * 32}
    assert churn(recs, 1, 2) == 1.0


def test_windows_never_take_a_partial_tail():
    recs = _trace([[[0, 1]]] * 20)
    assert len(windows_of(recs, 1, 16)[0]) == 1                 # 20 tokens: one full window of 16, tail dropped


def test_a_row_is_gate_up_down_in_nf4_with_an_fp32_absmax_per_64():
    h, i = 2048, 1024                                           # OLMoE-1B-7B
    n = 3 * h * i
    assert row_bytes(h, i) == n // 2 + n // 64 * 4 == 3_538_944
