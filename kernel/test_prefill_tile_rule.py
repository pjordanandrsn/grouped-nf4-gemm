"""The prefill M-tile height rule: ``cost`` (the default) minimises tiles x (D + BLOCK_M) over the actual group sizes;
``max`` (GNF4_PREFILL_TILE_RULE=max) keys on the largest group, so one hot expert no longer puts every group on 128-row tiles.
The rule is a speed choice: under either rule the launch computes the same product (checked on CUDA)."""
import pytest
import torch

import nf4_grouped as ng
from nf4_grouped import _prefill_block_m, _prefill_block_m_cost, _prefill_tile_rule

CUDA = torch.cuda.is_available()


def _brute(sizes, d):
    """The rule restated the slow way, for the comparison."""
    best = None
    for bm in (16, 32, 64, 128):
        c = sum(-(-int(r) // bm) for r in sizes) * (d + bm)
        if best is None or c < best[0] or (c == best[0] and bm > best[1]):
            best = (c, bm)
    return best[1]


@pytest.mark.parametrize("sizes", [[24] * 127 + [400], [1] * 64, [5, 500], [64] * 128, [256] * 128, [3, 17, 33, 65, 129, 1000],
                                   [24, 30, 18, 77, 26, 22, 41, 9] * 16])
@pytest.mark.parametrize("d", [16.0, 64.0, 96.0, 128.0, 512.0])
def test_cost_rule_is_the_argmin_it_claims(sizes, d):
    assert _prefill_block_m_cost(sizes, d) == _brute(sizes, d)


def test_one_hot_expert_no_longer_sets_every_tile():
    """Training-sized routing: ~24 rows per expert and one hot one. ``max`` takes 128 for all; ``cost`` does not."""
    sizes = [24] * 127 + [400]
    assert _prefill_block_m(max(sizes)) == 128
    assert _prefill_block_m_cost(sizes, 96.0) in (32, 64)


def test_large_groups_keep_the_tall_tile():
    """Prefill-sized groups: both rules land on 128."""
    sizes = [256] * 128
    assert _prefill_block_m(max(sizes)) == 128 and _prefill_block_m_cost(sizes, 96.0) == 128


def test_ties_go_to_the_taller_tile_and_d_moves_the_pick():
    assert _prefill_block_m_cost([16] * 8, 0.0) == 16              # at D 0 only padding counts, and 16-row tiles pad nothing
    assert _prefill_block_m_cost([128] * 8, 0.0) == 128            # tiles x bm = 1024 at every height: the tie goes tall
    assert _prefill_block_m_cost([100] * 10, 0.0) == 16            # cheap tiles: the least padding wins
    assert _prefill_block_m_cost([100] * 10, 1000.0) == 128        # dear tiles: the fewest tiles win


def test_rule_env(monkeypatch):
    monkeypatch.delenv("GNF4_PREFILL_TILE_RULE", raising=False)
    assert _prefill_tile_rule() == "cost"                         # the default since TC1 amendment 14's 5090 A/B
    monkeypatch.setenv("GNF4_PREFILL_TILE_RULE", "max")
    assert _prefill_tile_rule() == "max"
    monkeypatch.setenv("GNF4_PREFILL_TILE_RULE", "fastest")
    with pytest.raises(ValueError, match="expected max | cost"):
        _prefill_tile_rule()
    monkeypatch.setenv("GNF4_PREFILL_TILE_D", "512")
    assert _prefill_block_m_cost([20] * 100 + [200]) == _brute([20] * 100 + [200], 512.0)


@pytest.mark.skipif(not CUDA, reason="the fused kernel needs CUDA")
def test_both_rules_compute_the_same_product(monkeypatch):
    from nf4_pack_ref import quantize_pack_nf4
    torch.manual_seed(0)
    E, N, K = 16, 256, 512
    p, a = quantize_pack_nf4(torch.randn(E * N, K) * 0.02)
    B, A = p.reshape(E, N, K // 2).cuda(), a.reshape(E, N, K // 64).float().cuda()
    sizes = [24] * 15 + [150]
    eids = list(range(E))
    x = torch.randn(sum(sizes), K, device="cuda", dtype=torch.bfloat16)
    monkeypatch.setenv("GNF4_PREFILL_TILE_RULE", "max")
    y_max = ng.gemm_4bit_grouped(x, B, A, sizes, eids)
    monkeypatch.setenv("GNF4_PREFILL_TILE_RULE", "cost")
    y_cost = ng.gemm_4bit_grouped(x, B, A, sizes, eids)
    assert _prefill_block_m(max(sizes)) != _prefill_block_m_cost(sizes)   # the two rules really launch different tiles here
    assert torch.equal(y_max, y_cost)
    before = dict(ng.PREFILL_BM_STATS)
    ng.gemm_4bit_grouped(x, B, A, sizes, eids)                            # rule still cost: the counter records its height
    after = ng.PREFILL_BM_STATS
    assert {k: after[k] - before.get(k, 0) for k in after if after[k] != before.get(k, 0)} == {_prefill_block_m_cost(sizes): 1}
