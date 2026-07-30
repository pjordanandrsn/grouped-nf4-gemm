"""Gates for the MoE-block patch.

The stand-in below reproduces `KimiSparseMoeBlock`'s real interface as shipped
in modeling_kimi_linear.py -- `moe_infer(x, topk_ids, topk_weight)`, the
argsort/`tokens_per_expert` sort, the per-expert loop, and the unsort/weight/sum
tail. Testing against a made-up interface would prove nothing about the patch.

The load-bearing gate is `calls > 0`. A patch count is not a call count: both
prior inert fixes in this project's history patched cleanly and never executed.
"""
import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))

from arena_moe_patch import (arena_call_stats, disable_arena_experts,  # noqa: E402
                             enable_arena_experts)


class FakeSparseMoeBlock(nn.Module):
    """Upstream's moe_infer, verbatim in structure."""

    def __init__(self, n_experts, hidden, inter):
        super().__init__()
        self.experts = nn.ModuleList(
            [nn.Linear(hidden, hidden, bias=False) for _ in range(n_experts)])
        self.ep_rank, self.experts_per_rank = 0, n_experts

    @torch.no_grad()
    def moe_infer(self, x, topk_ids, topk_weight):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]
        tpe = tokens_per_expert.cpu().numpy()
        outputs, start = [], 0
        for i, n in enumerate(tpe):
            if n == 0:
                continue
            outputs.append(self.experts[i](sorted_tokens[start:start + n]))
            start += n
        outs = torch.cat(outputs, 0) if outputs else sorted_tokens.new_empty(0)
        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        return (new_x.view(*topk_ids.shape, -1).type(topk_weight.dtype)
                .mul_(topk_weight.unsqueeze(-1)).sum(1).type(new_x.dtype))


class FakeLayer(nn.Module):
    def __init__(self, moe=None):
        super().__init__()
        if moe is not None:
            self.block_sparse_moe = moe


class FakeModel(nn.Module):
    """Layer 0 dense (first_k_dense_replace), 1..3 routed — K3's shape."""

    def __init__(self, n_routed=3, n_experts=6, hidden=64, inter=32):
        super().__init__()
        inner = nn.Module()
        inner.layers = nn.ModuleList(
            [FakeLayer()] +
            [FakeLayer(FakeSparseMoeBlock(n_experts, hidden, inter))
             for _ in range(n_routed)])
        self.model = inner


class StubSource:
    """Stands in for ArenaExpertSource: same surface, no disk."""

    def __init__(self, layers, n_experts=6):
        self._layers, self._n = list(layers), n_experts

    @property
    def layers(self):
        return self._layers

    @property
    def n_experts(self):
        return self._n


def test_patches_only_routed_layers_present_in_the_arena():
    m = FakeModel()
    enable_arena_experts(m, "unused", source=StubSource([1, 2, 3]), verbose=False)
    s = arena_call_stats(m)
    assert s["patched"] == 3, s          # layer 0 is dense and must be skipped
    assert not hasattr(m.model.layers[0], "block_sparse_moe")


def test_layers_absent_from_the_arena_are_left_alone():
    """A partial bake must not silently route missing layers to nothing."""
    m = FakeModel()
    enable_arena_experts(m, "unused", source=StubSource([1, 2]), verbose=False)
    assert arena_call_stats(m)["patched"] == 2
    assert not hasattr(m.model.layers[3].block_sparse_moe,
                       "_arena_orig_moe_infer")


def test_refuses_when_nothing_matches():
    m = FakeModel()
    with pytest.raises(RuntimeError, match="patched 0 MoE blocks"):
        enable_arena_experts(m, "unused", source=StubSource([40, 41]),
                             verbose=False)


def test_refuses_a_model_without_layers():
    with pytest.raises(AttributeError, match="no `.layers`"):
        enable_arena_experts(nn.Linear(2, 2), "unused",
                             source=StubSource([1]), verbose=False)


def test_disable_restores_upstream_exactly():
    m = FakeModel()
    before = m.model.layers[1].block_sparse_moe.moe_infer
    enable_arena_experts(m, "unused", source=StubSource([1, 2, 3]), verbose=False)
    assert m.model.layers[1].block_sparse_moe.moe_infer is not before
    assert disable_arena_experts(m) == 3
    assert m.model.layers[1].block_sparse_moe.moe_infer == before
    assert arena_call_stats(m)["patched"] == 0


def test_call_stats_start_at_zero_so_inertness_is_visible():
    """patched > 0 with calls == 0 is exactly the inert-patch signature; the
    stats must be able to express it rather than implying success."""
    m = FakeModel()
    enable_arena_experts(m, "unused", source=StubSource([1, 2, 3]), verbose=False)
    s = arena_call_stats(m)
    assert s["patched"] == 3 and s["calls"] == 0 and s["experts_read"] == 0


def test_patched_block_calls_the_arena_and_counts_it():
    """The positive control for the above: a forward must move calls off zero."""
    m = FakeModel()
    seen = {}

    def fake_forward(src, layer, a_cat, sizes, expert_ids, **kw):
        seen["layer"], seen["sizes"] = layer, list(sizes)
        seen["ids"] = list(expert_ids)
        return torch.zeros(a_cat.shape[0], a_cat.shape[1],
                           dtype=a_cat.dtype, device=a_cat.device)

    import arena_moe_patch
    orig = arena_moe_patch.moe_layer_forward
    arena_moe_patch.moe_layer_forward = fake_forward
    try:
        enable_arena_experts(m, "unused", source=StubSource([1, 2, 3]),
                             verbose=False)
        blk = m.model.layers[2].block_sparse_moe
        T, H, k = 4, 64, 2
        x = torch.randn(T, H)
        topk_ids = torch.tensor([[0, 1], [1, 2], [0, 2], [1, 2]])
        topk_w = torch.full((T, k), 0.5)
        out = blk.moe_infer(x, topk_ids, topk_w)
    finally:
        arena_moe_patch.moe_layer_forward = orig

    assert out.shape == (T, H)
    assert seen["layer"] == 2, seen          # the block's own index, not 0
    assert sum(seen["sizes"]) == T * k       # every routed slot accounted for
    assert seen["ids"] == sorted(seen["ids"])
    assert 0 not in [n for n in seen["sizes"]], "empty groups must be dropped"
    s = arena_call_stats(m)
    assert s["calls"] == 1 and s["experts_read"] == len(seen["ids"])


# ---------------------------------------------- geometry invariants (K3) -----
# The experts contract over `routed_expert_hidden_size` (3584), NOT
# `hidden_size` (7168): "Stable LatentMoE" down-projects OUTSIDE the experts, in
# forward(), before moe_infer is reached. Patching forward() instead of
# moe_infer would hand the arena a 7168-wide activation against 3584-wide
# weights. Measured from the real release:
#     w1/w3 [3072, 1792] -> N=moe_inter 3072, K=3584
#     w2    [3584, 1536] -> N=3584,           K=moe_inter 3072
K3_REAL = {"hidden": 7168, "latent": 3584, "inter": 3072,
           "w1": (3072, 1792), "w1s": (3072, 112),
           "w2": (3584, 1536), "w2s": (3584, 96)}


def test_k3_expert_geometry_contracts_over_the_latent_dim():
    """If this ever reads `hidden` the patch is attached at the wrong level."""
    assert K3_REAL["w1"][1] * 2 == K3_REAL["latent"], "gate/up K must be latent"
    assert K3_REAL["w1"][0] == K3_REAL["inter"]
    assert K3_REAL["w2"][1] * 2 == K3_REAL["inter"], "down K must be moe_inter"
    assert K3_REAL["w2"][0] == K3_REAL["latent"], "down N returns to latent"
    assert K3_REAL["w1"][1] * 2 != K3_REAL["hidden"], (
        "gate/up contracted over hidden -> patched above the down-projection")


def test_scale_columns_are_K_over_32_on_both_projections():
    """A group-size slip shows up here before it shows up as garbage output."""
    assert K3_REAL["w1s"][1] == K3_REAL["latent"] // 32 == 112
    assert K3_REAL["w2s"][1] == K3_REAL["inter"] // 32 == 96


def test_patch_passes_the_activation_through_untouched():
    """moe_infer's `x` is already down-projected; the patch must not reshape or
    re-project it, only route it to the kernel."""
    import arena_moe_patch
    m = FakeModel()
    got = {}

    def spy(src, layer, a_cat, sizes, expert_ids, **kw):
        got["K"] = a_cat.shape[-1]
        got["ptr"] = a_cat.data_ptr()
        return torch.zeros(a_cat.shape[0], a_cat.shape[1], dtype=a_cat.dtype)

    orig = arena_moe_patch.moe_layer_forward
    arena_moe_patch.moe_layer_forward = spy
    try:
        enable_arena_experts(m, "unused", source=StubSource([1, 2, 3]),
                             verbose=False)
        blk = m.model.layers[1].block_sparse_moe
        x = torch.randn(4, 64)
        blk.moe_infer(x, torch.tensor([[0, 1], [1, 2], [0, 2], [1, 2]]),
                      torch.full((4, 2), 0.5))
    finally:
        arena_moe_patch.moe_layer_forward = orig
    assert got["K"] == 64, "activation width changed before reaching the kernel"
