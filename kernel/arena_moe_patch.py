"""Patch a Kimi-K3-lineage MoE block to compute from an NVMe arena.

`KimiSparseMoeBlock.moe_infer` already does the hard part for us. It sorts
tokens by routed expert and counts them per expert:

    idxs = topk_ids.view(-1).argsort()
    sorted_tokens = x[idxs // topk_ids.shape[1]]     # group-sorted [T, K]
    tokens_per_expert = cnts.sum(dim=0)              # per-group counts

...and then runs a **Python loop with one matmul per active expert**, each of
which must first materialise that expert's weights. That is the exact shape
`gemm_mxfp4_grouped` exists to collapse: one launch across all active experts,
computing on the packed bytes.

So this module swaps the loop, not the routing. Sorting, weighting, unsorting
and the shared-expert path stay upstream's — patching those would change the
model. What changes is where expert weights come from (an arena row, read on
demand) and how they are multiplied (packed, no dequantize).

Deliberately NOT handled here, because they are the caller's:
  * `use_latent_moe` up/down projections around the routed block
  * shared experts and the residual
Both live in `forward`, outside `moe_infer`, and are untouched.
"""

from __future__ import annotations

import torch

from arena_experts import ArenaExpertSource, moe_layer_forward


def _arena_moe_infer(self, x, topk_ids, topk_weight):
    """Drop-in for KimiSparseMoeBlock.moe_infer, arena-backed.

    Mirrors upstream's sort/unsort exactly; only the expert matmuls change.
    """
    src = self._arena_source
    layer = self._arena_layer

    cnts = topk_ids.new_zeros((topk_ids.shape[0], self._arena_n_experts))
    cnts.scatter_(1, topk_ids, 1)
    tokens_per_expert = cnts.sum(dim=0)
    idxs = topk_ids.view(-1).argsort()
    sorted_tokens = x[idxs // topk_ids.shape[1]]

    counts = tokens_per_expert.tolist()
    active = [(i, n) for i, n in enumerate(counts) if n > 0]
    if not active:
        outs = sorted_tokens.new_empty(0)
    else:
        expert_ids = [i for i, _n in active]
        sizes = [n for _i, n in active]
        self._arena_calls += 1
        self._arena_experts_read += len(expert_ids)
        outs = moe_layer_forward(src, layer, sorted_tokens, sizes, expert_ids)

    # unsort + combine: upstream's arithmetic, verbatim
    new_x = torch.empty_like(outs)
    new_x[idxs] = outs
    return (new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype))


def enable_arena_experts(model, arena: str, *, qd: int = 16,
                         source: ArenaExpertSource = None, verbose: bool = True):
    """Route every routed-MoE layer's expert matmuls through `arena`.

    Returns the ArenaExpertSource. **A patch count is not a call count** — this
    reports how many blocks were rebound, which says nothing about whether the
    kernel ever ran. Use :func:`arena_call_stats` after a forward pass to check
    that it did; a block whose `moe_infer` is shadowed by a subclass would
    patch cleanly and never fire.
    """
    src = source or ArenaExpertSource(arena, qd=qd, device="cuda")
    arena_layers = set(src.layers)
    patched, skipped = [], []

    layers = getattr(getattr(model, "model", model), "layers", None)
    if layers is None:
        raise AttributeError(
            "no `.layers` on the model or `model.model`; pass the language "
            "model itself (K3 wraps it as `language_model`)")

    for i, layer in enumerate(layers):
        blk = getattr(layer, "block_sparse_moe", None)
        if blk is None:                      # dense layer (first_k_dense_replace)
            continue
        if i not in arena_layers:
            skipped.append(i)
            continue
        blk._arena_source = src
        blk._arena_layer = i
        blk._arena_n_experts = len(getattr(blk, "experts", [])) or src.n_experts
        blk._arena_calls = 0
        blk._arena_experts_read = 0
        blk._arena_orig_moe_infer = blk.moe_infer
        blk.moe_infer = _arena_moe_infer.__get__(blk, type(blk))
        patched.append(i)

    if skipped and verbose:
        print(f"arena: {len(skipped)} MoE layers NOT in the arena "
              f"(first: {skipped[:4]}) — those still use resident experts")
    if not patched:
        raise RuntimeError(
            "patched 0 MoE blocks: no layer had `block_sparse_moe`, or none "
            "matched the arena's layers. Was the arena baked for this model?")
    if verbose:
        print(f"arena: patched {len(patched)} MoE layers "
              f"({patched[0]}..{patched[-1]}), {src.n_experts} experts each")
    return src


def disable_arena_experts(model) -> int:
    """Restore upstream `moe_infer`. Returns how many blocks were restored."""
    layers = getattr(getattr(model, "model", model), "layers", [])
    n = 0
    for layer in layers:
        blk = getattr(layer, "block_sparse_moe", None)
        if blk is not None and hasattr(blk, "_arena_orig_moe_infer"):
            blk.moe_infer = blk._arena_orig_moe_infer
            del blk._arena_orig_moe_infer
            n += 1
    return n


def arena_call_stats(model) -> dict:
    """Did the arena path actually run? `patched` counts rebound blocks;
    `calls` counts blocks that executed. calls == 0 with patched > 0 means the
    patch is inert — the failure mode a patch count cannot see."""
    layers = getattr(getattr(model, "model", model), "layers", [])
    patched = calls = experts = 0
    for layer in layers:
        blk = getattr(layer, "block_sparse_moe", None)
        if blk is not None and hasattr(blk, "_arena_orig_moe_infer"):
            patched += 1
            calls += blk._arena_calls
            experts += blk._arena_experts_read
    return {"patched": patched, "calls": calls, "experts_read": experts}
