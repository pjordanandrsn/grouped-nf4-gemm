#!/usr/bin/env python3
"""Routing-faithful activation groups, from this repo's own measured histograms.

Every leg so far has used one of two fictions:

  `decode_m*`      exactly `top_k` groups of M rows each. A real forward at 32
                   tokens spreads the SAME rows over ~58 of 64 experts with
                   1-17 rows each (sampled from the measured distribution;
                   operator datum for OLMoE at training shape is 57 of 64).
  `tokbudget_*`    all E experts with EQUAL counts. Occupancy is right at
                   T >= 512 (measured 1.000) but the counts are not: the
                   measured OLMoE histogram runs 31 to 795 against a uniform
                   256, cv 0.506.

Both fictions move BOTH arms and in opposite ways, which is why the correction
has to be measured rather than reasoned about:

  * the dequant-on-forward arm pays one `dequantize_4bit` per HIT expert, so it
    goes from `top_k` calls to ~58;
  * the fused arm goes from `top_k` groups to ~58 groups, and phase-1 found
    `n_groups` is the grouped path's cost knob -- which is precisely the
    insensitivity the fused single-launch design claims.

WHERE THE ROUTING COMES FROM. `bench/phase1/results/routing_olmoe.json` and
`routing_qwen.json`, measured on the real models with the gate hooked and a
per-expert bincount. Only those two census models have measured routing;
Gemma-4 was gated and GPT-OSS-120B needed an 80 GB card, so those are recorded
NOT-RUN rather than given invented routing.

Assignments are drawn per token, `top_k` distinct experts per token (top-k is
without replacement within a token), weighted by the measured per-expert
frequency of the chosen layer. Seeded, so a cell is reproducible.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

ROUTING_FILES = {
    "OLMoE": "routing_olmoe.json",
    "Qwen3-30B": "routing_qwen.json",
}


def routing_for(model: str, results_dir: Path):
    """Return (per_layer_counts, E, k) or None when the model has no measured
    routing. None means NOT-RUN, never 'substitute something plausible'."""
    for key, fname in ROUTING_FILES.items():
        if key.lower() in model.lower():
            p = Path(results_dir) / fname
            if not p.exists():
                return None
            d = json.loads(p.read_text())
            return d["per_layer_counts"], d["E"], d["k"]
    return None


def sample_group_sizes(counts, tokens, top_k, seed=0):
    """Draw `tokens` tokens, each hitting `top_k` DISTINCT experts weighted by
    the measured frequencies. Returns {expert_id: rows}, empties dropped."""
    total = sum(counts)
    p = [c / total for c in counts]
    idx = list(range(len(counts)))
    rng = random.Random(seed)
    hit: dict[int, int] = {}
    for _ in range(tokens):
        chosen: set[int] = set()
        # top-k is without replacement WITHIN a token; rejection sampling keeps
        # the marginal weighting while enforcing that.
        guard = 0
        while len(chosen) < top_k and guard < top_k * 200:
            chosen.add(rng.choices(idx, weights=p, k=1)[0])
            guard += 1
        for e in chosen:
            hit[e] = hit.get(e, 0) + 1
    return dict(sorted(hit.items()))


def routed_groups(spec, tokens, results_dir, device, seed=0, layer=None,
                  act_seed=7):
    """Groups as `make_activations` returns them -- [(expert_id, A[M,K])] -- but
    with measured occupancy and skew. Raises LookupError when the model has no
    measured routing, so a caller records NOT-RUN instead of inventing one."""
    import torch

    r = routing_for(spec.model, results_dir)
    if r is None:
        raise LookupError(f"no measured routing for {spec.model}")
    per_layer, E, k = r
    if E != spec.E or k != spec.top_k:
        raise LookupError(
            f"routing E={E} k={k} != census E={spec.E} k={spec.top_k}")
    counts = per_layer[len(per_layer) // 2] if layer is None else per_layer[layer]
    sizes = sample_group_sizes(counts, tokens, k, seed=seed)

    g = torch.Generator(device="cpu").manual_seed(act_seed)
    out = []
    for e, m in sizes.items():
        a = (torch.randn(m, spec.K, generator=g, dtype=torch.float32) * 0.5).to(
            device=device, dtype=torch.bfloat16)
        out.append((e, a))
    return out


def summarise(groups, E):
    rows = [a.shape[0] for _, a in groups]
    n = len(rows)
    mean = sum(rows) / n
    var = sum((x - mean) ** 2 for x in rows) / n
    return {"hit_experts": n, "of": E, "occupancy": n / E,
            "total_rows": sum(rows), "min_rows": min(rows),
            "max_rows": max(rows), "mean_rows": mean,
            "cv": (var ** 0.5) / mean if mean else None}
