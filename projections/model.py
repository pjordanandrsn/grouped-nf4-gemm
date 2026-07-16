# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Cross-vendor decode-throughput projection model for grouped-nf4-gemm.

Two regimes, two formulas — the arithmetic lifted verbatim from the flagship
receipts (bench/phase3/flagship/), where the streaming form is measured-anchored
to <2% on two model sizes and two link speeds.

  STREAMING (discrete GPU, experts stream from host RAM over the link):
      tok/s = link_GBps / bytes_per_token
    The per-token expert bytes must cross the host link every token; at bs=1
    decode the compute hides under the copy, so the link is the ceiling.

  UNIFIED (APU/SoC, weights resident in shared memory — Strix Halo, GB10, Thor):
      tok/s = mem_bw_GBps / (active_params_B * bytes_per_param * (1 - reuse))
    No streaming; each token reads the active weights from shared memory, so
    the memory bandwidth over the active-weight bytes is the ceiling.

In BOTH regimes NF4 vs bf16 is a 4x byte reduction on the binding resource
(0.5 vs 2.0 bytes/param) — modeled explicitly, not folded in.

TIER: every output of this model is 'projected' (R3) until a confirmatory
passes on that platform. Streaming has measured anchors (test_anchors.py);
unified has NO measured anchor yet — its rows carry higher uncertainty and
must say so.
"""

NF4_BYTES_PER_PARAM = 0.5
BF16_BYTES_PER_PARAM = 2.0
_ABSMAX_BYTES = 4          # fp32 blockwise absmax
_BLOCKSIZE = 64            # one absmax per 64 NF4 elements (locked e4b design)


def streaming_waterfall_toks(link_gbps: float, gb_per_token: float) -> float:
    """Discrete-GPU offload ceiling. Reproduces the published flagship anchors
    exactly (44.3 GB/s, 7.98 GB/tok -> 5.55; 55.5, 12.09 -> 4.59)."""
    return link_gbps / gb_per_token


def gb_per_token(layers: int, topk: int, hidden: int, inter: int,
                 bytes_per_param: float = NF4_BYTES_PER_PARAM) -> float:
    """Routed-expert bytes moved per decode token for a fused-MoE geometry:
    gate_up [2*inter, hidden] + down [hidden, inter] per active expert, times
    topk active experts, times layers. fp32 blockwise absmax added for NF4.
    Reproduces the measured 7.98 GB/token for Qwen3-235B-A22B geometry."""
    def wb(N: int, K: int) -> float:
        b = N * K * bytes_per_param
        if abs(bytes_per_param - NF4_BYTES_PER_PARAM) < 1e-9:
            b += N * (K // _BLOCKSIZE) * _ABSMAX_BYTES
        return b
    per_expert = wb(2 * inter, hidden) + wb(hidden, inter)
    return layers * topk * per_expert / 1e9


def effective_bytes_per_param(bytes_per_param: float) -> float:
    """Real bytes read per weight param, INCLUDING NF4's fp32 blockwise absmax
    (one fp32 per 64 elems = 0.0625 B/param). This is why the NF4-vs-bf16 byte
    reduction is ~3.56x, not a round 4x — absmax overhead is carried, not hidden."""
    if abs(bytes_per_param - NF4_BYTES_PER_PARAM) < 1e-9:
        return bytes_per_param + _ABSMAX_BYTES / _BLOCKSIZE
    return bytes_per_param


def unified_waterfall_toks(mem_bw_gbps: float, active_params_b: float,
                           bytes_per_param: float = NF4_BYTES_PER_PARAM,
                           reuse: float = 0.0) -> float:
    """Unified-memory ceiling. active_params_b in BILLIONS of activated params
    per token (e.g. 22 for Qwen3-235B-A22B). reuse in [0,1) discounts weight
    re-reads within a token (0 = conservative worst case). Absmax-inclusive.
    NO measured anchor yet — projected tier only."""
    denom = active_params_b * effective_bytes_per_param(bytes_per_param) * (1.0 - reuse)
    return mem_bw_gbps / denom


def project(regime: str, **kw) -> float:
    if regime == "streaming":
        return streaming_waterfall_toks(kw["link_gbps"], kw["gb_per_token"])
    if regime == "unified":
        return unified_waterfall_toks(kw["mem_bw_gbps"], kw["active_params_b"],
                                      kw.get("bytes_per_param", NF4_BYTES_PER_PARAM),
                                      kw.get("reuse", 0.0))
    raise ValueError(f"unknown regime {regime!r}")
