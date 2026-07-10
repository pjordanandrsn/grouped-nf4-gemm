#!/usr/bin/env python3
"""Shape census for the grouped W4A16 GEMM (Phase 0.2).

Emits shape_census.json: per-model fused-expert GEMM shapes (from the HF configs, the same
source the experts4bit-qlora loader reads) + tokens/expert statistics per regime. The census
— not intuition — picks Phase-2 tile configs; the skinny fine-grained shapes are the
battlefield (Marlin's documented failure mode).

Distributions are analytic at Phase 0 (uniform-routing binomial); Phase 1's harness replaces
them with measured routing histograms. Both are recorded under distinct keys so the
refinement is visible, not silent.
"""

import json
import math
from pathlib import Path

# (model, hidden H, moe intermediate I, experts E, top-k, layers L)
# From config.json of each repo (text_config where applicable), fetched 2026-07-10:
#   OLMoE-1B-7B-0924: hidden 2048, inter 1024, E 64, k 8, L 16
#   Qwen3-30B-A3B:    hidden 2048, moe_inter 768, E 128, k 8, L 48
#   gemma-4-26B-A4B:  hidden 2816, moe_inter 704, E 128, top_k_experts 8, L 30
#   gpt-oss-120b:     hidden 2880, inter 2880, E 128 (num_local_experts), k 4, L 36
MODELS = [
    ("allenai/OLMoE-1B-7B-0924", 2048, 1024, 64, 8, 16),
    ("Qwen/Qwen3-30B-A3B", 2048, 768, 128, 8, 48),
    ("google/gemma-4-26B-A4B", 2816, 704, 128, 8, 30),
    ("openai/gpt-oss-120b", 2880, 2880, 128, 4, 36),
]

PREFILL_S = 2048  # matches the published bench configs (seq_len 2048, packed)


def binom_quantile(n, p, q):
    """Quantile of Binomial(n, p) — tokens landing on one expert under uniform routing."""
    mu, var = n * p, n * p * (1 - p)
    # normal approx with continuity correction; exact enough for tile planning
    from math import erf, sqrt

    def cdf(k):
        return 0.5 * (1 + erf((k + 0.5 - mu) / sqrt(2 * var)))

    k = 0
    while cdf(k) < q and k < n:
        k += 1
    return k


def census():
    out = []
    for name, H, I, E, k, L in MODELS:
        gemms = {
            # transformers fused stacks: gate_up [E, 2I, H] -> N=2I, K=H ; down [E, H, I] -> N=H, K=I
            "gate_up": {"N": 2 * I, "K": H},
            "down": {"N": H, "K": I},
        }
        # per-expert token count M by regime (uniform-routing analytic; Phase-1 measures)
        n_draws = PREFILL_S * k  # routed (token, expert) assignments
        p = 1.0 / E
        regimes = {
            "decode_bs1": {
                "tokens_total": 1,
                "active_experts": k,
                "M_per_active_expert": 1,
                "note": "k experts, one token each — the skinny extreme; launch amortization is the design center",
            },
            "prefill_s2048": {
                "tokens_total": PREFILL_S,
                "M_mean": round(n_draws / E, 1),
                "M_p50": binom_quantile(n_draws, p, 0.50),
                "M_p95": binom_quantile(n_draws, p, 0.95),
                "M_p99": binom_quantile(n_draws, p, 0.99),
                "assumption": "uniform routing (analytic); replaced by measured histograms in Phase 1",
            },
            "train_microbatch": {
                "same_shape_as": "prefill_s2048",
                "note": "mb=1 seq 2048 packed (published bench config); backward stays dequant in v1",
            },
        }
        packed_gb = sum(g["N"] * g["K"] for g in gemms.values()) * E * L * 0.5 / 1e9
        absmax_gb = sum(g["N"] * g["K"] for g in gemms.values()) * E * L / 64 * 4 / 1e9
        out.append(
            {
                "model": name,
                "hidden": H,
                "moe_intermediate": I,
                "experts": E,
                "top_k": k,
                "layers": L,
                "per_expert_gemms": gemms,
                "regimes": regimes,
                "packed_nf4_gb": round(packed_gb, 2),
                "absmax_fp32_gb": round(absmax_gb, 3),
                "absmax_overhead_pct_of_packed": round(100 * absmax_gb / packed_gb, 1),
            }
        )
    return out


if __name__ == "__main__":
    data = {"generated_by": "census/make_census.py", "prefill_s": PREFILL_S, "models": census()}
    p = Path(__file__).parent / "shape_census.json"
    p.write_text(json.dumps(data, indent=2) + "\n")
    print(f"wrote {p}")
    for m in data["models"]:
        g = m["per_expert_gemms"]
        print(
            f"{m['model']:32} E={m['experts']:>3} k={m['top_k']} L={m['layers']:>2} "
            f"gate_up {g['gate_up']['N']}x{g['gate_up']['K']}  down {g['down']['N']}x{g['down']['K']}  "
            f"prefill M~{m['regimes']['prefill_s2048']['M_mean']}"
        )
