# Flagship bnb-CUDA-dequant baseline — the registered prediction is REFUTED: the standard path does NOT hide under the PCIe shadow. The fused advantage is architectural: 2.3× speed, 2.2× energy vs bnb's own CUDA dequant

**2026-07-14 · Protocol:** `kernel/prereg_flagship_bnb_baseline.json` (OTS
pre-data) · **Frozen:** `dc86375` · **Host:** FRESH RunPod SECURE H100
(2015 GB RAM; on-box link 55.1 GB/s → waterfall 6.91 tok/s), A-B-C-A
fresh-process sequence, 94L/E128/k8 synthetic Phase-A pipeline, pynvml
energy over the timed window.

## What was registered

Phase A measured the torch-dequant path at 34% of waterfall and the fused
kernel at 102–103%. The open attribution question: implementation (slow
torch LUT decode) or architecture (dequantize-then-matmul itself)? We
registered the prediction **(X2) that bnb's CUDA dequant kernel + cuBLAS
would ALSO hide under the ~180 ms/token copy shadow (≥ 0.90× waterfall)** —
a prediction that, if confirmed, would have narrowed our own speed claim to
the torch comparison and re-based the flagship advantage on energy/VRAM
alone. It was run blind either way.

## Result

| mode | tok/s | % of waterfall | VRAM | J/token | mean W |
|---|---|---|---|---|---|
| fused (run 1) | **6.466** | **93.6%** | 13.63 GB | **26.8** | 173 |
| **bnb_dequant** | **2.773** | **40.2%** | 13.69 GB | **59.1** | 164 |
| dequant (torch, Phase-A continuity) | 1.800 | 26.0% | 13.94 GB | 213.0 | 383 |
| fused (run 2, drift guard) | 6.461 | 93.5% | 13.63 GB | 26.8 | 173 |

| criterion | outcome |
|---|---|
| X1 self-check (staged bytes ↔ bnb layout) | **PASS** (gated in-run) |
| **X2 registered prediction (bnb ≥ 0.90× waterfall)** | **REFUTED** — 0.402× |
| X3 energy (fused ≤ 0.90× bnb J/token) | **PASS** — 0.453× |
| X4 drift (fused runs within 5%) | **PASS** — 0.1% |
| X5 VRAM | fused 13.63 ≤ bnb 13.69 (transient materialization visible, small) |

## What it means

**The dequant→matmul gap at offload scale is architectural, not an
implementation artifact.** Swapping the torch LUT decode for bnb's native
CUDA kernel recovers only 26% → 40% of waterfall. The per-layer compute
chain of the standard path — k×2 dequant kernel launches + k×2 GEMMs +
their materialization traffic (~3.3 ms/layer against a ~1.9 ms/layer copy
shadow) — exceeds the shadow, so compute, not the wire, sets the token
rate. The fused single-launch path (~0.35 ms/layer of MoE compute) hides
completely and rides the wire at 93–94%.

The flagship comparison, now against the strongest standard comparator:
**2.33× tokens/s and 2.21× J/token versus bnb's own CUDA dequant on the
identical pipeline** (and 3.6× / 7.9× versus the torch path). Phase A's
torch-dequant reading replicates (1.81 → 1.80 tok/s on a different pod).
A side observation carried for completeness: the torch path burns 383 W
(many small ops keep the SMs hot) versus bnb-dequant's 164 W — the torch
baseline was not merely slow, it was hot; bnb's kernel is power-efficient
but still can't close the launch/materialization wall-clock.

One honest caveat: this pod's fused fraction reads 93.5–93.6% of waterfall
(Phase A read 102–103% on a slower 44.3 GB/s link). On this faster link the
same serialized non-MoE costs occupy a larger fraction of the shorter token
budget — consistent with the B3 finding that fixed costs weigh more as
links get faster. Both readings stand; the fraction is link-dependent, the
ordering is not.

## Evidence / teardown

`bench/phase3/flagship/bnb_baseline/`: `bnbbl_{fused,bnb_dequant,dequant,fused2}.json`,
`bnbbl_bnb_dequant.log` (self-check line), `BNBBL_STATE`, `SHA256SUMS`.
Pod `4g6t03hfg0pqgx` DELETE → 404-verified, **0 pods remaining**.
