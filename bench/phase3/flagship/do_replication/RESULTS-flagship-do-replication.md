# DRAFT — Flagship replication on DigitalOcean (Phase A + Phase B, first non-RunPod provider)

> **DRAFT / addendum-hold.** Receipts banked in this directory
> (`flag_*.json`, `phaseB_do.json`); not stamped, not linked from the
> flagship RESULTS docs until reviewed. EXPLORATORY tier — Phase A here is
> **qualified** by a suite-environment anomaly disclosed below.

All six prior flagship hosts were RunPod virtual pods. These two runs put the
pipeline on a different provider (DigitalOcean GPU droplets, `gpu-h200x1-141gb`,
image `gpu-h100x1-base`), hosts #6 (Phase A) and #7 (Phase B).

## Phase A — synthetic-weight A-B-C-A (H200, link 55.5 GB/s → waterfall 6.94 tok/s)

| mode | tok/s | fraction of waterfall |
|---|---|---|
| fused (A) | 6.49 | **93.5%** |
| bnb_dequant (B) | 5.67 | **81.7%** |
| torch dequant (C) | 1.82 | 26.3% |
| fused (A, re-run) | 6.49 | 93.5% |

- **Fused replicates**: 93.5% of the per-box waterfall, A-B-C-A bracketing
  stable to 0.03% — same fraction as the RunPod 55 GB/s pod (93.5–93.6%).
- **Torch dequant replicates**: 1.82 tok/s ≈ the 1.80–1.81 seen on every host.
- **The bnb-CUDA gap compresses on faster compute.** On the RunPod H100 the bnb
  baseline reached only 40% of waterfall (fused 2.33× tok/s); on this H200 it
  reaches 81.7% (fused 1.14×). Direction is exactly what the architecture
  predicts: the bnb path is compute-bound (per-expert dequant+GEMM must fit
  under the copy shadow), so faster compute hides more of it. The *ordering*
  (fused > bnb > torch) and the fused path's ceiling fraction are
  arch-invariant; the *margin* over bnb is a function of the box's
  compute-to-link ratio. The registered H100 result stands as registered; this
  is the second point on that curve, not a contradiction.
- **Suite anomaly (disclosure).** The property suite on this droplet ran
  18/44: every failure is `AttributeError: triton.language has no attribute
  'gather'` — the DO image ships a pre-3.4 triton with its preinstalled torch,
  and the run bootstrap only reinstalls torch when CUDA is broken, so the stale
  pair survived. `tl.gather` is the v6 register-LUT **prefill** mainloop; the
  26 failures are exactly the V1-mainloop-dependent tests (boundaries +
  prefill-config), and the 18 passes are the decode path — the only path Phase
  A times. The four timed modes all ran rc=0. Treat these Phase-A numbers as
  replication-grade for decode-mode ordering and fractions, with the suite
  anomaly open until a pinned-stack rerun (`pip install -U "triton>=3.4"` —
  runner fix queued: verify `tl.gather` at bootstrap, not just
  `torch.cuda.is_available()`).

## Phase B — real 438 GB checkpoint, coherent decode (H200, link 55.4 GB/s → waterfall 6.93 tok/s)

Real Qwen3-235B-A22B-Instruct-2507, stream-quantized to NF4 in place,
generated coherent text through the fused kernel (`phaseB_do.json`):

| config | tok/s | notes |
|---|---|---|
| prefetch-off | **4.42–4.45** | 0.64× per-box ceiling; identity 6/6 |
| gpu (resident ids) | 4.40–4.43 | ×0.994–0.995 ≈ parity, as on RunPod |
| gpu_early (speculative) | 3.70–3.73 | ×0.835–0.843 at hit 0.79–0.80 |

- **VRAM peak 15.2 GB** — identical to all five RunPod Phase-B hosts.
- **The wire law (2−H) replicates on a new provider to ~1%**: measured hit
  0.79–0.80 → predicted slowdown ×1/(2−H) = 0.833–0.826; observed 0.835–0.843.
- **The router-serialization tax grows with link speed, again**: off-mode =
  0.64× ceiling here (55 GB/s box) vs 0.76× on the 45 GB/s RunPod pod —
  matching the B3 observation on RunPod's 55 GB/s pod (0.63×).
- Download→first-token entirely on DO infrastructure; total wall ~55 min
  (DO's ingress makes the 438 GB pull ~20–30 min).

## Ops note

Same-night lane incidents (watcher destroying SSH-dark-but-healthy droplets
during large downloads; stale DO API reads mislabeling a provisioning droplet)
are recorded in the ops log; the lane watchers now follow the autopsy law —
**never destroy on SSH-fail alone; consult the API, and cross-check the lane
log before judging a droplet wedged.**
