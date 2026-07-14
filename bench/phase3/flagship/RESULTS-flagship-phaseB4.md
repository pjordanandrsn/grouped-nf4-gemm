# Flagship B4 — VERDICT: NOT CONFIRMED (0.57×); Python threading is the wrong substrate, and the prefetch problem is now fully characterized

**2026-07-14 · Protocol:** `kernel/prereg_flagship_phaseB4.json` (OTS pre-data)
· **Frozen:** `c584f96` · **Host:** RunPod SECURE H100 (link 51.8 GB/s →
waterfall 6.49). Smoke-at-4-layers passed; zero wasted builds.

## Result

| prompt | off → early2 | paired | agreement | identical |
|---|---|---|---|---|
| all three | 4.35–4.36 → 2.47–2.51 | **0.568–0.577×** | 0.926–0.931 | True 3/3 |

F1 identity PASS, F4 VRAM PASS (15.2), **F2/F3 FAIL** — and much worse than
B3's inline version (0.87–1.07×).

## Diagnosis

The issuer thread removed the main thread's *CUDA* syncs but replaced them
with **GIL contention**: per layer, the worker's `event.synchronize()` +
32 small `copy_` launches interleave with the main thread's ~50 launch
calls, chopping the launch run-ahead far more aggressively than B3's single
added sync did. Uniform 0.57× across prompts (vs B3's noisy 0.87–1.07×)
says this is structural overhead, not scheduling luck. Python threads
trade a sync tax for a GIL tax; at ~2.4 ms/layer budgets, both lose.

## The prefetch problem is now fully characterized — three measured failure modes, two measured constants

| attempt | mechanism | constant measured | outcome |
|---|---|---|---|
| B2 speculation | last-token routing | stickiness **44%** | 0.73× — bandwidth inflation (2−H) |
| B3 early routing | pre-attn router, inline | agreement **93%** | 0.87–1.07× — +94 CUDA syncs/token |
| B4 early routing | pre-attn router, threaded | (same 93%) | 0.57× — GIL contention |

The predictor is right (93%); the bytes are nearly free (1.07×); every
CPU-mediated issuance mechanism loses more than the overlap gains. The
conclusion is architectural: **the copy must be GPU-driven end-to-end** — a
gather kernel that reads the pinned host store directly over UVA/PCIe
(zero-copy), indexed by the GPU-resident early-router ids, with no CPU in
the per-layer loop at all. That also deletes the 32-small-memcpy pattern.
It is a real kernel-engineering task (B5, not attempted tonight), and it is
the only remaining shape for this idea — everything else has been measured
out.

## Standing state

Shipped configuration: **prefetch OFF, 4.30–4.36 tok/s** (now replicated on
four pods), 15.2 GB VRAM, ~0.63–0.76× of per-box waterfall ceilings with the
gap fully attributed (router serialization + launch-loop costs). The
93%-agreement predictor and the identity-preserving slot machinery are
proven components awaiting a GPU-driven issuer.

## Evidence / teardown

`phaseB4.json`, `phaseB4_smoke.json`. Pod DELETE → 404-verified, 0 pods.
