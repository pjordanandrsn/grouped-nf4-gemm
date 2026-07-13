# Confirmatory v4 — VERDICT: NOT CONFIRMED (P-axis passes clean; F-axis fails on a measurement-boundary artifact; E1 by one instance-cell)

**Date:** 2026-07-13 · **Frozen code:** `80bda12` · **Protocol:** `kernel/prereg_v4_confirmatory.json` (OTS pre-data; exploratory calibrations committed pre-stamp) · **Reducer:** `bench/phase1/reduce_confirmatory_v4.py` · **Evidence:** `bench/phase1/confirmatory_v4/`

v4 tested the post-v3 queue: the min-bytes dispatch floor (+ `fused_routed`
product backend), the split-K work floor, the universal-constant revert, and
the group-size-keyed prefill config. Fresh post-stamp A5000
(`x1ht322rvrk7zz`, driver 550.127.08, torn down + 404-verified) + home A2000;
per device: suite once, then 3 decode runs (census + floor/split dev cells)
and 3 prefill runs (census), n=3 fresh-process.

## Registered criteria — outcomes

| criterion | outcome |
|---|---|
| P1 prefill config (paired ≥1.15 on ≥6/8 per device) | **PASS** — 16/16 cells at **1.22–1.74×**, both devices |
| P2 prefill floor (vs dequant ≥0.75 on ≥6/8 A5000, ≥4/8 A2000) | **PASS** — A5000 6/8, A2000 5/8 |
| P2b prefill wins (≥1.3 on ≥2/8, A5000) | **PASS** — Qwen down 1.86×, gemma down 1.91× |
| F1 no-catastrophe (routed vs dequant ≥0.55 everywhere) | **FAIL** — Switch dn 0.47/0.36, Switch gu 0.64/0.39, Scout dn 0.48 (A5000) |
| F2 floor identity (Switch cells ∈ [0.75, 1.35], A5000) | **FAIL** — 0.47 / 0.64 |
| F3 eligible identity (paired routed vs fused ∈ [0.85, 1.18]) | **FAIL** — 0.57–0.95 across eligible cells, both devices |
| E1 census energy (routed < dequant, 8/8 per device) | **FAIL on A5000** — 7/8; gpt-oss down at 1.194 on this instance (its routed-vs-dequant speed was 0.67× the same day). A2000 8/8. |
| Q1 suite 44/44 both devices | **PASS** |
| **V4_CONFIRMED** | **FALSE** |

## The F-axis finding: dispatch is load-time state, and a per-call benchmark boundary manufactures a regression

`fused_routed` was written as a per-call product path: every invocation
recomputes `decode_dispatch(...)` (plus `get_device_properties`, list
construction, and a nested backend call) before launching the same kernel
the `fused_nf4` backend launches directly. That python costs **~40–100 µs
per call** — invisible at prefill, but 15–40% of a 100–500 µs decode cell.
The result: the F3 "identity" bands failed everywhere, and on the ~50–100 µs
Switch cells the wrapper alone pushed routed-vs-dequant to 0.36–0.64 even
though routed IS the dequant path there.

This is the op-boundary lesson for the third time (v1: fixture assembly
charged to the kernel; v3: latency-bound cells only support paired claims;
v4: **path dispatch must not be timed per call, because no real integration
pays it per call**). MoE expert shapes are static at model load — an
integration calls `decode_dispatch()` once per layer and caches the branch.
The registered criteria bound a per-call implementation, the per-call
implementation is the wrong product shape, and the criteria correctly
failed it. The correct product claim (floor cells run the dequant path;
eligible cells run the fused kernel; dispatch amortized to zero) is
**by-construction** once dispatch is hoisted to load time — a v5 harness
backend with per-stack dispatch caching would measure it, and is queued.

The mechanism is visible in the numbers: the F3 "overhead ratio" reconstructs
to a roughly constant ~40–100 µs per call across cells of very different
sizes — a python floor, not a kernel effect.

## What v4 confirmed (the P-axis, clean on both devices)

The group-size-keyed M-tile config (128/128/w8/s3 for m ≥ 128-row groups,
64-row tiles at w4/s2; sweep basis committed) is **blind-confirmed**:

- Paired vs the retired config: **1.22–1.74× on all 16 prefill cells**
  (A5000 1.22–1.73, A2000 1.23–1.74) — the full sweep-predicted gain
  transferred to fresh runs on both devices.
- Absolute: down-projections at **1.86×/1.91×** the dequant path on the
  A5000 (Qwen3-30B, gemma-4), Qwen gate_up at parity (0.99); the weakest
  cells remain OLMoE/gemma gate_up (0.35–0.49 on the A2000, 0.39–0.80
  A5000) — the mainloop rewrite stays the path to full prefill parity.

Also reported (no criteria): the universal-constant revert showed no
regressions anywhere (as v3 predicted — noise-equivalent on the A2000), and
the split work-floor kept census cells split-free while Scout/Hunyuan down
retained their split plans.

## The cursed cell, sixth reading

gpt-oss `down` (2880×2880) vs dequant at decode: **1.75 → 1.03 → 1.00 →
1.47 → 0.69 → 0.67** across six instance-contexts on functionally identical
kernels. On this pod it took E1 down with it (energy 1.194 follows speed).
The cell is listed in the README as instance-sensitive; v4 adds another
point to that curve and nothing about the v4 changes touches it (its paired
ratios were ~1.0 throughout).

## Evidence

`bench/phase1/confirmatory_v4/` — 12 rep JSONs (2 devices × {3 decode, 3
prefill}), suite logs, states, `reduction_v4.json`, `SHA256SUMS`. A2000
incident disclosed: the leg first launched during peak home-service hours
and its suite OOM'd (6 cells, co-resident VRAM pressure — vLLM + evening
SDXL/voice-tts held ~9.7 GB); the run was killed unread, deferred behind a
VRAM gate, and re-ran clean (44/44) once ≥8.4 GB was free. The A5000 leg was
unaffected.

## Queue after v4

1. **v5: load-time dispatch in the harness** (per-stack cached branch —
   measures the product shape; expected to convert F1/F2/F3 to
   by-construction passes).
2. Prefill mainloop rewrite (gate_up parity; the config ceiling is reached).
3. sm_120 (Phase 4).
