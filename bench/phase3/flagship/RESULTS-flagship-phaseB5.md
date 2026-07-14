# Flagship B5 — VERDICT: NOT CONFIRMED (G2 1.015×, G3 0.84×); the prefetch program is CLOSED by an exact wire law, and `gpu` mode is the new best configuration

**2026-07-14 · Protocol:** `kernel/prereg_flagship_phaseB5.json` (OTS pre-data)
· **Frozen:** `3305a83` (+ disclosed smoke-gate fix `82f069b`) · **Host:**
RunPod SECURE H100 80GB HBM3 (on-box link **45.0 GB/s** → waterfall
**5.63 tok/s** — the slowest-linked of the five flagship pods; every ratio
shifts accordingly).

## The design under test

The copy is a **kernel**: `kernel/host_gather.py` reads the pinned host
expert store directly over UVA (zero-copy PCIe) from a triton kernel, indexed
by **GPU-resident** router ids — the CPU never knows the ids and never
synchronizes. Two arms vs the same OFF baseline: **`gpu`** = true
post-attention ids, gather serialized on the main stream (removes the
baseline's ~94 router syncs/token and its 32-memcpy/layer pattern);
**`gpu_early`** = pre-attention early-router ids gathered speculatively on a
prefetch stream overlapping attention, with a post-attention correction
gather that re-fetches only mispredicted slots.

## Results (94 layers, 3 prompts, off 64 / on 128 greedy tokens, one process)

| prompt | off | gpu | gpu_early | paired gpu | paired gpu_early | hit (positional) |
|---|---|---|---|---|---|---|
| MoE-vs-dense | 4.343 | 4.409 | 3.640 | **1.015×** | **0.838×** | 0.776 |
| haiku | 4.329 | 4.390 | 3.655 | **1.014×** | **0.844×** | 0.798 |
| quantization | 4.330 | 4.400 | 3.684 | **1.016×** | **0.851×** | 0.790 |

| criterion | outcome |
|---|---|
| G1 identity (both arms × 3 prompts) | **PASS 6/6** — byte-identical greedy text |
| G2 `gpu` ≥ 1.05× off | **FAIL** (1.014–1.016×) |
| G3 `gpu_early` ≥ 1.15× off and ≥ 0.80× waterfall | **FAIL** (0.838–0.851×; 0.646–0.654× waterfall) |
| G4 VRAM ≤ 20 GB | **PASS** (15.16 GB) |

Waterfall fractions: off **0.771×**, gpu **0.782×**, gpu_early 0.646–0.654×.
Off baseline 4.33–4.34 — the **fifth pod** inside the 4.30–4.36 replication
band.

## What was learned (each item at full volume)

**1. The sync-tax hypothesis is REFUTED as the residual gap.** B3 attributed
the ~23% off-vs-waterfall gap to router serialization + per-layer syncs. `gpu`
mode deletes *every* added sync and *every* per-layer memcpy launch — and buys
**+1.5%**. The baseline's entire sync + launch overhead is ~1.5% of token
time; the remaining ~22% is compute (attention/router/MoE) that a serialized
schedule cannot hide under the wire.

**2. The wire law closes the prefetch program.** Speculation at hit rate H
moves (2−H)× the bytes (B2's law, now at its best-case H). Measured
positional hit 0.776–0.798 → predicted gpu_early/gpu = 1/(2−H) =
**0.817–0.832**; observed **0.826–0.837** — the bytes law predicts the loss
to ~1%. On a pipeline whose wire already runs at ~78% duty there is only
~22% idle link for speculation to exploit, while (2−H) inflates traffic by
~21%: near-exact cancellation, minus stream contention → guaranteed small
loss. Break-even needs H ≳ 0.95; the best predictor this model offers is
0.93 set-agreement (0.78–0.80 positional). **No speculative prefetch can win
on this pipeline.** B2 (bandwidth, H=0.44) → B3 (sync tax) → B4 (GIL tax) →
B5 (wire law at the predictor's ceiling, with zero implementation excuses
left): the idea is measured out, end to end.

**3. `gpu` mode is the new best configuration.** 4.39–4.41 tok/s, +1.5% over
prefetch-off, byte-identical output, and architecturally cleaner: no
per-layer cudaMemcpyAsync launches, no GPU→CPU id syncs, the token loop
enqueues bounded work and runs ahead. Small, real, replicated across all
three prompts.

**4. The UVA zero-copy substrate is validated.** A triton kernel bitcasting
an int64 to a global pointer (`tl.cast(..., tl.pointer_type(...))`) reads
cudaHostAlloc'd memory over PCIe at ≥ memcpy-engine throughput (`gpu` ≥ off
proves it end-to-end at 7.98 GB/token). Reusable primitive; smoke-validated
before the full build, as registered.

**5. Positional accounting behaved exactly as registered:** 0.78–0.80
positional vs the known 0.926–0.931 set-agreement — sorted-alignment shift
undercounts set intersection, and the reported number equals the PCIe
traffic actually saved.

## Disclosed deviations

The first smoke FAILED (identity break + CUDA illegal access in `gpu_early`):
the speculative gather could execute before the main-stream `pred_ids_buf`
write landed — reading stale ids, or the −1 generation-reset (out-of-bounds
host read). Fixed by `pred_ready` events ordering the prefetch stream after
the ids write (`82f069b`, zero added CPU syncs); smoke2 re-gated identity
before the full build. `gpu` passed identity in the *failed* smoke too — the
primitive was never in question. The adjudicated full run executed **once**.
Smoke1 log kept in evidence (`phaseB5_smoke.log`).

## Standing state after the flagship program

Ship **`gpu` mode**: 235B-A22B real-checkpoint decode at **4.39–4.41 tok/s**
on a 15.2 GB VRAM working set (H100 + 45 GB/s link; 0.78× waterfall). The
fused NF4 grouped GEMM remains the sole MoE compute. Prefetch: **closed,
negative, with the constants to prove it** (stickiness 0.44, agreement 0.93,
sync tax 1.5%, wire law (2−H)).

## Evidence / teardown

`phaseB5.json`, `phaseB5_smoke2.json`, `phaseB5_smoke.log` (failed smoke1),
`phaseB5_smoke2.log`, `phaseB5.log`, `SHA256SUMS.phaseB5`. Pod
`s7kphbzgn5gs3i` DELETE → 404-verified, **0 pods remaining**; hardcap
sleeper killed by PID.
