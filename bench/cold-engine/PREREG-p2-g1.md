# Preregistration — P2-G1: the promotion mechanics pay what the model says

Registered before implementation and before any measurement. The promotion
harness does not exist yet.

## What this gates

[`SPEC-elastic-phase2.md`](SPEC-elastic-phase2.md) §11: before any controller
is built, the bare mechanics — copy a DRAM expert up on a side stream, execute
it on GPU, retain it — must deliver the saving the calibrated model predicts.
G1 is that check, on real components end to end:

* the **same packed MXFP4 bytes** consumed by both tiers — the phase-2 CPU
  kernel (`gemv_mxfp4_grouped_cpu`) from the DRAM arena, and the
  oracle-adjudicated GPU kernel (`gemm_mxfp4_grouped`) after promotion;
* pinned staging, H2D on a **side stream**, first GPU use gated on the copy
  event (spec I2/I4);
* retention in a transient `DevRowCache` with `protected = rows − k` (I1).

## Design choice, registered: a no-reuse stream

The routing stream feeds **fresh experts every step**, so every promotion is
a pure copy-plus-execute with zero amortisation — the n\* = 1 regime, and the
conservative one. Gate E already established amortisation; G1 isolates the
mechanics. Retention is verified by **counters, not wall**: a targeted
re-invocation of a promoted expert must produce zero additional H2D
(`host_to_cache_rows` unchanged) and a cache hit.

## The arms

Per step, `m = 16` cold expert invocations at the `gptossish_gateup` shape.

* **Arm A (baseline):** all 16 execute on CPU via the phase-2 kernel.
* **Arm B (promote `p`):** `p ∈ {1, 2, 4, 8}` of them are promoted; the
  remaining `16 − p` execute on CPU concurrently — the CPU work is what the
  copies hide under.
* Paired, alternating, ≥ 5 repeats per point, medians.

At calibrated rates 16 CPU invocations ≈ 2 ms against tens of µs of GPU work,
so the CPU is the long pole by construction at every swept `p`.

## G1 — the registered prediction

Per-row predicted saving from **this box's own calibration** (run first,
`elastic_e3.py`, and gated: `n*_direct ∈ [2, 5]` un-hidden or the box is
rejected):

```
Δ = t_cpu_row − (1 − hide) · t_link_row − t_gpu_row
```

> Realised per-row saving — `(wall_A − wall_B) / p` — is **≥ 0.70 × Δ** at
> every swept `p`, on the medians.

* **Refuted** at any `p` — the promotion path has overheads the model missed
  (allocator, event, dispatch), and per the spec **the controller is not
  built until the mechanics pay**. Reported per-`p`.
* 0.70 is the slack for unmodelled contention; the E3b measurement already
  charges DRAM contention (7.8% joint inflation), so most of the gap is
  budgeted, not hoped away.

## Correctness, registered — no new tolerance is invented

* **Weight bytes:** every promoted row's device bytes are **identical**
  (byte-compare) to its DRAM source. Exact, every row, every repeat.
* **CPU outputs:** bit-exact against `ref_gemv_grouped`, the kernel's
  existing committed contract.
* **GPU outputs:** within the bound the GPU kernel's **own committed tests**
  assert against the fp32 reference, at the commit carrying this file.
  Referencing the existing contract rather than inventing a tolerance here.
* **Retention:** re-invoking a promoted expert produces a hit and zero H2D.

A failure of any correctness check voids the wall numbers for that run
entirely — a fast wrong answer is not a saving.

## Falsifiability, demonstrated before the real measurement is read

The registered spoiler arm: the identical harness with the side stream
replaced by **default-stream synchronous copies**. Its realised/predicted
ratio must fall **below** the 0.70 bar — the synchronous regime is the one
phase 0 showed loses, and if the instrument cannot distinguish it from the
side-stream arm, the instrument is not measuring the mechanics. If the
spoiler passes the bar, G1 is reported as uninformative and not scored.

## Harness validation, before trust

* `p = 0` reduces Arm B to Arm A: walls within repeat spread, zero H2D.
* Counter accounting: Arm B's H2D row count equals `p × steps` exactly on the
  no-reuse stream.
* The A/B alternates inside the repeat loop (thermal drift lands on both
  arms), the `run_r2_wall` convention.
* *(Amended in review, pre-data, no measurements taken)*: a 1 GiB untimed
  read is inserted before **each** timed arm, so neither arm inherits the
  other's L3/DRAM cache state — review found arm A's pass over the same
  expert rows would otherwise pre-warm arm B and bias the paired difference
  toward PASS.

## What would count as a miss

* G1 refuted at any `p` ⇒ reported per-`p`, controller deferred, the overhead
  profiled before anything else is attempted.
* Spoiler arm not failing ⇒ uninformative, no score.
* Any correctness failure ⇒ that run void; reported as a correctness bug, not
  a performance result.
* Box fails the calibration gate ⇒ failed run, nothing scored, new box.
* The box is destroyed when the run ends; receipts committed before it.

## Out of scope

The controller (G2), elasticity under pressure (G3), end-to-end wall (G4),
any reuse-bearing routing stream, and the e4b executor. One gate, mechanics
only.
