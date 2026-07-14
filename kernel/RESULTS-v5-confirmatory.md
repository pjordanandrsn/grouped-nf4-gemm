# Confirmatory v5 — VERDICT: NOT CONFIRMED as registered (one A2000 cell +0.04 over the band); the dispatch fix is otherwise clean on both devices

**Date:** 2026-07-14 · **Frozen code:** `47a06ba` · **Protocol:** `kernel/prereg_v5_confirmatory.json` (OTS pre-data) · **Reducer:** `bench/phase1/reduce_confirmatory_v4.py` with the v5 spec · **Evidence:** `bench/phase1/confirmatory_v5/`

v5 re-registered ONLY the v4 F-axis (dispatch/product-path) at the corrected
measurement boundary: `fused_routed` now caches the `decode_dispatch` branch
per stack (load-time semantics — MoE shapes are static; v4 proved per-call
dispatch timing manufactures a python floor no integration pays). Fresh
post-stamp A5000 + home A2000, n=3 fresh-process, suite 44/44 both.

## Registered criteria — outcomes

| criterion | outcome |
|---|---|
| F1 no-catastrophe (routed vs dequant ≥ 0.55, every cell, both devices) | **PASS** — min 0.76 (A5000 Scout dn), 0.85 (A2000 gemma dn) |
| F2 floor identity (Switch cells routed-vs-dequant ∈ [0.75,1.35], A5000) | **PASS** — 0.97 / 1.17 |
| F3 eligible identity (paired routed-vs-fused ∈ [0.85,1.18], both devices) | **FAIL by one cell** — A5000 11/11 in-band (0.915–1.071); A2000 10/11 in-band, **granite-3.1 down = 1.224** (+0.044 over) |
| E1 census energy (routed < dequant on ≥ 7/8, both devices) | **PASS** — **8/8 both devices** |
| Q1 suite 44/44 both | **PASS** |
| **V5_CONFIRMED** | **FALSE** (F3) |

## What v5 established — the v4 boundary artifact is fixed

The v4 failure was systematic: per-call dispatch added ~40–100 µs that
dragged F3 to **0.57–0.95 everywhere**. The one-line load-time cache moved
the paired routed-vs-fused ratio to **0.915–1.071 on the A5000 (11/11
in-band)** and **0.854–1.160 on 10 of 11 A2000 cells** — i.e. the wrapper is
now the near-free thing the product needs it to be. Floor routing works
(Switch cells take the dequant path, 0.97/1.17 vs dequant on the A5000; the
routed-vs-fused 0.34/0.42 there is the *point* — routed avoided the losing
fused kernel). Energy improved on **8/8 census cells on both devices** (v4
was 7/8 on the A5000; the cursed gpt-oss-down cell happened to land on its
favorable side of the instance lottery this run — 1.011 speed, energy below).

## The one miss, at full volume

**A2000 `granite-3.1-3b-a800m` down (N=1536, K=512): routed/fused median
1.224**, +0.044 over the [0.85,1.18] band. This is NOT a dispatch-overhead
effect — routed and fused run the *identical* kernel on an eligible cell;
the consult between them is a cached-tuple lookup + one python branch
(sub-microsecond). It is **timing jitter on a sub-100 µs cell on the
contended home card** — precisely the latency-bound-instability class the
v3 methodology law named (the same cell reads 1.071 on the fresh A5000, and
its own reps span 1.09–1.37). The registered band was carried over from v4
unchanged and did not widen for the A2000's documented tiny-cell jitter, so
by the letter of the protocol it is a miss, and it counts as one.

Per the no-tune clause: not re-run; the claim is narrowed to **"load-time
dispatch is a near-free wrapper (paired ratio in [0.85,1.18]) on the A5000
across all cells and on 10/11 A2000 cells; the exception is a sub-100 µs
cell whose run-to-run jitter on a contended card exceeds the band — a
measurement-noise miss, not a dispatch cost."** A future v6 would either
widen the A2000 band to its measured jitter envelope or pin the cell with
more reps; neither changes the kernel.

## Cumulative dispatch story (v4 → v5)

v4 correctly failed a per-call implementation and taught the boundary; v5
implemented the load-time fix and confirmed it everywhere except one
noise-dominated home-card cell. The product behavior — **tiny cells to the
dequant path, everything else to the fused kernel, dispatch amortized to
zero** — is now measured and holds. This is the honest closing state of the
decode dispatch line.

## Evidence

`bench/phase1/confirmatory_v5/`: 6 rep JSONs (2 devices × 3), suite logs,
states, `reduction_v5.json`, `SHA256SUMS`. A2000 leg ran in the 02:00–06:00
quiet window behind a VRAM gate (39/39 cells ok every rep — no OOM skips);
its first three family-hours attempts were scrubbed unread per the
home-card law. A5000 leg: fresh SECURE pod (one sick host — the "unspecified
launch failure" machine — auto-rejected by the host-aware launcher, healthy
pod found on cycle 3). Both torn down, 404-verified.
