# Flagship B2 — VERDICT: NOT CONFIRMED (S2/S3 fail); the experiment measured why: Qwen3-235B expert stickiness is 0.43–0.45, and speculation at that hit rate is bandwidth-negative

**2026-07-14 · Protocol:** `kernel/prereg_flagship_phaseB2.json` (OTS pre-data)
· **Frozen:** `e765cad` · **Host:** RunPod SECURE H100 80GB (fresh pod,
link measured on-box: **45.4 GB/s** → waterfall **5.69 tok/s** — the
Phase-B B1 gap is closed; this harness now adjudicates its own ceiling).

## The design under test

While layer L computes, speculatively stream layer L+1's 8 experts using
**last token's routing for that layer**; on router resolve, fetch only the
mispredicted experts into their slots (slot-permutation absorbed by
slot-aligned router weights).

## Registered criteria

| criterion | outcome |
|---|---|
| S1 identity (ON greedy tokens ≡ OFF for the paired prefix, every prompt) | **PASS** — 3/3 identical. Speculation changed *when* bytes moved, never the math. |
| S2 paired speedup (ON ≥ 1.05× OFF) | **FAIL** — ON is **0.73–0.76×** OFF (4.25–4.32 → 3.16–3.28 tok/s) |
| S3 waterfall (ON ≥ 0.80 × 5.69 = 4.55) | **FAIL** — 3.16–3.28 |
| S4 VRAM ≤ 20 GB | **PASS** — 15.2 GB |
| REPORT hit rate | **0.452 / 0.428 / 0.435** across the three prompts |

## Why it lost — the arithmetic closes

**Measured for the first time here: a production 235B MoE's token-to-token
expert persistence at k=8/128 is ~44%** (per-layer, greedy decode, three
prompts, ~26k routing decisions). With hit rate H, naive full-k speculation
makes the PCIe link carry the speculative k experts *plus* (1−H)·k
corrections — **(2−H)× the baseline bytes**. At H = 0.44 that is 1.56×,
predicting ~0.64× throughput before overlap credit; measured 0.73–0.76×
(slightly better than pure-bytes because speculative copies do overlap
compute the baseline serializes). On a link-saturated pipeline,
**mispredicted speculation is not free — the wrong bytes occupy the same
link the corrections need.** Break-even for this design needs H ≳ 0.6–0.7.

A compounding implementation choice, owned here: corrections for layer L
were queued on the same copy stream *behind* layer L+1's speculative set, so
the stalled layer waited for ~85 MB of next-layer speculation before its own
~47 MB of fixes. A priority-stream fix would help — but the bytes arithmetic
caps even the fixed version at ~5.69/1.56 ≈ 3.65 tok/s < the 4.30 tok/s
no-speculation baseline. **At 44% stickiness, full-k speculation loses on
fundamentals, not implementation.** No re-run (no-tune clause).

## What stands after B2

- The no-speculation pipeline remains the shipped configuration:
  **4.25–4.32 tok/s real-checkpoint generation on 15.2 GB VRAM** (~0.76× the
  on-box waterfall; the residual gap is the router-serialization cost B
  registered).
- The identity machinery (slot-mapped speculative buffers) is correct and
  kept — it is the substrate any future prefetch variant needs.
- **The measured 44% stickiness is the design input the next attempt was
  missing.** Two registered directions that respect the bandwidth-neutrality
  law: (1) **early routing** — compute layer L+1's router on L+1's
  *pre-attention* hidden (available immediately after L's MoE); if its top-8
  agreement with the true post-attention router is high (same token, same
  residual stream — plausibly ≫ 44%), the copy overlaps attention with near
  zero wasted bytes; (2) **partial speculation** — stream only experts whose
  measured per-slot persistence clears the (2−H) break-even, padding the
  rest after resolve.

## Evidence / teardown

`phaseB2.json` (per-prompt off/on tok/s, hit rates, identity flags, texts,
link microbench, VRAM), `phaseB2.log`. Pod DELETE → 404-verified, 0 pods.
