# Phase 2 v1.1 — a failed optimization iteration (reverted; v1 stands)

**2026-07-13** · A5000 SECURE pod (sm_86, triton 3.4.0) · ~$0.5 · pod
404-verified. Recorded as a negative so the same three changes aren't retried
blind. v1 (`nf4_grouped.py` @ 809b395) remains the shipped kernel.

## What was tried, all at once

Three "improvements" to the v1 kernel, motivated by the v1 open items (wide-N
decode gap, prefill parity):

1. **`tl.interleave` byte-dedup** — load the packed tile once as `[BN, BK/2]`
   and split hi/lo nibbles, replacing v1's per-element `kk//2` gather (which
   loads each byte twice).
2. **`triton.autotune`** on both paths (BLOCK_N/K/warps/stages, keyed on N,K),
   to close the wide-N decode gap without hand-picking.
3. **bf16-MMA prefill mainloop** (decode reduction unchanged) — decode the
   weight tile to fp32, absmax-scale, downcast once to bf16 for `tl.dot`,
   replacing v1's TF32 dot. Motivated by "bf16 tensor cores are full-rate on
   sm_86, TF32 is half-rate" → expected prefill speedup.

## Result: regressed on every axis

| axis | v1 | v1.1 | verdict |
|---|---|---|---|
| property suite | 35/35 | **34/35** — `test_all_codes_both_nibble_positions` fails | correctness regression |
| P-fid (m=128, M-tile path) | 0.61–0.79× dequant | **1.00×** | the "more accurate" win destroyed |
| decode (GPT-OSS gate_up) | 0.318 ms | **0.615 ms** | ~2× slower |
| decode (Qwen gate_up) | 0.285 ms | 0.565 ms | ~2× slower |
| prefill (OLMoE gate_up) | 17.7 ms | **25.9 ms** | slower |

**Root cause, per change:**

- **bf16-MMA killed P-fid, and it was the whole point of the kernel.** Rounding
  the weight tile to bf16 for the MMA is the *same* rounding the dequant path's
  global-memory materialization does — so fused error vs fp64 collapses from
  0.61–0.79× the dequant path to exactly 1.00× (measured, m=128). The registered
  P-fid claim ("fused ≤ dequant, and *stronger* because fp32 accumulation") only
  holds because v1 keeps fp32/TF32 inputs. bf16 inputs forfeit it. And it broke
  the exact-decode test (one-hot GEMM no longer bit-exact at bf16). **The
  fidelity edge is load-bearing and dtype-bought; do not trade it for MMA rate.**
- **bf16-MMA didn't even buy prefill speed** — 25.9 vs 17.7 ms. At these
  tiny-M grouped shapes the mainloop is memory/decode-bound, not MMA-rate-bound,
  so the extra per-tile downcast + the `tl.interleave`/tail-mask overhead cost
  more than the faster tensor core saved. The prefill bottleneck is *not* the
  dot dtype.
- **autotune picked worse configs than the hand-tuned BLOCK_N=128** for the
  decode path — the config set / key wasn't right, and the winner it cached was
  ~2× the v1 hand-pick on several cells. Autotune is not free here; the search
  space needs to actually contain and select the hand-tuned point.

## Disposition

Reverted entirely — v1 is strictly better on correctness, fidelity, and speed.
The v1 open items stand and need *different* approaches than this bundle:

- **Prefill parity** is not a dtype problem. Profile where v1's fp32 mainloop
  actually spends time at M≈256 (likely the per-element nibble gather + no
  K-pipelining) before touching the dot. Keep fp32/TF32 inputs to preserve
  P-fid; a bf16 variant, if ever added, must be a *separate* opt-in path that
  the fidelity gate is allowed to hold to a looser (documented) bound.
- **Wide-N decode gap** (GPT-OSS gate_up vs gemv): retune BLOCK_N by hand per N
  (or fix the autotune search to include and select the winner) — but as an
  isolated change, measured against v1, not bundled.
- **Lesson:** change one thing at a time against the v1 baseline. Bundling three
  made the regression un-attributable until each was reasoned back out.

Evidence: `~/gnf4-p2v11/` on the mini (v11_decode.json, v11_prefill.json,
v11_suite.log). No commit of the v1.1 kernel; this note + the v1 test fix are
the only artifacts kept.
