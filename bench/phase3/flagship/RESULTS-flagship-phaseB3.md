# Flagship B3 — VERDICT: NOT CONFIRMED (E2/E3); the predictor is validated at 93% agreement — the cost is in the plumbing, and it's identified

**2026-07-14 · Protocol:** `kernel/prereg_flagship_phaseB3.json` (OTS pre-data)
· **Frozen:** `5f17cd4` · **Host:** RunPod SECURE H100 (fresh pod; on-box
link **55.1 GB/s** → waterfall **6.90 tok/s** — note this pod's link is 21%
faster than B2's, which moves every ratio). Smoke-at-4-layers passed before
the full build (the B2 rebuild-cycle lesson, applied — zero wasted builds).

## The design under test

The moment layer L's MoE lands, run **layer L+1's real router weights on
L+1's pre-attention hidden** and stream that prediction (overlapping L+1's
attention); the true post-attention router adjudicates, mispredictions are
fetched on a **priority stream** (the owned B2 queueing flaw, fixed).

## Results

| prompt | off → early tok/s | paired | agreement A | identical |
|---|---|---|---|---|
| MoE-vs-dense | 4.33 → 3.93 | 0.907× | **0.926** | True |
| haiku | 4.33 → 3.78 | 0.872× | **0.931** | True |
| quantization | 4.30 → 4.59 | 1.066× | **0.930** | True |

| criterion | outcome |
|---|---|
| E1 identity | **PASS** 3/3 |
| E2 paired ≥ 1.05× every prompt | **FAIL** (1 of 3) |
| E3 ≥ 0.80× waterfall | **FAIL** (0.55–0.66×) |
| E4 VRAM ≤ 20 GB | **PASS** (15.2) |

## The science: the predictor is right

**A production 235B router's top-8 choice is 92.6–93.1% predictable from
the pre-attention residual stream** — attention barely moves the routing
decision at decode. Against B2's 44% last-token stickiness, this is the
predictor early-fetch needs: bytes overhead is only (2−A) ≈ **1.07×**, an
order less waste than speculation's 1.56×.

## Why it still lost — the plumbing, not the predictor

With A = 0.93 the bytes argument permits a win; two prompts lost anyway,
and one won. The identified suspect (stated as the leading hypothesis, with
the reasoning): the harness's `route()` ends in `topk(...).tolist()` — **a
GPU→CPU synchronization** — and early routing adds one per layer **on the
critical path** (94 extra syncs/token, each stalling the main stream while
copies are mid-flight). That sync overhead is of the same order as the
overlap the design buys, so the outcome degenerates to which side of noise
a given prompt's schedule lands on (0.87–1.07× observed). Secondary
context: on this pod's faster link (55 GB/s) the copy is only ~145 ms/token,
so fixed compute/sync costs weigh relatively more — the off-baseline itself
reads 0.63× of THIS pod's ceiling (vs 0.76× on B2's slower-linked pod);
the router-serialization tax grows as links get faster.

Per the no-tune clause: no re-run. The fix direction is registered for a
future B4, not improvised tonight: **keep expert ids GPU-resident
end-to-end** (or pipeline the D2H ids transfer one layer ahead), so early
routing costs zero main-stream syncs. The identity machinery and the
93%-agreement measurement carry forward.

## Standing state after B2 + B3

The shipped configuration remains **prefetch OFF**: 4.30–4.33 tok/s
real-checkpoint generation (replicated across three pods now), VRAM
15.2 GB. Two prefetch designs have been registered, run blind, and
reported: speculation fails on hit rate (44%), early routing validates its
predictor (93%) but pays its win back in synchronization. The remaining gap
to the waterfall is now precisely characterized: router serialization +
per-layer sync costs, with a GPU-resident-ids implementation as the
identified path.

## Evidence / teardown

`phaseB3.json`, `phaseB3_smoke.json`, `phaseB3.log`. Pod DELETE →
404-verified, 0 pods.
