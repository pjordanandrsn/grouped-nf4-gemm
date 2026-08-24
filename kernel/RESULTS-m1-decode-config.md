# RESULTS — K1 M=1 decode config: PARTIAL, baked with full disclosure

Run 2026-08-24 against `PREREG-m1-decode-config.md` (#241), EPYC 9655 +
RTX 5090 (NUMA pre-gate; one earlier box was auto-refused at SSH,
destroyed + verified zero — as was this one after receipts). gnf4 main
`9728fdb` (the sweep measured with the REGISTERED median estimator —
Bugbot #241 caught the mean/median mismatch pre-run). Receipts in
`receipts-m1-config/`; verdict by `k1_verdict.py` (self-tested, 9
branches).

## Verdict table

| gate/bar | registered | measured | result |
|---|---|---|---|
| sweep noise gate | plan drift ≤ 5% | pass both shapes | PASS |
| H-K kernel | winner ≤ 2/3 plan | ratio **0.794** | **PARTIAL** (band (0.667, 0.826]) |
| G0 e2e (g0/gk) | < 7.5% | 0.081% / 0.014% | PASS |
| fidelity | props suite + agreement ≥ 100 | **48/48 numeric props; tokens 127/127** | PASS¹ |
| H-E e2e | graph step ≤ 13.8 ms | **13.46 ms** | PASS |
| GS B=16 | certified band | 130.4 ms | PASS |

¹ The suite's 49th test (`test_plan_thresholds`) asserts the PLAN's own
outputs and saw the env override — an expected test-environment
conflict, not a numerical failure; the bake adds sm_170 census
assertions to that test instead.

**⇒ PARTIAL — baked with full disclosure** (the prereg's registered
consequence for this exact band): the per-shape winners land in
`_decode_plan` behind an `sm_count ≥ 160` + exact-census-shape guard;
everything else keeps the universal constant, asserted by the extended
plan test. Cross-config tokens agreed 127/127 — better than the
investigate floor required.

## Winners and what they say

| shape | plan | winner | per-call |
|---|---|---|---|
| gate_up (1536×2048, T=8) | 64,2,sk4 | **64,2,sk16** | 55.4 → 44.5 µs |
| down (2048×768, T=8) | 64,2,sk4 | **32,2,sk1** | 35.9 → 27.9 µs |

Config space bought **20.6%** of the kernel (4.37 → 3.48 ms/step
equivalent) — under the 33% bar. The two shapes want OPPOSITE
treatments (deeper split vs narrower unsplit tiles), which is exactly
why the universal constant was leaving time on the table, and also why
config knobs alone cannot reach roofline: at 3.48 ms/step the GEMV
still runs at ~18% of the measured 1573.9 GB/s.

## e2e (through the b1d graph loop)

Baseline graph 14.95 → **13.46 ms/step = 74.3 tok/s single-stream**
(the wall gained 1.49 ms — more than the sweep's 0.91 ms/step
prediction; the deeper-split gate_up likely also overlaps better
inside the graph). B=16 untouched (130.4, flag-free path).

## K2 (registered follow-on, bars now fixed by these receipts)

The kernel-body lane: vectorized dual-nibble mainloop (each packed
byte is currently loaded twice, per-element uint8). Baseline for K2 =
this cert's 3.48 ms/step equivalent at ~18% of roofline; K2's prereg
sets its bar from here (floor: 0.65 ms/step). Single-stream ladder:
14.1 → 20.1 → 65.8 → **74.3 tok/s**; remaining rungs to 425: K2 +
elementwise fusion (~250-400 ceiling), then speculative decoding.
