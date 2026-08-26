# PREREG — M3: flip the two certified knobs ON by default

Registered 2026-08-26, before measurement. Both knobs are measured,
certified, and shipping **OFF**, so the default install runs
**7.37 ms (130–142 tok/s)** while **6.281 ms (159.2 tok/s)** is proven
and one env var away. This cycle asks whether the defaults may move —
not whether the speed is real, which is already settled.

| knob | certified by | measured |
|---|---|---|
| `GNF4_GEMV_DOTPAD=1` | K6-B PARTIAL, SV1 composed | 7.25 → 6.476 ms, 127/127 tokens identical |
| `GNF4_ATTN_COMPUTE=fp8` | K8 PASS | 6.498 → 6.281 ms, ppl +0.0092 |

Neither prereg licensed a flip. K6-B shipped OFF pending a P-fid
stage; K8 shipped OFF pending "a longer-horizon quality window than
1024 scored tokens". This is that registration.

## What is NOT in question

The speed. Both cuts are same-box A/A-clean measurements with
committed receipts. M3 measures **quality at a longer horizon**, and
nothing else.

## The horizon problem, stated plainly

Every quality receipt so far is short: K6-B's identity ran 127 greedy
tokens; K8's perplexity scored 1024. A default is what every user
runs on every prompt, so the question is whether the divergence these
paths permit stays bounded when the window grows — not whether it was
invisible at 127 steps.

Both knobs are numerics-changing, and differently:
- dot-pad rounds both MMA operands to bf16 (~2^-8 relative), and came
  out token-identical at 127.
- fp8-COMPUTE pays an e4m3 rounding on q and on p (p99 ~5e-2), and
  also came out token-identical at 127 — which the K8 RESULTS
  explicitly declined to rely on.

## Arms (one box, all four combinations)

`OFF/OFF`, `DOTPAD`, `FP8`, `BOTH` — each an A/A pair, plus a
long-horizon quality window per arm:

- **perplexity over 8192 teacher-forced decode tokens** (8× K8's
  window), through the paged decode path, same text and budget in
  every arm (the K8 instrument and its amendment).
- **first-divergence step** against OFF/OFF over a 1024-token greedy
  continuation — RECORDED, not gated, for the same reason K8 gave.

## Bars

- **Q (BAR, per knob and for BOTH)**: perplexity ≤ OFF/OFF + **0.05**
  — the epsilon TR2 and K8 both used, unchanged so this cycle is
  comparable to them.
- **Q2 (BAR, composition)**: `BOTH` must be ≤ OFF/OFF + 0.05 as well.
  Two individually-passing knobs are not licensed to compose; their
  errors could add.
- **S (BAR)**: `BOTH` must be at least as fast as either knob alone,
  outside A/A noise. A default that is slower than its parts is not a
  default.
- **PASS** flips both. **PARTIAL** flips only the knob(s) whose own Q
  passes when the other is off. **REFUTED** flips nothing and says
  which bar failed.

## REFUSE gates

- A/A spread ≤ 2% and tokens identical within each arm.
- Anchor: the OFF/OFF arm inside `decode_anchor`'s committed gate
  (`GATE_LO_MS`..`GATE_HI_MS`) — note M2 measured 8.5% inter-box
  dispersion, so this EXCLUDES outliers rather than certifying a
  class.
- Same box, one provisioning, all four arms.
- Identical text digest and token budget across every quality arm.

## Frame note

This cycle cannot raise the ceiling — 159.2 tok/s is already
measured, and K11 closed what lay beyond it. It can only change what
users get without reading documentation, which is the entire point.

## Receipts

`kernel/receipts-m3/` — four A/A pairs, four perplexity receipts with
their shared text digest, divergence logs, box_meta with the anchor
probe. `m3_verdict.py` (self-tested) is committed BEFORE the box.
