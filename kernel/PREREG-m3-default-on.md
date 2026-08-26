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
- **PASS** flips both — and requires Q, Q2 and S all to hold.
- **PARTIAL** flips **at most ONE** knob. This is the important
  wording: "the knobs whose own Q passed" would let a Q2 or S failure
  still ship the composed configuration those bars just refused
  (review, gnf4#284). If the composition is not licensed, only one
  default may move.
- **Tie-break, registered here so it is not chosen after seeing the
  numbers**: when the composition is unlicensed and BOTH knobs pass
  their own Q, flip the one with the **larger measured step cut**.
  Quality has already been established for both at that point, so
  speed is the remaining discriminator.
- **REFUTED** flips nothing and names the bar that failed.

Note what S does and does not gate: it constrains shipping **both**
as the default, not whether a solo knob may flip. An S failure
therefore demotes to the PARTIAL path rather than refuting the cycle
(review, gnf4#284).

## REFUSE gates

- A/A spread ≤ 2% and tokens identical within each arm.
- Anchor: the OFF/OFF arm inside `decode_anchor`'s committed gate
  (`GATE_LO_MS`..`GATE_HI_MS`) — note M2 measured 8.5% inter-box
  dispersion, so this EXCLUDES outliers rather than certifying a
  class.
- Same box, one provisioning, all four arms.
- Identical text digest and token budget across every quality arm.

## AMENDMENT (2026-08-26, after registration, before the box)

**Every arm must carry a mechanism receipt.**

All four arms are selected by env vars, and an env var is a *request*.
`GNF4_GEMV_DOTPAD=1` engages the dot-pad kernel only if the shape is
in `_DOTPAD_CONFIGS` **and** the part carries >= 160 SMs; miss either
and the call quietly takes the certified scalar path. That arm would
then match OFF/OFF in step time and — because K6-B measured dot-pad
**token-identical** at 127 tokens — in perplexity too. This cycle
would read it as a knob that costs nothing and flip a default on an
arm that never ran the mechanism. Recording `os.environ` is no
defence: that records the request again.

So `nf4_grouped.dispatch_counts()` and
`fp8_paged_attn.compute_counts()` are recorded per arm, and
`m3_verdict` REFUSES unless each arm's tally shows exactly the
mechanism it names — and shows a non-zero tally at all, since an
all-zero receipt proves nothing about which path ran.

Why this is admissible after registration: **the gate can only
REFUSE.** It cannot turn a REFUTED into a PASS, cannot move a bar,
and cannot change which knob a PARTIAL names. It can only stop a
verdict from being read off arms that did not exercise the treatment.
Verified by mutation: with the check disabled, an arm whose knob was
silently ignored returns `PASS — flip both`.

The counters and their discrimination tests were validated on real
silicon before the box — an RTX A2000 (26 SMs, below the guard), where
the env var set and the tally showing scalar is the exact failure this
receipt exists to catch.

## Frame note

This cycle cannot raise the ceiling — 159.2 tok/s is already
measured, and K11 closed what lay beyond it. It can only change what
users get without reading documentation, which is the entire point.

## Receipts

`kernel/receipts-m3/` — four A/A pairs, four perplexity receipts with
their shared text digest, divergence logs, box_meta with the anchor
probe. `m3_verdict.py` (self-tested) is committed BEFORE the box.
