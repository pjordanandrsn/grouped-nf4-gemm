# RESULTS — M3: PASS, both defaults may move

Measured 2026-08-26 on one RTX 5090 (machine 34046), one provisioning,
all four arms. Receipts in `receipts-m3/`; the verdict reproduces from
them with `python kernel/m3_verdict.py kernel/receipts-m3/m3_report.json`.

```
M3 VERDICT: PASS  (flip both)
  off     7.843 ms   (baseline)
  dotpad  7.032 ms   ppl -0.0133 OK
  fp8     7.620 ms   ppl -0.0058 OK
  both    6.803 ms   ppl -0.0021 OK
```

## The measurement

| arm | step a / b | A/A | ppl (8192 tok) | Δppl | first div |
|---|---|---|---|---|---|
| off | 7.841 / 7.844 | 0.04% | 8.04843 | — | — |
| dotpad | 7.032 / 7.032 | 0.00% | 8.03510 | −0.0133 | 471 |
| fp8 | 7.619 / 7.621 | 0.03% | 8.04263 | −0.0058 | none |
| both | 6.800 / 6.805 | 0.07% | 8.04630 | −0.0021 | none |

All four quality arms scored the **same 8192 tokens** (`text_sha`
`5e3656e01e50`) through the paged decode path. Anchor: OFF/OFF at
7.843 ms, inside `decode_anchor`'s committed gate [7.004, 7.906].

- **Q** (each knob ≤ OFF + 0.05): −0.0133 and −0.0058. Pass.
- **Q2** (composition ≤ OFF + 0.05): −0.0021. Pass.
- **S** (BOTH at least as fast as either alone, outside noise):
  6.803 vs 7.032 and 7.620, at ≤ 0.07% A/A. Pass.

**Read the deltas as zero, not as improvements.** All four arms sit
within ±0.02 of each other on a perplexity of 8.05. bf16 and e4m3
rounding are unbiased; a negative sign here is noise, not a gain. The
claim this cycle supports is *no measurable quality cost at 8× K8's
horizon* — which is exactly what the bar asked and no more.

## What the longer horizon changed

K8 measured fp8's quality delta as **+0.0092** over 1024 scored
tokens and explicitly declined to rely on it, registering that a
longer window was owed. Over 8192 tokens the delta is **−0.0058** —
the sign flipped. The short-window number was noise around zero, and
the cost does not grow with the horizon. That is the finding M3 was
registered to produce.

## The mechanism receipts

Every arm proves which kernel dispatched, not which env var was set
(AMENDMENT, `nf4_grouped.dispatch_counts` / `fp8_paged_attn.compute_counts`):

| arm | dot-pad | scalar | fp8 | f32 |
|---|---|---|---|---|
| off | 0 | 384 | 0 | 192 |
| dotpad | 384 | 0 | 0 | 192 |
| fp8 | 0 | 384 | 192 | 0 |
| both | 384 | 0 | 192 | 0 |

The counts reconcile: 384 = 48 layers × 2 GEMVs × 4 passes (3 warmup
+ 1 capture), 192 = 48 × 4. In the eager perplexity arms they scale
with steps instead (786432 = 8192 × 96), because eager re-enters
Python every step while graph replays do not.

This mattered. Without it a `dotpad` arm whose knob was silently
ignored — the shape unregistered, or the part below the 160-SM guard —
would have matched OFF in step time *and*, since K6-B measured
dot-pad token-identical at 127 tokens, in perplexity too. Mutation
testing showed that arm returns `PASS — flip both`.

## Divergence: recorded, and it behaves like a threshold

`dotpad` diverges from OFF at step 471 (544 of the remaining 552
tokens differ). `fp8` and `both` are **byte-identical to OFF across
all 1023 tokens** — including `both`, which carries dot-pad's
perturbation.

That is not a contradiction, and the receipts confirm every div arm
carried its knob. Token streams are **argmaxes**, not logits: two
different logit vectors share an argmax unless a near-tie flips. fp8
flips none in 1023 steps. Dot-pad flips exactly one, and that flip is
a knife-edge property of the *f32* hidden state — adding fp8 shifts
the state so the same step is no longer tied.

One flip in ~3069 arm-steps is why this measure is **recorded, not
gated**: it is chaotic and not monotone in perturbation size.
Perplexity over 8192 tokens is the instrument that resolves quality,
and it is the one the bars are written against.

## Scope, stated plainly

- **One box.** M2 measured 8.5% inter-box dispersion, so the absolute
  numbers here are this box's. This box's OFF is 7.84 ms against the
  7.25 of the box that certified dot-pad — 8% slower, near the top of
  the committed gate. Every bar above is a **same-box** comparison,
  which is what M2 said protects a verdict.
- The cuts replicate independently at that offset: dot-pad's certified
  same-box ratio 6.476/7.25 = 0.8932 predicts 7.00 here; measured
  7.03. fp8's certified cut was 0.217 ms; measured 0.22 ms — an
  absolute constant, not a proportional one, as fixed attention work
  per step should be.
- **A note on the anchor gate.** Under K8's old ±3% screen around the
  uncertified 7.39, this box (7.85) would have been destroyed as
  non-compliant. M2 widened the gate precisely because a tight window
  destroys normal boxes for being normal. The gate earned its width.

## What PASS does not license

PASS says the defaults may move on **quality and speed**. It does not
say fp8-COMPUTE is *applicable* everywhere: that path asserts
`sm_89+`, `v_groups == 1`, `k_groups in (1, 2, 4)`, and more, none of
which this cycle varied. Flipping the default unconditionally would
turn a working f32 install into an assertion error on any pre-Ada GPU.

The flip must therefore be **capability-conditional** — default to
fp8 where fp8 can run, f32 otherwise, with an *explicit*
`GNF4_ATTN_COMPUTE=fp8` still failing loudly, because a user who asks
for it by name should be told it is unavailable rather than silently
given something else. That is a productization decision beyond these
bars, recorded here and implemented separately so it is reviewed as
the behaviour change it is.

Dot-pad needs no such guard: its dispatch is already gated on the
shape table and the 160-SM count, so a non-qualifying part silently
takes the certified scalar path — which is exactly the fallback the
mechanism receipt above exists to detect.
