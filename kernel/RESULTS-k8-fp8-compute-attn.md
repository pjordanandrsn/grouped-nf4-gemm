# RESULTS — K8: fp8-COMPUTE attention PASSES. 159.2 tok/s certified.

## The quality debt is paid; the 250 frame still does not close.

Measured 2026-08-26 under PREREG-k8-fp8-compute-attn (+ its
decode-path amendment). Receipts in `receipts-k8/` (RTX 5090, anchor
7.26 ms, all arms knob-ON; instance destroyed, vast zero by-id and
list).

```
K8 VERDICT: PASS  (speed+quality)
  step: f32 6.498 -> fp8 6.281 ms = +0.217 ms (PASS >= 0.15, PARTIAL >= 0.05)
  quality: ppl 4.9425 -> 4.9517 = +0.0092 vs eps 0.05 -> OK
```

## Every gate passed

| gate | result |
|---|---|
| G1 A/A, f32 pair | 6.498 / 6.500 ms, token streams identical |
| G1 A/A, fp8 pair | 6.281 / 6.281 ms, token streams identical |
| G2 anchor health | f32 arm 0.3% off the certified 6.476 knob point |
| G3 same-box | one box, one provisioning, both arms |
| G4 error bound ON THIS BOX | mean 0.00445, p99 0.01953, max 0.10547 (bars 5e-3 / 5e-2 / 2e-1) |
| G5 budget + text parity | 1024 tokens both, text sha `456e4ede9b4b` both |

## The result

**0.217 ms off the step at a perplexity cost of +0.0092 (+0.19%).**
The lane SV2 priced at 0.15–0.2 ms delivered **0.217 ms** —
marginally ABOVE its estimate, which matters for how much the
remaining estimates should be trusted (see below).

Single-stream ladder, all quality-certified:

| configuration | ms/step | tok/s |
|---|---|---|
| certified default | 7.35–7.39 | ~136 |
| `GNF4_GEMV_DOTPAD=1` (SV1) | 6.476 | 154.4 |
| **+ `GNF4_ATTN_COMPUTE=fp8` (K8)** | **6.281** | **159.2** |

## Disclosure: the streams were identical, and that is not the gate

The prereg deliberately refused a token-identity bar — this path's
p99 element error is ~5e-2 by mechanism, so identity could not be
guaranteed and a borrowed identity gate would have been
unsatisfiable by construction. **It came out identical anyway: all
127 greedy tokens matched between the f32 and fp8 arms.**

That is reported as disclosure, not as vindication of a gate we did
not set. One 127-token window at one prompt offset does not
establish identity in general, the mechanism still says divergence is
permitted, and the perplexity delta (+0.0092 over 1024 scored
tokens) is the evidence the verdict actually rests on.

## The instrument nearly certified nothing at all

The first attempt's two quality arms **ran, exited 0, and wrote
plausible JSON while scoring nothing**. The scoring branch lives
inside `_b1d_stage_a`, which only executes under `--b1d-loop`; the
arms passed `--ppl-steps` alone and fell through to an ordinary
decode. Nothing errored. The only thing that caught it was the
compose step asserting on `attn_compute` — a key the ppl receipt
uniquely carries — and a looser assertion would have produced a
verdict from two arms' worth of nothing.

`--ppl-steps` without `--b1d-loop` now REFUSES (e4b#279), and the
re-run verified the refusal on the same box before scoring anything
(`k8_guard.log`). This is the second time in this cycle that the
registered quality question was nearly answered by a measurement
that could not have failed: the prereg amendment already recorded
that a teacher-forced forward with `use_cache=False` never calls the
paged decode kernel at all.

## What this does to the 250 frame

Two of SV2's four lanes are now measured:

| lane | SV2 framed | measured |
|---|---|---|
| MoE GEMV round 2 | ~1.5–1.8 ms | **0.24 ms** (K7, REFUTED — ~7x short) |
| fp8-COMPUTE attention | 0.15–0.2 ms | **0.217 ms** (K8, PASS — slightly over) |
| attn-proj GEMV | 0.22 ms | unmeasured (already 1.14x its floor) |
| fusion/norm/router residue | 0.7–0.8 ms | unmeasured, unregistered |
| **pool** | **~2.5–2.7 ms** | **~1.38–1.48 ms** |

Against the registered 2.48 ms bar, **250-by-composition still does
not clear**. But the two measurements disagree in an informative way:
the frame was not uniformly optimistic — K8's lane came in slightly
ABOVE its estimate while K7's came in 7x below. The dominant term was
the wrong one, not every term.

Honest reading: reaching 250 now requires the fusion residue
(0.7–0.8 ms framed, unmeasured) to hold AND roughly 1 ms to come
from somewhere not yet registered. The MoE GEMV's 3.8x-floor gap
remains the largest single opportunity on the device and remains
unexplained — K7 refuted one hypothesis about its cause, not the gap.

## Ship posture

`GNF4_ATTN_COMPUTE=fp8` ships **OFF by default** with its receipts,
the K6-B posture. PASS licenses registering a default flip, not the
flip itself: that additionally needs a composed cert on a fresh box
and a longer-horizon quality window than 1024 scored tokens. An
unset env remains byte-identical to the certified path, and an
unrecognised value refuses rather than silently running f32.
