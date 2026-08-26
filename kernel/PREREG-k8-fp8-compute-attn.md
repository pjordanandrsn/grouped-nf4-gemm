# PREREG — K8: fp8-COMPUTE attention, the quality debt called in

Registered 2026-08-26, before measurement. Calls in a debt the code
states in its own docstring: `compute="fp8"` is built and sm_89+
gated, and *"fp8 COMPUTE quality is owed separately if it becomes the
default — the G7 oracle certified storage."* RESULTS-sv2 (e4b #277)
registered this slice as one of the four treatments on the 250 tok/s
composition route: `_fp8_paged_decode_split` measures **0.520 ms of
the 6.46 ms step**, and the SV2 frame priced this lane at ~0.15–0.2 ms
realistic.

## What is ALREADY certified (cited, not re-derived)

- **Structural correctness** across permuted tables, ragged tails,
  grouped K scales, small-GQA padding, and the k_groups refusal —
  `test_fp8_paged_attn.py`, every shape run through the `f8dot` and
  `pf8` modes at the measured 1.5e-1 worst-element envelope.
- **Serving-shape error distribution**, CI-gated
  (`test_f8dot_error_is_bounded_and_reported`, lengths
  512/731/288/512, k_groups=4): mean < 5e-3, p99 < 5e-2, max < 2e-1.
- **Mechanism**: q and p each pay one e4m3 rounding; p dominates and
  is worst when one or two tokens carry the softmax mass — probed
  exact at T=1, 0.087 worst-element at T=2, 0.034 at T=33, shrinking
  as contributions average. Decode over hundreds of KV tokens is the
  averaging regime, which is why a serving cert is the missing piece
  rather than a tighter tensor bound.

## What is OWED, and the frame for it

Tensor tolerance is not the open question; **model behaviour** is.
Two arms on ONE box, same prompt, same greedy decode, differing only
in the attention compute mode.

### Quality (the ship gate)

**Identity is deliberately NOT the bar.** Unlike dot-pad — whose
bf16-MMA rounding left 127/127 tokens identical — this path's p99
element error is ~5e-2, and a borrowed identity gate would be
unsatisfiable by construction
([[correctness-bars-derive-from-the-mechanism]]: a style-borrowed
bar once rejected 72 configs whose delta was inherent rounding). The
mechanism-appropriate question is whether lossier attention makes the
model worse:

- **Q1 (BAR)**: held-out perplexity under `compute="fp8"` must be
  **<= f32 + 0.05** on the same text and token budget — the epsilon
  TR2 used for the training-quality gate, borrowed because it is the
  same question (does the cheaper path degrade modelling).
- **Q2 (RECORDED, not a bar)**: first-divergence step and top-1
  agreement rate over 127 greedy steps. Divergence is EXPECTED here;
  it is disclosure, not a gate. A trace that degenerates
  ([[check-traces-for-degeneracy]]: 8-gram / distinct-token laws)
  is excluded from Q2 and the exclusion is reported.

### Speed (the SV2 composition input)

Step-level delta on the certified knob point, both arms graph-replayed
on the same box:

- **PASS**: step cut **>= 0.15 ms** AND Q1 passes — the low end of the
  SV2 frame's price for this lane; it stays in the 250 pool.
- **PARTIAL**: step cut **>= 0.05 ms** AND Q1 passes — ships as a knob,
  but the RESULTS must recompute the SV2 composition sum with the
  measured figure rather than the frame's estimate.
- **REFUTED**: cut < 0.05 ms, OR Q1 fails. Either way the lane is dead
  as a 250 lever and SV2's addressable pool loses this 0.15–0.2 ms;
  the RESULTS says so instead of retaining the estimate.

## REFUSE gates (checked before any number is read)

- **G1 A/A**: each arm timed twice; spread <= 2% and token streams
  identical within an arm (per-arm determinism).
- **G2 anchor health**: the f32 arm's step within +/-5% of the
  certified knob point for its configuration; outside it, the
  ms-denominated bars do not transfer.
- **G3 same-box**: both arms from one box, one provisioning. Cross-box
  step deltas are not adjudicated.
- **G4 the CI error bound passes ON THE BOX**
  (`test_f8dot_error_is_bounded_and_reported`) — the tensor-level
  guarantee this frame cites must hold on the silicon being measured,
  not merely in CI ([[verify-the-instrument-claim]]).
- **G5 budget parity**: both arms evaluate the SAME token budget and
  the same held-out text; a mismatch refuses rather than being
  normalised away.

## Implementation in scope

`compute=` already reaches the kernel through the e4b shim's `**kw`,
but nothing selects it. In scope: an env-gated selector
(`GNF4_ATTN_COMPUTE=fp8`, default `f32` — an unset env must leave the
certified path byte-identical) plus the arms. NOT in scope: any
change to the fp8 kernels themselves; this cycle certifies what is
built.

## Ship gate

PASS or PARTIAL ships the knob OFF-by-default with its receipts. A
default flip is a SEPARATE registration and additionally requires the
composed step cert on a fresh box — the K6-B posture, unchanged.

## Receipts

`kernel/receipts-k8/` — both arms' step receipts and token logs, the
perplexity pair, the on-box error-bound output, and box_meta with the
anchor probe. `k8_verdict.py` (self-tested refusal directions) is
committed BEFORE the box cycle.
