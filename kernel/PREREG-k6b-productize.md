# PREREG — K6-B: productize the dot-pad GEMV (opt-in knob)

Registered 2026-08-25, licensed by AMENDMENT-k6-frame (PARTIAL on both
boxes: pair ratios 0.589 and 0.668, Stage A numerics certified under
the relative gate). Before any measurement of this stage.

## What ships in this PR (product change, default OFF)

- `_gemv_nf4_dotpad` — the bench kernel moved verbatim into
  `nf4_grouped.py` (wide uint32 loads; dequant identical to the wide
  scalar path; `tl.dot` reduce with x in M-row 0).
- Routing knob: `GNF4_GEMV_DOTPAD=1` routes the decode GEMV through
  dot-pad at the re-gate winners' configs — gate_up-shaped cells
  (N=1536, K=2048): bn=16/w2/s2; down-shaped (N=2048, K=768):
  bn=16/w4/s2 — gated exactly like K1's baked winners (sm_count ≥ 160,
  exact census shapes; anything else falls through to the certified
  scalar path). **Default OFF until the Stage A gates below pass.**

## Why token IDENTITY is not the gate here (registered up front)

Every previous shipped default (b1c, b1d, K1 configs, B1, B2) was
bitwise- or token-identical by construction. K6-B is the campaign's
first numerics-CHANGING candidate: bf16-input MMA differs from the
fp32-scalar chain by ~2⁻⁸ relative per GEMV (Stage A receipts:
max|Δ|=0.5 on refs ~85–147, argmax 99.4% per call). Demanding exact
greedy identity would refuse the mechanism for being what it is. The
K1-class P-fid framing applies instead.

## Stage A gates (one box, e4b harness, no new e4b code)

Two b1d graph runs, knob OFF then ON, same prompt/seed (the graph arm
already captures tokens):

1. **Step gain, gain-frame** (bars-follow-the-claim): step ratio
   ON/OFF ≤ **0.85** = PASS (≥15% step cut; the receipts' 33–41% GEMV
   cut predicts ~0.80–0.84); 0.85–0.95 PARTIAL under A/A; > 0.95
   REFUTED (the kernel's µs win does not survive the full step).
2. **Quality**: first token divergence at step ≥ **32** of 128 (a
   2⁻⁸-per-GEMV perturbation that flips the argmax inside 32 steps is
   too hot to ship), AND agreement-to-first-divergence reported, AND
   the OFF-arm trace non-degenerate (check-traces law).
3. A/A on the OFF arm (frame-relative margin as amended everywhere).
4. If PASS+quality: the knob's default flips ON in the RESULTS PR
   with the receipts, disclosure of the numerics change in the
   README's fidelity section, and the scalar path retained as
   `GNF4_GEMV_DOTPAD=0` rollback. PARTIAL ships the knob OFF with the
   receipts (available, not default).

## Verdict calculator

`k6b_verdict.py`, self-tested both directions; receipts in
`kernel/receipts-k6b/`.
