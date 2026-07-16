# Intel notes — PR #1984 (zaid646, "Experts4bit: fused MoE weights in 4-bit NF4/FP4")

Captured 2026-07-16 (private-lineage). Primary source, not testimony.

## Technical read (for the record; do NOT post)
- zaid646's implementation is **storage + an on-the-fly dequant loop — NO fused kernel.**
  His weights are stored 4-bit and "dequantized on-the-fly during the forward pass"
  (his words, PR body). That is the exact two-pass path grouped-nf4-gemm's fused
  single-launch kernel replaces.
- His **~5000 tok/s** is a module-level dequant-loop microbench, **not comparable**
  to grouped-nf4-gemm decode-bs1 numbers (different measurement boundary: his is a
  standalone forward-pass loop; ours is op-boundary decode vs the dequant→GEMM path
  under blind confirmatories).
- His **RTX 3090 + NF4 error figures (MAE 0.073 / RMSE 0.092)** are candidate FLEET
  contributions — usable only once he appears on **#1965** (the implementation of
  record). Not to be cited as ours; not to be solicited.

## Disposition
- #1984 CLOSED as `Duplicate` of #1965 by the maintainer (matthewdouglas).
- No agent comments on #1984 (standing rail). Operator's reply is operator-posted.
