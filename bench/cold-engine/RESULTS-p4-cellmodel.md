# The partial-cell offline test: hypothesis REFUTED for the serving regime — and the cell model, confirmed where it lives, hands over a better lever

The registered next hypothesis from [RESULTS-p4-rowblock.md](RESULTS-p4-rowblock.md)
("partial register-cells… testable offline against existing receipts
before any box"), tested offline as registered. Instrument:
[p4_cellmodel_fit.py](p4_cellmodel_fit.py); receipts:
[p4-receipts/cellmodel_fit.json](p4-receipts/cellmodel_fit.json), the
committed e4b `b16close/rows_curve.json` points, and this repo's 9V74
A/B/A receipts. No box.

## The cell model is real — 5× better fit, kinks exactly at boundaries

On the single-expert T-sweep (cell path, T ≥ 2):

| model | RMS | max residual |
|---|---|---|
| `c0 + c1·ceil(T/8) + c2·rows` | **21.6 µs** | 38 µs |
| `a0 + a1·rows` (the null) | 107.3 µs | 137 µs |

The null's residuals flip sign exactly at the 8/16/32 cell boundaries.
Fitted constants: **~58 µs of decode per expert-cell** (c1/8 experts) and
**c2 ≈ 0.013 µs/row within a cell — extra rows inside a cell are free.**
(The b16close "flat to T=32" law and its later refutation both live here:
flat *within* cells, stepped *between* them.)

## But the serving regime never leaves the first cell — so partial cells cannot be its penalty

Serving calls run T̄ ≈ 2.2–4.4 rows per unique (64–128 rows over 29
uniques): `ceil(T/8) = 1` on both sides of the A/B — zero cell-count
difference — yet the measured marginal cost is **2.37 µs/row** (9V74:
290.9 → 442.6 µs for +64 rows), against the sweep's ~0.01. **The
hypothesis is refuted**: the rows-penalty is a **multi-expert call
interaction** — ~2.4 µs/row of cost that exists only when many groups
share the call — with the narrowed candidates being per-thread group
interleave (activation-slab reloads as work items alternate groups),
out-scatter across group-strided destinations, and per-group dispatch
inside the call. Each is now bounded by the measured 2.37 µs/row × rows.

## The constructive lever the confirmed model hands over

Call cost ≈ fixed + **58 µs × uniques** + (multi-expert interaction) ×
rows. Uniques, not rows, carry the decode bill — so **concentrating rows
per unique expert** (solver/routing side, e4b) attacks the B=16 wall
directly: 128 rows over ~15 uniques instead of 29 removes ~0.8 ms of
decode per call at zero kernel changes, and within-cell rows stay free up
to T = 8. This is an e4b placement-objective statement with kernel-model
receipts behind it — registered here as the surviving P4-line lever,
alongside the unexplained-but-bounded 2.37 µs/row interaction term.

Per the standing discipline: the multi-expert interaction hypothesis, if
pursued, gets its own registration; the router-concentration lever
belongs to e4b's solver constants (the same `cpu_us_fixed`/`per_row`
family the campaign already tunes) and should enter as a solver-objective
experiment there.
