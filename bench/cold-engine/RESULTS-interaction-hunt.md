# RESULTS — the interaction-term hunt: MODEL-REFUTED — and the residuals name the missing physics

Registered in [PREREG-interaction-hunt.md](PREREG-interaction-hunt.md)
(#234; bars frozen first). Run 2026-08-23 on an EPYC 9654 (triad
155.6 GB/s), bit-exact test passing on-box before any timing. Receipt:
[p4-receipts/ixhunt-2026-08-23.json](p4-receipts/ixhunt-2026-08-23.json).

**Scored verdict: MODEL-REFUTED** — H1 failed (fit RMS 31.6% of the
median cell against the 8% bar) and H2 failed (holdout G=29/rows=64
missed at −13.8% against ±10%; rows=128 hit at −3.4%). Per the
registered rule, no attribution is claimed; the residual structure is
the deliverable.

## The residual structure: an occupancy term the family cannot express

The worst residuals concentrate exactly where **work items = G × tiles**
starves the 32-thread pool:

| cell | med | model err |
|---|---|---|
| G=1, rows=128, N=256 (8 items) | 801.9 µs | −45% |
| G=4, rows=128, N=256 (32 items) | 240.8 µs | +79% |
| G=64, rows=64, N=256 (512 items) | 165.4 µs | −49% |

And the headline pathology, plain in the raw slice (rows=128, N=768):
**G=1 takes 820.7 µs while G=64 takes 733.7 µs with 64× the weight
bytes.** One expert × 24 column tiles = 24 work items for 32 threads,
each item serially walking 128 rows — the call is parallelism-starved,
not byte-bound. A linear family `F + a·GN + b·rowsN + c·G + d·rows` has
no `work / min(G·tiles, threads)` term; the fit's negative dispatch
coefficient (c_G = −4.1 µs) is that hole soaking up starvation, and the
31.6% RMS is the refutation the prereg anticipated.

## What stands after the refutation

* **The serving shape is well-fed** (G=29 × 24 tiles = 696 items ≫ 32
  threads), so occupancy starvation is NOT the serving-regime mechanism —
  the hunt eliminates it for the 2.4 µs/row question even as it
  dominates the grid. The serving residual remains bounded and
  unattributed; on this host it measures 4.1 µs/row (446.1 → 709.6 µs
  for 64 → 128 rows at G=29), host-dependent as law 7 predicts.
* **A new engine-relevant fact, free of charge**: single-expert
  large-rows calls (the decode-adjacent shape G ≈ 1–4 at small N) run
  parallelism-starved in the current (group × tile) item decomposition —
  the item grid should split rows when G × tiles < threads. That is a
  concrete kernel improvement with its own falsifiable prediction
  (G=1/rows=128/N=256 should drop from ~800 µs toward ~work/32), and per
  the standing discipline it gets its own registration or none.
* **The next model family, if pursued**: occupancy-aware —
  `med = max(serial_item_work · items / min(items, threads)) + …` — fits
  only well-fed cells or models the min() explicitly. Registered
  separately or not at all (the hard stop).

One box, one grid, ~$0.12; box destroyed; zero instances; program total
~$1.81.
