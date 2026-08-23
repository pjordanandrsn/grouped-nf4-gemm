# PREREG — the multi-expert interaction-term hunt

Registered before any measurement. The bounded residual from
[RESULTS-p4-cellmodel.md](RESULTS-p4-cellmodel.md): serving-shape calls
pay ~2.4 µs/row marginal while the committed T-sweep shows ~0.013 µs/row
within a cell — a term that exists only in multi-expert calls, worth
~2 ms of the 22.9 ms B=16 wall. Three candidates were named (per-thread
group-slab interleave, group-strided out-scatter, per-group dispatch).
The committed receipts cannot separate them — decode and slab traffic
both scale as G × N in every existing sweep — so this is a one-box
registered design in which each mechanism owns a distinct axis.

## The discriminating model

Kernel-only grouped NF4 calls (the P4 harness pattern, medians of 50,
32 threads, rows split evenly across G uniques):

```
med(G, rows, N) = F + a·(G·N) + b·(rows·N) + c·G + d·rows
```

* `a·(G·N)` — weight decode: bytes ∝ uniques × columns, rows-independent
  (the confirmed cell model).
* `b·(rows·N)` — activation/slab re-streaming per column tile: hops
  reload slabs whose total traffic is rows × N-tiles, G-independent at
  fixed rows.
* `c·G` — per-group dispatch (bookkeeping per group regardless of size).
* `d·rows` — per-row epilogue (out writes, weight application).

**Design**: full grid G ∈ {1, 2, 4, 8, 16, 32, 64} × rows ∈ {32, 64,
128} × N ∈ {256, 768} at K = 2048 (42 cells), E = 64 arena experts,
fit by least squares. **Holdout**: the serving-shape points (G = 29,
rows ∈ {64, 128}, N = 768) are measured but excluded from the fit.

## Registered claims

* **H1 (identifiability)**: the 5-term fit explains the grid with
  RMS ≤ 8% of the grid's median cell time (else the model family is
  wrong and no attribution is claimed).
* **H2 (prediction)**: the fitted model predicts both held-out G = 29
  serving points within **±10%** (else the fit does not transfer to the
  shape that matters and no attribution is claimed).
* **H3 (attribution)**: passing H1∧H2, the interaction mechanism is
  named by the largest non-decode fitted term at the serving shape
  (b·rows·N vs c·G vs d·rows evaluated at G=29, rows=128, N=768) —
  provided it exceeds the runner-up by ≥ 1.5×; a closer race is
  reported as a mixed attribution, not forced.
* Box gates: AVX-512, triad ≥ 100 GB/s (CPU-only; no GPU gates). The
  bit-exact multi-row test must pass on-box first (correctness voids).
* **Hard stop**: one box, one grid. H1 or H2 failing ⇒ the model family
  is reported refuted with residual structure attached; any next family
  is registered separately.

The named mechanism's FIX (if any) is out of scope — a separate
registration after the attribution.
